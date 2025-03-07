# Import necessary libraries
import os
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import torch
import yaml
from pprint import pprint
from utils import(
    set_seed,
    print_trainable_params_info,
    apply_chat_template,
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":

    # Set up logging and tracking
    wandb.login()
    
    # get training configuration
    with open('training_config.yaml') as file:
        config = yaml.safe_load(file)
    
    print('-'*50)
    print("Training configuration:")
    pprint(config)
    print('-'*50)
    
    MODELS_DICT = config['MODELS_DICT']
    
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    
    # Training hyperparameters
    num_train_epochs = config['hyperparameters']['num_train_epochs']
    lr = config['hyperparameters']['lr']
    batch_size = config['hyperparameters']['batch_size']
    gradient_accumulation_steps = config['hyperparameters']['gradient_accumulation_steps']
    max_grad_norm = config['hyperparameters']['max_grad_norm']
    warmup_steps = config['hyperparameters']['warmup_steps']
    warmup_ratio = config['hyperparameters']['warmup_ratio']
    MAX_LEN = config['hyperparameters']['MAX_LEN']
    
    # Logging and saving
    logging_steps = config['hyperparameters']['logging_steps']
    save_steps = config['hyperparameters']['save_steps']
    eval_steps = config['hyperparameters']['eval_steps']

    # Training data path
    TRAIN_DATA_PATH = config['DATASET_PATH']
    TASKS = config['TASKS']

    # base model path
    BASE_MODEL = config['BASE_MODEL']
    MODEL_PATH = MODELS_DICT[BASE_MODEL]['MODEL_PATH']
    IS_CAUSAL_LM = MODELS_DICT[BASE_MODEL]['CAUSAL_LM']
    IS_SFT_TRAINING = MODELS_DICT[BASE_MODEL]['SFT_TRAINING']
    FP16_TRAINING = config['FP16_TRAINING']
    
    # max training samples
    MAX_TRAINING_SAMPLES = config['MAX_TRAINING_SAMPLES']
    
    if FP16_TRAINING:
        torch_dtype=torch.bfloat16 # bfloat16 has better precission than float16 thanks to bigger mantissa. Though not available with all GPUs architecture.
    else:
        torch_dtype=torch.float32
    
    # set seed
    SEED = config['SEED']
    set_seed(SEED)
       
    # Load tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = 'left'
    
    if config['hyperparameters']['USE_LORA']:
        # Apply LoRA
        print(f"[INFO] Training with LoRA")
        
        # Define LoRA configuration
        lora_config = LoraConfig(
            r=config['hyperparameters']['lora_r'],
            lora_alpha=config['hyperparameters']['lora_alpha'],
            lora_dropout=config['hyperparameters']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM" if IS_CAUSAL_LM else "SEQ_2_SEQ_LM",  # Adjust for your task
            target_modules=config['hyperparameters']['target_modules'],  # Specify target modules if required
        )
        
        # Wrap the model with LoRA
        model = get_peft_model(model, lora_config)

        # Log trainable parameters for verification
        print_trainable_params_info(model)

        print('-'*50)
    
    # Set reasonable default for models without max length
    tokenizer.model_max_length = config['hyperparameters']['MAX_LEN']

    # Set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print(f'[INFO] Model and Tokenizer loaded: {MODEL_PATH}, version: {BASE_MODEL}, IS_SFT_TRAINING: {IS_SFT_TRAINING}, FP16_TRAINING: {FP16_TRAINING}')
    print('-'*50)
    
    # Project name for loggings and savings
    project_name = config["PROJECT_NAME"]
    fp16 = '-FP16' if FP16_TRAINING else ''
    sft = '-SFT' if IS_SFT_TRAINING else ''
    # LoRA params
    # lora_training = f'-lo' if config['hyperparameters']['USE_LORA'] else ''
    lora_r = f'-r-{config['hyperparameters']['lora_r']}' if config['hyperparameters']['USE_LORA'] else ''
    lora_alpha = f'-a-{config['hyperparameters']['lora_alpha']}' if config['hyperparameters']['USE_LORA'] else ''
    lora_dropout = f'-d-{config['hyperparameters']['lora_dropout']}' if config['hyperparameters']['USE_LORA'] else ''
    
    run_name = f'{MODEL_PATH.split("/")[-1]}-bs-{batch_size}-lr-{lr}-ep-{num_train_epochs}-wp-{warmup_ratio}-gacc-{gradient_accumulation_steps}-gnm-{max_grad_norm}{fp16}{sft}-mx-{config['hyperparameters']['MAX_LEN']}{lora_r}{lora_alpha}-v5'
    assert '--' not in run_name, f"[WARN] Detected -- in run_name. This will cause a push_to_hub error! Found run_name={run_name} "
    assert len(run_name) < 96, f"[WARN] run_name too long, found len(run_name)={len(run_name)} > 96. This will cause a push_to_hub error! Consider squeezing it. Found run_name={run_name}"

    # Where to save the model
    #MODEL_RUN_SAVE_PATH = f"BounharAbdelaziz/{run_name}"
    MODEL_RUN_SAVE_PATH = f"ybendou/{run_name}"
    
    # Save the configuration to a .txt file
    output_filename = f"./run_configs/{run_name}.txt"
    with open(output_filename, 'w') as output_file:
        for key, value in config.items():
            output_file.write(f"{key}: {value}\n")

    print(f"Configuration saved to {output_filename}")
    
    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged, all runs will be under this project
        project=project_name,   
        # Group runs by model size
        group=MODEL_PATH,       
        # Unique run name
        name=run_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            # "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # "weight_decay": weight_decay,
            "dataset": TRAIN_DATA_PATH,
            "tasks": TASKS,
            "chat_template": DEFAULT_CHAT_TEMPLATE,
            "max_context_length": config['hyperparameters']['MAX_LEN'],
        }
    )

    # Set chat template
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    dataset = get_datasets(TRAIN_DATA_PATH, TASKS)  # Replace with your dataset path
    # Transform the dataset into a conversational format
    dataset = dataset.map(
        apply_chat_template,
        num_proc=os.cpu_count(),
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["messages"],
        desc="Applying chat template..."
    )

    # Create the splits
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_RUN_SAVE_PATH,
        evaluation_strategy="steps",
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
        bf16=config['FP16_TRAINING'],
        fp16_full_eval=config['FP16_TRAINING'],
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        report_to="wandb",
        push_to_hub=False,
        metric_for_best_model=config['METRIC_FOR_BEST_MODEL'],
        gradient_checkpointing=True,
        # use_cache = False, # as we set gradient_checkpointing=True
        load_best_model_at_end=True,
        optim=config['hyperparameters']['optimizer'],
        gradient_checkpointing_kwargs={"use_reentrant": False} if config['hyperparameters']['USE_LORA'] else None,  # Avoids gradient issues in backprop when LoRA is set to True. # https://discuss.huggingface.co/t/how-to-combine-lora-and-gradient-checkpointing-in-whisper/50629
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        dataset_text_field="text",
        # compute_metrics=lambda x : compute_metrics_causal_lm(x, tokenizer),
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics, # avoids OOM in eval
    )
    

    # Train the model
    trainer.train()

    # Push to Hugging Face Hub
    print("[INFO] Preparing to push to hub...")

    if config['hyperparameters']['USE_LORA']:
        print("[INFO] Merging LoRA weights before pushing...")
        from peft import merge_and_unload
        model = merge_and_unload(model)
        
    # Save the model and tokenizer locally before pushing
    trainer.save_model(MODEL_RUN_SAVE_PATH)  # This saves the model, tokenizer, and config
    tokenizer.save_pretrained(MODEL_RUN_SAVE_PATH)

    # Push to the hub
    print("[INFO] Pushing model and tokenizer to Hugging Face Hub...")
    trainer.push_to_hub()
    tokenizer.push_to_hub(MODEL_RUN_SAVE_PATH)
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print(f'[INFO] Results on test set: {test_results}')