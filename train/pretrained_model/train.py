# Import necessary libraries
import numpy as np
import wandb
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from evaluate import load
from nltk.tokenize import RegexpTokenizer
import torch
import yaml
from pprint import pprint

# Load metrics
rouge_metric = load("rouge")
bleu_metric = load("bleu")

# Metrics computation function

def compute_metrics_causal_lm(eval_pred):
    preds, labels = eval_pred
    
    # If preds is a tuple of logits, extract token IDs
    if isinstance(preds, tuple):
        preds = preds[0]  # Assuming the first element is logits
    if len(preds.shape) == 3:  # If preds is (batch_size, seq_length, vocab_size)
        preds = np.argmax(preds, axis=-1)  # Convert logits to token IDs
    
    # Ensure labels are not masked
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Post-process text for ROUGE and BLEU
    text_preds = [(p if p.endswith(("!", "؟", "۔")) else p + "۔") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "؟", "۔")) else l + "۔") for l in text_labels]
    
    sent_tokenizer = RegexpTokenizer(u'[^!؟۔]*[!؟۔]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer.tokenize(l))) for l in text_labels]
    
    # Compute metrics
    rouge_results = rouge_metric.compute(predictions=text_preds, references=text_labels)
    bleu_results = bleu_metric.compute(predictions=text_preds, references=text_labels)
    
    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleu": bleu_results["bleu"]
    }
    
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process text for ROUGE and BLEU
    text_preds = [(p if p.endswith(("!", "؟", "。")) else p + "۔") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "؟", "。")) else l + "۔") for l in text_labels]

    sent_tokenizer = RegexpTokenizer(u'[^!؟۔]*[!؟۔]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer.tokenize(l))) for l in text_labels]

    rouge_results = rouge_metric.compute(predictions=text_preds, references=text_labels)
    bleu_results = bleu_metric.compute(predictions=text_preds, references=text_labels)

    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleu": bleu_results["bleu"]
    }

def preprocess_function_causal_lm_sft_training(examples, input_column_name="text", target_column_name="summary"):
    return NotImplementedError("Not implemented yet!")

def preprocess_function_causal_lm(examples, input_column_name="text", target_column_name="summary"):
    # Tokenize inputs with padding and truncation
    model_inputs = tokenizer(
        examples[input_column_name], 
        truncation=True, 
        padding=True,  # Add padding
        max_length=512
    )
    
    # Tokenize labels with padding and truncation
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples[target_column_name], 
            truncation=True, 
            padding=True,  # Add padding
            max_length=128
        )
    
    # Ensure consistent batch processing
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function(examples, input_column_name="text", target_column_name="summary"):
    inputs = [doc for doc in examples[input_column_name]]
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples[target_column_name], 
            max_length=128, 
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    
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
    
    
    MODELS_DICT = {
        "bert-base-arabic": {
            "MODEL_PATH": "asafaya/bert-base-arabic",
            "CAUSAL_LM": False,
            "SFT_TRAINING": False,
        },
        "gpt2": {
            "MODEL_PATH": "openai-community/gpt2",
            "CAUSAL_LM": True,
            "SFT_TRAINING": False,
        },
        "mt5-small": {
            "MODEL_PATH": "google/mt5-small",
            "CAUSAL_LM": False,
            "SFT_TRAINING": False,
        },
        "mt5-base": {
            "MODEL_PATH": "google/mt5-base",
            "CAUSAL_LM": False,
            "SFT_TRAINING": False,
        },
        "Qwen2.5-0.5B": {
            "MODEL_PATH": "Qwen/Qwen2.5-0.5B",
            "CAUSAL_LM": True,
            "SFT_TRAINING": False,
        },
        "Qwen2.5-0.5B-Instruct": {
            "MODEL_PATH": "Qwen2.5-0.5B-Instruct",
            "CAUSAL_LM": True,
            "SFT_TRAINING": True,
        },
    }
    
    # Training hyperparameters
    num_train_epochs = config['hyperparameters']['num_train_epochs']
    lr = config['hyperparameters']['lr']
    batch_size = config['hyperparameters']['batch_size']
    gradient_accumulation_steps = config['hyperparameters']['gradient_accumulation_steps']
    max_grad_norm = config['hyperparameters']['max_grad_norm']
    warmup_steps = config['hyperparameters']['warmup_steps']
    
    # Logging and saving
    logging_steps = config['hyperparameters']['logging_steps']
    save_steps = config['hyperparameters']['save_steps']
    eval_steps = config['hyperparameters']['eval_steps']

    # Training data path
    TRAIN_DATA_PATH = config['DATASET_PATH']
    
    # base model path
    BASE_MODEL = config['BASE_MODEL']
    MODEL_PATH = MODELS_DICT[BASE_MODEL]['MODEL_PATH']
    IS_CAUSAL_LM = MODELS_DICT[BASE_MODEL]['CAUSAL_LM']
    IS_SFT_TRAINING = MODELS_DICT[BASE_MODEL]['SFT_TRAINING']
    
    # max training samples
    MAX_TRAINING_SAMPLES = config['MAX_TRAINING_SAMPLES']
    
    # set seed
    SEED = config['SEED']
    set_seed(SEED)

    # Where to save the model
    MODEL_RUN_SAVE_PATH = f"BounharAbdelaziz/Arabic-Summarization/{MODEL_PATH.split('/')[-1]}-bs-{batch_size}-lr-{lr}-ep-{num_train_epochs}-wmp-{warmup_steps}-gacc-{gradient_accumulation_steps}-gnorm-{max_grad_norm}"
    
    # Load dataset
    dataset = load_dataset(TRAIN_DATA_PATH)  # Replace with your dataset path
    
    # truncate training dataset to observe data size impact on performance
    print(f'[INFO] Truncating training samples to: {MAX_TRAINING_SAMPLES}...')
    dataset['train'] = dataset['train'].select(range(min(len(dataset['train']), MAX_TRAINING_SAMPLES)))
    print(f'[INFO] Dataset loaded: {dataset}')
    print('-'*50)
    
    # Load tokenizer and model
    if IS_CAUSAL_LM:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16, # bfloat16 has better precission than float16 thanks to bigger mantissa
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16, # bfloat16 has better precission than float16 thanks to bigger mantissa
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f'[INFO] Model and Tokenizer loaded: {MODEL_PATH}')
    print('-'*50)
    
    project_name = "arabic-summarization"
    run_name = f'{MODEL_PATH.split("/")[-1]}-bs-{batch_size}-lr-{lr}-ep-{num_train_epochs}-wmp-{warmup_steps}-gacc-{gradient_accumulation_steps}-gnorm-{max_grad_norm}'

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
            # "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # "weight_decay": weight_decay,
            "dataset": TRAIN_DATA_PATH,
        }
    )

    if IS_CAUSAL_LM:
        
        if IS_SFT_TRAINING:
            # Apply preprocessing
            tokenized_datasets = dataset.map(preprocess_function_causal_lm_sft_training, batched=True)
        else:
            
            # Apply preprocessing
            tokenized_datasets = dataset.map(preprocess_function_causal_lm, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=MODEL_RUN_SAVE_PATH,
                evaluation_strategy="steps",
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_train_epochs,
                save_total_limit=1,
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                report_to="wandb",
                push_to_hub=False,
            )
        
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                compute_metrics=compute_metrics_causal_lm
            )
        
    else:
        
        # Apply preprocessing
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=MODEL_RUN_SAVE_PATH,
            evaluation_strategy="steps",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            save_total_limit=1,
            predict_with_generate=True,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            report_to="wandb",
            push_to_hub=False,
        )
    
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=compute_metrics,
        )

    # Train the model
    trainer.train()

    # Push to Hugging Face Hub
    print("[INFO] Pushing to hub...")
    trainer.push_to_hub(MODEL_RUN_SAVE_PATH)
    
    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f'[INFO] Results on test set: {test_results}')


    