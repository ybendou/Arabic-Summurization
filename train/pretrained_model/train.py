# Import necessary libraries
import numpy as np
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
from datasets import load_dataset
from evaluate import load
from nltk.tokenize import RegexpTokenizer
import torch
import yaml
from pprint import pprint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from torchmetrics.text.rouge import ROUGEScore
# from torchmetrics.text.bleu import BLEUScore

# rouge_metric = ROUGEScore()
# bleu_metric = BLEUScore()


from evaluate import load

rouge = load('rouge')
bleu = load('bleu')

@torch.no_grad()
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
    rouge_results = rouge.compute(predictions=text_preds, references=text_labels)
    bleu_results = bleu.compute(predictions=text_preds, references=text_labels)
    
    return {
        "rouge1": rouge_results["rouge1"] * 100,
        "rouge2": rouge_results["rouge2"] * 100,
        "rougeL": rouge_results["rougeL"] * 100,
        "rougeLsum": rouge_results["rougeLsum"] * 100,
        "bleu": bleu_results["bleu"] * 100,
    }
    
@torch.no_grad()
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    # Clip token IDs to valid range
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
    
    # Ensure labels are not masked
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Post-process text for ROUGE and BLEU
    text_preds = [(p if p.endswith(("!", "؟", "۔")) else p + "۔") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "؟", "۔")) else l + "۔") for l in text_labels]

    sent_tokenizer = RegexpTokenizer(u'[^!؟۔]*[!؟۔]')
    text_preds = ["\n".join([s.strip() for s in sent_tokenizer.tokenize(p)]) for p in text_preds]
    text_labels = ["\n".join([s.strip() for s in sent_tokenizer.tokenize(l)]) for l in text_labels]

    # Compute metrics
    rouge_results = rouge.compute(predictions=text_preds, references=text_labels)
    bleu_results = bleu.compute(predictions=text_preds, references=text_labels)
    
    return {
        "rouge1": rouge_results["rouge1"] * 100,
        "rouge2": rouge_results["rouge2"] * 100,
        "rougeL": rouge_results["rougeL"] * 100,
        "rougeLsum": rouge_results["rougeLsum"] * 100,
        "bleu": bleu_results["bleu"] * 100,
    }


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed. 
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/22
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels
    
def preprocess_function_causal_lm_sft_training(examples, input_column_name="text", target_column_name="summary"):
    # Format data in instruction-following style
    formatted_inputs = [
        f"### System: You are a summarization tool designed exclusively to generate concise summaries of Arabic input text.\n### Human: Summarize the following text in Arabic:\n{source_text}\n### Assistant: {summary}" 
        for source_text, summary in zip(examples[input_column_name], examples[target_column_name])
    ]
    
    # Tokenize inputs with truncation and padding
    model_inputs = tokenizer(
        formatted_inputs, 
        truncation=True, 
        padding='max_length',  # Use max_length padding
        max_length=tokenizer.model_max_length,  # Ensure consistent max length
        return_tensors='pt'  # Return PyTorch tensors
    )
    
    # Set labels to be the same as input_ids for SFT
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # Replace padding tokens with -100 in labels
    model_inputs["labels"] = torch.where(
        model_inputs["labels"] == tokenizer.pad_token_id, 
        torch.tensor(-100), 
        model_inputs["labels"]
    )
    
    return model_inputs

def preprocess_function_causal_lm(examples, input_column_name="text", target_column_name="summary"):
    # Tokenize inputs with truncation and padding
    model_inputs = tokenizer(
        examples[input_column_name], 
        truncation=True, 
        padding='max_length',  # Use max_length padding instead of True
        max_length=tokenizer.model_max_length,  # Ensure consistent max length
        return_tensors='pt'  # Return PyTorch tensors
    )
    
    # Tokenize labels with padding and truncation
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples[target_column_name], 
            truncation=True, 
            padding='max_length',  # Use max_length padding
            max_length=tokenizer.model_max_length,
            return_tensors='pt'  # Return PyTorch tensors
        )
    
    # Ensure labels have the same shape as input_ids
    model_inputs["labels"] = labels["input_ids"]
    
    # Replace -100 for padding tokens in labels
    model_inputs["labels"] = torch.where(
        model_inputs["labels"] == tokenizer.pad_token_id, 
        torch.tensor(-100), 
        model_inputs["labels"]
    )
    
    return model_inputs

def preprocess_function(examples, input_column_name="text", target_column_name="summary"):
    # Tokenize inputs with truncation and padding
    model_inputs = tokenizer(
        examples[input_column_name], 
        max_length=tokenizer.model_max_length, 
        truncation=True,
        padding='max_length',  # Use max_length padding
        return_tensors='pt'  # Return PyTorch tensors
    )

    # Tokenize labels with padding and truncation
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples[target_column_name], 
            max_length=tokenizer.model_max_length, 
            truncation=True,
            padding='max_length',  # Use max_length padding
            return_tensors='pt'  # Return PyTorch tensors
        )

    # Ensure labels have the same shape as input_ids
    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding tokens with -100 in labels
    model_inputs["labels"] = torch.where(
        model_inputs["labels"] == tokenizer.pad_token_id, 
        torch.tensor(-100), 
        model_inputs["labels"]
    )
    
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
        "mt5-small-SFT": {
            "MODEL_PATH": "google/mt5-small",
            "CAUSAL_LM": False,
            "SFT_TRAINING": True,
        },
        "mt5-base": {
            "MODEL_PATH": "google/mt5-base",
            "CAUSAL_LM": False,
            "SFT_TRAINING": False,
        },
        "mt5-base-SFT": {
            "MODEL_PATH": "google/mt5-base",
            "CAUSAL_LM": False,
            "SFT_TRAINING": True,
        },
        "Qwen2.5-0.5B": {
            "MODEL_PATH": "Qwen/Qwen2.5-0.5B",
            "CAUSAL_LM": True,
            "SFT_TRAINING": False,
        },
        "Qwen2.5-0.5B-SFT": {
            "MODEL_PATH": "Qwen/Qwen2.5-0.5B",
            "CAUSAL_LM": True,
            "SFT_TRAINING": True,
        },
        "Qwen2.5-0.5B-Instruct": {
            "MODEL_PATH": "Qwen/Qwen2.5-0.5B-Instruct",
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
    warmup_ratio = config['hyperparameters']['warmup_ratio']
    
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
    FP16_TRAINING = config['FP16_TRAINING']
    
    # max training samples
    MAX_TRAINING_SAMPLES = config['MAX_TRAINING_SAMPLES']
    
    if FP16_TRAINING:
        torch_dtype=torch.bfloat16 # bfloat16 has better precission than float16 thanks to bigger mantissa
    else:
        torch_dtype=torch.float32
    
    # set seed
    SEED = config['SEED']
    set_seed(SEED)
   
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
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype, 
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Set a maximum length for tokenization
    tokenizer.model_max_length = config['hyperparameters']['MAX_LEN']
    if BASE_MODEL == "gpt2":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print(f'[INFO] Model and Tokenizer loaded: {MODEL_PATH}')
    print('-'*50)
    
    # Project name for loggings and savings
    project_name = "arabic-summarization"
    fp16 = 'FP16' if FP16_TRAINING else ''
    sft = 'SFT' if IS_SFT_TRAINING else ''
    run_name = f'{MODEL_PATH.split("/")[-1]}-bs-{batch_size}-lr-{lr}-ep-{num_train_epochs}-wmp-{warmup_steps}-gacc-{gradient_accumulation_steps}-gnorm-{max_grad_norm}-{fp16}-{sft}-mxln-{config['hyperparameters']['MAX_LEN']}'
    
    # Where to save the model
    MODEL_RUN_SAVE_PATH = f"BounharAbdelaziz/Arabic-Summarization/{run_name}"
    
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
        }
    )

    if IS_CAUSAL_LM:
        
        if IS_SFT_TRAINING:
            # Apply preprocessing
            tokenized_datasets = dataset.map(preprocess_function_causal_lm_sft_training, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=MODEL_RUN_SAVE_PATH,
                evaluation_strategy="steps",
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                # eval_accumulation_steps=config['hyperparameters']['eval_accumulation_steps'],  # Helps with memory management during evaluation
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
            )
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                compute_metrics=compute_metrics_causal_lm,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics, # avoids OOM in eval
            )
            
        else:
            
            print(f'[INFO] Running preprocess_function_causal_lm')
            # Apply preprocessing
            tokenized_datasets = dataset.map(preprocess_function_causal_lm, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=MODEL_RUN_SAVE_PATH,
                evaluation_strategy="steps",
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                # eval_accumulation_steps=config['hyperparameters']['eval_accumulation_steps'],  # Helps with memory management during evaluation => tooo slow!
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
            )
        
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                compute_metrics=compute_metrics_causal_lm,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics, # avoids OOM in eval
            )
        
    else:
        
        # Apply preprocessing
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=MODEL_RUN_SAVE_PATH,
            evaluation_strategy="steps",
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
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
            metric_for_best_model=config['METRIC_FOR_BEST_MODEL'],
            gradient_checkpointing=True,
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