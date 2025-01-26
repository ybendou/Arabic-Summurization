# Import necessary libraries
import numpy as np
from evaluate import load
from nltk.tokenize import RegexpTokenizer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from torchmetrics.text.rouge import ROUGEScore
# from torchmetrics.text.bleu import BLEUScore

# rouge_metric = ROUGEScore()
# bleu_metric = BLEUScore()

from evaluate import load
rouge = load('rouge')
bleu = load('bleu')


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed. 
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/22
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

@torch.no_grad()
def compute_metrics_causal_lm(eval_pred, tokenizer):
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
def compute_metrics(eval_pred, tokenizer):
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
   
def preprocess_function_causal_lm_sft_training(examples, tokenizer, input_column_name="text", target_column_name="summary"):
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

def preprocess_function_causal_lm(examples, tokenizer, input_column_name="text", target_column_name="summary"):
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

def preprocess_function(examples, tokenizer, input_column_name="text", target_column_name="summary"):
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
    """ Sets the seed for reproducibility """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_params_info(model):
    """
    Prints the total and trainable parameters in the model, 
    along with the percentage reduction in trainable parameters.
    
    Parameters:
    - model: The PyTorch model (could be wrapped with LoRA).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction_percent = (1 - trainable_params / total_params) * 100

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Reduction in Trainable Parameters: {reduction_percent:.2f}%")
    
# Add LoRA configuration
def apply_lora(model, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=None, IS_CAUSAL_LM=True):
    """
    Applies LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.
    
    Parameters:
    - model: The base model to be adapted with LoRA.
    - r: LoRA rank.
    - lora_alpha: Scaling factor for LoRA layers.
    - lora_dropout: Dropout probability for LoRA layers.
    - target_modules: List of modules to apply LoRA (e.g., `["q_proj", "v_proj"]`).
    
    Returns:
    - The model wrapped with LoRA.
    """
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # task_type="CAUSAL_LM" if IS_CAUSAL_LM else "SEQ_2_SEQ_LM",  # Adjust for your task
        target_modules=target_modules,  # Specify target modules if required
    )
    
    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)

    # Log trainable parameters for verification
    print_trainable_params_info(model)
    return model



