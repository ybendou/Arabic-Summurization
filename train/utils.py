# Import necessary libraries
import numpy as np
from evaluate import load
from nltk.tokenize import RegexpTokenizer
import torch
import csv
import re
# from torchmetrics.text.rouge import ROUGEScore
# from torchmetrics.text.bleu import BLEUScore

# rouge_metric = ROUGEScore()
# bleu_metric = BLEUScore()

from evaluate import load
rouge = load('rouge')
bleu = load('bleu')
meteor = load('meteor')
bertscore = load('bertscore')

def create_conversation(example):
    """
    Transform the dataset into a conversational format.
    The user provides the text, and the assistant provides the summary.
    """
    # Create a conversation with user and assistant roles
    messages = [
        {"role": "user", "content": example["text"]},  # User provides the text
        {"role": "assistant", "content": example["summary"]}  # Assistant provides the summary
    ]
    # Return the conversation as a dictionary
    return {"messages": messages}

def create_testing_conversation(example):
    """
    Transform the dataset into a conversational format.
    The user provides the text, and the assistant must output summary.
    """
    # Create a conversation with user and assistant roles
    messages = [
        {"role": "user", "content": example["text"]},  # User provides the text
        {"role": "assistant", "content": ""}  # Assistant must output the summary
    ]
    # Return the conversation as a dictionary
    return {"messages": messages}


def apply_chat_template(example, tokenizer):
    """ Apply the chat template to the dataset. """
    example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return example

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed. 
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/22
    """
    
    # Handle tuple logits (happens when the model is trained using LoRA)
    if isinstance(logits, tuple):
        logits = logits[1]          # logits[0] is the loss value and logits[1] are the logits used to compute loss
                                    # logits: (tensor(2.0426, device='cuda:0'), tensor([[[ 7.8750,  5.3750,  7.0938,  ..., -4.2500, -4.2500, -4.2500],
                                    #          [ 5.0938,  5.0625,  7.3750,  ..., -1.5312, -1.5312, -1.5312],
                                    #          [ 2.6562, -0.9609,  0.0728,  ..., -2.0312, -2.0312, -2.0312],
                                    #          ...,
                                    #          [ 4.1562,  1.4375, -3.6250,  ..., -2.1250, -2.1250, -2.1250],
                                    #          [ 3.7344, -1.6641, -3.8125,  ..., -1.9688, -1.9688, -1.9688],
                                    #          [ 8.1875, -1.2344, -1.6094,  ..., -3.0938, -3.0938, -3.0938]]],
                                    #        device='cuda:0'))

    # Proceed with argmax
    pred_ids = torch.argmax(logits, dim=-1)

    return pred_ids

@torch.no_grad()
def compute_metrics_causal_lm(eval_pred, tokenizer):
    """Compute ROUGE and BLEU scores for evaluation."""
    predictions, references = eval_pred

    # Clip token IDs to the valid range
    vocab_size = tokenizer.vocab_size

    def clip_token_ids(token_ids):
        """Clip token IDs to the valid range [0, vocab_size - 1]."""
        return [min(max(token_id, 0), vocab_size - 1) for token_id in token_ids]

    # Decode predictions and references
    decoded_preds = [
        tokenizer.decode(clip_token_ids(pred), skip_special_tokens=True)
        for pred in predictions
    ]
    decoded_refs = [
        tokenizer.decode(clip_token_ids(ref), skip_special_tokens=True)
        for ref in references
    ]
    
    # Clean summaries
    def clean_summary(text):
        special_tokens = ["<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>"]
        for token in special_tokens:
            text = text.replace(token, "")
        return re.sub(r"\s+", " ", text).strip()
    
    pred_summaries = []
    for pred in decoded_preds:
        if "<|assistant|>" in pred:
            summary = pred.split("<|assistant|>")[-1].strip()
            summary = clean_summary(summary)
            pred_summaries.append(summary)
        else:
            summary = pred.strip()
            summary = clean_summary(summary)
            pred_summaries.append(summary)
            
    # apply the same to the references
    ref_summaries = []
    for ref in decoded_refs:
        if "<|assistant|>" in ref:
            summary = ref.split("<|assistant|>")[-1].strip()
            summary = clean_summary(summary)
            ref_summaries.append(summary)
        else:
            summary = ref.strip()
            summary = clean_summary(summary)
            ref_summaries.append(summary)
            
    # print(f'0 - ref_summaries[0]: {ref_summaries[0]}')
    
    # Convert to token IDs
    pred_token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in pred_summaries]
    ref_token_ids = [tokenizer.encode(r, add_special_tokens=False) for r in ref_summaries]

    # Use the exact same metric function from training
    eval_pred = (pred_token_ids, ref_token_ids)
    
    predictions, references = eval_pred

    # Clip token IDs to the valid range
    vocab_size = tokenizer.vocab_size

    # Decode predictions and references in batches
    decoded_preds = tokenizer.batch_decode([clip_token_ids(pred) for pred in predictions], skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode([clip_token_ids(ref) for ref in references], skip_special_tokens=True)
    
    # Print decoded examples to inspect issues
    print(f'decoded_preds[0]: {decoded_preds[0]}')
    print(f'decoded_refs[0]: {decoded_refs[0]}')

    # Compute ROUGE and BLEU scores
    rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_refs, use_stemmer=True)
    bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_refs)

    metrics = {key: rouge_results[key] * 100 for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    metrics["bleu"] = bleu_results["bleu"] * 100

    return metrics


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
    
    # Compute metrics
    rouge_results = rouge.compute(predictions=text_preds, references=text_labels)
    bleu_results = bleu.compute(predictions=text_preds, references=text_labels)
    # meteor_results = meteor.compute(
    #     predictions=text_preds, 
    #     references=text_labels
    # )
    # bertscore_results = bertscore.compute(
    #     predictions=text_preds, 
    #     references=text_labels, 
    #     lang='ar'
    # )
    
    return {
        "rouge1": rouge_results["rouge1"] * 100,
        "rouge2": rouge_results["rouge2"] * 100,
        "rougeL": rouge_results["rougeL"] * 100,
        "rougeLsum": rouge_results["rougeLsum"] * 100,
        "bleu": bleu_results["bleu"] * 100,
        # "meteor": meteor_results["meteor"] * 100,
        # "bertscore_precision": sum(bertscore_results['precision']) / len(bertscore_results['precision']) * 100,
        # "bertscore_recall": sum(bertscore_results['recall']) / len(bertscore_results['recall']) * 100,
        # "bertscore_f1": sum(bertscore_results['f1']) / len(bertscore_results['f1']) * 100
    }
   
# def preprocess_function_causal_lm_sft_training(examples, tokenizer, input_column_name="text", target_column_name="summary"):
#     # Format data in instruction-following style
#     formatted_inputs = [
#         f"### System: You are a summarization tool designed exclusively to generate concise summaries of Arabic input text.\n### Human: Summarize the following text in Arabic:\n{source_text}\n### Assistant: {summary}" 
#         for source_text, summary in zip(examples[input_column_name], examples[target_column_name])
#     ]
    
#     # Tokenize inputs with truncation and padding
#     model_inputs = tokenizer(
#         formatted_inputs, 
#         truncation=True, 
#         padding='max_length',  # Use max_length padding
#         max_length=tokenizer.model_max_length,  # Ensure consistent max length
#         return_tensors='pt'  # Return PyTorch tensors
#     )
    
#     # Set labels to be the same as input_ids for SFT
#     model_inputs["labels"] = model_inputs["input_ids"].clone()
    
#     # Replace padding tokens with -100 in labels
#     model_inputs["labels"] = torch.where(
#         model_inputs["labels"] == tokenizer.pad_token_id, 
#         torch.tensor(-100), 
#         model_inputs["labels"]
#     )
    
#     return model_inputs

# def preprocess_function_causal_lm_sft_training(examples, tokenizer):
#     inputs = []
#     targets = []
    
#     for text, summary in zip(examples['text'], examples['summary']):
#         # Create structured prompt
#         full_prompt = f"### System: You are a summarization tool designed exclusively to generate concise summaries of Arabic input text.\n### Human: {text}\n### Assistant: "
#         target = f"{summary}{tokenizer.eos_token}"
#         inputs.append(full_prompt)
#         targets.append(target)  # Add EOS
    
#     # Tokenize inputs and targets separately
#     tokenized_inputs = tokenizer(
#         inputs, 
#         max_length=tokenizer.model_max_length,
#         truncation=True, 
#         padding='max_length',
#         return_tensors="pt"
#     )
    
#     # Tokenize labels with padding and truncation
#     with tokenizer.as_target_tokenizer():
#         tokenized_targets = tokenizer(
#             targets,
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#             padding='max_length', 
#             return_tensors="pt"
#         )
    
#     # Create combined sequence
#     input_ids = tokenized_inputs.input_ids.clone()
#     labels = tokenized_targets.input_ids.clone()
    
    
#     # Mask everything except the assistant response
#     for i in range(len(input_ids)):
#         # Calculate prompt length
#         prompt_length = len(tokenizer.encode(inputs[i])) - 1  # Exclude EOS
        
#         # Mask input prompt in labels
#         labels[i, :prompt_length] = -100
        
#         # Mask padding
#         labels[i][labels[i] == tokenizer.pad_token_id] = -100
        
#     inputs = tokenizer.batch_decode(tokenized_inputs.input_ids, skip_special_tokens=True)
#     text_labels = tokenizer.batch_decode(tokenized_targets.input_ids, skip_special_tokens=True)
#     print(f'inputs: {inputs}')
#     print(f'target: {text_labels}')
#     print(f'-'*50)
    
#     return {
#         "input_ids": input_ids,
#         "attention_mask": tokenized_inputs.attention_mask,
#         "labels": labels
#     }

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