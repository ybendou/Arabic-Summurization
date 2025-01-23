from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import yaml
from pprint import pprint
from tqdm import tqdm

def get_summary_batch(model, tokenizer, queries, system_prompt, model_name, batch_size, max_new_tokens=512):
    """
    This function generates summaries for a batch of queries using the model.
    
    -----
    Args:
        model: the preloaded model to use
        tokenizer: the preloaded tokenizer to use
        queries: list of input texts to summarize
        system_prompt: the system prompt tuned for the model
        model_name: the model name
        batch_size: number of queries to process in parallel
        max_new_tokens: maximum number of new tokens to generate
        
    ---------
    Returns:
        list of dictionaries with summaries and model names
    """
    
    # Set model to eval mode
    model.eval()
    
    # Process in batches
    results = []
    
    # Use torch.no_grad() to disable gradient computation
    with torch.no_grad():
        # Iterate through queries in batches
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            
            # Prepare messages for the entire batch
            batch_messages = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ] for query in batch_queries
            ]
            
            # Apply chat template to the entire batch
            batch_texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ) for messages in batch_messages
            ]
            
            # Tokenize the batch with left padding
            model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side='left').to(model.device)
            
            # Generate summaries for the batch
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
            
            # Decode only the generated tokens (excluding input)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Batch decode the generated tokens
            summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Create results for this batch
            batch_results = [
                {"summary": summary, "summary_model_name": model_name} 
                for summary in summaries
            ]
            
            results.extend(batch_results)
    
    return results

if __name__ == "__main__":
    
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # get configuration
    with open('config.yaml') as file:
        config = yaml.safe_load(file)
        
    # Pretty-print the entire configuration
    print('-'*50)
    print("[INFO] Loaded configuration:")
    pprint(config)
    print('-'*50)
    
    # load the dataset
    dataset = load_dataset(config['DATASET_PATH'])
    print(f'[INFO] dataset: {dataset}')
    print('-'*50)
    
    # set the model to use
    MODEL_NAME = config['MODEL_NAME']
    
    # set the system prompt
    SYSTEM_PROMPT = config['SYSTEM_PROMPT']
    
    # Batched processing
    batch_size = config['BATCH_SIZE']
    
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use float16 for faster inference
        device_map="auto",
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2 for faster inference
    )
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = 'left'
    
    updated_dataset = DatasetDict()
    # Process each split in the dataset
    for split in dataset:
        print('-'*50)
        # # select a subset of the dataset for debugging
        # dataset[split] = dataset[split].select(range(config['MAX_SAMPLES']))
        
        # Prepare texts for batched processing
        texts = dataset[split]['text']
        
        # Process in batches with progress bar
        updated_results = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {split} batches"):
            batch_texts = texts[i:i+batch_size]
            batch_results = get_summary_batch(
                model, 
                tokenizer, 
                batch_texts, 
                SYSTEM_PROMPT, 
                MODEL_NAME, 
                batch_size=batch_size
            )
            updated_results.extend(batch_results)
        
        # Save results in the dataset
        dataset[split] = dataset[split].add_column("summary", [result['summary'] for result in updated_results])
        dataset[split] = dataset[split].add_column("summary_model_name", [result['summary_model_name'] for result in updated_results])

        # change column order for better readability
        new_column_order = ["text", "summary", "summary_model_name", "tokenizer_name", "dataset_source", "sequence_length"]

        # Reorder the columns
        updated_dataset[split] = dataset[split].select_columns(new_column_order)
        
    print('-'*50)
      
    # Push the updated dataset to the hub
    print(f"[INFO] Pushing the updated dataset to the hub: {config['SUMMARY_DATASET_SAVE_PATH']}")
    updated_dataset.push_to_hub(config['SUMMARY_DATASET_SAVE_PATH'], private=True, commit_message=f"Add synthetic summarization using {MODEL_NAME}")