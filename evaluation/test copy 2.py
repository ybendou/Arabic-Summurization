import os
import yaml
import torch
import logging
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

import evaluate

class SummarizationEvaluator:
    def __init__(self, config: Dict[str, Any]):
        """Initialize summarization evaluator"""
        self.config = config
        self.logger = self._setup_logger()
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load evaluation metrics
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.bertscore = evaluate.load('bertscore')
        
        # init to none
        self.model = None
        self.tokenizer = None
        self.data_collator = None
    
    def _setup_logger(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_dataset(self, subset: bool = False) -> Dataset:
        """Load dataset from configuration"""
        self.logger.info("Loading dataset...")
        dataset = load_dataset(
            self.config['DATASET_PATH'], 
            split="validation"
        )
        
        if subset:
            dataset = Dataset.from_dict(dataset[:10])
        
        return dataset
    
    def load_model_tokenizer_dataloader(self, model_path: str, tokenizer_path: str, max_len: int):
        """Load model and tokenizer"""
        self.logger.info(f"Loading model: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        # put in eval mode
        self.model.eval()
        
        # Load tokenizer for preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.model_max_length = max_len
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True)
        
    def preprocess_dataset(self, dataset: Dataset, text_column: str = "text", max_length:int = 1024) -> Dataset:
        """Tokenize and preprocess the dataset"""
        self.logger.info("Preprocessing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def preprocess_function_causal_lm_sft(self, examples, input_column_name="text", target_column_name="summary"):
        # Format data in instruction-following style
        formatted_inputs = [
            f"### System: You are a summarization tool designed exclusively to generate concise summaries of Arabic input text.\n### Human: Summarize the following text in Arabic:\n{source_text}\n### Assistant: {summary}" 
            for source_text, summary in zip(examples[input_column_name], examples[target_column_name])
        ]
        
        # Tokenize inputs with truncation and padding
        model_inputs = self.tokenizer(
            formatted_inputs, 
            truncation=True, 
            padding='max_length',  # Use max_length padding
            max_length=self.tokenizer.model_max_length,  # Ensure consistent max length
            return_tensors='pt'  # Return PyTorch tensors
        )
        
        # Set labels to be the same as input_ids for SFT
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        # Replace padding tokens with -100 in labels
        model_inputs["labels"] = torch.where(
            model_inputs["labels"] == self.tokenizer.pad_token_id, 
            torch.tensor(-100), 
            model_inputs["labels"]
        )
        
        return model_inputs
    
    # def generate_summaries(
    #     self, 
    #     dataset: Dataset, 
    #     model_path: str, 
    #     text_column: str = "text",
    #     batch_size: int = 16,
    #     is_sft: bool = True,
    # ) -> Dataset:
    #     """Generate summaries in batches using the model directly"""
    #     model_name = model_path.split('/')[-1]
        
    #     input_texts = dataset[text_column]
    #     summaries = []
        
    #     for i in tqdm(range(0, len(input_texts), batch_size), desc=f'Summarizing {model_name}'):
    #         batch_inputs = input_texts[i:i+batch_size]
            
    #         # Generate summaries
    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 **batch_inputs,
    #                 max_new_tokens=2048, # summary cannot be larger than that.. (in our dataset)
    #                 do_sample=False,
    #             )
            
    #         # Decode outputs
    #         batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #         summaries.extend(batch_summaries)
        
    #     dataset = dataset.add_column(f"summary_{model_name}", summaries)
    #     return dataset

    def generate_summaries(
        self, 
        dataset: Dataset, 
        model_path: str, 
        text_column: str = "text",
        batch_size: int = 16,
        is_sft: bool = True,
    ) -> Dataset:
        """Generate summaries in batches using the model directly"""
        model_name = model_path.split('/')[-1]
        
        # Extract tokenized inputs from the dataset
        input_ids = dataset["input_ids"]
        attention_mask = dataset["attention_mask"]
        summaries = []
        
        for i in tqdm(range(0, len(input_ids), batch_size), desc=f'Summarizing using {model_name}'):
            # Prepare the batch
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]
            
            # Move batch to the correct device
            batch_input_ids = torch.tensor(batch_input_ids).to(self.device)
            batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
            
            # Generate summaries
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=2048,  # summary cannot be larger than this
                    do_sample=False,
                )
            
            # Decode outputs
            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(batch_summaries)
        
        # Add summaries to the dataset
        dataset = dataset.add_column(f"summary_{model_name}", summaries)
        return dataset

    def compute_metrics(
        self, 
        dataset: Dataset, 
        models: List[str], 
        reference_column: str = "summary"
    ) -> pd.DataFrame:
        """Compute comprehensive metrics"""
        results = []
        
        for model_path in models:
            model_name = model_path.split('/')[-1]
            summary_column = f"summary_{model_name}"
            
            # Extract predictions and references
            predictions = dataset[summary_column]
            references = dataset[reference_column]
            
            # Compute ROUGE
            rouge_results = self.rouge.compute(
                predictions=predictions, 
                references=references
            )
            
            # Compute BLEU
            bleu_results = self.bleu.compute(
                predictions=predictions, 
                references=references
            )
            
            # Compute METEOR
            meteor_results = self.meteor.compute(
                predictions=predictions, 
                references=references
            )
            
            # Compute BERTScore
            bertscore_results = self.bertscore.compute(
                predictions=predictions, 
                references=references, 
                lang='ar'
            )
            
            # Aggregate results
            model_metrics = {
                "model": model_name,
                "rouge1": rouge_results["rouge1"] * 100,
                "rouge2": rouge_results["rouge2"] * 100,
                "rougeL": rouge_results["rougeL"] * 100,
                "rougeLsum": rouge_results["rougeLsum"] * 100,
                "bleu": bleu_results["bleu"] * 100,
                "meteor": meteor_results["meteor"] * 100,
                "bertscore_precision": sum(bertscore_results['precision']) / len(bertscore_results['precision']) * 100,
                "bertscore_recall": sum(bertscore_results['recall']) / len(bertscore_results['recall']) * 100,
                "bertscore_f1": sum(bertscore_results['f1']) / len(bertscore_results['f1']) * 100
            }
            
            results.append(model_metrics)
        
        return pd.DataFrame(results)
    
    def save_results(
        self, 
        dataset: Dataset, 
        metrics_df: pd.DataFrame, 
        output_path: str
    ):
        """Save results to disk"""
        os.makedirs(output_path, exist_ok=True)
        
        self.logger.info(f"Saving results to {output_path}...")
        
        dataset_df = dataset.to_pandas()
        dataset_df.to_csv(os.path.join(output_path, "summarized_dataset.csv"), index=False)
        metrics_df.to_csv(os.path.join(output_path, "metrics.csv"), index=False)
    
    def run_evaluation(self, subset: bool = False):
        """Run complete evaluation pipeline"""
        dataset = self.load_dataset(subset)
        
        for version in self.config['EVALUATE_MODELS']:
            
            # loadings
            self.load_model_tokenizer_dataloader(
                model_path=self.config['EVALUATE_MODELS'][version]['model_path'],
                tokenizer_path=self.config['EVALUATE_MODELS'][version]['tokenizer_path'], 
                max_len=self.config['EVALUATE_MODELS'][version]['MAX_LEN']
            )
            
            # # Preprocess dataset (tokenize upfront)
            # tokenized_dataset = self.preprocess_dataset(
            #     dataset, 
            #     text_column=self.config['TEXT_COLUMN'], 
            #     max_length=self.config['EVALUATE_MODELS'][version]['MAX_LEN']
            # )
            
            # Apply preprocessing
            tokenized_dataset = dataset.map(self.preprocess_function_causal_lm_sft, batched=True)
        
            dataset = self.generate_summaries(
                tokenized_dataset, 
                model_path=self.config['EVALUATE_MODELS'][version]['model_path'],
                text_column=self.config['TEXT_COLUMN'],
                batch_size=self.config['batch_size'],
                is_sft=self.config['EVALUATE_MODELS'][version]['IS_SFT']
            )
            
            # Free GPU memory after each model evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        metrics_df = self.compute_metrics(
            dataset, 
            self.config['EVALUATE_MODELS']
        )
        
        self.save_results(
            dataset, 
            metrics_df, 
            self.config.get('OUTPUT_PATH', './results')
        )
        
        return dataset, metrics_df
    
with open('eval_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
evaluator = SummarizationEvaluator(config)
dataset, metrics = evaluator.run_evaluation(subset=False)

print("Model Performance Metrics:")
print(metrics)