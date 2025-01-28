import os
import yaml
import torch
import logging
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer, DataCollatorForSeq2Seq
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

import evaluate

class SummarizationEvaluator:
    def __init__(self, config: Dict[str, Any]):
        """Initialize summarization evaluator"""
        self.config = config
        self.logger = self._setup_logger()
        
        # Set up device
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load evaluation metrics
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.bertscore = evaluate.load('bertscore')
        
        # Load tokenizer for preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['MODEL_PATH'])
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True)
    
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
    
    def preprocess_dataset(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        """Tokenize and preprocess the dataset"""
        self.logger.info("Preprocessing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def generate_summaries(
        self, 
        dataset: Dataset, 
        model_path: str, 
        text_column: str = "text",
        batch_size: int = 16
    ) -> Dataset:
        """Generate summaries in batches"""
        summarizer = self.load_summarization_pipeline(model_path)
        model_name = model_path.split('/')[-1]
        
        input_texts = dataset[text_column]
        summaries = []
        
        is_sft = 'SFT' in model_name or 'Instruct' in model_name
        
        for i in tqdm(range(0, len(input_texts), batch_size), desc=f'Summarizing {model_name}'):
            batch_inputs = input_texts[i:i+batch_size]
            
            # Apply the formatting
            if is_sft:
                formatted_inputs = [
                    f"### System: You are a summarization tool designed exclusively to generate concise summaries of Arabic input text.\n### Human: Summarize the following text in Arabic:\n{source_text}\n### Assistant:" 
                    for source_text in batch_inputs
                ]
            else:
                formatted_inputs = batch_inputs
            
            # Generate summaries
            batch_summaries = summarizer(
                formatted_inputs, 
                do_sample=False, 
                max_new_tokens=128
            )
            
            batch_summaries = [
                summary[0]['generated_text'] for summary in batch_summaries
            ]
            
            summaries.extend(batch_summaries)
        
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
        
        # Preprocess dataset (tokenize upfront)
        tokenized_dataset = self.preprocess_dataset(dataset, text_column=self.config['TEXT_COLUMN'])
        
        for model_path in self.config['EVALUATE_MODELS']:
            dataset = self.generate_summaries(
                tokenized_dataset, 
                model_path, 
                text_column=self.config['TEXT_COLUMN']
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