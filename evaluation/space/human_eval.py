import gradio as gr
from collections import defaultdict
import os
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from datasets import load_dataset
import random
import pandas as pd
from collections import defaultdict


human_eval_dataset = load_dataset("BounharAbdelaziz/Arabic-Summarization-Human-Eval-Summaries", split='test').to_csv('human_eval_dataset.csv')

# chat template for SFT models
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
# precision
torch_dtype = torch.float16

# inference device
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

# maximum number of tokens for generate()
MAX_NEW_TOKENS = 256


############ Preload summarization models
mayofid_qwen_path = "BounharAbdelaziz/MaYofid-Qwen2.5-3B-Instruct"

mayofid_qwen = AutoModelForCausalLM.from_pretrained(mayofid_qwen_path, torch_dtype=torch_dtype).to(device)
mayofid_qwen.use_cache = True

# load tokenizer
tokenizer_mayofid_qwen = AutoTokenizer.from_pretrained(mayofid_qwen_path)
    
# Set chat template
tokenizer_mayofid_qwen.chat_template = DEFAULT_CHAT_TEMPLATE

mayofid_qwen_awq_path = "BounharAbdelaziz/Qwen2.5-3B-Instruct-Summarizer-AWQ"

mayofid_qwen_awq = AutoModelForCausalLM.from_pretrained(mayofid_qwen_awq_path, torch_dtype=torch_dtype).to(device)
mayofid_qwen_awq.use_cache = True

# load tokenizer
tokenizer_mayofid_qwen_awq = AutoTokenizer.from_pretrained(mayofid_qwen_awq_path)
    
# Set chat template
tokenizer_mayofid_qwen_awq.chat_template = DEFAULT_CHAT_TEMPLATE

mayofid_falcon_path = "BounharAbdelaziz/MaYofid-Falcon3-3B-Instruct"

mayofid_falcon = AutoModelForCausalLM.from_pretrained(mayofid_qwen_awq_path, torch_dtype=torch_dtype).to(device)
mayofid_falcon.use_cache = True

# load tokenizer
tokenizer_mayofid_falcon = AutoTokenizer.from_pretrained(mayofid_qwen_awq_path)
    
# Set chat template
tokenizer_mayofid_falcon.chat_template = DEFAULT_CHAT_TEMPLATE

SUMMARY_MODELS_OPTIONS = [
    "MaYofid-Qwen2.5-3B-Instruct",
    "MaYofid-Qwen2.5-3B-Instruct-AWQ",
    "MaYofid-Falcon3-3B-Instruct",
]

MODELS_DICT = {
    "MaYofid-Qwen2.5-3B-Instruct": {
        "model": mayofid_qwen,
        "tokenizer": tokenizer_mayofid_qwen,
    },
    "MaYofid-Qwen2.5-3B-Instruct-AWQ": {
        "model": mayofid_qwen_awq,
        "tokenizer": mayofid_qwen_awq,
    },
    "MaYofid-Falcon3-3B-Instruct": {
        "model": mayofid_falcon,
        "tokenizer": tokenizer_mayofid_falcon,
    }
}

def encode_image_to_base64(image_path):
    """Encode an image or GIF file to base64."""
    with open(image_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    return encoded_string

def create_html_media(media_path, is_gif=False):
    """Create HTML for displaying an image or GIF."""
    media_base64 = encode_image_to_base64(media_path)
    media_type = "gif" if is_gif else "jpeg"
    
    html_string = f"""
    <div style="display: flex; justify-content: center; align-items: center; width: 100%; text-align: center;">
        <div style="max-width: 450px; margin: auto;">
            <img src="data:image/{media_type};base64,{media_base64}"
                 style="max-width: 75%; height: auto; display: block; margin: 0 auto; margin-top: 50px;"
                 alt="Displayed Media">
        </div>
    </div>
    """
    return html_string

def summarize_batch(batch_texts, model, tokenizer, max_length=1024, max_new_tokens=256, batch_size=1, device="cuda"):
    
    model.eval()
    model.to(device)
    
    # Prepare the messages for the model using the tokenizer's chat template
    messages = [
        [{"role": "user", "content": text}] for text in batch_texts
    ]
    
    # Apply the chat template
    input_ids = tokenizer.apply_chat_template(
        messages, 
        truncation=True,
        max_length=max_length,  # adjust based on your model's context window
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(device)
    # Create attention mask based on non-zero tokens
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
            
    generation_config = model.generation_config
    
    # Generate summaries
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            bos_token_id=generation_config.bos_token_id,
            eos_token_id=generation_config.eos_token_id,
            pad_token_id=generation_config.pad_token_id,
        )
    
    # Decode the generated outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract summaries (text after <|assistant|>)
    summaries = []
    for text in generated_texts:
        if "<|assistant|>" in text:
            summaries.append(text.split("<|assistant|>")[-1].strip())
        else:
            summaries.append(text.strip())
    
    # Clean summaries
    def clean_summary(text):
        special_tokens = ["<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>"]
        for token in special_tokens:
            text = text.replace(token, "")
        return re.sub(r"\s+", " ", text).strip()
    
    cleaned_summaries = [clean_summary(summary) for summary in summaries]
    
    return cleaned_summaries[0]

class SummarizationBattleArena:
    def __init__(self, dataset_path):
        """Initialize battle arena with dataset"""
        self.df = pd.read_csv(dataset_path)
        self.current_index = 0
        self.evaluation_results = []
        self.model_scores = defaultdict(lambda: {'wins': 0, 'total_comparisons': 0})
    
    def get_next_battle_pair(self):
        """Retrieve next pair of summaries for comparison"""
        if self.current_index >= len(self.df):
            return None
        
        row = self.df.iloc[self.current_index]
        
        model_summary_cols = [col for col in row.index if col.startswith("MaYofid")]
        selected_models = random.sample(model_summary_cols, 2)
        
        battle_data = {
            'input_text': row['text'],
            'summary1': row[selected_models[0]],
            'summary2': row[selected_models[1]],
            'model1_name': selected_models[0],
            'model2_name': selected_models[1]
        }
        
        self.current_index += 1
        return battle_data
    
    def record_evaluation(self, preferred_models, input_text, summary1, summary2, model1_name, model2_name):
        """Record user's model preference and update scores"""
        self.model_scores[model1_name]['total_comparisons'] += 1
        self.model_scores[model2_name]['total_comparisons'] += 1
        
        if preferred_models == "Both Good":
            self.model_scores[model1_name]['wins'] += 1
            self.model_scores[model2_name]['wins'] += 1
        elif preferred_models == "Summary A":  # Maps to first model
            self.model_scores[model1_name]['wins'] += 1
        elif preferred_models == "Summary B":  # Maps to second model
            self.model_scores[model2_name]['wins'] += 1
        # "Both Bad" case - no wins recorded
        
        evaluation = {
            'input_text': input_text,
            'summary1': summary1,
            'summary2': summary2,
            'model1_name': model1_name,
            'model2_name': model2_name,
            'preferred_models': preferred_models
        }
        self.evaluation_results.append(evaluation)
        
        return self.get_model_scores_df()
    
    def get_model_scores_df(self):
        """Convert model scores to DataFrame"""
        scores_data = []
        for model, stats in self.model_scores.items():
            win_rate = (stats['wins'] / stats['total_comparisons'] * 100) if stats['total_comparisons'] > 0 else 0
            scores_data.append({
                'Model': model,
                'Wins': stats['wins'],
                'Total Comparisons': stats['total_comparisons'],
                'Win Rate (%)': round(win_rate, 2)
            })
        return pd.DataFrame(scores_data).sort_values('Win Rate (%)', ascending=False)
    
def summarize_text(input_text, model_name):
    """Dummy summarization function (replace with actual model calls)"""
    
    model = MODELS_DICT[model_name]["model"]
    tokenizer = MODELS_DICT[model_name]["tokenizer"]
    return summarize_batch([input_text], model, tokenizer, max_length=1024, max_new_tokens=256, batch_size=1, device="cuda")

def create_battle_arena(dataset_path, is_gif):
    arena = SummarizationBattleArena(dataset_path)
    
    def battle_round():
        battle_data = arena.get_next_battle_pair()
        
        if battle_data is None:
            return "No more texts to evaluate!", "", "", "", "", gr.DataFrame(visible=False)
        
        return (
            battle_data['input_text'], 
            battle_data['summary1'], 
            battle_data['summary2'],
            battle_data['model1_name'], 
            battle_data['model2_name'],
            gr.DataFrame(visible=True)
        )
    
    def submit_preference(input_text, summary1, summary2, model1_name, model2_name, preferred_models):
        scores_df = arena.record_evaluation(
            preferred_models, input_text, summary1, summary2, model1_name, model2_name
        )
        next_battle = battle_round()
        return (*next_battle[:-1], scores_df)

    with gr.Blocks(css="footer{display:none !important}") as demo:
        
        base_path = os.path.dirname(__file__)
        local_image_path = os.path.join(base_path, 'sum_battle_leaderboard.gif')
        gr.HTML(create_html_media(local_image_path, is_gif=is_gif))
        
        with gr.Tabs():
            with gr.Tab("Battle Arena"):
                gr.Markdown("# 🤖 Summarization Model Battle Arena")
                
                input_text = gr.Textbox(label="Input text", interactive=False)
                
                with gr.Row():
                    summary1 = gr.Textbox(label="Summary A", interactive=False)
                    model1_name = gr.State()  # Hidden state for model1 name
                
                with gr.Row():
                    summary2 = gr.Textbox(label="Summary B", interactive=False)
                    model2_name = gr.State()  # Hidden state for model2 name
                
                preferred_models = gr.Radio(
                    label="Which summary is better?",
                    choices=["Summary A", "Summary B", "Both Good", "Both Bad"]
                )
                submit_btn = gr.Button("Vote", variant="primary")
                
                scores_table = gr.DataFrame(
                    headers=['Model', 'Wins', 'Total Comparisons', 'Win Rate (%)'],
                    label="🏆 Leaderboard"
                )
                
                submit_btn.click(
                    submit_preference,
                    inputs=[input_text, summary1, summary2, model1_name, model2_name, preferred_models],
                    outputs=[input_text, summary1, summary2, model1_name, model2_name, scores_table]
                )
                
                demo.load(battle_round, outputs=[input_text, summary1, summary2, model1_name, model2_name, scores_table])
            
            # Summarization Tab
            with gr.Tab("Summarization"):
                gr.Markdown("# 📝 Arabic Text Summarization")
                
                with gr.Row():
                    with gr.Column():
                        input_text_summary = gr.Textbox(
                            label="Input Text", 
                            lines=8, 
                            placeholder="Enter Arabic text to summarize...",
                            elem_classes=["rtl"]
                        )
                        model_selector = gr.Dropdown(
                            label="Select Model",
                            choices=SUMMARY_MODELS_OPTIONS,
                            value="MaYofid-Qwen2.5-3B-Instruct"
                        )
                        summarize_btn = gr.Button("Generate Summary", variant="primary")
                    
                    with gr.Column():
                        output_summary = gr.Textbox(
                            label="Generated Summary", 
                            lines=5, 
                            interactive=False,
                            elem_classes=["rtl"]
                        )
                
                summarize_btn.click(
                    summarize_text,
                    inputs=[input_text_summary, model_selector],
                    outputs=output_summary
                )
    return demo

if __name__ == "__main__":
    dataset_path = 'human_eval_dataset.csv'
    is_gif = True
    demo = create_battle_arena(dataset_path, is_gif)
    demo.launch(debug=True, share=True)