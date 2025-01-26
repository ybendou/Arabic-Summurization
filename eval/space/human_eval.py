import gradio as gr
import random
import pandas as pd
from collections import defaultdict
import os
import base64

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

def generate_dummy_arabic_data(num_samples=50):
    """Generate dummy Arabic-like text for testing"""
    def generate_arabic_text(min_length=50, max_length=500):
        arabic_chars = [
            'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 
            'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 
            'ن', 'ه', 'و', 'ي', ' '
        ]
        length = random.randint(min_length, max_length)
        return ''.join(random.choices(arabic_chars, k=length))
    
    data = {
        'input_text': [generate_arabic_text() for _ in range(num_samples)],
        'summary_extractive': [generate_arabic_text(20, 100) for _ in range(num_samples)],
        'summary_abstractive': [generate_arabic_text(20, 100) for _ in range(num_samples)],
        'summary_transformer': [generate_arabic_text(20, 100) for _ in range(num_samples)],
        'summary_lstm': [generate_arabic_text(20, 100) for _ in range(num_samples)]
    }
    
    return pd.DataFrame(data)

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
        
        model_summary_cols = [col for col in row.index if 'summary_' in col]
        
        selected_models = random.sample(model_summary_cols, 2)
        
        battle_data = {
            'input_text': row['input_text'],
            'summary1': row[selected_models[0]],
            'summary2': row[selected_models[1]],
            'model1_name': selected_models[0].replace('summary_', ''),
            'model2_name': selected_models[1].replace('summary_', '')
        }
        
        self.current_index += 1
        return battle_data
    
    def record_evaluation(self, preferred_models, input_text, summary1, summary2, model1_name, model2_name):
        """Record user's model preference and update scores"""
        # Update comparison counts
        self.model_scores[model1_name]['total_comparisons'] += 1
        self.model_scores[model2_name]['total_comparisons'] += 1
        
        # Update wins based on user preference
        if preferred_models == "Both Good":
            # Both models get a win
            self.model_scores[model1_name]['wins'] += 1
            self.model_scores[model2_name]['wins'] += 1
        elif preferred_models == "Model 1":
            self.model_scores[model1_name]['wins'] += 1
        elif preferred_models == "Model 2":
            self.model_scores[model2_name]['wins'] += 1
        # "Both Bad" case doesn't award any wins
        
        # Record evaluation details
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
    # Replace this with actual model inference
    return f"Summary from {model_name}: {input_text[:50]}..."  # Dummy summary

def create_battle_arena(dataset_path, is_gif=True):
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
            preferred_models, 
            input_text, 
            summary1, 
            summary2, 
            model1_name, 
            model2_name
        )
        next_battle = battle_round()
        return (*next_battle[:-1], scores_df)

    with gr.Blocks(css="footer{display:none !important}") as demo:
        base_path = os.path.dirname(__file__)
        local_image_path = os.path.join(base_path, 'sum_battle_leaderboard.gif')
        gr.HTML(create_html_media(local_image_path, is_gif=is_gif))
        
        with gr.Tabs():
            # Battle Arena Tab
            with gr.Tab("Battle Arena"):
                gr.Markdown("# 🤖 Arabic Summarization Model Battle Arena", elem_classes=["arena-title"])
                
                with gr.Row():
                    input_text = gr.Textbox(
                        label="Input text", 
                        interactive=False, 
                        elem_classes=["summary-card"]
                    )
                
                with gr.Row():
                    with gr.Column():
                        summary1 = gr.Textbox(
                            label="Summary 1", 
                            interactive=False, 
                            elem_classes=["summary-card"]
                        )
                        model1_name = gr.Textbox(label="Model 1", interactive=False)
                    
                    with gr.Column():
                        summary2 = gr.Textbox(
                            label="Summary 2", 
                            interactive=False, 
                            elem_classes=["summary-card"]
                        )
                        model2_name = gr.Textbox(label="Model 2", interactive=False)
                
                with gr.Row(elem_classes=["preference-section"]):
                    preferred_models = gr.Radio(
                        label="Which summary is better?",
                        choices=["Model 1", "Model 2", "Both Good", "Both Bad"],
                        elem_classes="voting-options"
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
                            lines=5, 
                            placeholder="Enter Arabic text to summarize...",
                            elem_classes=["rtl"]
                        )
                        model_selector = gr.Dropdown(
                            label="Select Model",
                            choices=["extractive", "abstractive", "transformer", "lstm"],
                            value="extractive"
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
    # Generate dummy data and launch app
    dataset_path = 'dummy_arabic_summaries.csv'
    dummy_data = generate_dummy_arabic_data()
    dummy_data.to_csv(dataset_path, index=False)
    is_gif = True
    demo = create_battle_arena(dataset_path, is_gif)
    demo.launch(debug=True, share=True)