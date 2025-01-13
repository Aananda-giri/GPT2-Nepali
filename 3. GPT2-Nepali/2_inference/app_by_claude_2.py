# app.py
import gradio as gr
import torch
from model import GPTModel
from transformers import PreTrainedTokenizerFast

# Load model and tokenizer once at startup
def load_model_n_tokenizer():
    model = GPTModel.from_pretrained("Aananda-giri/GPT2-Nepali")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/NepaliBPE")
    return model, tokenizer

# Initialize at startup
model, tokenizer = load_model_n_tokenizer()
model.eval()

def generate(prompt, max_new_tokens, top_k, top_p, temperature, repetition_penalty, penalize_len_below):
    device = next(model.parameters()).device
    
    # Convert top_k to None if using top_p
    if top_p > 0:
        top_k = None
    else:
        top_p = None
    
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_length=penalize_len_below,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create Gradio interface
with gr.Blocks(title="Nepali GPT-2 Text Generator") as interface:
    gr.Markdown("# Nepali GPT-2 Text Generator")
    gr.Markdown("Enter Nepali text to generate content using the custom GPT-2 model.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter Nepali text here...")
            max_tokens = gr.Slider(minimum=1, maximum=512, value=50, step=1, label="Max New Tokens")
            
            with gr.Row():
                with gr.Column():
                    top_k = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top K (set to 0 to use Top P)")
                    temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                with gr.Column():
                    top_p = gr.Slider(minimum=0, maximum=1.0, value=0, step=0.05, label="Top P (set above 0 to use instead of Top K)")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.2, step=0.1, label="Repetition Penalty")
            
            min_length = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Minimum Length Penalty")
            generate_btn = gr.Button("Generate Text")
        
        with gr.Column():
            output = gr.Textbox(label="Generated Text", lines=10)
    
    # Add examples if you have any
    gr.Examples(
        examples=[
            ["रामले भात खायो", 50, 50, 0, 0.7, 1.2, 50],
            ["नेपाल एउटा", 100, 0, 0.9, 0.8, 1.2, 100],
        ],
        inputs=[prompt, max_tokens, top_k, top_p, temperature, repetition_penalty, min_length],
        outputs=output,
        fn=generate,
        cache_examples=True,
    )
    
    generate_btn.click(
        fn=generate,
        inputs=[prompt, max_tokens, top_k, top_p, temperature, repetition_penalty, min_length],
        outputs=output
    )

interface.launch()