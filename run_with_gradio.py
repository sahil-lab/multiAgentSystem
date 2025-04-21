import os
import gradio as gr

# List available models
def list_models(model_dir="D:/llm_models"):
    """List all GGUF models in the specified directory"""
    models = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.gguf'):
                models.append(os.path.join(root, file))
    return models

def generate_response(model_path, prompt, max_tokens=200, temperature=0.7):
    """Generate a response using the selected model"""
    try:
        from llama_cpp import Llama
        
        # Load the model
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,       # Context window size
            n_threads=4,      # CPU threads to use
            n_gpu_layers=0    # Set higher if you have a GPU
        )
        
        # Generate response
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            echo=False
        )
        
        return output["choices"][0]["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Llama 3.1 8B Interface") as demo:
    gr.Markdown("# Llama 3.1 8B Model Interface")
    
    # List available models
    models = list_models()
    if not models:
        gr.Markdown("❌ No GGUF models found in D:/llm_models")
    else:
        gr.Markdown(f"✅ Found {len(models)} GGUF model(s)")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=models,
                label="Select Model",
                value=models[0] if models else None
            )
            
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="Write a short poem about artificial intelligence.",
                lines=4
            )
            
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=50, 
                    maximum=1000,
                    value=200,
                    step=50,
                    label="Max Tokens"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
            
            submit_btn = gr.Button("Generate Response")
        
        with gr.Column():
            output_text = gr.Textbox(label="Response", lines=10)
    
    submit_btn.click(
        fn=generate_response,
        inputs=[model_dropdown, prompt_input, max_tokens, temperature],
        outputs=output_text
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False) 