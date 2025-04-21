import os
from huggingface_hub import hf_hub_download, list_repo_files
import tempfile

# Set HuggingFace cache to D drive
cache_dir = "D:/hf_cache"  # Custom cache on D drive
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
print(f"Using custom HuggingFace cache at: {cache_dir}")

# Model details
model_id = "Pabitra09/Llama-3_8b_fine_tuning_with_cpp_to_python_conversion_gguf_encoding"

# List all files in the repository
print(f"Listing all files in the repository: {model_id}")
try:
    files = list_repo_files(model_id)
    print("Available files:")
    for file in files:
        print(f"  - {file}")
    
    # Look for smaller GGUF files (try Q4_K_M instead of F16)
    gguf_files = [f for f in files if f.endswith('.gguf') and 'Q4_K_M' in f]
    if not gguf_files:
        gguf_files = [f for f in files if f.endswith('.gguf')]
    
    if gguf_files:
        print("\nFound GGUF files:")
        for gguf_file in gguf_files:
            print(f"  - {gguf_file}")
        
        # Use the first GGUF file found
        filename = gguf_files[0]
        print(f"\nUsing model file: {filename}")
        
        # Create a models directory if it doesn't exist
        models_dir = "D:/llm_models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Download the model
        print(f"Downloading {filename} from {model_id}...")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            cache_dir=cache_dir
        )
        print(f"Model downloaded to: {model_path}")
        
        # Load the model
        print("Loading model with llama-cpp-python...")
        from llama_cpp import Llama
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,         # Context window size
            n_threads=4,        # CPU threads to use
            n_gpu_layers=0      # Set higher if you have a GPU
        )
        
        # Run inference
        prompt = "Write a short poem about artificial intelligence."
        print("\nGenerating response for prompt:", prompt)
        print("\nResponse:")
        output = llm(
            prompt,
            max_tokens=200,
            temperature=0.7,
            top_p=0.95,
            echo=False
        )
        
        print(output["choices"][0]["text"])
    else:
        print("\nNo GGUF files found in the repository. Available files:")
        for file in files:
            print(f"  - {file}")
        
except Exception as e:
    print(f"Error: {e}") 