import os
from huggingface_hub import hf_hub_download

# Set HuggingFace cache to D drive
cache_dir = "D:/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
print(f"Using custom HuggingFace cache at: {cache_dir}")

# Set model output directory
model_dir = "D:/llm_models"
os.makedirs(model_dir, exist_ok=True)

# Model details - using Mistral 7B instead
model_id = "TheBloke/Mistral-7B-v0.1-GGUF"
filename = "mistral-7b-v0.1.Q4_K_M.gguf"  # Smaller 4-bit quantized model

print(f"Downloading {filename} from {model_id}...")
model_path = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    cache_dir=cache_dir
)
print(f"Model downloaded to: {model_path}")
print("This model should work better with llama-cpp-python on Windows.") 