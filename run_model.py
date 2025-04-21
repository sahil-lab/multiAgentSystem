import os
from llama_cpp import Llama

# Find the model file
model_dir = "D:/llm_models"
model_pattern = "unsloth"

# Look for the model in the models directory
model_files = []
for root, dirs, files in os.walk(model_dir):
    for file in files:
        if model_pattern in file and file.endswith('.gguf'):
            model_files.append(os.path.join(root, file))

if not model_files:
    print(f"Model file containing '{model_pattern}' not found in {model_dir}")
    exit(1)

# Use the first found model
model_path = model_files[0]
print(f"Found model at: {model_path}")

# Load the model
print("Loading model with llama-cpp-python...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,       # Context window size
    n_threads=4,      # CPU threads to use
    n_gpu_layers=0    # Set higher if you have a GPU
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