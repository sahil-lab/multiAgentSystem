#!/bin/bash

# Script to set up the Multi-Agent AI System
echo "Setting up Multi-Agent AI System environment..."

# Create necessary directories
mkdir -p models
mkdir -p chroma_db

# Check if Python is installed
if command -v python3 &>/dev/null; then
    echo "Python found, installing dependencies..."
    
    # Install dependencies using pip
    python3 -m pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "Dependencies installed successfully!"
    else
        echo "Error installing dependencies. Please check requirements.txt and try again."
        exit 1
    fi
else
    echo "Python 3 not found. Please install Python 3.8 or higher and try again."
    exit 1
fi

# Function to download a model
download_model() {
    MODEL_NAME=$1
    MODEL_URL=$2
    
    if [ ! -f "models/$MODEL_NAME" ]; then
        echo "Downloading $MODEL_NAME..."
        
        # Check if curl or wget is available
        if command -v curl &>/dev/null; then
            curl -L "$MODEL_URL" -o "models/$MODEL_NAME"
        elif command -v wget &>/dev/null; then
            wget "$MODEL_URL" -O "models/$MODEL_NAME"
        else
            echo "Error: Neither curl nor wget is available. Please install one of them and try again."
            return 1
        fi
        
        if [ $? -eq 0 ]; then
            echo "$MODEL_NAME downloaded successfully!"
        else
            echo "Error downloading $MODEL_NAME."
            return 1
        fi
    else
        echo "$MODEL_NAME already exists, skipping download."
    fi
    
    return 0
}

echo "Do you want to download model files now? (y/n)"
read download_choice

if [ "$download_choice" = "y" ] || [ "$download_choice" = "Y" ]; then
    # Prompt for model selection
    echo "Which model would you like to download?"
    echo "1. Llama 2 7B Chat (4-bit quantized, ~4GB)"
    echo "2. Mistral 7B Instruct v0.2 (4-bit quantized, ~4GB)"
    echo "3. NeuralChat 7B v3.1 (4-bit quantized, ~4GB)"
    echo "4. Phi-2 (4-bit quantized, ~2GB)"
    echo "5. All of the above"
    echo "0. None/Skip"
    
    read model_choice
    
    case $model_choice in
        1)
            download_model "llama-2-7b-chat.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
            ;;
        2)
            download_model "mistral-7b-instruct-v0.2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            ;;
        3)
            download_model "neural-chat-7b-v3-1.Q4_K_M.gguf" "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf"
            ;;
        4)
            download_model "phi-2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
            ;;
        5)
            download_model "llama-2-7b-chat.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
            download_model "mistral-7b-instruct-v0.2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            download_model "neural-chat-7b-v3-1.Q4_K_M.gguf" "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf"
            download_model "phi-2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
            ;;
        0)
            echo "Skipping model download. You'll need to place model files in the 'models' directory manually."
            ;;
        *)
            echo "Invalid choice. Skipping model download."
            ;;
    esac
else
    echo "Skipping model download. You'll need to place model files in the 'models' directory manually."
fi

echo ""
echo "Setup complete! To start the system, run: python app.py"
echo "Then open your web browser to http://localhost:7860" 