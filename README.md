# Multi-Agent Iterative AI System

A locally-running AI system that uses multiple agents to iteratively refine responses, producing higher quality outputs than a single agent could achieve alone.

## Features

- **Multiple LLM Agents**: Run 1 to potentially millions of local AI agents that build upon each other's responses
- **Fully Local**: Works entirely on your machine with no API calls or cloud dependencies
- **Convergence Detection**: Automatically detects when responses have stabilized
- **Vector Storage**: Keeps track of all iterations using vector embeddings
- **Response Benchmarking**: Evaluates responses based on semantic coherence, structure, and other metrics
- **Web UI**: Easy-to-use interface for configuring and using the system

## How It Works

1. The system loads multiple instances of the same local LLM model as separate "agents"
2. When you ask a question, the first agent generates an initial response
3. Subsequent agents iteratively refine this response, each building upon the previous agent's output
4. The system can use different temperature settings per agent to encourage diversity of thought
5. The result is a higher-quality answer than what a single agent would produce

This approach mimics the human thought process of drafting and refining ideas, leveraging the strengths of the underlying model multiple times.

## Technical Overview

- Uses llama.cpp and the GGUF model format for efficient local inference
- Implements semantic search via SentenceTransformers for embedding generation
- Stores iteration history in a ChromaDB vector database
- Provides real-time quality metrics using a custom benchmarking system
- Gradio-based UI that displays progress and allows dynamic configuration

## Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- GGUF model file(s) - can be downloaded using the setup scripts
- GPU acceleration supported but optional

## Installation

### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-ai.git
cd multi-agent-ai

# Make setup script executable 
chmod +x setup.sh

# Run setup script (will install dependencies and help download models)
./setup.sh
```

### Windows

```
# Clone the repository
git clone https://github.com/yourusername/multi-agent-ai.git
cd multi-agent-ai

# Run setup script (will install dependencies and help download models)
setup_windows.bat
```

## Running the System

After setup is complete:

```bash
python app.py
```

Then open your web browser to http://localhost:7860

## Usage

1. **Initialize the System**:
   - Select your model
   - Choose the number of agents (start with 3-5 for testing)
   - Configure temperature strategy and convergence settings
   - Click "Initialize System"

2. **Query the System**:
   - Enter your question
   - Optionally provide specific improvement instructions
   - Click "Process Query"
   - View the refined result or individual iterations

3. **Adjust as Needed**:
   - Increase agent count for more iterations
   - Modify convergence threshold to control refinement depth
   - Experiment with different temperature strategies

## Advanced Configuration

The system can be further configured by modifying the following:

- `agent.py`: Customize agent behavior and inference settings
- `orchestrator.py`: Adjust the orchestration logic between agents
- `benchmarking.py`: Create custom metrics for response evaluation

## Models

This system works with any GGUF format models. Recommended models:

- Llama 2 7B Chat (Q4_K_M quantized)
- Mistral 7B Instruct v0.2 (Q4_K_M quantized)
- NeuralChat 7B v3.1 (Q4_K_M quantized)
- Phi-2 (Q4_K_M quantized)

The setup scripts can automatically download these models.

## Performance Notes

- Start with fewer agents (3-5) to test the system
- The UI limits you to 50 agents, but the backend supports up to 1 million (hardware permitting)
- Response time scales roughly linearly with the number of agents
- GPU acceleration significantly improves performance

## License

MIT License 