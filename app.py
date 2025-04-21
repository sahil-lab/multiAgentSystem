import os
import json
import time
import gradio as gr
import numpy as np
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agent import Agent
from orchestrator import Orchestrator
from benchmarking import ResponseBenchmark

# Try importing Mistral adapter
try:
    from mistral_adapter import MistralAgent, MISTRAL_NATIVE_AVAILABLE
    MISTRAL_SUPPORT = True
except ImportError:
    MISTRAL_SUPPORT = False
    MISTRAL_NATIVE_AVAILABLE = False

# Try importing vLLM adapter
try:
    from vllm_adapter import VLLMAgent, VLLM_AVAILABLE
    VLLM_SUPPORT = True
except ImportError:
    VLLM_SUPPORT = False
    VLLM_AVAILABLE = False

# Constants
DEFAULT_MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_M.gguf"  # Update with your model path
DEFAULT_NUM_AGENTS = 3
MAX_AGENTS = 1000000  # Increased to 1 million as requested
VECTOR_DB_PATH = "./chroma_db"
MODEL_OPTIONS = [
    "./models/llama-2-7b-chat.Q4_K_M.gguf",
    "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
    "./models/neural-chat-7b-v3-1.Q4_K_M.gguf",
    "./models/phi-2.Q4_K_M.gguf",
    "mistralai/Mistral-7B-Instruct-v0.2"  # HF model path for direct loading
]

# HF credentials from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

# Create benchmark instance
benchmark = ResponseBenchmark()

# Global orchestrator instance
orchestrator = None

def initialize_orchestrator(
    model_path: str, 
    num_agents: int, 
    temperature_strategy: str,
    convergence_threshold: float,
    max_iterations: int,
    use_mistral_native: bool = False,
    use_vllm: bool = False,
    use_hf_auth: bool = False,
    tensor_parallel_size: int = 1
) -> Dict:
    """Initialize the orchestrator with the selected parameters.
    
    Args:
        model_path: Path to the model file
        num_agents: Number of agents to create
        temperature_strategy: Temperature strategy to use
        convergence_threshold: Convergence threshold
        max_iterations: Maximum iterations
        use_mistral_native: Whether to use Mistral native libraries
        use_vllm: Whether to use vLLM engine
        use_hf_auth: Whether to use Hugging Face authentication
        tensor_parallel_size: Number of GPUs for tensor parallelism with vLLM
        
    Returns:
        Dict: Status information
    """
    global orchestrator
    
    start_time = time.time()
    
    try:
        # Check if model file exists or is a HF model path
        is_hf_model = not os.path.exists(model_path) and "/" in model_path
        
        if not is_hf_model and not os.path.exists(model_path):
            return {
                "status": "error",
                "message": f"Model file not found at {model_path}. Please download the model or update the path."
            }
            
        if is_hf_model and not HF_TOKEN and use_hf_auth:
            return {
                "status": "error",
                "message": f"Hugging Face token not found in environment. Please set HF_TOKEN in .env file."
            }
        
        # Determine agent factory
        agent_factory = None
        
        # vLLM takes precedence if selected
        if use_vllm and VLLM_SUPPORT:
            agent_factory = lambda model_path, agent_id, temperature, verbose: VLLMAgent(
                model_path=model_path,
                agent_id=agent_id,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                use_auth=use_hf_auth,
                verbose=verbose
            )
            agent_type = "VLLMAgent"
        # Mistral agent if selected and available
        elif use_mistral_native and MISTRAL_SUPPORT and ("mistral" in model_path.lower() or is_hf_model):
            if MISTRAL_NATIVE_AVAILABLE:
                agent_factory = lambda model_path, agent_id, temperature, verbose: MistralAgent(
                    model_path=model_path,
                    agent_id=agent_id,
                    temperature=temperature,
                    use_native=True,
                    verbose=verbose,
                    use_auth=use_hf_auth
                )
                agent_type = "MistralAgent (native)"
            else:
                agent_factory = lambda model_path, agent_id, temperature, verbose: MistralAgent(
                    model_path=model_path,
                    agent_id=agent_id,
                    temperature=temperature,
                    use_native=False,
                    verbose=verbose,
                    use_auth=use_hf_auth
                )
                agent_type = "MistralAgent (transformers)"
        else:
            # Default to regular Agent if not HF model
            if is_hf_model and not VLLM_SUPPORT and not MISTRAL_SUPPORT:
                return {
                    "status": "error",
                    "message": f"For HF models, please enable VLLM or Mistral support. Regular Agent only supports local GGUF models."
                }
                
            agent_factory = None
            agent_type = "Agent (llama.cpp)"
            
        # Create new orchestrator
        orchestrator = Orchestrator(
            model_path=model_path,
            num_agents=num_agents,
            vector_db_path=VECTOR_DB_PATH,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            temperature_strategy=temperature_strategy,
            verbose=True,
            agent_factory=agent_factory
        )
        
        init_time = time.time() - start_time
        
        auth_info = f" with HF auth as {HF_USERNAME}" if use_hf_auth else ""
        gpu_info = f" using {tensor_parallel_size} GPU(s)" if use_vllm and tensor_parallel_size > 1 else ""
        
        return {
            "status": "success",
            "message": f"Initialized system with {num_agents} agents ({agent_type}{auth_info}{gpu_info}) in {init_time:.2f}s",
            "model_path": model_path,
            "num_agents": num_agents,
            "agent_type": agent_type,
            "using_hf_auth": use_hf_auth,
            "using_vllm": use_vllm
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error initializing system: {str(e)}"
        }

def process_query(
    query: str,
    improvement_instruction: str,
    max_tokens: int,
    stop_on_convergence: bool,
    show_all_iterations: bool
) -> Union[str, Dict, List]:
    """Process a query through the agent system.
    
    Args:
        query: User query to process
        improvement_instruction: Optional improvement instruction
        max_tokens: Maximum tokens to generate
        stop_on_convergence: Whether to stop when converged
        show_all_iterations: Whether to show all iterations
        
    Returns:
        Union[str, Dict, List]: Results
    """
    global orchestrator
    
    if orchestrator is None:
        return "Error: System not initialized. Please initialize the system first."
    
    try:
        # Process the query
        results = orchestrator.process_query(
            query=query,
            max_tokens=max_tokens,
            improvement_instruction=improvement_instruction if improvement_instruction else None,
            stop_on_convergence=stop_on_convergence,
            benchmark_fn=benchmark.get_benchmark_fn()
        )
        
        if show_all_iterations:
            # Return all iterations for display
            iterations_data = []
            
            for i, iteration in enumerate(results["iterations"]):
                iter_data = {
                    "iteration": i + 1,
                    "agent_id": iteration["agent_id"],
                    "text": iteration["text"],
                    "generation_time": f"{iteration['generation_time']:.2f}s",
                    "tokens": iteration["tokens_used"]
                }
                
                if "benchmark_score" in iteration:
                    iter_data["score"] = f"{iteration['benchmark_score']:.4f}"
                    
                if "similarity_to_previous" in iteration:
                    iter_data["similarity"] = f"{iteration['similarity_to_previous']:.4f}"
                    
                iterations_data.append(iter_data)
                
            return iterations_data
        else:
            # Return just the best result
            best_result = results["best_result"]
            best_index = results["best_iteration"]
            
            response = f"Best result (iteration {best_index + 1}):\n\n{best_result['text']}"
            
            metrics = f"\n\nMetrics:\n"
            metrics += f"- Selection method: {results['selection_method']}\n"
            metrics += f"- Processing time: {results['processing_time']:.2f}s\n"
            metrics += f"- Iterations: {results['num_iterations']}\n"
            
            if "benchmark_score" in best_result:
                metrics += f"- Score: {best_result['benchmark_score']:.4f}\n"
                
            if results["converged"]:
                metrics += f"- Converged at iteration: {results['convergence_iteration'] + 1}\n"
                
            return response + metrics
            
    except Exception as e:
        return f"Error processing query: {str(e)}"

def update_num_agents(num_agents: int) -> str:
    """Update the number of agents in the orchestrator.
    
    Args:
        num_agents: New number of agents
        
    Returns:
        str: Status message
    """
    global orchestrator
    
    if orchestrator is None:
        return "Error: System not initialized. Please initialize the system first."
    
    try:
        start_time = time.time()
        orchestrator.set_num_agents(num_agents)
        update_time = time.time() - start_time
        
        return f"Updated to {num_agents} agents in {update_time:.2f}s"
    except Exception as e:
        return f"Error updating agents: {str(e)}"

# Build the Gradio interface
with gr.Blocks(title="Multi-Agent AI System") as app:
    gr.Markdown("# Multi-Agent Iterative AI System")
    gr.Markdown("A locally-running system that iteratively refines responses using multiple agents")
    
    with gr.Tab("Initialize System"):
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    label="LLM Model",
                    value=MODEL_OPTIONS[0] if MODEL_OPTIONS else DEFAULT_MODEL_PATH,
                    info="Select the LLM model to use for all agents"
                )
                
                custom_model_path = gr.Textbox(
                    label="Custom Model Path (optional)",
                    placeholder="Path to your GGUF model file or HF model name",
                    info="Leave empty to use the selected model from dropdown"
                )
                
                num_agents_slider = gr.Slider(
                    minimum=1,
                    maximum=MAX_AGENTS,
                    value=DEFAULT_NUM_AGENTS,
                    step=1,
                    label="Number of Agents",
                    info=f"Number of agents (1-{MAX_AGENTS}) to create for iterative refinement"
                )
                
                temp_strategy = gr.Radio(
                    choices=["fixed", "decreasing", "random"],
                    value="decreasing",
                    label="Temperature Strategy",
                    info="How to set temperatures across multiple agents"
                )
                
                convergence_threshold = gr.Slider(
                    minimum=0.8,
                    maximum=0.99,
                    value=0.98,
                    step=0.01,
                    label="Convergence Threshold",
                    info="Similarity threshold to consider result converged"
                )
                
                max_iterations = gr.Slider(
                    minimum=2,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Maximum Iterations",
                    info="Maximum iterations regardless of convergence"
                )
                
                with gr.Group():
                    gr.Markdown("### Backend Options")
                    
                    use_vllm = gr.Checkbox(
                        label="Use vLLM Engine",
                        value=False,
                        info="Use vLLM for efficient inference (fastest option)",
                        visible=VLLM_AVAILABLE
                    )
                    
                    tensor_parallel_size = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=1,
                        step=1,
                        label="Tensor Parallel Size (GPUs)",
                        info="Number of GPUs to use for tensor parallelism with vLLM",
                        visible=VLLM_AVAILABLE
                    )
                    
                    use_mistral_native = gr.Checkbox(
                        label="Use Mistral Libraries",
                        value=False,
                        info="Use Mistral native libraries for Mistral models",
                        visible=MISTRAL_SUPPORT
                    )
                    
                    use_hf_auth = gr.Checkbox(
                        label=f"Use HF Authentication ({HF_USERNAME})" if HF_USERNAME else "Use HF Authentication",
                        value=False,
                        info="Use Hugging Face token for accessing models (requires token in .env)",
                        visible=HF_TOKEN is not None
                    )
                
                init_btn = gr.Button("Initialize System", variant="primary")
                
            with gr.Column():
                system_status = gr.JSON(
                    label="System Status",
                    value={"status": "not_initialized", "message": "System not initialized"}
                )
                
    with gr.Tab("Query System"):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter your question here...",
                    lines=5
                )
                
                improvement_instruction = gr.Textbox(
                    label="Improvement Instruction (optional)",
                    placeholder="Specific instructions for improvement (e.g., 'Make it more concise')",
                    lines=2
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=128,
                    maximum=4096,
                    value=512,
                    step=128,
                    label="Max Tokens per Generation"
                )
                
                stop_on_convergence = gr.Checkbox(
                    label="Stop on Convergence",
                    value=True,
                    info="Stop iterating when results converge"
                )
                
                show_iterations = gr.Checkbox(
                    label="Show All Iterations",
                    value=False,
                    info="Display all iterations instead of just the best result"
                )
                
                update_agents_slider = gr.Slider(
                    minimum=1,
                    maximum=MAX_AGENTS,
                    value=DEFAULT_NUM_AGENTS,
                    step=1,
                    label="Update Number of Agents"
                )
                
                update_agents_btn = gr.Button("Update Agent Count")
                
                query_btn = gr.Button("Process Query", variant="primary")
                
            with gr.Column():
                output = gr.Textbox(
                    label="Result",
                    lines=20
                )
                
                iterations_table = gr.Dataframe(
                    headers=["iteration", "agent_id", "text", "generation_time", "tokens", "score", "similarity"],
                    label="All Iterations",
                    visible=False
                )
    
    # Connect components with functions
    init_btn.click(
        fn=initialize_orchestrator,
        inputs=[
            model_dropdown,
            num_agents_slider,
            temp_strategy,
            convergence_threshold,
            max_iterations,
            use_mistral_native,
            use_vllm,
            use_hf_auth,
            tensor_parallel_size
        ],
        outputs=[system_status]
    )
    
    def handle_model_path(dropdown_path, custom_path):
        if custom_path and os.path.exists(custom_path):
            return custom_path
        return dropdown_path
        
    model_path_fn = handle_model_path
    
    update_agents_btn.click(
        fn=update_num_agents,
        inputs=[update_agents_slider],
        outputs=[output]
    )
    
    def route_query_output(
        query, improvement, max_tokens, stop_convergence, show_iterations
    ):
        result = process_query(
            query, improvement, max_tokens, stop_convergence, show_iterations
        )
        
        if show_iterations:
            return gr.update(visible=False), gr.update(visible=True, value=result)
        else:
            return gr.update(visible=True, value=result), gr.update(visible=False)
    
    query_btn.click(
        fn=route_query_output,
        inputs=[
            query_input,
            improvement_instruction,
            max_tokens_slider,
            stop_on_convergence,
            show_iterations
        ],
        outputs=[
            output,
            iterations_table
        ]
    )
    
    show_iterations.change(
        fn=lambda x: (gr.update(visible=not x), gr.update(visible=x)), 
        inputs=[show_iterations],
        outputs=[output, iterations_table]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0") 