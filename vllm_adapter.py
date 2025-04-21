import os
import json
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get HF token
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

try:
    from vllm import LLM, SamplingParams
    from huggingface_hub import login
    VLLM_AVAILABLE = True
    
    # Login to Hugging Face if token is available
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print(f"Logged in to Hugging Face as {HF_USERNAME}")
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available, using fallback options")
except AttributeError as e:
    # Handle errors like missing torch._inductor attribute
    if "_inductor" in str(e):
        print(f"vLLM initialization error with torch: {str(e)}")
        print("This may be due to PyTorch version incompatibility.")
        VLLM_AVAILABLE = False
    else:
        raise

# Import SentenceTransformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class VLLMAgent:
    """Agent implementation using vLLM for high-throughput inference."""
    
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        tensor_parallel_size: int = 1,
        use_hf_auth: bool = False,
        **kwargs
    ):
        """Initialize vLLM Agent.
        
        Args:
            model_path: Path or name of the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            use_hf_auth: Whether to use Hugging Face authentication
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed or incompatible. Please install it with `pip install vllm`")
            
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.llm = None
        
        # Load the model
        try:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            # Limit tensor parallel size to available GPUs
            effective_tp_size = min(tensor_parallel_size, gpu_count) if gpu_count > 0 else tensor_parallel_size
            
            # Setup Hugging Face token if needed
            if use_hf_auth and HF_TOKEN:
                hf_token = HF_TOKEN
            elif use_hf_auth and "HUGGING_FACE_TOKEN" in os.environ:
                hf_token = os.environ["HUGGING_FACE_TOKEN"]
            else:
                hf_token = None
                
            # Initialize the LLM
            try:
                self.llm = LLM(
                    model=model_path,
                    tensor_parallel_size=effective_tp_size,
                    trust_remote_code=True,
                    download_dir="./models/vllm_cache",
                    token=hf_token
                )
                print(f"Successfully loaded vLLM with model: {model_path}")
                print(f"Using tensor parallel size: {effective_tp_size}")
            except AttributeError as e:
                if "_inductor" in str(e):
                    print(f"Error initializing vLLM due to PyTorch compatibility issue: {str(e)}")
                    print("This may be fixed by installing a compatible PyTorch version for vLLM.")
                    raise ImportError(f"vLLM initialization failed: {str(e)}")
                else:
                    raise
            
            # Initialize embedding model if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Loaded embedding model: all-MiniLM-L6-v2")
            else:
                self.embedding_model = None
                print("SentenceTransformers not available, embeddings will not be generated")
            
        except Exception as e:
            print(f"Error initializing vLLM: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using vLLM.
        
        Args:
            prompt: The prompt to generate text from
            
        Returns:
            Dict with generated text and metadata
        """
        # Verify that LLM was initialized successfully
        if self.llm is None:
            return {
                "text": "",
                "error": "vLLM model was not properly initialized",
                "generation_time": 0,
                "embedding": []
            }
        
        # Override parameters with kwargs if provided
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        previous_result = kwargs.get("previous_result")
        improvement_instruction = kwargs.get("improvement_instruction")
        
        # Format prompt if needed
        formatted_prompt = self._format_prompt(prompt, previous_result, improvement_instruction)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=kwargs.get("stop", None)
        )
        
        # Measure generation time
        start_time = time.time()
        
        try:
            # Generate outputs
            outputs = self.llm.generate(formatted_prompt, sampling_params)
            generation_time = time.time() - start_time
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                
                # Get token counts
                prompt_tokens = len(outputs[0].prompt_token_ids)
                generated_tokens = len(outputs[0].outputs[0].token_ids)
                
                # Generate embedding if embedding model is available
                embedding = None
                if self.embedding_model is not None:
                    embedding = self.embedding_model.encode(generated_text).tolist()
                
                # Prepare result
                result = {
                    "text": generated_text,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "total_tokens": prompt_tokens + generated_tokens,
                    "generation_time": generation_time,
                    "embedding": embedding
                }
                
                return result
            else:
                return {
                    "text": "",
                    "error": "No output generated",
                    "generation_time": generation_time,
                    "embedding": []
                }
                
        except Exception as e:
            generation_time = time.time() - start_time
            print(f"Error during vLLM generation: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "generation_time": generation_time,
                "embedding": []
            }
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts in a batch.
        
        Args:
            prompts: List of prompts to generate text from
            
        Returns:
            List of result dictionaries
        """
        # Verify that LLM was initialized successfully
        if self.llm is None:
            return [{
                "text": "",
                "error": "vLLM model was not properly initialized",
                "generation_time": 0,
                "embedding": []
            } for _ in prompts]
        
        # Override parameters with kwargs if provided
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        
        # Get previous results and improvement instructions, if provided
        previous_results = kwargs.get("previous_results", [None] * len(prompts))
        improvement_instructions = kwargs.get("improvement_instructions", [None] * len(prompts))
        
        # Format prompts if needed
        formatted_prompts = [
            self._format_prompt(prompt, prev, instr)
            for prompt, prev, instr in zip(
                prompts, 
                previous_results[:len(prompts)], 
                improvement_instructions[:len(prompts)]
            )
        ]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=kwargs.get("stop", None)
        )
        
        # Measure generation time
        start_time = time.time()
        
        try:
            # Generate outputs in batch
            batch_outputs = self.llm.generate(formatted_prompts, sampling_params)
            generation_time = time.time() - start_time
            
            results = []
            
            for output in batch_outputs:
                if output.outputs and len(output.outputs) > 0:
                    generated_text = output.outputs[0].text
                    
                    # Get token counts
                    prompt_tokens = len(output.prompt_token_ids)
                    generated_tokens = len(output.outputs[0].token_ids)
                    
                    # Generate embedding if embedding model is available
                    embedding = None
                    if self.embedding_model is not None:
                        embedding = self.embedding_model.encode(generated_text).tolist()
                    
                    # Prepare result
                    result = {
                        "text": generated_text,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "total_tokens": prompt_tokens + generated_tokens,
                        "generation_time": generation_time / len(prompts),  # Approximate per-request time
                        "embedding": embedding
                    }
                    
                    results.append(result)
                else:
                    results.append({
                        "text": "",
                        "error": "No output generated",
                        "generation_time": generation_time / len(prompts),
                        "embedding": []
                    })
            
            return results
                
        except Exception as e:
            generation_time = time.time() - start_time
            print(f"Error during vLLM batch generation: {str(e)}")
            
            # Return error for all prompts
            return [{
                "text": "",
                "error": str(e),
                "generation_time": generation_time / len(prompts),
                "embedding": []
            } for _ in prompts]
    
    def _format_prompt(self, prompt: str, previous_result: Optional[str] = None, improvement_instruction: Optional[str] = None) -> str:
        """Format the prompt for various model types.
        
        Args:
            prompt: The input prompt
            previous_result: Optional previous result to improve
            improvement_instruction: Optional improvement instruction
            
        Returns:
            str: Formatted prompt
        """
        is_mistral = "mistral" in self.model_path.lower()
        
        if previous_result:
            if improvement_instruction:
                if is_mistral:
                    return f"<s>[INST] Previous response: {previous_result}\n\nOriginal query: {prompt}\n\nImprovement instruction: {improvement_instruction} [/INST]"
                else:
                    return f"Previous response: {previous_result}\n\nOriginal query: {prompt}\n\nImprovement instruction: {improvement_instruction}\n\nYour improved response:"
            else:
                if is_mistral:
                    return f"<s>[INST] Previous response: {previous_result}\n\nOriginal query: {prompt}\n\nPlease improve the previous response. [/INST]"
                else:
                    return f"Previous response: {previous_result}\n\nOriginal query: {prompt}\n\nPlease improve the previous response:\n\n"
        else:
            # No previous result, just format the prompt for the model
            if is_mistral:
                return f"<s>[INST] {prompt} [/INST]"
            else:
                return prompt
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(similarity)
        
    def __del__(self):
        """Clean up resources when this agent is deleted."""
        # Release GPU memory
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                del self.llm
        except:
            pass 