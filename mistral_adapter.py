import os
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Hugging Face credentials from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

try:
    # Import Mistral-specific libraries if available
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_inference.transformer import Transformer
    from mistral_inference.generate import generate
    MISTRAL_NATIVE_AVAILABLE = True
except ImportError:
    MISTRAL_NATIVE_AVAILABLE = False
    
# Fallback to transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
    
    # Login to Hugging Face if token is available
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print(f"Logged in to Hugging Face as {HF_USERNAME}")
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class MistralAgent:
    """Agent implementation specific to Mistral models using native libraries when available"""
    
    def __init__(
        self,
        model_path: str,
        agent_id: int = 0,
        temperature: float = 0.7,
        use_native: bool = True,
        device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        verbose: bool = False,
        use_auth: bool = True
    ):
        """Initialize Mistral agent.
        
        Args:
            model_path: Path to model folder or HF model name
            agent_id: Unique identifier for this agent
            temperature: Temperature parameter for generation
            use_native: Whether to use native Mistral libraries if available
            device: Device to run inference on
            verbose: Whether to print detailed logs
            use_auth: Whether to use Hugging Face authentication
        """
        self.agent_id = agent_id
        self.verbose = verbose
        self.temperature = temperature
        self.device = device
        self.model_path = model_path
        self.use_auth = use_auth and HF_TOKEN is not None
        
        # Check if native Mistral libraries are available and should be used
        self.use_native = use_native and MISTRAL_NATIVE_AVAILABLE
        
        if self.verbose:
            print(f"Agent {agent_id}: Initializing Mistral with {model_path}")
            print(f"Using {'native Mistral' if self.use_native else 'transformers'} libraries")
            if self.use_auth:
                print(f"Using Hugging Face authentication as {HF_USERNAME}")
        
        if self.use_native:
            # Initialize with native Mistral libraries
            self.tokenizer = MistralTokenizer.v1()
            self.model = Transformer.from_folder(model_path)
        elif TRANSFORMERS_AVAILABLE:
            # Initialize with HuggingFace transformers
            if os.path.exists(model_path):
                # Local model path
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                # Remote model, potentially requiring authentication
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    token=HF_TOKEN if self.use_auth else None
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    token=HF_TOKEN if self.use_auth else None
                )
            
            # Move model to specified device
            if self.device != "cpu":
                self.model.to(self.device)
        else:
            raise ImportError("Neither Mistral native libraries nor transformers are available")
        
        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self.embedding_model = None
            if self.verbose:
                print("SentenceTransformer not available, embeddings will not be generated")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        previous_result: Optional[str] = None,
        improvement_instruction: Optional[str] = None
    ) -> Dict:
        """Generate a response using Mistral model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            previous_result: Optional result from a previous agent to improve upon
            improvement_instruction: Optional specific instruction for improvement
            
        Returns:
            dict: Response containing the generated text and metadata
        """
        import time
        start_time = time.time()
        
        # Prepare conversation
        if previous_result:
            if improvement_instruction:
                messages = [
                    {"role": "user", "content": f"Previous response: {previous_result}\n\nOriginal query: {prompt}\n\nImprovement instruction: {improvement_instruction}"},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": previous_result},
                    {"role": "user", "content": "Please improve your previous response. Make it more accurate, detailed and helpful."}
                ]
        else:
            messages = [{"role": "user", "content": prompt}]
            
        # Generate response
        if self.use_native:
            # Use native Mistral libraries
            mistral_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    mistral_messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    mistral_messages.append(AssistantMessage(content=msg["content"]))
                    
            completion_request = ChatCompletionRequest(messages=mistral_messages)
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            
            out_tokens, _ = generate(
                [tokens], 
                self.model, 
                max_tokens=max_tokens, 
                temperature=self.temperature,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
            
            text = self.tokenizer.decode(out_tokens[0])
            tokens_used = len(tokens) + len(out_tokens[0])
            
        else:
            # Use transformers
            encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
            
            if self.device != "cpu":
                encodeds = encodeds.to(self.device)
                
            generated_ids = self.model.generate(
                encodeds, 
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.temperature
            )
            
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            # Find the last assistant response in the decoded text
            parts = text.split("[/INST]")
            if len(parts) > 1:
                text = parts[-1].strip()
                
            tokens_used = len(encodeds[0]) + len(generated_ids[0]) - len(encodeds[0])
        
        generation_time = time.time() - start_time
        
        # Create embedding for the response
        if self.embedding_model:
            embedding = self.embedding_model.encode(text)
            embedding_list = embedding.tolist()
        else:
            # Create a dummy embedding if no embedding model
            embedding_list = [0.0] * 384  # Standard embedding size
        
        result = {
            "agent_id": self.agent_id,
            "text": text,
            "embedding": embedding_list,
            "generation_time": generation_time,
            "tokens_used": tokens_used,
            "improved_from": self.agent_id - 1 if previous_result else None
        }
        
        if self.verbose:
            print(f"Agent {self.agent_id} generated {len(text)} chars in {generation_time:.2f}s")
            
        return result
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string.
        
        Args:
            text: The text to embed
            
        Returns:
            List[float]: The vector embedding
        """
        if self.embedding_model:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        else:
            # Return dummy embedding if no embedding model
            return [0.0] * 384
        
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(similarity) 