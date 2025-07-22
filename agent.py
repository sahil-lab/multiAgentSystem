import os
import time
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import gc

class Agent:
    """Agent class that wraps a local LLM for inference with improvement capabilities"""
    

    def __init__(
        self,
        model_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        context_size: int = 4096,
        agent_id: int = 0,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """Initialize an Agent with a local LLM.
        
        Args:
            model_path: Path to the GGUF model file
            embedding_model: Name or path to sentence transformer model for embeddings
            context_size: Context window size for the LLM
            agent_id: Unique identifier for this agent
            temperature: Temperature parameter for generation
            verbose: Whether to print detailed logs
        """
        self.agent_id = agent_id
        self.verbose = verbose
        self.temperature = temperature
        
        if self.verbose:
            print(f"Agent {agent_id}: Initializing with {model_path}")
        
        # define global parameters for usage in prompt generation later
        self.model_path = model_path
        self.context_size = context_size
        self.verbose = verbose
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        previous_result: Optional[str] = None,
        improvement_instruction: Optional[str] = None
    ) -> Dict:
        """Generate a response to the prompt, optionally refining a previous result.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            previous_result: Optional result from a previous agent to improve upon
            improvement_instruction: Optional specific instruction for improvement
            
        Returns:
            dict: Response containing the generated text and metadata
        """
        start_time = time.time()
        
        # If we have a previous result to improve upon
        if previous_result:
            if improvement_instruction:
                full_prompt = f"""
Previous agent generated this response:
'''
{previous_result}
'''

Instruction for improvement: {improvement_instruction}

Original prompt: {prompt}

Your improved response:
"""
            else:
                full_prompt = f"""
Previous agent generated this response:
'''
{previous_result}
'''

Review and improve the response above to make it more accurate, detailed, and helpful.
Original prompt: {prompt}

Your improved response:
"""
        else:
            full_prompt = prompt

        # Load LLM
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_batch=512,
            verbose=self.verbose
        )

        # Generate response
        response = self.llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stop=["</s>", "Human:", "User:"],
            echo=False
        )

        # Remove LLM from memory
        del self.llm
        gc.collect()
        
        generation_time = time.time() - start_time
        
        # Extract text
        text = response["choices"][0]["text"].strip()
        
        # Create embedding for the response
        embedding = self.embedding_model.encode(text)
        
        result = {
            "agent_id": self.agent_id,
            "text": text,
            "embedding": embedding.tolist(),
            "generation_time": generation_time,
            "tokens_used": response["usage"]["total_tokens"],
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
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
        
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