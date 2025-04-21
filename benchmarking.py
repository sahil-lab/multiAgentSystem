import re
import numpy as np
from typing import Dict, List, Union, Optional, Callable
from sentence_transformers import SentenceTransformer

class ResponseBenchmark:
    """Benchmarking tools for evaluating agent responses"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the benchmarking system.
        
        Args:
            embedding_model: Model to use for semantic similarity calculations
        """
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def semantic_coherence(self, query: str, response: str) -> float:
        """Measure semantic coherence between query and response.
        
        Args:
            query: The original query
            response: The generated response
            
        Returns:
            float: Coherence score between 0-1
        """
        query_embedding = self.embedding_model.encode(query)
        response_embedding = self.embedding_model.encode(response)
        
        # Compute cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        response_norm = np.linalg.norm(response_embedding)
        
        if query_norm == 0 or response_norm == 0:
            return 0.0
            
        similarity = np.dot(query_embedding, response_embedding) / (query_norm * response_norm)
        return float(similarity)
    
    def length_quality(self, response: str, min_length: int = 50, optimal_length: int = 500) -> float:
        """Score based on length - penalizes too short or excessively long responses.
        
        Args:
            response: The generated response
            min_length: Minimum acceptable length
            optimal_length: Optimal response length
            
        Returns:
            float: Length quality score between 0-1
        """
        length = len(response)
        
        if length < min_length:
            # Penalize very short responses
            return length / min_length
        elif length <= optimal_length:
            # Reward responses approaching optimal length
            return 1.0
        else:
            # Gradually penalize overly verbose responses
            return max(0.5, 1.0 - 0.5 * ((length - optimal_length) / optimal_length))
    
    def text_structure(self, response: str) -> float:
        """Evaluate text structure quality.
        
        Args:
            response: The generated response
            
        Returns:
            float: Structure quality score between 0-1
        """
        # Check for paragraphs
        paragraphs = response.split('\n\n')
        if len(paragraphs) < 2:
            paragraphs = response.split('\n')
        
        # Reasonable number of paragraphs
        paragraph_score = min(1.0, len(paragraphs) / 3)
        
        # Check for sentences
        sentences = re.split(r'[.!?]+', response)
        sentence_score = min(1.0, len(sentences) / 5)
        
        # Check for lists or structured elements
        has_lists = 1.0 if (re.search(r'\n[-*] ', response) or 
                           re.search(r'\n\d+\.', response)) else 0.5
        
        # Calculate final structure score
        structure_score = (paragraph_score * 0.4 + 
                          sentence_score * 0.3 + 
                          has_lists * 0.3)
                          
        return structure_score
    
    def evaluate_response(
        self, 
        query: str, 
        response: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Comprehensive evaluation of a response using multiple metrics.
        
        Args:
            query: The original query
            response: The generated response
            weights: Optional custom weights for different metrics
            
        Returns:
            Dict: Detailed evaluation scores
        """
        if weights is None:
            weights = {
                "coherence": 0.5,
                "structure": 0.3,
                "length": 0.2
            }
            
        # Calculate individual scores
        coherence_score = self.semantic_coherence(query, response)
        structure_score = self.text_structure(response)
        length_score = self.length_quality(response)
        
        # Calculate weighted total
        total_score = (
            coherence_score * weights["coherence"] +
            structure_score * weights["structure"] +
            length_score * weights["length"]
        )
        
        return {
            "total_score": total_score,
            "coherence": coherence_score,
            "structure": structure_score,
            "length": length_score
        }
    
    def get_benchmark_fn(self) -> Callable:
        """Return a function that can be used for benchmarking responses.
        
        Returns:
            Callable: A function that takes (query, response) and returns a score
        """
        def benchmark_fn(query: str, response: str) -> float:
            results = self.evaluate_response(query, response)
            return results["total_score"]
            
        return benchmark_fn 