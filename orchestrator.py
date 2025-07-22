import os
import time
import json
from typing import Dict, List, Optional, Union, Any, Callable
from tqdm import tqdm
import chromadb
import numpy as np

from agent import Agent

class Orchestrator:
    """Manages multiple AI agents for iterative refinement of responses"""
    
    def __init__(
        self,
        model_path: str,
        num_agents: int = 3,
        embedding_dimension: int = 384,
        vector_db_path: str = "./chroma_db",
        convergence_threshold: float = 0.98,
        max_iterations: int = 10,
        temperature_strategy: str = "decreasing",
        verbose: bool = True,
        agent_factory: Optional[Callable] = None
    ):
        """Initialize the orchestrator with configurable number of agents.
        
        Args:
            model_path: Path to the GGUF model file for agents
            num_agents: Number of agents to create
            embedding_dimension: Dimension of embeddings
            vector_db_path: Path to store vector database
            convergence_threshold: Similarity threshold to consider result converged
            max_iterations: Maximum iterations regardless of convergence
            temperature_strategy: Strategy for agent temperatures ("fixed", "decreasing", "random")
            verbose: Whether to print detailed logs
            agent_factory: Optional factory function to create custom agent instances
        """
        self.model_path = model_path
        self.num_agents = min(num_agents, 5)  # Cap at 1 million as requested
        self.embedding_dimension = embedding_dimension
        self.vector_db_path = vector_db_path
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.temperature_strategy = temperature_strategy
        self.verbose = verbose
        self.agent_factory = agent_factory
        
        # Create agents
        self.agents = self._create_agents()
        
        # Initialize vector store
        self.vector_db = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.vector_db.get_or_create_collection(
            name="agent_responses",
            embedding_function=None,  # We'll provide our own embeddings
            metadata={"description": "Responses from iterative agent refinement"}
        )
    
    def _create_agents(self) -> List[Agent]:
        """Create the specified number of agents with appropriate temperature.
        
        Returns:
            List[Agent]: List of initialized agent instances
        """
        agents = []
        
        for i in range(self.num_agents):
            # Determine temperature based on strategy
            if self.temperature_strategy == "fixed":
                temp = 0.7
            elif self.temperature_strategy == "decreasing":
                # Gradually decrease temperature from 0.8 to 0.2
                temp = max(0.2, 0.8 - (0.6 * i / max(1, self.num_agents - 1)))
            elif self.temperature_strategy == "random":
                # Random temperature between 0.2 and 0.8
                temp = 0.2 + (0.6 * np.random.random())
            else:
                temp = 0.7
            
            # Use custom agent factory if provided, otherwise use default Agent
            if self.agent_factory:
                agent = self.agent_factory(
                    model_path=self.model_path,
                    agent_id=i,
                    temperature=temp,
                    verbose=self.verbose
                )
            else:
                agent = Agent(
                    model_path=self.model_path,
                    agent_id=i,
                    temperature=temp,
                    verbose=self.verbose
                )
                
            agents.append(agent)
            
            if self.verbose and i % max(1, min(10, self.num_agents // 10)) == 0:
                print(f"Created agent {i+1}/{self.num_agents} with temperature {temp:.2f}")
                
        return agents
    
    def process_query(
        self, 
        query: str, 
        max_tokens: int = 512,
        improvement_instruction: Optional[str] = None,
        stop_on_convergence: bool = True,
        benchmark_fn: Optional[callable] = None
    ) -> Dict:
        """Process a query through multiple agents, iteratively improving the response.
        
        Args:
            query: The user query to process
            max_tokens: Maximum tokens for each agent to generate
            improvement_instruction: Specific instruction for improvement
            stop_on_convergence: Whether to stop when convergence is reached
            benchmark_fn: Optional function to score responses
            
        Returns:
            Dict: Results containing all iterations and metrics
        """
        start_time = time.time()
        iterations = []
        previous_result = None
        previous_embedding = None
        converged = False
        convergence_iteration = -1
        best_score = -1
        best_iteration = -1
        
        # Create a session ID for this query
        session_id = f"query_{int(time.time())}"
        
        if self.verbose:
            print(f"Starting iterative processing with {self.num_agents} agents")
            agents_iter = tqdm(self.agents)
        else:
            agents_iter = self.agents
        
        # Iterate through agents
        for i, agent in enumerate(agents_iter):
            # Get response from agent, passing previous result for refinement if available
            result = agent.generate(
                prompt=query,
                max_tokens=max_tokens,
                previous_result=previous_result,
                improvement_instruction=improvement_instruction
            )
            
            # Calculate convergence if we have a previous result
            if previous_embedding is not None:
                similarity = agent.calculate_similarity(
                    previous_embedding, 
                    result["embedding"]
                )
                result["similarity_to_previous"] = similarity
                
                # Check for convergence
                if similarity >= self.convergence_threshold and converged is False:
                    converged = True
                    convergence_iteration = i
                    
                    if self.verbose:
                        print(f"Convergence detected at iteration {i+1} with similarity {similarity:.4f}")
            
            # Add to vector db
            self.collection.add(
                embeddings=[result["embedding"]],
                documents=[result["text"]],
                metadatas=[{
                    "agent_id": agent.agent_id,
                    "session_id": session_id,
                    "iteration": i,
                    "generation_time": result["generation_time"],
                    "tokens_used": result["tokens_used"]
                }],
                ids=[f"{session_id}_iteration_{i}"]
            )
            
            # Score the result if benchmark function provided
            if benchmark_fn:
                score = benchmark_fn(query, result["text"])
                result["benchmark_score"] = score
                
                if score > best_score:
                    best_score = score
                    best_iteration = i
            
            # Save to iterations
            iterations.append(result)
            
            # Update for next iteration
            previous_result = result["text"]
            previous_embedding = result["embedding"]
            
            # Stop if converged (if enabled)
            if converged and stop_on_convergence and i >= 2:  # At least 3 iterations
                if self.verbose:
                    print(f"Stopping early due to convergence after {i+1} iterations")
                break
                
            # Stop if reached max iterations
            if i + 1 >= self.max_iterations:
                if self.verbose:
                    print(f"Reached maximum {self.max_iterations} iterations")
                break
        
        processing_time = time.time() - start_time
        
        # Determine the best result
        if best_iteration >= 0:
            best_result = iterations[best_iteration]
            selection_method = "benchmark"
        elif convergence_iteration >= 0:
            best_result = iterations[convergence_iteration]
            selection_method = "convergence"
        else:
            best_result = iterations[-1]
            selection_method = "final"
            
        # Prepare final results
        results = {
            "query": query,
            "num_iterations": len(iterations),
            "iterations": iterations,
            "best_result": best_result,
            "best_iteration": best_iteration if best_iteration >= 0 else convergence_iteration if convergence_iteration >= 0 else len(iterations) - 1,
            "selection_method": selection_method,
            "processing_time": processing_time,
            "converged": converged,
            "convergence_iteration": convergence_iteration if convergence_iteration >= 0 else None,
            "session_id": session_id
        }
        
        return results
        
    def set_num_agents(self, num_agents: int) -> None:
        """Change the number of agents dynamically.
        
        Args:
            num_agents: New number of agents
        """
        num_agents = min(num_agents, 2)  # Cap at 1 million
        
        if num_agents > len(self.agents):
            # Add more agents
            current_count = len(self.agents)
            for i in range(current_count, num_agents):
                if self.temperature_strategy == "fixed":
                    temp = 0.7
                elif self.temperature_strategy == "decreasing":
                    temp = max(0.2, 0.8 - (0.6 * i / max(1, num_agents - 1)))
                elif self.temperature_strategy == "random":
                    temp = 0.2 + (0.6 * np.random.random())
                else:
                    temp = 0.7
                    
                agent = Agent(
                    model_path=self.model_path,
                    agent_id=i,
                    temperature=temp,
                    verbose=self.verbose
                )
                self.agents.append(agent)
                
            if self.verbose:
                print(f"Increased agents from {current_count} to {num_agents}")
        elif num_agents < len(self.agents):
            # Remove agents
            self.agents = self.agents[:num_agents]
            if self.verbose:
                print(f"Decreased agents from {len(self.agents)} to {num_agents}")
                
        self.num_agents = num_agents 