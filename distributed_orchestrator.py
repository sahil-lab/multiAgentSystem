import os
import time
import json
import uuid
import redis
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
import threading
import chromadb
from dotenv import load_dotenv
from orchestrator import Orchestrator

# Load environment variables
load_dotenv()

from benchmarking import ResponseBenchmark

class DistributedOrchestrator(Orchestrator):
    """Orchestrator that distributes work across multiple worker nodes using Redis.
    
    This implementation allows scaling to large numbers of agents by distributing
    the workload across many worker processes or machines.
    """
    
    def __init__(
        self, 
        num_agents: int = 3,
        max_tokens: int = 512, 
        temperature: float = 0.7,
        model_path: str = None,
        model_type: str = "llama",
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
        use_hf_auth: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        agent_factory: Optional[Callable] = None
    ):
        """Initialize the distributed orchestrator.
        
        Args:
            num_agents: Number of agents to use
            max_tokens: Maximum tokens for each generation
            temperature: Temperature for generation
            model_path: Path to the model (passed to workers)
            model_type: Type of model ("llama" or "mistral")
            use_vllm: Whether to use vLLM for inference
            tensor_parallel_size: Number of GPUs for tensor parallelism
            use_hf_auth: Whether to use Hugging Face authentication
            redis_host: Redis host address
            redis_port: Redis port number
            agent_factory: Optional factory function for creating agents
        """
        super().__init__(num_agents, max_tokens, temperature, agent_factory)
        
        self.model_path = model_path or os.environ.get("MODEL_PATH", "./models/llama-2-7b-chat.Q4_K_M.gguf")
        self.model_type = model_type
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.use_hf_auth = use_hf_auth
        
        # Connect to Redis
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        self.pending_jobs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Create pubsub for status updates
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe("agent_status")
        
        # Initialize vector store
        self.vector_db = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.vector_db.get_or_create_collection(
            name="agent_responses",
            embedding_function=None,  # We'll provide our own embeddings
            metadata={"description": "Responses from distributed agent refinement"}
        )
        
        # Create benchmark instance
        self.benchmark = ResponseBenchmark()
        
        # Clear any stale jobs/results
        self._clear_stale_jobs()
        
        hf_info = f" with HF authentication" if self.use_hf_auth else ""
        if self.verbose:
            print(f"Initialized distributed orchestrator with {num_agents} agents using model type {model_type}{hf_info}")
    
    def _clear_stale_jobs(self):
        """Clear any stale jobs and results from Redis."""
        # Clear job and result queues
        self.redis_client.delete("agent_jobs")
        self.redis_client.delete("agent_results")
    
    def _generate_temperature(self, agent_id: int) -> float:
        """Generate temperature based on strategy and agent ID.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            float: Temperature value
        """
        if self.temperature_strategy == "fixed":
            return 0.7
        elif self.temperature_strategy == "decreasing":
            # Gradually decrease temperature from 0.8 to 0.2
            return max(0.2, 0.8 - (0.6 * agent_id / max(1, self.num_agents - 1)))
        elif self.temperature_strategy == "random":
            # Random temperature between 0.2 and 0.8
            return 0.2 + (0.6 * np.random.random())
        else:
            return 0.7
    
    def _submit_job(self, job_data: Dict) -> str:
        """Submit a job to the Redis queue.
        
        Args:
            job_data: Job data
            
        Returns:
            str: Job ID
        """
        job_id = job_data.get("job_id", str(uuid.uuid4()))
        job_data["job_id"] = job_id
        
        # Add model information to job data
        job_data["model_type"] = self.model_type
        job_data["model_path"] = self.model_path
        job_data["use_hf_auth"] = self.use_hf_auth
        
        # Push job to queue
        self.redis_client.rpush("agent_jobs", json.dumps(job_data))
        
        if self.verbose:
            print(f"Submitted job {job_id} for agent {job_data.get('agent_id', 'unknown')}")
            
        return job_id
    
    def _wait_for_result(self, job_id: str, timeout: int = None) -> Optional[Dict]:
        """Wait for a job result.
        
        Args:
            job_id: Job ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            Optional[Dict]: Job result or None if timed out
        """
        if timeout is None:
            timeout = self.result_timeout
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check all results in the queue
            result_count = self.redis_client.llen("agent_results")
            
            for i in range(result_count):
                # Get result without removing it
                result_json = self.redis_client.lindex("agent_results", i)
                
                if result_json:
                    result = json.loads(result_json)
                    
                    if result.get("job_id") == job_id:
                        # Found our result, remove it from queue
                        self.redis_client.lrem("agent_results", 1, result_json)
                        return result
            
            # No result yet, sleep a bit
            time.sleep(0.1)
        
        if self.verbose:
            print(f"Timeout waiting for job {job_id}")
            
        return None
    
    def process_query(
        self, 
        query: str, 
        max_tokens: int = 512,
        improvement_instruction: Optional[str] = None,
        stop_on_convergence: bool = True,
        benchmark_fn: Optional[callable] = None
    ) -> Dict:
        """Process a query through distributed agents.
        
        Args:
            query: The user query to process
            max_tokens: Maximum tokens for each agent to generate
            improvement_instruction: Optional specific instruction for improvement
            stop_on_convergence: Whether to stop when convergence is reached
            benchmark_fn: Optional function to score responses
            
        Returns:
            Dict: Results containing all iterations and metrics
        """
        if benchmark_fn is None:
            benchmark_fn = self.benchmark.get_benchmark_fn()
            
        start_time = time.time()
        session_id = f"query_{int(time.time())}"
        iterations = []
        previous_result = None
        previous_embedding = None
        converged = False
        convergence_iteration = -1
        best_score = -1
        best_iteration = -1
        
        # Process through agents
        for i in range(min(self.num_agents, self.max_iterations)):
            # Determine temperature
            temperature = self._generate_temperature(i)
            
            # Prepare job data
            job_data = {
                "job_id": f"{session_id}_agent_{i}",
                "agent_id": i,
                "prompt": query,
                "max_tokens": max_tokens,
                "previous_result": previous_result,
                "improvement_instruction": improvement_instruction,
                "temperature": temperature,
                "timestamp": time.time()
            }
            
            # Submit job
            job_id = self._submit_job(job_data)
            
            # Wait for result
            result = self._wait_for_result(job_id)
            
            if result is None or not result.get("success", False):
                # Job failed
                error_msg = result.get("error", "Unknown error") if result else "Timeout"
                
                if self.verbose:
                    print(f"Agent {i} failed: {error_msg}")
                
                # Skip this iteration but continue with others
                continue
                
            # Calculate similarity if we have a previous result
            if previous_embedding is not None:
                similarity = self._calculate_similarity(previous_embedding, result["embedding"])
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
                    "agent_id": result["agent_id"],
                    "session_id": session_id,
                    "iteration": i,
                    "generation_time": result["generation_time"],
                    "tokens_used": result["tokens_used"]
                }],
                ids=[f"{session_id}_iteration_{i}"]
            )
            
            # Score with benchmark
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
        
        processing_time = time.time() - start_time
        
        # Determine the best result
        if best_iteration >= 0:
            best_result = iterations[best_iteration]
            selection_method = "benchmark"
        elif convergence_iteration >= 0:
            best_result = iterations[convergence_iteration]
            selection_method = "convergence"
        else:
            best_result = iterations[-1] if iterations else None
            selection_method = "final"
            
        # Prepare final results
        results = {
            "query": query,
            "num_iterations": len(iterations),
            "iterations": iterations,
            "best_result": best_result,
            "best_iteration": best_iteration if best_iteration >= 0 else convergence_iteration if convergence_iteration >= 0 else len(iterations) - 1 if iterations else -1,
            "selection_method": selection_method,
            "processing_time": processing_time,
            "converged": converged,
            "convergence_iteration": convergence_iteration if convergence_iteration >= 0 else None,
            "session_id": session_id
        }
        
        return results
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
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
        
    def set_num_agents(self, num_agents: int) -> None:
        """Change the number of agents.
        
        Args:
            num_agents: New number of agents
        """
        self.num_agents = min(num_agents, 1000000)
        
        if self.verbose:
            print(f"Updated to {self.num_agents} agents")
            
    def shutdown_workers(self):
        """Send shutdown command to all workers."""
        self.redis_client.publish("agent_control", json.dumps({"command": "shutdown"}))
        
        if self.verbose:
            print("Sent shutdown command to workers")
            
    def run_iteration(self, 
                     prompt: str,
                     prev_outputs: Optional[List[str]] = None,
                     improvement_instruction: Optional[str] = None,
                     iteration_num: int = 0) -> List[Dict]:
        """Run a single iteration of the orchestration process.
        
        Args:
            prompt: The initial prompt
            prev_outputs: Previous outputs from agents (if not first iteration)
            improvement_instruction: Instruction for improving the previous outputs
            iteration_num: Current iteration number
            
        Returns:
            List of results with their metrics
        """
        if not self.redis_client.ping():
            raise ConnectionError("Cannot connect to Redis server")
            
        job_ids = []
        
        # Create and submit jobs for each agent
        for agent_id in range(self.num_agents):
            job_id = str(uuid.uuid4())
            job_data = {
                "job_id": job_id,
                "agent_id": agent_id,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "model_path": self.model_path,
                "model_type": self.model_type,
                "use_vllm": self.use_vllm,
                "tensor_parallel_size": self.tensor_parallel_size
            }
            
            # If not first iteration, include previous outputs and improvement instruction
            if prev_outputs and iteration_num > 0:
                if agent_id < len(prev_outputs):
                    job_data["previous_result"] = prev_outputs[agent_id]
                    
                if improvement_instruction:
                    job_data["improvement_instruction"] = improvement_instruction
            
            # Store job in pending jobs
            self.pending_jobs[job_id] = job_data
            
            # Submit job to Redis queue
            self.redis_client.rpush("agent_jobs", json.dumps(job_data))
            job_ids.append(job_id)
            
        print(f"Submitted {len(job_ids)} jobs for iteration {iteration_num}")
        
        # Wait for all results
        results = self._collect_results(job_ids)
        
        # Process results
        result_list = []
        for job_id in job_ids:
            if job_id in results:
                result = results[job_id]
                agent_id = result.get("agent_id", 0)
                
                # Structure the result
                processed_result = {
                    "agent_id": agent_id,
                    "output": result.get("output", ""),
                    "tokens": result.get("tokens", 0),
                    "generation_time": result.get("generation_time", 0),
                    "similarity": result.get("similarity", 0.0) if "similarity" in result else None,
                    "backend": result.get("backend", "unknown")
                }
                
                result_list.append(processed_result)
                
        return result_list
    
    def _collect_results(self, job_ids: List[str], timeout: float = 60.0) -> Dict[str, Dict]:
        """Collect results for the given job IDs.
        
        Args:
            job_ids: List of job IDs to collect results for
            timeout: Maximum time to wait for results in seconds
            
        Returns:
            Dictionary of job ID to result
        """
        start_time = time.time()
        collected_results = {}
        
        while len(collected_results) < len(job_ids) and (time.time() - start_time) < timeout:
            # Check for results in the results queue
            result_data_raw = self.redis_client.lpop("agent_results")
            
            if result_data_raw:
                try:
                    result_data = json.loads(result_data_raw)
                    job_id = result_data.get("job_id")
                    
                    if job_id in job_ids:
                        collected_results[job_id] = result_data
                        
                        # Remove from pending jobs
                        if job_id in self.pending_jobs:
                            del self.pending_jobs[job_id]
                except Exception as e:
                    print(f"Error parsing result: {str(e)}")
            
            # If not all results collected, sleep a bit
            if len(collected_results) < len(job_ids):
                time.sleep(0.1)
                
        # Check for timeout
        if len(collected_results) < len(job_ids):
            print(f"Warning: Timed out waiting for results. Collected {len(collected_results)}/{len(job_ids)}")
            
        return collected_results
    
    def shutdown(self) -> None:
        """Shut down the orchestrator and send shutdown signal to workers."""
        control_message = {
            "command": "shutdown",
            "timestamp": time.time()
        }
        
        # Send shutdown command to workers
        self.redis_client.publish("agent_control", json.dumps(control_message))
        
        # Clean up Redis connection
        self.redis_client.close() 