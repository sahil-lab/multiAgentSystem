import os
import json
import time
import torch
import redis
import argparse
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("worker")

# Load environment variables
load_dotenv()

# Default Redis connection info
DEFAULT_REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
DEFAULT_REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
DEFAULT_REDIS_DB = int(os.environ.get("REDIS_DB", 0))
DEFAULT_REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Model configuration
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))

# Import agent modules
try:
    from agent import Agent
    from vllm_adapter import VLLMAgent, VLLM_AVAILABLE
    from mistral_adapter import MistralAgent
except ImportError as e:
    logger.error(f"Error importing agent modules: {str(e)}")
    raise

def create_agent(model_path: str, **kwargs) -> Any:
    """Create an agent based on the model type.
    
    Args:
        model_path: Path or name of the model
        
    Returns:
        Agent instance
    """
    model_path_lower = model_path.lower()
    max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
    
    # Check for GPU availability
    tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    logger.info(f"Creating agent for model: {model_path}")
    logger.info(f"Available GPUs: {gpu_count}")
    
    # Determine which agent implementation to use
    try:
        # Try to use vLLM if available and we have GPU
        if VLLM_AVAILABLE and gpu_count > 0:
            logger.info(f"Using vLLM agent for {model_path}")
            return VLLMAgent(
                model_path=model_path,
                max_tokens=max_tokens,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                use_hf_auth=True
            )
        # For Mistral models, use MistralAgent
        elif "mistral" in model_path_lower:
            logger.info(f"Using Mistral agent for {model_path}")
            return MistralAgent(
                model_path=model_path,
                max_tokens=max_tokens,
                temperature=temperature
            )
        # Fallback to the default Agent
        else:
            logger.info(f"Using default agent for {model_path}")
            return Agent(
                model_path=model_path,
                max_tokens=max_tokens,
                temperature=temperature
            )
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        # If vLLM fails, try fallback options
        if VLLM_AVAILABLE and "mistral" in model_path_lower:
            logger.info("Falling back to MistralAgent")
            return MistralAgent(
                model_path=model_path,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            logger.info("Falling back to default Agent")
            return Agent(
                model_path=model_path,
                max_tokens=max_tokens,
                temperature=temperature
            )

def connect_to_redis(host: str, port: int, db: int, password: Optional[str] = None) -> redis.Redis:
    """Connect to Redis server.
    
    Args:
        host: Redis host
        port: Redis port
        db: Redis database
        password: Redis password
        
    Returns:
        Redis connection
    """
    try:
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=30
        )
        r.ping()  # Test connection
        logger.info(f"Connected to Redis at {host}:{port}/{db}")
        return r
    except redis.RedisError as e:
        logger.error(f"Redis connection error: {str(e)}")
        raise

def process_job(agent: Any, job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a job using the agent.
    
    Args:
        agent: Agent instance
        job_data: Job data dictionary
        
    Returns:
        Result dictionary
    """
    job_id = job_data.get("id", "unknown")
    prompt = job_data.get("prompt", "")
    max_tokens = job_data.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = job_data.get("temperature", DEFAULT_TEMPERATURE)
    previous_result = job_data.get("previous_result")
    improvement_instruction = job_data.get("improvement_instruction")
    
    logger.info(f"Processing job {job_id}")
    start_time = time.time()
    
    try:
        # Generate response
        result = agent.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            previous_result=previous_result,
            improvement_instruction=improvement_instruction
        )
        
        # Add job ID and processing time
        result["job_id"] = job_id
        result["processing_time"] = time.time() - start_time
        
        logger.info(f"Job {job_id} completed in {result['processing_time']:.2f}s")
        return result
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing job {job_id}: {error_message}")
        
        # Return error result
        return {
            "job_id": job_id,
            "text": "",
            "error": error_message,
            "processing_time": time.time() - start_time
        }

def start_worker(redis_conn: redis.Redis, agent: Any, worker_id: str = "worker"):
    """Start the worker loop to process jobs.
    
    Args:
        redis_conn: Redis connection
        agent: Agent instance
        worker_id: Worker ID
    """
    logger.info(f"Starting worker {worker_id}")
    job_queue = "job_queue"
    result_queue = "result_queue"
    
    while True:
        try:
            # Get job from queue with timeout
            job = redis_conn.blpop(job_queue, timeout=1)
            
            if job is None:
                continue
                
            # Parse job data
            _, job_data_str = job
            job_data = json.loads(job_data_str)
            
            # Process job
            result = process_job(agent, job_data)
            
            # Send result back
            redis_conn.rpush(result_queue, json.dumps(result))
            
        except redis.RedisError as e:
            logger.error(f"Redis error: {str(e)}")
            time.sleep(5)  # Wait before retrying
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            time.sleep(1)  # Prevent rapid spinning on errors

def main():
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(description="AI Agent Worker")
    
    parser.add_argument("--redis-host", type=str, default=DEFAULT_REDIS_HOST, help="Redis host")
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=DEFAULT_REDIS_DB, help="Redis database")
    parser.add_argument("--redis-password", type=str, default=DEFAULT_REDIS_PASSWORD, help="Redis password")
    
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Model path or name")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size for vLLM")
    
    parser.add_argument("--worker-id", type=str, default=f"worker-{os.getpid()}", help="Worker ID")
    
    args = parser.parse_args()
    
    try:
        # Connect to Redis
        redis_conn = connect_to_redis(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            password=args.redis_password
        )
        
        # Create agent
        agent = create_agent(
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel
        )
        
        # Start worker loop
        start_worker(redis_conn, agent, args.worker_id)
        
    except KeyboardInterrupt:
        logger.info("Worker shutdown requested")
        
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 