import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    """
    def __init__(self, rate=1.0):
        self.rate = rate
        self.tokens = 1.0
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            current = time.time()
            time_passed = current - self.last_update
            self.tokens = min(1.0, self.tokens + time_passed * self.rate)
            
            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 0.0
                self.last_update = time.time()
            else:
                self.tokens -= 1.0
                self.last_update = current

# Initialize the Together client and rate limiter
MODEL = "meta-llama/Llama-Vision-Free"
thread_local = threading.local()
rate_limiter = RateLimiter(rate=1/6.1)  # 10 requests per minute

def get_client():
    """
    Get the Together client, creating it if it doesn't exist.
    """
    if not hasattr(thread_local, "client"):
        thread_local.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return thread_local.client

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def solve_problem(problem, data, output_path, model=MODEL):
    """
    Solve the problem using a single model call and save the output.
    """
    try:
        start_time = time.time()
        
        # Create prompt
        prompt = f"{problem}"
        messages = [{"role": "user", "content": prompt}]
        
        # Acquire rate limit token before making API call
        rate_limiter.acquire()
        
        # Make API call
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=250,
            temperature=0.7,
        )
        
        solution = response.choices[0].message.content.strip()
        total_tokens = response.usage.total_tokens
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Prepare output data
        output_data = {
            "problem": problem,
            "solution": data.get("solution", ""),
            "total_tokens": total_tokens,
            "time_taken": time_taken,
            "file_name": data.get("file_name", ""),
            "model_solution": solution,
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save output
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
            
        logger.info(f"Saved output to {output_path}")
        return output_data
        
    except Exception as e:
        logger.error(f"Error solving problem: {str(e)}")
        raise

def solve_problems(problem_files, output_path, model=MODEL, max_workers=1):
    """
    Solve problems in parallel using a thread pool.
    """
    os.makedirs(output_path, exist_ok=True)
    
    def process_problem(problem_file):
        try:
            with open(problem_file, 'r') as f:
                data = json.load(f)
            
            problem = data['problem']
            filename = os.path.basename(problem_file)
            data['file_name'] = problem_file
            output_file = os.path.join(output_path, filename)
            
            logger.info(f"Processing problem: {filename}")
            return solve_problem(problem, data, output_file, model)
        except Exception as e:
            logger.error(f"Error processing {problem_file}: {str(e)}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_problem, pf): pf for pf in problem_files}
        
        for future in as_completed(future_to_file):
            problem_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"Completed processing {problem_file}")
            except Exception as e:
                logger.error(f"Problem {problem_file} generated an exception: {str(e)}")

def all_problem_files(data_dir):
    """
    Get a list of all problem files in the specified data directory.
    """
    problem_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                problem_files.append(os.path.join(root, file))
    return problem_files

def main():
    data_dir = "./MATH"
    file_safe_model_name = MODEL.replace("/", "-")
    output_dir = f"{file_safe_model_name}_baseline_simple_output"
    problem_files = all_problem_files(data_dir)
    solve_problems(problem_files, output_dir, max_workers=8)

if __name__ == "__main__":
    main()
