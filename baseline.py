import json
import logging
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
import threading
import random
from environment import (
    MODEL, get_client, RateLimiter, rate_limiter,
    check_existing_runs, all_problem_files
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def solve_problem(problem, data, output_path, model=MODEL):
    """
    Solve the problem using a single model call and save the output.
    """
    try:
        start_time = time.time()
        
        # Create prompt
        prompt = problem + " Ensure your answer is in the format $\\boxed{answer}$."
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

        # logger.info(path_parts)
        path_parts = data["file_name"].split(os.sep)
        problem_type = path_parts[-2]  # Assumes structure like .../subject/problem.json

        subject_path = os.path.join(output_path, problem_type)
        problem_number = os.path.splitext(os.path.basename(data["file_name"]))[0]
        os.makedirs(subject_path, exist_ok=True)
        output_file = os.path.join(subject_path, f"{problem_number}.json")
        logger.info(output_file)

        # Save output
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
            
        logger.info(f"Saved output to {output_file}")
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
            # Check if output already exists
            if check_existing_runs(problem_file, output_path):
                return None
                
            with open(problem_file, 'r') as f:
                data = json.load(f)
            
            problem = data['problem']
            filename = os.path.basename(problem_file)
            data['file_name'] = problem_file
            # output_file = os.path.join(output_path, filename)
            
            logger.info(f"Processing problem: {filename}")
            return solve_problem(problem, data, output_path, model)
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

def main():
    data_dir = "./MATH_subsample_uniform"
    file_safe_model_name = MODEL.replace("/", "-")
    output_dir = f"{file_safe_model_name}_baseline_output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    problem_files = all_problem_files(data_dir)
    random.seed(42)
    random.shuffle(problem_files)
    solve_problems(problem_files, output_dir, max_workers=8)

if __name__ == "__main__":
    main()
