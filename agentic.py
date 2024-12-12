import json
import logging
import os
import threading
import time
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
from environment import (
    MODEL, get_client, RateLimiter, rate_limiter, 
    check_existing_runs, all_problem_files, type_to_filepath
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Together client and rate limiter
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Thread-local storage for API client
thread_local = threading.local()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_step(problem, previous_steps, verbosity=False, model=MODEL):
    """
    Generate a reasoning step using the primary model with retry logic.
    """
    try:
        prompt = f"Solve this step-by-step:\n{problem}\n"
        if previous_steps:
            prompt += "Previous steps:\n" + "\n".join(previous_steps) + "\n\n"
            prompt += "Next step (if final step use $\\boxed{answer}$):"
        else:
            prompt += "First step only:"
        messages = [{"role": "user", "content": prompt}]

        # Acquire rate limit token before making API call using global rate limiter
        rate_limiter.acquire()
        
        # Generate the next step using thread-local client
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=250,
            temperature=0.7,
        )
        step = response.choices[0].message.content.strip()
        step_tokens = response.usage.total_tokens
        if verbosity:
            logger.info(f"Generated step: {step}")
        return step, step_tokens
    except Exception as e:
        logger.error(f"Error generating step: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def critique_step(problem, step, previous_steps, verbosity=False, model=MODEL, temperature=0.7):
    """
    Use the critique model to assess the current step's correctness with retry logic.
    """
    try:
        critique_prompt = (
            f"You are evaluating steps in solving the following problem:\n"
            f"Problem: {problem}\n\n"
            f"Previous steps:" + "\n".join(previous_steps) + "\n\n"
            f"Current step:\n{step}\n\n"
            f"Please respond only with one of the following:\n"
            f"- 'Incorrect' if the current step has any errors. Critically check any reasoning in the step.\n"
            f"- 'Correct' if the current step is accurate.\n"
        )

        messages = [{"role": "user", "content": critique_prompt}]
        
        # Acquire rate limit token before making API call using global rate limiter
        rate_limiter.acquire()
        
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=5,
            temperature=temperature,
        )
        critique = response.choices[0].message.content.strip()
        total_tokens = response.usage.total_tokens
        if verbosity:
            logger.info(f"Critique: {critique}")
        return critique, total_tokens
    except Exception as e:
        logger.error(f"Error in critique step: {str(e)}")
        raise

def solve_problem(problem, data, output_path, verbosity=False, model=MODEL, max_steps=10):
    """
    Solve the problem using a primary generator model and a critique model, and save the output.
    """
    steps = []
    full_steps = []
    final_answer_tokens = 0
    total_tokens = 0
    is_solved = False
    start_time = time.time()

    while not is_solved:
        new_step, step_tokens = generate_step(problem, steps, verbosity=verbosity)
        total_tokens += step_tokens
        full_steps.append((new_step, 1))
        steps.append(new_step)

        critique, critique_tokens = critique_step(problem, new_step, steps[:-1], verbosity=verbosity)
        total_tokens += critique_tokens

        if len(steps) >= max_steps:
            if verbosity:
                logger.info("Problem unsolved: maximum number of steps reached. Stopping.")
            break
        if "incorrect" in critique.lower():
            steps.pop()
            full_steps[-1] = (new_step, 0)
            if verbosity:
                logger.info("Correction required based on critique. Revising step.")
        elif "correct" in critique.lower():
            # if "final answer" in new_step.lower() or len(set(steps)) < len(steps):
            if "\\boxed" in new_step or len(set(steps)) < len(steps):
                is_solved = True
                final_answer_tokens = step_tokens
                if verbosity:
                    logger.info("Final answer reached.")
        else:
            if verbosity:
                logger.info("Unknown critique response. Stopping.")
            break

    end_time = time.time()
    time_taken = end_time - start_time
    output_data = {
        "problem": data["problem"], # Problem text (from MATH dataset)
        "level": data["level"], # Problem difficulty level (from MATH dataset)
        "type": data["type"], # Problem type (from MATH dataset)
        "solution": data["solution"], # Correct ground truth solution (from MATH dataset)
        "is_solved": is_solved, # Whether the problem was solved
        "model_solution": "\n".join(steps), # Model-generated solution
        "steps": steps, # "Correct" steps generated by the model
        "full_steps": full_steps, # Full steps (including incorrect) with correctness indicator
        "time": time_taken, # Time taken to solve the problem
        "total_tokens": total_tokens, # Total tokens used for all completions
        "final_answer_tokens": final_answer_tokens, # Tokens used to generate the final answer, important for answer verification cost estimation
        "model": model, # Model used for generation and critique
    }

    subject_path = os.path.join(output_path, type_to_filepath[data["type"]])
    logger.info(subject_path)
    problem_number = os.path.splitext(os.path.basename(data["file_name"]))[0]
    logger.info(problem_number)
    os.makedirs(subject_path, exist_ok=True)
    output_file = os.path.join(subject_path, f"{problem_number}.json")
    logger.info(output_file)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    if verbosity:
        logger.info(f"Saved output to {output_file}")

def solve_problems(problem_files, output_path="./output", verbosity=False, model=MODEL, max_workers=2, skip_existing=True):
    """
    Solve problems in parallel using a thread pool, skipping problems that already have runs.
    """
    os.makedirs(output_path, exist_ok=True)
    
    def process_problem(problem_file):
        try:
            # Skip if runs already exist
            if skip_existing and check_existing_runs(problem_file, output_path):
                return None
                
            with open(problem_file, 'r') as f:
                data = json.load(f)
            
            problem = data['problem']
            filename = os.path.basename(problem_file)
            data['file_name'] = problem_file
            # output_file = os.path.join(output_path, filename)
            
            logger.info(f"Processing problem: {filename}")
            return solve_problem(problem, data, output_path, verbosity, model)
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
    all_files = all_problem_files("./MATH_subsample_uniform")
    random.seed(42)
    random.shuffle(all_files)
    file_safe_model_name = MODEL.replace("/", "-")
    output_path = f"{file_safe_model_name}_agentic_output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    solve_problems(all_files, output_path=output_path, verbosity=True, model=MODEL, max_workers=8)

if __name__ == "__main__":
    main()