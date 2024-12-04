import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    """
    def __init__(self, rate=1.0):  # rate is in requests per second
        self.rate = rate
        self.tokens = 1.0  # Start with one token
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        """
        Acquire a token, blocking if necessary.
        """
        with self.lock:
            current = time.time()
            # Add new tokens based on time elapsed
            time_passed = current - self.last_update
            self.tokens = min(1.0, self.tokens + time_passed * self.rate)
            
            if self.tokens < 1.0:
                # Need to wait
                sleep_time = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 0.0
                self.last_update = time.time()
            else:
                # Consume one token
                self.tokens -= 1.0
                self.last_update = current

# Initialize the Together client and rate limiter
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
MODEL = "meta-llama/Llama-Vision-Free"
type_to_filepath = {"Algebra": "algebra",
                    "Counting & Probability": "counting_and_probability",
                    "Geometry": "geometry",
                    "Number Theory": "number_theory",
                    "Intermediate Algebra": "intermediate_algebra",
                    "Precalculus": "precalculus",
                    "Prealgebra": "prealgebra",}

# Thread-local storage for API client
thread_local = threading.local()

# Global rate limiter shared across all threads
rate_limiter = RateLimiter(rate=1/6.1)  # 10 requests per minute = 1 request per 6 seconds

def get_client():
    """
    Get the Together client, creating it if it doesn't exist.
    """
    if not hasattr(thread_local, "client"):
        thread_local.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return thread_local.client

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_step(problem, previous_steps, verbosity=False, model=MODEL):
    """
    Generate a reasoning step using the primary model with retry logic.
    """
    try:
        prompt = f"Problem: {problem}\n"
        if previous_steps:
            prompt += "Steps so far:\n" + "\n".join(previous_steps) + "\n\n"
            prompt += "Generate only the next step in solving the problem. Be succinct and do not explain. If this or the previous step is the final step, you must indicate it with \"Final Answer\" in your response."
        else:
            prompt += "Generate only the first step in solving the problem. Be succinct and do not explain. Do not give the final answer."
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
def critique_step(problem, step, previous_steps, verbosity=False, model=MODEL):
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
            temperature=0.7,
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

        if "correct" in critique.lower():
            if "final answer" in new_step.lower() or len(set(steps)) < len(steps):
                is_solved = True
                final_answer_tokens = step_tokens
                if verbosity:
                    logger.info("Final answer reached.")
        elif "incorrect" in critique.lower():
            steps.pop()
            full_steps[-1] = (new_step, 0)
            if verbosity:
                logger.info("Correction required based on critique. Revising step.")
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
    problem_number = os.path.splitext(os.path.basename(data["file_name"]))[0]
    run_number = 0
    while os.path.exists(os.path.join(subject_path, problem_number, f"run_{run_number}.json")):
        run_number += 1
    problem_output_path = os.path.join(subject_path, problem_number)
    os.makedirs(problem_output_path, exist_ok=True)
    output_file = os.path.join(problem_output_path, f"run_{run_number}.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    if verbosity:
        logger.info(f"Saved output to {output_file}")

def check_existing_runs(problem_file, output_path):
    """
    Check if any runs exist for a given problem file in the output directory.
    Returns True if runs exist, False otherwise.
    """
    try:
        filename = os.path.basename(problem_file)
        problem_output_path = os.path.join(output_path, filename.replace('.json', ''))
        
        # Check if the problem directory exists and has any run files
        if os.path.exists(problem_output_path):
            run_files = [f for f in os.listdir(problem_output_path) if f.startswith('run_') and f.endswith('.json')]
            if run_files:
                logger.info(f"Skipping {filename} - {len(run_files)} existing runs found")
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking existing runs for {problem_file}: {str(e)}")
        return False

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
            output_file = os.path.join(output_path, filename)
            
            logger.info(f"Processing problem: {filename}")
            return solve_problem(problem, data, output_file, verbosity, model)
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

all_files = all_problem_files("./MATH")
# problem_files = ["./MATH/train/algebra/0.json", "./MATH/train/algebra/3.json"]
problem_files = all_files
solve_problems(problem_files, output_path="./output", verbosity=True, model=MODEL)
