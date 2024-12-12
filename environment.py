import json
import logging
import os
import threading
import time
from datetime import datetime
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

# Model configuration
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
thread_local = threading.local()

# Initialize rate limiter with optimal rate (9.5-9.9 requests per second)
rate_limiter = RateLimiter(rate=9.9)

def get_client():
    """
    Get the Together client, creating it if it doesn't exist.
    """
    if not hasattr(thread_local, "client"):
        thread_local.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    return thread_local.client

def check_existing_runs(problem_file, output_path):
    """
    Check if an output file exists for a given problem file.
    Returns True if output exists, False otherwise.
    """
    file_safe_model_name = MODEL.replace("/", "-")
    output_dir = os.path.join(output_path, file_safe_model_name)
    
    # Extract problem ID from filename
    problem_id = os.path.splitext(os.path.basename(problem_file))[0]
    output_file = os.path.join(output_dir, f"{problem_id}.json")
    
    return os.path.exists(output_file)

def all_problem_files(data_dir):
    """
    Get a list of all problem files in the specified data directory.
    """
    problem_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                problem_files.append(os.path.join(root, file))
    return problem_files

# Type to filepath mapping
type_to_filepath = {
    "Algebra": "algebra",
    "Counting and Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus"
}
