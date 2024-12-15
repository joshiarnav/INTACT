import os
import json
from sys import version
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
import threading
import random
from environment import ( #pylint: disable=import-error
    MODEL, get_client, RateLimiter, rate_limiter,
    check_existing_runs, all_problem_files
) 
from tqdm import tqdm
from solver import solver #pylint: disable=import-error
from cot import generate_cot_sample #pylint: disable=import-error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def generate_cot_sample(problem, data, model=MODEL, temperature=0.9):
#     """
#     Generate a single CoT sample for a problem with higher temperature for diversity.
#     """
#     rate_limiter.acquire()
#     client = get_client()
    
#     # Format the problem with CoT prompt
# #     prompt = f"""Problem: {problem}
# # Indicate your answer in the format $\\boxed{{answer}}$.
# # Let's think step by step:
# # """
#     prompt = problem + " Ensure your answer is in the format $\\boxed{answer}$. Let's think step by step:"

#     try:
#         response = client.complete(
#             prompt=prompt,
#             model=model,
#             max_tokens=1024,
#             temperature=temperature,
#             top_k=50,
#             top_p=0.7,
#             repetition_penalty=1.1,
#             stop=['Problem:', '\n\n']
#         )
        
#         solution_text = response['output']['choices'][0]['text'].strip()
#         # Format the complete solution
#         complete_solution = "1)" + solution_text
        
#         # Extract final answer - assuming it's in the last line after "Therefore" or similar
#         lines = complete_solution.split('\n')
#         answer = lines[-1]
#         for line in reversed(lines):
#             if any(word in line.lower() for word in ['therefore', 'thus', 'so', 'answer', 'final']):
#                 answer = line
#                 break
                
#         return {
#             'cot_solution': complete_solution,
#             'final_answer': answer
#         }
        
#     except Exception as e:
#         logger.error(f"Error generating CoT sample: {str(e)}")
#         return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def critique_solutions(solutions, problem, model=MODEL, version="full", temperature=0.5):
    """
    Have the model critique and select the best solution from N samples.
    """
    rate_limiter.acquire()
    client = get_client()
    
    # Format solutions for critique
    
    
    try:
        solutions_text = ""
        if version == "full":
            for i, sol in enumerate(solutions, 1):
                if sol:
                    solutions_text += f"\nSolution {i}: {sol}\n"
            prompt = (
                f"Below are {len(solutions)} different solution attempts for a math problem.\n"
                f"Problem: {problem}\n\n"
                f"{solutions_text}\n"
                f"Return strictly the index number (1 to {len(solutions)}) of the best solution.\n"
            )
        if version == "solution_only":
            for i, sol in enumerate(solutions, 1):
                if sol:
                    # Find the final answer which will be contained within $\\boxed{answer}$
                    final_answer = sol.split('\\boxed{')[1].split('}')[0]
                    solutions_text += f"\nSolution {i}: {final_answer}\n"
            prompt = (
                f"Below are {len(solutions)} different solutions for a math problem.\n"
                f"Problem: {problem}\n\n"
                f"{solutions_text}\n"
                f"Return strictly the index number (1 to {len(solutions)}) of the best solution.\n"
            )

        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=250,
            temperature=temperature,
        )
    
        critique = response.choices[0].message.content.strip()
                
        # Extract the chosen solution index
        # for line in critique.split('\n'):
        #     if any(x in line.lower() for x in ['best solution is', 'solution', 'choose', 'select']):
        #         for i in range(len(solutions)):
        #             if str(i+1) in line:
        #                 return i, critique

        # Check for a choice in the critique which should be a very brief answer with a number as an integer or language representing a number
        # solution_to_critique_text = {0: ["first", "first solution", "1", "one"],
        #                             1: ["second", "second solution", "2", "two"],
        #                             2: ["third", "third solution", "3", "three"],
        #                             3: ["fourth", "fourth solution", "4", "four"],
        #                             4: ["fifth", "fifth solution", "5", "five"],
        #                             5: ["sixth", "sixth solution", "6", "six"]}
        # critique_text_to_solution = {("first", "first solution", "1", "one"): 0,
        #                             ("second", "second solution", "2", "two"): 1,
        #                             ("third", "third solution", "3", "three"): 2,
        #                             ("fourth", "fourth solution", "4", "four"): 3,
        #                             ("fifth", "fifth solution", "5", "five"): 4,
        #                             ("sixth", "sixth solution", "6", "six"): 5}
        # critique = response.choices[0].message.content.strip()



        for i in range(1, len(solutions) + 1):
            if str(i) in critique:
                return i, critique
            # for key in solution_to_critique_text[i]:
                # if key in critique.lower():
                #     return i, critique
                        
        # Default to first solution if no clear choice found
        return -1, critique
        
    except Exception as e:
        logger.error(f"Error in critique: {str(e)}")
        return 0, "Error in critique"

def generate_solution(problem, model, temperature):
    """Helper function to generate a single solution"""
    solution, total_tokens = generate_cot_sample(problem, model, temperature=temperature)
    return solution, total_tokens

def solve_problem(problem_file, output_path, model=MODEL):
    """
    Solve a single problem using best-of-N approach.
    """
    try:
        start_time = time.time()
        # Load problem data
        with open(problem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Generate N solutions in parallel
        solutions = []
        num_samples = 3  # Number of solutions to generate
        # logger.info(f"Generating {num_samples} solutions for {problem_file}")
        total_tokens = 0
        max_retries = 7
        
        with ThreadPoolExecutor(max_workers=num_samples) as executor:
            future_to_solution = {
                executor.submit(generate_solution, data['problem'], model, 0.9): i 
                for i in range(num_samples)
            }
            
            for future in as_completed(future_to_solution):
                solution, tokens = future.result()
                if solution:
                    if "boxed" not in solution and max_retries > 0:
                        max_retries -= 1
                        # add this generation back into the pool
                        # future_to_solution[executor.submit(generate_solution, data['problem'], model, 0.9)] = future_to_solution.pop(future)
                        logger.info(f"Retrying generation for {problem_file}")
                        future_to_solution.update({executor.submit(generate_solution, data['problem'], model, 0.7): future_to_solution.pop(future)})
                    elif max_retries <= 0:
                        logger.error(f"Failed to generate a valid solution for {problem_file}")
                        return
                    else:
                        solutions.append(solution)
                        total_tokens += int(tokens)
        
        if not solutions:
            logger.error(f"No valid solutions generated for {problem_file}")
            return
            
        # Critique and select best solution
        best_idx, critique = critique_solutions(solutions, data['problem'], model)
        # logger.info(f"Best solution index: {best_idx}")
        # logger.info(f"Critique: {critique}")
        if best_idx < 0:
            logger.info(f"Critique generated for no valid solution: {critique}")
        
        end_time = time.time()
        time_taken = end_time - start_time
        # Prepare output
        output_data = {
            'problem': data['problem'],
            'all_solutions': solutions,
            'model_solution': solutions[best_idx] if best_idx >= 0 else "$\\boxed{}$",
            'critique': critique,
            'solution': data.get('solution', ''),
            'level': data.get('level', ''),
            'type': data.get('type', ''),
            'total_tokens': total_tokens,
            'time_taken': time_taken,
        }
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Completed {problem_file}")
        
    except Exception as e:
        logger.error(f"Error processing {problem_file}: {str(e)}")

def solve_problems_parallel(problem_files, output_path="./output", model=MODEL, max_workers=2, skip_existing=True):
    """
    Solve problems in parallel using a thread pool.
    """
    file_safe_model_name = model.replace("/", "-")
    output_base = os.path.join(output_path, f"{file_safe_model_name}_bestofn_output_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for problem_file in problem_files:
            if skip_existing and check_existing_runs(problem_file, output_base):
                continue
                
            # Create output path
            path_parts = problem_file.split(os.sep)
            problem_type = path_parts[-2]
            problem_filename = os.path.basename(problem_file)
            output_file = os.path.join(output_base, problem_type, problem_filename)
            
            future = executor.submit(solve_problem, problem_file, output_file, model=model)
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Thread generated an exception: {str(e)}")

def main():
    data_dir = "../datasets/MATH_subsample_uniform"
    file_safe_model_name = MODEL.replace("/", "-")
    output_path = f"../main_experiment_results/{file_safe_model_name}_bestofn_output_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Get all problem files
    problem_files = all_problem_files(data_dir)
    random.seed(42)
    random.shuffle(problem_files)
    # sample_size = 50
    # problem_files = problem_files[:sample_size]
    
    # Solve problems
    solve_problems_parallel(problem_files, output_path=output_path, model=MODEL, max_workers=16)
    
    # Run the solver on the output directory
    solver(output_path)

if __name__ == "__main__":
    main()
