import os
import json
from together import Together
import time

# Initialize the Together client
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
# MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MODEL = "meta-llama/Llama-Vision-Free"
type_to_filepath = {"Algebra": "algebra",
                    "Counting & Probability": "counting_and_probability",
                    "Geometry": "geometry",
                    "Number Theory": "number_theory",
                    "Intermediate Algebra": "intermediate_algebra",
                    "Precalculus": "precalculus",
                    "Prealgebra": "prealgebra",}

def generate_step(problem, previous_steps, verbosity=False, model=MODEL):
    """
    Generate a reasoning step using the primary model.
    """
    prompt = f"Problem: {problem}\n"
    if previous_steps:
        prompt += "Steps so far:\n" + "\n".join(previous_steps) + "\n\n"
        prompt += "Generate only the next step in solving the problem. Be succinct and do not explain. If this is the final step, indicate it with \"Final Answer\" in your response."
    else:
        prompt += "Generate only the first step in solving the problem. Be succinct and do not explain."
    messages = [{"role": "user", "content": prompt}]

    # Generate the next step
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=250,
        temperature=0.7,
    )
    step = response.choices[0].message.content.strip()
    step_tokens = response.usage.total_tokens
    if verbosity:
        print(f"Generated step: {step}")
    return step, step_tokens

def critique_step(problem, step, previous_steps, verbosity=False, model=MODEL):
    """
    Use the critique model to assess the current step's correctness.
    """
    critique_prompt = (
        f"You are evaluating steps in solving the following problem:\n"
        f"Problem: {problem}\n\n"
        f"Previous steps:" + "\n".join(previous_steps) + "\n\n"
        f"Current step:\n{step}\n\n"
        f"Please respond only with one of the following:\n"
        f"- 'Incorrect' if the current step has any errors. Critically check any reasoning in the step.\n"
        f"- 'Correct' if the current step is accurate but not the final answer.\n"
        f"- 'Complete' if the current step is both correct and the final answer.\n"
    )

    messages = [{"role": "user", "content": critique_prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=5,
        temperature=0.7,
    )
    critique = response.choices[0].message.content.strip()
    total_tokens = response.usage.total_tokens
    if verbosity:
        print(f"Critique: {critique}")
    return critique, total_tokens

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
                print("Problem unsolved: maximum number of steps reached. Stopping.")
            break

        if "correct" in critique.lower():
            continue
        #     if "final answer" in new_step.lower() or "final answer" in critique.lower():
        #         is_solved = True
        #         if verbosity:
        #             print("Final answer reached.")
        elif "incorrect" in critique.lower():
            steps.pop()
            full_steps[-1] = (new_step, 0)
            if verbosity:
                print("Correction required based on critique. Revising step.")
        elif "complete" in critique.lower():
            is_solved = True
            final_answer_tokens = step_tokens
            if verbosity:
                print("Final answer reached.")
        else:
            if verbosity:
                print("Unknown critique response. Stopping.")
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
        print(f"Saved output to {output_file}")

def solve_problems(problem_files, output_path="./output", verbosity=False, model=MODEL):
    """
    Solve a list of problem files and save outputs.
    """
    for problem_file in problem_files:
        with open(problem_file) as f:
            data = json.load(f)
            data["file_name"] = problem_file  # Store filename in data for output structure
        if verbosity:
            print(f"\nSolving problem from file: {problem_file}\n")
        solve_problem(data["problem"], data, output_path, verbosity=verbosity, model=model)
        if verbosity:
            print("\n" + "="*50 + "\n")

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
problem_files = ["./MATH/train/algebra/0.json", "./MATH/train/algebra/3.json"]
solve_problems(problem_files, output_path="./output", verbosity=True, model=MODEL)
