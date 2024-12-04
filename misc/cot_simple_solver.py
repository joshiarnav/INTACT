import os
import json

def all_solution_files(data_dir):
    """
    Get a list of all problem files in the specified data directory.
    """
    problem_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            problem_files.append(os.path.join(data_dir, file))
    return problem_files

def check_problems(filepaths):
    problem_to_correctness = {}
    total_tokens = 0
    for filepath in filepaths:
        with open(filepath, "r") as f:
            # Check if the part of the JSON marked "model_solution" contains the section in the part of the JSON marked "solution" within the $\\boxed{}$ section.
            data = json.load(f)
            model_solution = data["model_solution"]
            solution = data["solution"]
            total_tokens += data["total_tokens"]
            # Find the part of the solution that is within the $\\boxed{}$ section.
            boxed_solution = solution[solution.find("$\\boxed{") + len("$\\boxed{"):solution.find("}$")]
            problem_to_correctness[filepath] = boxed_solution in model_solution

    return problem_to_correctness, total_tokens

def main():
    data_dir = "./meta-llama-Llama-Vision-Free_cot_simple_output"
    problem_files = all_solution_files(data_dir)
    problem_to_correctness, total_tokens = check_problems(problem_files)
    # Save filenames of problems that were solved correctly.
    with(open("correct_problems_cot_simple.txt", "w")) as f:
        for problem, correctness in problem_to_correctness.items():
            if correctness:
                f.write(f"{problem}\n")
                print(f"{problem}: {correctness}")
    # Print the number of problems that were solved correctly and the number of problems that were solved incorrectly and the total number of problems and the percentage of problems that were solved correctly.
    num_correct = sum(problem_to_correctness.values())
    num_incorrect = len(problem_to_correctness) - num_correct
    total = len(problem_to_correctness)
    print(f"Correct: {num_correct}")
    print(f"Incorrect: {num_incorrect}")
    print(f"Total: {total}")
    print(f"Percentage: {num_correct / total * 100:.2f}%")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
