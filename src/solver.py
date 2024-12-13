import os
import json
import argparse

def all_solution_files(data_dir):
    """
    Get a list of all problem files in the specified data directory.
    """
    problem_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                problem_files.append(os.path.join(root, file))
    return problem_files

def check_problems(filepaths):
    problem_to_correctness = {}
    total_tokens = 0
    for filepath in filepaths:
        with open(filepath, "r") as f:
            # Skip if not a JSON file
            if not filepath.endswith(".json"):
                continue

            # Check if the part of the JSON marked "model_solution" contains the section in the part of the JSON marked "solution" within the $\\boxed{}$ section.
            data = json.load(f)
            model_solution = data["model_solution"]
            solution = data["solution"]
            total_tokens += data["total_tokens"]

            # Find the part of the solution that is within the $\\boxed{}$ section.
            # boxed_solution = solution[solution.find("\\boxed{") + len("\\boxed{"):solution.find("}")]
            boxed_solution = solution.split("\\boxed{")[1].split("}")[0]
            boxed_solution_with_box = "\\boxed{" + boxed_solution + "}"
            # if boxed_solution_with_box not in model_solution:
            #     print("Filepath:", filepath)
            #     print("Boxed solution not in model solution:", boxed_solution_with_box)
            #     print("Model solution:", model_solution)
            problem_to_correctness[filepath] = boxed_solution_with_box in model_solution

    return problem_to_correctness, total_tokens

def solver(data_dir):
    problem_files = all_solution_files(data_dir)
    problem_to_correctness, total_tokens = check_problems(problem_files)
    # Save filenames of problems that were solved correctly.
    with(open(os.path.join(data_dir, "correct_problems.txt"), "w")) as f:
        for problem, correctness in problem_to_correctness.items():
            if correctness:
                f.write(f"{problem}\n")

    # # Calculate statistics once
    # num_correct = sum(problem_to_correctness.values())
    # num_incorrect = len(problem_to_correctness) - num_correct
    # total_problems = len(problem_to_correctness)
    # percent_correct = (num_correct / total_problems) * 100 if total_problems > 0 else 0
    # Calculate statistics
    num_problems = len(problem_files)
    num_correct = sum(1 for x in problem_to_correctness.values() if x)
    accuracy = num_correct / num_problems if num_problems > 0 else 0
    avg_tokens = total_tokens / num_problems if num_problems > 0 else 0

    # # Save metadata to a separate file
    # with(open(os.path.join(data_dir, "metadata.txt"), "w")) as f:
    #     f.write(f"Number of problems solved correctly: {num_correct}\n")
    #     f.write(f"Number of problems solved incorrectly: {num_incorrect}\n")
    #     f.write(f"Total number of problems: {total_problems}\n")
    #     f.write(f"Percentage of problems solved correctly: {percent_correct:.2f}%\n")
    #     f.write(f"Total tokens used: {total_tokens}\n")
    #     f.write(f"\nDetailed problem results:\n")
    #     for problem, correctness in problem_to_correctness.items():
    #         f.write(f"{problem}: {'Correct' if correctness else 'Incorrect'}\n")

    # print(f"Results saved to {data_dir}/correct_problems.txt and {data_dir}/metadata.txt")
    # print(f"Number of problems solved correctly: {num_correct}")
    # print(f"Number of problems solved incorrectly: {num_incorrect}")
    # print(f"Total number of problems: {total_problems}")
    # print(f"Percentage of problems solved correctly: {percent_correct:.2f}%")
    # print(f"Total tokens used: {total_tokens}")
    # Save statistics
    stats = {
        "num_problems": num_problems,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens
    }
    with open(os.path.join(data_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    return problem_to_correctness, stats

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='Check solutions for math problems')
    # parser.add_argument('--data-dir', type=str, default='meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo_agentic_output',
    #                   help='Directory containing the solution files (default: meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo_agentic_output)')

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing problem files")
    args = parser.parse_args()
    # main(args.data_dir)
    solver(args.dir)
