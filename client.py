import os
import json
from together import Together
import time

# Initialize the Together client
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

def generate_step(problem, previous_steps, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    """
    Generate a reasoning step using the primary model.
    """
    # Construct the prompt with the problem and previous steps
    prompt = f"Problem: {problem}\n"
    if previous_steps:
        prompt += "Steps so far:\n" + "\n".join(previous_steps) + "\n\n"
        prompt += "Generate only the next step in solving the problem. If this is the final step, indicate it with \"Final Answer\"."
    else:
        prompt += "Generate only the first step in solving the problem."

    # print(f"Prompt: {prompt}")

    # Prepare the full chat query to the model
    messages = [{"role": "user", "content": prompt}]

    # Generate the next step
    # response = client.completions.create(
    #     model=model,
    #     prompt=prompt,
    #     max_tokens=100,
    #     temperature=0.7,
    #     stop=["\n"]
    # )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=250,
        temperature=0.7,
        # stop=["\n"]
    )
    # Extract and return the generated text
    # return response.choices[0].text.strip()
    # print(response.choices[0].message)
    return response.choices[0].message.content.strip()


def critique_step(problem, step, previous_steps, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    """
    Use the critique model to assess the current step's correctness.
    """
    # Construct the critique prompt
    critique_prompt = (
        f"You are evaluating steps in solving the following problem:\n"
        f"Problem: {problem}\n\n"
        f"Previous steps:" + "\n".join(previous_steps) + "\n\n"
        f"Current step:\n{step}\n\n"
        f"Please respond with one of the following:\n"
        f"- 'Correct' if the step is accurate.\n"
        # f"- 'Incorrect' with a correction if the step has an error.\n"
        f"- 'Incorrect' if the step has an error.\n"
        f"- 'Complete' if this is the correct final answer.\n"
        # f"- 'Backtrack' if an earlier step needs revision."
    )

    # print(f"Critique Prompt: {critique_prompt}")

    # Prepare the full chat query to the model
    messages = [{"role": "user", "content": critique_prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        # stop=["\n"]
    )

    # # Generate the critique
    # response = client.completions.create(
    #     model=model,
    #     prompt=critique_prompt,
    #     max_tokens=100,
    #     temperature=0.0,
    #     stop=["\n"]
    # )
    # # Extract and return the critique text
    # return response.choices[0].text.strip()
    return response.choices[0].message.content.strip()

def solve_problem(problem):
    """
    Solve the problem using a primary generator model and a critique model.
    """
    steps = []
    is_solved = False
    max_steps = 10

    while not is_solved:
        # Generate a new step
        new_step = generate_step(problem, steps)
        print(f"Generated step: {new_step}")

        # Add the step to the list
        steps.append(new_step)

        # Critique the step
        critique = critique_step(problem, new_step, steps[:-1])
        print(f"Critique: {critique}")

        # Check if the maximum number of steps has been reached
        if len(steps) >= max_steps:
            print("Maximum number of steps reached. Stopping.")
            break

        # Process the critique response
        # if critique.startswith("Correct"):
        if "correct" in critique.lower():
            # Check if this is the final answer or if we need another step
            if "final answer" in new_step.lower() or "Final Answer" in critique:
                is_solved = True
                print("Final answer reached.")
            else:
                print("Proceeding to the next step.")

        # elif critique.startswith("Incorrect"):
        elif "incorrect" in critique.lower():
            # Handle correction as suggested by critique
            print("Correction required based on critique. Revising step.")
            steps.pop()  # Remove the incorrect step

        # elif critique.startswith("Backtrack"):
        #     # Backtrack to the previous step
        #     print("Backtracking as per critique suggestion.")
        #     if len(steps) > 1:
        #         steps.pop()  # Remove the last incorrect step to retry the previous context
        #     else:
        #         print("Backtracking not possible; starting over from the first step.")
        #         steps.clear()

        elif "complete" in critique.lower():
            is_solved = True
            print("Final answer reached.")

        else:
            print("Unknown critique response. Stopping.")
            break

    print("\nSolution Steps:")
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")

def solve_problems(problems):
    """
    Solve a list of problems.
    """
    for problem in problems:
        print(f"\nSolving problem: {problem}\n")
        solve_problem(problem)
        print("\n" + "="*50 + "\n")

# Example problem from MATH dataset
# math_problem = "If Sarah has 10 apples and gives 3 to Tom, then buys 5 more, how many apples does she have?"

# Pull problem from the "problem" section of the JSON file 0.json located at ./MATH/train/algebra/0.json
# with open("./MATH/train/algebra/0.json") as f:
#     data = json.load(f)
#     math_problem = data["problem"]

# # Solve the problem
# solve_problem(math_problem)
