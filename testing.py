import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


def generate_step(problem, previous_steps, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    """
    Generate a reasoning step using the primary model.
    """
    # Construct the prompt with the problem and previous steps
    prompt = f"Problem: {problem}\n"
    if previous_steps:
        prompt += "Steps so far:\n" + "\n".join(previous_steps) + "\n\n"
    prompt += "What should be the next step?"

    print(f"Prompt: {prompt}")

    # Prepare the full chat query to the model
    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        # stop=["\n"]
    )
    print(response.choices[0].message)
    return response.choices[0].message.content.strip()

math_problem = "If Sarah has 10 apples and gives 3 to Tom, then buys 5 more, how many apples does she have?"

step1 = generate_step(math_problem, [])

print(f"Step 1: {step1}")