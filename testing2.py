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
    prompt += "Generate only the next step. If this is the final step, please indicate it with \"Final Step\"."

    print(f"Prompt: {prompt}")

    # Prepare the full chat query to the model
    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        # repetition_penalty=0.5,
        # stop=["\n"]
    )
    # print(response.choices[0].message.content.strip())
    print(response)
    return response.choices[0].message.content.strip()

math_problem = "If Sarah has 10 apples and gives 3 to Tom, then buys 5 more, how many apples does she have?"

step1 = generate_step(math_problem, ["Step 1: To find out how many apples Sarah has, \
                                     we need to follow these steps: \n Step 1: Determine \
                                     the initial number of apples Sarah has, which is 10. \
                                     Then we subtract the number of apples she gives to Tom, \
                                     which is 3. \n10 (initial apples) - 3 (apples given to Tom) = 7 apples.", 
                                     "Step 2: After determining that Sarah has 7 apples, we need \
                                     to add the number of apples she buys, which is 5. \n7 (apples remaining) \
                                     + 5 (apples bought) = 12 apples."])

print(f"Step 1: {step1}")