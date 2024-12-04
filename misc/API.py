import os
from together import Together

client = Together()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    messages=[
      {"role": "user", "content": "What are some fun things to do in New York?"},
      {"role": "assistant", "content": "You could go to the Empire State Building!"},
      {"role": "user", "content": "That sounds fun! Where is it?"},
    ],
)

print(response.choices[0].message.content)