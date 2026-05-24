from ollama import chat

response = chat(
    model='qwen2.5',
    messages=[{'role': 'user', 'content': 'What is the currency of China?'}],
)
print(response.message.content)