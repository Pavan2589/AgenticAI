import ollama

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "system",
            "content": "You are an expert Python tutor. Explain concepts simply."
        },
        {
            "role": "user",
            "content": "What is a Python decorator?"
        }
    ],
    think=False,
)

print(response.message.content)