from huggingface_hub import InferenceClient

client = InferenceClient(base_url="http://127.0.0.1:8000")
output = client.chat.completions.create(
    model="qwen2",
    messages=[
        {"role": "user", "content": "Who are you?"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
