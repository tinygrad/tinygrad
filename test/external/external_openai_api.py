from openai import OpenAI

client = OpenAI(
  base_url='http://localhost:11434/v1/',
  api_key='ollama',  # required but ignored
)

response = client.chat.completions.create(
  model='qwen3-vl:8b',
  messages=[
    {
      'role': 'user',
      'content': [
        {'type': 'text', 'text': "What's in this image?"},
      ],
    }
  ],
  max_tokens=300,
)
print(response.choices[0].message.content)