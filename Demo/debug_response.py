import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个知识渊博且风趣的AI助手。"},
        {"role": "user", "content": "用一句话给一款新上市的智能手表写个Slogan。"}
    ],
    temperature=0.7,
    max_tokens=300
)

print("=== Full response object ===")
print(response)
print("\n=== choices[0] ===")
print(response.choices[0])
print("\n=== message ===")
print(response.choices[0].message)
print("\n=== content ===")
print(repr(response.choices[0].message.content))
print("\n=== reasoning_content ===")
rc = getattr(response.choices[0].message, "reasoning_content", "<no attr>")
print(repr(rc))
