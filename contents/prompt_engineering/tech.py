from dotenv import load_dotenv
from openai import OpenAI
import os
import json
load_dotenv()

api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

prompt = '''
入力された内容がAIに関係するかを教えてください
'''

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt},

        {"role": "user", "content": "プロンプトハッキング"},
        {"role": "system","content": "true"},

        {"role": "user", "content": "こんにちは"},
        {"role": "system","content": "false"},

        {"role": "user", "content": "aiは便利です"}
    ]
)

print(response.choices[0].message.content)  # true

