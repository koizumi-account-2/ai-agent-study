from dotenv import load_dotenv
from openai import OpenAI
import os
import json
load_dotenv()

api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

prompt = '''
入力されたテーマに合ったメインカラーのコードを教えてください

前提条件: """
メインカラーは3色提示してください
"""

出力形式は以下のJSON形式で出力してください
```
[
    {
        "color": "メインカラーのコード1",
        "name": "メインカラーの名前1",
        "reason": "メインカラーの理由1"
    },
    {
        "color": "メインカラーのコード2",
        "name": "メインカラーの名前2",
        "reason": "メインカラーの理由2"
    },
    ...
]
```
'''

def get_main_color(theme:str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": theme}
        ]
    )
    return response.choices[0].message.content

print(get_main_color("元気が出る"))