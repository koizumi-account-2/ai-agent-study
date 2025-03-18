from dotenv import load_dotenv
from openai import OpenAI
import os
import json
load_dotenv()

api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

# systen_prompt = """
# ユーザーが入力した料理のレシピを教えてください
# """

# def get_recipe(food_name:str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": systen_prompt},
#             {"role": "user", "content": food_name}
#         ]
#     )
#     return response.choices[0].message.content

# recipe = get_recipe("ハンバーグ")
# print(recipe)

prompt = '''\
前提条件を踏まえて、ユーザが入力した料理のレシピを教えてください

前提条件: """
分量: 1人分
味の好み: 甘口
"""

出力形式は以下のJSON形式で出力してください
```
{
    "材料": ["材料名1","材料名2","材料名3"],
    "作り方": ["作り方1","作り方2","作り方3"]
}
```
'''

def get_recipe(food_name:str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "user", "content": food_name}
        ]
    )
    return response.choices[0].message.content

recipe = get_recipe("ハンバーグ")
print(recipe)