from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
# output = model.invoke("こんにちは")
# print(output)

messages = [
    SystemMessage(content="あなたはプロの料理人です。料理の作り方を教えてください。"),
    HumanMessage(content="チーズケーキの作り方を教えてください。")
]

output = model.invoke(messages)
for chunk in model.stream(messages):
    print(chunk)