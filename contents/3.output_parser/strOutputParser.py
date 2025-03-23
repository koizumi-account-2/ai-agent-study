
from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()

api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
output_parser = StrOutputParser()
ai_message = AIMessage(content="チーズケーキのレシピは以下の通りです。")
recipe = output_parser.invoke(ai_message)
print(type(recipe))
print(recipe)

