from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.retrievers import TavilySearchAPIRetriever
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
'''
以下の文脈を踏まえてユーザの質問に回答してください。

文脈:
"""
{context}
"""

ユーザの質問:
"""
{input}
"""
'''
)

retriever = TavilySearchAPIRetriever(api_key=tavily_api_key, k=3)

chain = (
    {
        "context":retriever,
        "input":RunnablePassthrough()
    }
    | RunnablePassthrough.assign(answer = prompt | model | output_parser)
)

result = chain.invoke("明日からの気温について")

print(result)




























