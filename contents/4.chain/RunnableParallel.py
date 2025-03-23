from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
output_parser = StrOutputParser()

optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは楽観的なアシスタントです。ユーザの質問に100文字程度で回答してください。"),
        ("human", "{input}")   
    ]
)

pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは悲観的なアシスタントです。ユーザの質問に100文字程度で回答してください。"),
        ("human", "{input}")   
    ]
)   

synthesis_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたはシンセシスアシスタントです。楽観的な回答と悲観的な回答を参考に、2つの意見をまとめて100文字程度で回答してください。"),
        ("human", "楽観的な回答:{optimistic_opinion}\n悲観的な回答:{pessimistic_opinion}")   
    ]
)

# 楽観的な回答
optimistic_chain = optimistic_prompt | model | output_parser
# 悲観的な回答
pessimistic_chain = pessimistic_prompt | model | output_parser
# 並列実行
synthesis_chain = (
    {
        "optimistic_opinion":optimistic_chain,
        "pessimistic_opinion":pessimistic_chain
    }
    | synthesis_prompt
    | model
    | output_parser
)

result = synthesis_chain.invoke({"input":"人口増加について"})

print(result)























