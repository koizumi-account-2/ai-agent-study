from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)

output_parser = StrOutputParser()


cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザの質問にステップバイステップで回答してください。"),
        ("human", "{input}")   
    ]
)

# チェーン
cot_chain = cot_prompt | model | output_parser



summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論を要約してください。"),
        ("human", "{text}")   
    ]
)

summarize_chain = summarize_prompt | model | output_parser

cot_summarize_chain = cot_chain | summarize_chain

result = cot_summarize_chain.invoke({"input":"10 + 2 * 3 - 5"})
print(result)
























