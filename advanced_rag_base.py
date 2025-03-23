from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from pydantic import BaseModel
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
output_parser = StrOutputParser()

def file_filter(file_path:str)->bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    file_filter=file_filter,
    branch="master"
)

docs = loader.load()

## ベクトル化によるindexing

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=api_key)
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください

文脈:"""
{context}
"""
                                        
質問:"""
{question}
"""
''')

langchain_document_retriever = retriever.with_config({"run_name":"langchain_document_retriever"})

langchain_web_retriever = retriever.with_config({"run_name":"langchain_web_retriever"})

from enum import Enum
class Route(str,Enum):
    DOCUMENT = "document"
    WEB = "web"

class RouteOutput(BaseModel):
    route: Route

route_prompt = ChatPromptTemplate.from_template('''
質問に回答するために適切なRetrieverを選択してください

質問:"""
{question}
"""
''')

route_chain = route_prompt | model.with_structured_output(RouteOutput) | (lambda x: x.route)

def routed_retriever(inp:dict[str,any])->list[Document]:
    question = inp["question"]
    route = inp["route"]
    if route == Route.DOCUMENT:
        return langchain_document_retriever.invoke(question)
    elif route == Route.WEB:
        return langchain_web_retriever.invoke(question)
    else:
        raise ValueError(f"Invalid route: {route}")
    
chain = {
    "question": RunnablePassthrough(),
    "route": route_chain
} | RunnablePassthrough.assign(context=routed_retriever) | prompt | model | output_parser

chain.invoke("明日の天気を教えてください")











