from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from pydantic import BaseModel, Field
from langchain_core.documents import Document
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

query_generator_prompt = ChatPromptTemplate.from_template('''
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザの質問に値して複数の視点を提供することが目標です。

質問:"""
{question}
"""
''')

class QueryGeneratorOutout(BaseModel):
    queries: list[str] = Field(...,description="検索クエリのリスト")

def reciprocal_rank_fusion(retriver_outputs:list[list[Document]],k : int)->list[str]:
    # 各ドキュメントのコンテンツとそのスコアの辞書を作成
    doc_scores = {}
    for i,docs in enumerate(retriver_outputs):
        for doc in docs:
            content = doc.page_content
            if content not in doc_scores:
                doc_scores[content] = 0
            else:
                doc_scores[content] += 1/(i+1)
                
    # スコアの高い順にソート
    sorted_docs = sorted(doc_scores.items(),key=lambda x:x[1],reverse=True)
    return [content for content,_ in sorted_docs]
    
query_generator_chain = query_generator_prompt | model.with_structured_output(QueryGeneratorOutout) | (lambda x: x.queries)
multi_query_rag_chain = {
    "question": RunnablePassthrough(),
    "context": query_generator_chain | retriever.map() | reciprocal_rank_fusion
} | prompt | model | output_parser

multi_query_rag_chain.invoke("LangChainの概要を教えてください")











