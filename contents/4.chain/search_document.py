from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import GitLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)

def file_filter(file_path:str)-> bool:
    return file_path.endswith(".mdx")  

loader = GitLoader(clone_url="https://github.com/langchain-ai/langchain",
                   branch="master",
                   repo_path="./langchain",
                   file_filter=file_filter)   

raw_docs = loader.load()
print(len(raw_docs)) #405

## チャンク分割
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

docs = text_splitter.split_documents(raw_docs)

print(len(docs)) #1500


# ベクトル化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=api_key)

# ドキュメント保存
vector_store = Chroma.from_documents(documents=docs,embedding=embeddings)

# Retriever
retriever = vector_store.as_retriever()

query = "S3からファイルを読み込む方法を教えてください"

context_docs = retriever.invoke(query)
print(f"ドキュメント数: {len(context_docs)}")

first_doc = context_docs[0]
print(f"ドキュメントMeta: {first_doc.metadata}")
print(f"ドキュメント内容: {first_doc.page_content}")


























