from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

class Recipe(BaseModel):
    name:str = Field(description="料理の名前")
    ingredients:list[str] = Field(description="料理の材料")
    steps:list[str] = Field(description="料理の作り方")
output_parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以下の料理のレシピを教えてください\n\n"
         "{format_instructions}"),
        ("human", "{input}")   
    ]
)
prompt_with_format_instructions = prompt.partial(format_instructions=output_parser.get_format_instructions())


model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)

chain = prompt_with_format_instructions | model | output_parser

result = chain.invoke({"input":"チーズケーキ"})
print(type(result))
print(result)
