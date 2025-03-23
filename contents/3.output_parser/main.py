
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()

api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
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
prompt_value  = prompt_with_format_instructions.invoke({"input":"チーズケーキ"})
print("====system prompt====")
print(prompt_value.messages[0].content)
print("====user prompt====")
print(prompt_value.messages[1].content)

ai_message = model.invoke(prompt_value)
print("====ai message====")
print(ai_message.content)

recipe = output_parser.parse(ai_message.content)
print("====recipe====")
print(type(recipe))
print(recipe)