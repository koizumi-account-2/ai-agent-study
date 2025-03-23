from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
llm = llm.configurable_fields(max_tokens=ConfigurableField(id = "max_tokens"))
output_parser = StrOutputParser()


# Role定義
ROLES = {
    "1":{
        "name":"一般知識エキスパート",
        "description":"幅広い分野の一般的な質問に対して回答する",
        "details":"幅広い分野の一般的な質問に対して、正確で渡りやすい回答を提供してください"
    },
    "2":{
        "name":"生成AI製品エキスパート",
        "description":"生成AI製品に関する質問に回答する",
        "details":"生成AIや関連製品、技術に関する質問に対して、製品の情報と深い洞察を提供してください"
    },
    "3":{
        "name":"カウンセラー",
        "description":"カウンセリングやコンサルティングに関する質問に回答する",
        "details":"カウンセリングやコンサルティングに関する質問に対して、専門的なアドバイスと実践的な解決策を提供してください"
    }
}

from pydantic import BaseModel,Field
from typing import Annotated
import operator

# ステート
class State(BaseModel):
    query: str = Field(...,description="ユーザーの質問")
    current_role: str = Field(default="",description="選定された回答ロール")
    messages: Annotated[list[str],operator.add] = Field(default=[],description="回答履歴")
    current_judge: bool = Field(default=False,description="品質チェックの結果")
    judge_reason: str = Field(default="",description="品質チェックの理由")

# selection node
from typing import Any
from langchain_core.prompts import ChatPromptTemplate

def selection_node(state:State)->dict[str,Any]:
    query = state.query
    role_options = "\n".join([f"{k}. {v['name']}" for k,v in ROLES.items()])
    prompt = ChatPromptTemplate.from_template(
        """
        質問を分析して最も適切な回答担当ロールを選択してください。

        選択肢:
        {role_options}
        
        回答は選択肢の番号(1,2,3)で返答してください。

        質問:
        {query}
        """.strip()
    )

    chain = prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    role_number = chain.invoke({"query":query,"role_options":role_options})
    role = ROLES[role_number.strip()]["name"]

    return {"current_role":role}

def answering_node(state:State)->dict[str,Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    prompt = ChatPromptTemplate.from_template(
        """
        あなたは{role}として回答してください。以下の質問に対してあなたの役割に基づいた適切な回答を提供してください。

        役割の詳細:
        {role_details}

        質問:{query}

        回答:""".strip()
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query":query,"role_details":role_details,"role":role})

    return {"messages":[answer]}

# check node

class Judgement(BaseModel):
    is_correct: bool = Field(default=False,description="判定結果")
    reason: str = Field(default="",description="判定理由")

def check_node(state:State)->dict[str,Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(
        """
        以下の回答の品質をチェックして、問題がある場合には'False'、問題がない場合には'True'を返答してください。
        また、その判断理由も返答してください。

        ユーザからの質問: {query}
        回答: {answer}
        """.strip()
    )

    chain = prompt | llm.with_structured_output(Judgement)
    judgement: Judgement = chain.invoke({"query":query,"answer":answer})

    return {"current_judge":judgement.is_correct,"judge_reason":judgement.reason}

# メッセージを追加するノード関数
from langchain_core.messages import SystemMessage,HumanMessage
def add_message(state:State)->dict[str,Any]:
    additional_messages = []
    if not state.messages:
        additional_messages.append(
            SystemMessage(content="あなたは最小限の応答をする対話エージェントです")
        )
    additional_messages.append(HumanMessage(content=state.query))
    return {"messages":additional_messages}

# LLMからの応答を追加するノード関数
def llm_response(state:State)->dict[str,Any]:
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
    answer = llm.invoke(state.messages)
    return {"messages":[answer]}

# チェックポイントの内容を表示する

from langchain_core.runnables import RunnableConfig
import pprint
from langgraph.checkpoint.base import BaseCheckpointSaver
def print_checkpoint_dump(checkpointer:BaseCheckpointSaver,config:RunnableConfig):
    checkpoint_tuple = checkpointer.get_tuple(config)
    print("チェックポイントデータ：")
    pprint.pprint(checkpoint_tuple.checkpoint)
    print("\nメタデータ：")
    pprint.pprint(checkpoint_tuple.metadata)
# グラフの作成
from langgraph.graph import StateGraph,END

graph = StateGraph(State)
graph.add_node("add_message",add_message)
graph.add_node("llm_response",llm_response)
graph.set_entry_point("add_message")
graph.add_edge("add_message","llm_response")
graph.add_edge("llm_response",END)

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

compiled_graph = graph.compile(checkpointer=checkpointer)
config = {"configurable":{"thread_id":"example_thread"}}

user_state = State(query="私の好物はざるそばです。覚えておいてね")
first_result = compiled_graph.invoke(user_state,config)
print_checkpoint_dump(checkpointer,config)
user_query = "私の好物は?"
user_state = State(query=user_query)
second_result = compiled_graph.invoke(user_state,config)
print(second_result)






