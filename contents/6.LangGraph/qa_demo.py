from langchain_core.pydantic_v1 import BaseModel,Field
from typing import Annotated
import operator

class State(BaseModel):
    query: str = Field(...,description="ユーザーの質問")
    current_role: str = Field(default="",description="選定された回答ロール")
    massages: Annotated[list[str],operator.add] = Field(default=[],description="回答履歴")
    current_judge: bool = Field(default=False,description="品質チェックの結果")
    judge_reason: str = Field(default="",description="品質チェックの理由")


from langgraph.graph import StateGraph
workflow = StateGraph(State)

from typing import Any
# def answering_node(state:State)->dict[str,Any]:
#     query = state.query
#     role = state.current_role

#     # ユーザからの質問内容と選択されたロールをもとに回答を生成するロジック
#     genereted_message= # ...生成処理...

#     # 生成された回答でステートを更新
#     return {"massages":[genereted_message]}

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
prompt = # ...queryとroleを引数にとるChatPropmptTemplate...
llm = # ...生成に使用するLLM...
answering_node = (
    RunnablePassthrough.assign(
        query= lambda x: x["query"],
        role= lambda x: x["current_role"],
    )
    | prompt 
    | llm 
    | StrOutputParser()
    | RunnablePassthrough.assign(
        massages=lambda x: [x["massages"]]
    )
)

from langgraph.graph import END
def check_node(state:State)->dict[str,Any]:
    query = state.query
    message = state.massages[-1]

    # 回答の品質をチェックするロジック
    # ...チェック処理...
    judge_result = # ...チェック結果...
    judge_reason = # ...チェック理由...

    # チェック結果でステートを更新
    return {"current_judge": judge_result,
        "judge_reason": judge_reason
    }

workflow.set_entry_point("selection")
workflow.add_edge("selection","answering")

workflow.add_conditional_edges(
    "check",
    lambda state:state.current_judge,
    {
        True: END,
        False: "selection"
    }
)

completed_graph = workflow.compile()