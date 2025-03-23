from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Annotated, Any,Optional
import operator 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph,END
import os

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)
output_parser = StrOutputParser()



class Purpose(BaseModel):
    tone: str = Field(default="",description="雰囲気・感情")
    user_context: str = Field(default="",description="UIの使用場面")
    target_user: str = Field(default="",description="想定ユーザー層")
    emotion: str = Field(default="",description="色の心理効果")

class ColorSuggestion(BaseModel):
    color: str = Field(...,description="色のコード")
    name: str = Field(...,description="色の名前")
    reason: str = Field(...,description="色選択の理由")

class ColorSuggestionList(BaseModel):
    suggestions:  Annotated[list[ColorSuggestion],operator.add] = Field(default_factory=list,description="提案履歴")

class State(BaseModel):
    query: str = Field(...,description="ユーザーの質問")
    purpose: Optional[Purpose] = Field(default=None, description="ユーザー要求の目的分析結果")
    current_role: str = Field(default="",description="選定された回答ロール")
    suggestions: ColorSuggestionList= Field(default=ColorSuggestionList(),description="提案履歴")
    current_judge: bool = Field(default=False,description="品質チェックの結果")
    judge_reason: str = Field(default="",description="品質チェックの理由")


from langgraph.graph import StateGraph
workflow = StateGraph(State)

from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

from langchain_core.prompts import ChatPromptTemplate
class PurposeGenerator:
    def __init__(self,llm:ChatOpenAI):
        self.llm = llm.with_structured_output(Purpose)

    def run(self,user_query:str)->Purpose:
        # prompt
        prompt = ChatPromptTemplate.from_template(
        """
以下のユーザー要求を読み取り、次の観点で分析してください：
- 雰囲気・感情（tone）
- 使用場面（user_context）
- 想定ユーザー層（target_user）
- 色の使用目的（emotion） 

ユーザーの要求:'''{user_query}'''
        """.strip()
        )
        chain = prompt | self.llm
        return chain.invoke({"user_query":user_query})


class SuggestionColor:
    def __init__(self,llm:ChatOpenAI):
        self.llm = llm.with_structured_output(ColorSuggestionList)

    def run(self,purpose:Purpose)->ColorSuggestionList:
        # prompt
        prompt = ChatPromptTemplate.from_template(
        """
あなたはUIデザイン向けのメインカラーの提案をするエージェントです。
以下の要求の分析結果に対して、色の提案をしてください。
提案は、色のコード、色の名前、色の理由を含めて3つ提示してください。

【分析結果】:'''
- 雰囲気・感情（tone）: {tone}
- UIの使用場面（user_context）: {user_context}
- 想定ユーザー層（target_user）: {target_user}
- 色の心理効果（emotion）: {emotion}
'''
        """.strip()
        )
        chain = prompt | self.llm
        return chain.invoke(purpose.model_dump())

class Judgement(BaseModel):
    current_judge: bool = Field(default=False,description="判定結果")
    reason: str = Field(default="",description="判定理由")

class CheckSuggestion:
    def __init__(self,llm:ChatOpenAI):
        self.llm = llm.with_structured_output(Judgement)

    def run(self,state: State) -> dict[str, Any]:
        from langchain.prompts import ChatPromptTemplate
        import json
        print(state.suggestions)
        # 直近の提案（最新の3つ）を取得
        suggestions = state.suggestions[-3:]
        
        # JSON文字列に変換（LLMに渡しやすいように）
        suggestions_text = json.dumps(
            [s.model_dump() for s in suggestions],
            ensure_ascii=False,
            indent=2
        )

        query = state.query  # 質問内容
        prompt = ChatPromptTemplate.from_template(
        """
以下はユーザーの質問と、それに対する色の提案です。
この提案の品質をチェックしてください。

- 提案が3つあり、それぞれ色コード、名前、理由が含まれているか。
- 色がユーザーの要求（雰囲気・用途・感情）に合っているか。
- 不適切な点があれば指摘してください。

問題がある場合には'False'、問題がない場合には'True'を返答してください。また、その判断理由も返答してください。

ユーザーの質問: {query}
色の提案:
{suggestions}
        """.strip()
        )

        chain = prompt | self.llm
        judgement: Judgement = chain.invoke({
            "query": query,
            "suggestions": suggestions_text
        })
        print(judgement)
        return {
            "current_judge": judgement.current_judge,
            "judge_reason": judgement.reason
        }
    

class ColorAgent:
    def __init__(self,llm:ChatOpenAI):
        self.purpose_generator = PurposeGenerator(llm)
        self.suggestion_color = SuggestionColor(llm)
        self.check_suggestion_node = CheckSuggestion(llm)

        self.graph = self._create_graph()
    
    # 目的分析
    def _purpose_node(self,state:State)->dict[str,Any]:
        purpose: Purpose = self.purpose_generator.run(state.query)
        print(purpose)
        return {"purpose":purpose}
    # 色提案
    def _suggestion_node(self,state:State)->dict[str,Any]:
        suggestions: ColorSuggestionList = self.suggestion_color.run(state.purpose)
        print(suggestions)
        return {"suggestions":suggestions.suggestions}
    
    # 品質チェック
    def _check_node(self,state:State)->dict[str,Any]:
        judgement: Judgement = self.check_suggestion_node.run(state)
        print(judgement)
        return {
            "current_judge": judgement["current_judge"],
            "judge_reason": judgement["judge_reason"]
        }
    
    # グラフの作成
    def _create_graph(self)->StateGraph:
        workflow = StateGraph(State)
        workflow.add_node("purpose_node",self._purpose_node)
        workflow.add_node("suggestion_node",self._suggestion_node)
        workflow.add_node("check_node",self._check_node)

        workflow.set_entry_point("purpose_node")

        workflow.add_edge("purpose_node","suggestion_node")
        workflow.add_edge("suggestion_node","check_node")
        workflow.add_conditional_edges(
            "check_node",
            lambda state:state.current_judge,
            {
                True: END,
                False: "purpose_node"
            }
        )
        print("graph created")
        return workflow.compile()
    
    def run(self,query:str)->str:
        init_state = State(query=query)
        result = self.graph.invoke(init_state)
        print(result["suggestions"][-1])
        return result
    
    
if __name__ == "__main__":
    agent = ColorAgent(model)
    agent.run("社内で使うシステムです。使用者は社員です。見ていて落ち着く色を提案してください")