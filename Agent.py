import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Literal
from langchain.tools import tool
from langgraph.graph import MessagesState

from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import pandas as pd
import numpy as np
from Datasources import ELabJobsDB
import PromptLibrary
import SupportFunctions
import toolsLibrary

class State(MessagesState):
    combined_output: str
    keywords: List[Tuple[str, str]]
    instigationids: List[int]
    correctOrNot: str
    sqlFeedback: str
    query: str
    query_result: pd.DataFrame
    instigation_texts: str
    answer: str
    contextRelevant:str
    contextFeedback:str


class Agent: 
    def __init__(self):

        api_key = os.environ.get("OPENAI_API_KEY")

        self.llm_light = init_chat_model("gpt-5-mini")
        self.llm_strong = init_chat_model("gpt-5")
        self.elab_DB= ELabJobsDB()
        self.queryLimiter = 0
        self.contextLimiter = 0
        self.MAXQUERYRETRIES = 3
        self.MAXCONTEXTRETRIES = 3

    def keyWordExtractor(self, State: State):
        # Vordefinierte Typen

        user_prompt = State["messages"][-1].content

        content = PromptLibrary.keyWordExtractionPrompt.format(
            user_prompt=user_prompt,
            keyword_descriptions=PromptLibrary.keyword_descriptions,
        )
        llm_structured = self.llm_light.with_structured_output(SupportFunctions.KeywordOutput)
        result = llm_structured.invoke(content)

        return {"keywords": [item.keyword for item in result.keywords]}




    def llm_call(self, State: State):
        user_prompt = State["messages"][-1].content
        keywords = State["keywords"]
        tools = [toolsLibrary.runQueryTool, toolsLibrary.get_SQL_Texts, toolsLibrary.createAnswer]
        self.tools_by_name = {tool.name: tool for tool in tools}
        llm_w_tools=self.llm_strong.bind_tools(tools)
        if State.get("contextFeedback"):
            sql = State["query"]
            feedback = State["query"]
            prompt=PromptLibrary.sql_feedback_prompt_template.format(sql=sql,feedback=feedback, keywords=keywords)
            conversation = [
                {"role": "system", "content": PromptLibrary.explain_query_to_llm},
                {"role": "user", "content": prompt},
            ]

        else:
            prompt=PromptLibrary.instigation_query_prompt_template.format(user_prompt=user_prompt,keywords=keywords)

            conversation =  PromptLibrary.explain_query_to_llm + "\n\n" + prompt


        return {
            "messages": [
                llm_w_tools.invoke(
                    [
                        SystemMessage(
                            content=conversation
                        )
                    ]
                    + State["messages"]
                )
            ]
        }

    def createAnswer(self, State: State):

        user_prompt = State["messages"][0].content
        instigation_texts=State["messages"].toolmessage

        prompt=PromptLibrary.answer_from_context_prompt_template.format(user_prompt=user_prompt, instigation_texts=instigation_texts)
        result = self.llm_light.invoke(prompt)
        return {"answer": result.content}

    def checkKeyWords(self, State: State):
        """Gate function to check if any keywords are present."""
        if State.get("keywords") and len(State["keywords"]) > 0:
            return "Pass"
        return "Fail"


    def checkContextRelevance(self, State: State):
        evaluator = self.llm_light.with_structured_output(SupportFunctions.ContextFeedback)
        grade = evaluator.invoke(
            f"This is the context:\n{State['instigation_texts']} \n\nThis is the users Prompt:\n{State['messages'][0]} ")
        return {"contextRelevant": grade.grade, "contextFeedback": grade.feedback}

    def routeisContextRelevant(self, State: State):
        """Route back to joke generator or end based upon feedback from the evaluator"""
        if State["contextRelevant"] == "relevant":
            return "createAnswer"
        elif State["sqlFeedback"] == "not relevant":
            if self.contextLimiter <= 2:
                self.contextLimiter += 1
                return "llm_call"
            else:
                return "createAnswer"

    def  toolcallTriggered(self, State:State)->Literal["tool_node", END]:
        """Route back to joke generator or end based upon feedback from the evaluator"""

        messages = State["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"
        # Otherwise, we stop (reply to the user)
        return END


    def instigationsRecieved(self, State: State):
        print(State)


    def tool_node(self, state: dict):
        """Performs the tool call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls: tool = self.tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}


    def createAgent(self):
        workflow = StateGraph(State)
        workflow.add_node("keyWordExtractor", self.keyWordExtractor)
        workflow.add_node("llm_call", self.llm_call)
        #workflow.add_node("get_SQL_Texts", self.get_SQL_Texts)
        workflow.add_node("checkContextRelevance", self.checkContextRelevance)
        workflow.add_node("tool_node", self.tool_node)


        workflow.add_edge(START, "keyWordExtractor")

        workflow.add_conditional_edges(
            "keyWordExtractor", self.checkKeyWords, {"Fail": END, "Pass": "llm_call"})
        # Add edges to connect nodes

        workflow.add_conditional_edges(
            "llm_call",
            self.toolcallTriggered,
            ["tool_node", END]

        )

        workflow.add_edge("tool_node", "llm_call")
        #workflow.add_edge("llm_call", "get_SQL_Texts")

        #workflow.add_edge("get_SQL_Texts", "checkContextRelevance")

        agent = workflow.compile()
        messages = [HumanMessage(content="Gibts Messungen zum 505?")]
        messages = agent.invoke({"messages": messages})
        for m in messages["messages"]:
            m.pretty_print()

if __name__ == "__main__":
    agent = Agent()
    agent.createAgent()

        

