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
from toolsLibrary import *

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
    def __init__(self, prompt: str):
        self.prompt = prompt
        api_key = os.environ.get("OPENAI_API_KEY")

        self.llm_light = init_chat_model("gpt-5-mini")
        self.llm_very_light = init_chat_model("gpt-5-nano")
        self.llm_strong = init_chat_model("gpt-5.1")

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

    def sql_agent(self, State: State):
        user_prompt = State["messages"][-1].content
        keywords = State["keywords"]

        tools = [runQueryTool, get_SQL_Texts_Tool]
        self.tools_by_name = {tool.name: tool for tool in tools}

        # LLM mit Tools binden
        llm_with_tools = self.llm_strong.bind_tools(tools)

        # Conversation vorbereiten
        if State.get("contextFeedback"):
            sql = State["query"]
            feedback = State["query"]
            prompt = PromptLibrary.sql_feedback_prompt_template.format(
                sql=sql,
                feedback=feedback,
                keywords=keywords
            )
            system_message = {
                "role": "system",
                "content": PromptLibrary.explain_query_to_llm
            }
            user_message = {"role": "user", "content": prompt}
            base_conversation = [system_message, user_message]

        else:
            system_message = {
                "role": "system",
                "content": PromptLibrary.explain_query_to_llm
            }
            user_message = {
                "role": "user",
                "content": PromptLibrary.explain_query_to_llm.format(
                    user_prompt=user_prompt,
                    keywords=keywords
                )
            }
            base_conversation = [system_message, user_message]

        next_step = llm_with_tools.invoke(base_conversation + State["messages"])

        return {"messages": [next_step]}



    def createAnswerAgent(self, State: State):
        user_prompt = State["messages"][-1].content
        keywords = State["keywords"]
        instigation_texts = State["messages"][-1]

        tools2 = [queryGlossar, queryDatasheets]

        self.tools_by_name2 = {tool.name: tool for tool in tools2}
        llm_w_tools=self.llm_strong.bind_tools(tools2)

        #prompt=PromptLibrary.instigation_query_prompt_template.format(user_prompt=user_prompt,keywords=keywords)
        conversation =  PromptLibrary.answer_Prompt_template.format(user_prompt=user_prompt,instigation_texts=instigation_texts)

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




    def  toolcallTriggered(self, State:State)->Literal["tool_node", "createAnswerAgent"]:
        """Route back to joke generator or end based upon feedback from the evaluator"""

        messages = State["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"
        # Otherwise, we stop (reply to the user)
        return "createAnswerAgent"


    def  toolcallTriggered2(self, State:State)->Literal["answer_tool_node", END]:
        """Route back to joke generator or end based upon feedback from the evaluator"""

        messages = State["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "answer_tool_node"
        # Otherwise, we stop (reply to the user)
        return END

    def routeToolCallResult(self, State: dict):
        last_message = State["messages"][-1]

        # 🔌 Datenbankverbindung prüfen
        if isinstance(last_message.content, str) and "08001" in last_message.content:
            print("❌ No connection to database!")
            return END

        # 🛠 Zähler für getSQLTexts hochzählen
        tool_calls = getattr(last_message, "tool_calls", [])
        if tool_calls:  # Wenn mindestens ein Tool aufgerufen wurde
            last_tool_name = tool_calls[-1]["name"]  # Name des letzten Tool-Calls
        else:
            last_tool_name = None

        # Zähler aus State holen oder initialisieren
        counter = State.get("getSQLTexts_counter", 0)

        if last_tool_name == "get_SQL_Texts_Tool":
            counter += 1
            State["getSQLTexts_counter"] = counter  # Wichtig: explizit speichern
            print(f"🔢 getSQLTexts wurde {counter} mal aufgerufen")

            if counter >= 3:
                print("✅ Maximal 3 Aufrufe erreicht, weiter zu createAnswerAgent")
                return "createAnswerAgent"

        # 🧩 Standardweg
        return "sql_agent"

    def instigationsRecieved(self, State: State):
        print(State)


    def tool_node(self, state: dict):
        """Performs the tool call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls: tool = self.tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}


    def answer_tool_node(self, state: dict):
        """Performs the tool call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls: tool = self.tools_by_name2[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}


    def createAgent(self):
        workflow = StateGraph(State)
        workflow.add_node("keyWordExtractor", self.keyWordExtractor)
        workflow.add_node("sql_agent", self.sql_agent)
        #workflow.add_node("get_SQL_Texts", self.get_SQL_Texts)
        workflow.add_node("checkContextRelevance", self.checkContextRelevance)
        workflow.add_node("tool_node", self.tool_node)
        workflow.add_node("answer_tool_node", self.answer_tool_node)
        workflow.add_node("routeToolCallResult", self.routeToolCallResult)
        workflow.add_node("createAnswerAgent", self.createAnswerAgent)
        workflow.add_edge(START, "keyWordExtractor")


        workflow.add_conditional_edges(
            "keyWordExtractor", self.checkKeyWords, {"Fail": "createAnswerAgent", "Pass": "sql_agent"})
        # Add edges to connect nodes


        workflow.add_conditional_edges(
            "sql_agent",
            self.toolcallTriggered,
            ["tool_node", "createAnswerAgent"]

        )

        workflow.add_conditional_edges(
            "createAnswerAgent",
            self.toolcallTriggered2,
            ["answer_tool_node", END]

        )


        #Route if no connection available or max runs of sql db was done
        workflow.add_conditional_edges("tool_node", self.routeToolCallResult, ["sql_agent","createAnswerAgent", END])

        #workflow.add_conditional_edges("answer_tool_node", self.routeToolCallResult, ["createAnswerAgent", END])


        workflow.add_edge("answer_tool_node", "createAnswerAgent")



        #workflow.add_edge("tool_node", "sql_agent")
        #workflow.add_edge("sql_agent", "get_SQL_Texts")

        #workflow.add_edge("get_SQL_Texts", "checkContextRelevance")

        agent = workflow.compile()
        messages = [HumanMessage(content=self.prompt)]
        messages = agent.invoke({"messages": messages})


if __name__ == "__main__":
    agent = Agent("Bestehen die 5xx den IP67 test?")
    agent.createAgent()

        

