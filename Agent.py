import os
import json
import traceback

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Literal
from langchain.tools import tool
from langgraph.graph import MessagesState

from langchain.messages import SystemMessage, HumanMessage, ToolMessage

import pandas as pd

import SupportFunctions
from Functions.classLibrary import PromptManager, Prompt
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
    def __init__(self):



        self.llm_strong = init_chat_model("gpt-5", temperature=0.0)
        self.llm_light = init_chat_model("gpt-5-mini", temperature=0.0)

        self.elab_DB= ELabJobsDB()
        self.queryLimiter = 0
        self.contextLimiter = 0
        self.MAXQUERYRETRIES = 3
        self.MAXCONTEXTRETRIES = 3


        #variables for evalaution
        self._userPrompt=""
        self._context=""
        self._answer=""
        self._instigationids=[]
        self._glossar=[]
        self._datasheets=[]
        
        

    def keyWordExtractor(self, State: State):
        # Vordefinierte Typen

        user_prompt = State["messages"][-1].content
        self._userPrompt=user_prompt

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
        llm_with_tools = self.llm_light.bind_tools(tools)

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
        llm_w_tools=self.llm_light.bind_tools(tools2)

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
        # Otherwise, we stop (reply to the user) and speichern die finale Antwort
        try:
            content = getattr(last_message, "content", None)
            # content kann ein String oder eine strukturierte Nachricht sein
            if isinstance(content, str):
                self._answer = content
            else:
                # Fallback: in lesbare Zeichenkette umwandeln
                self._answer = str(content)
        except Exception:
            pass
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
        """Performs the tool call for ALL pending tool_calls and returns one ToolMessage per call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            # Speichere Kontext und IDs, wenn get_SQL_Texts_Tool ausgeführt wurde
            try:
                if tool_call.get("name") == "get_SQL_Texts_Tool":
                    # Kontexttext sichern
                    if isinstance(observation, dict) and "instigation_texts" in observation:
                        self._context = observation.get("instigation_texts") or ""
                    # Übergebene IDs sichern
                    args = tool_call.get("args", {}) or {}
                    self._instigationids = args.get("instigationids", []) or []
            except Exception:
                # Keine harte Ausnahme werfen – Logging optional
                pass
            # Content als String serialisieren, falls es sich um komplexe Objekte handelt
            msg_content = observation if isinstance(observation, (str, bytes)) else str(observation)
            result.append(ToolMessage(content=msg_content, tool_call_id=tool_call["id"]))
        return {"messages": result}


    def answer_tool_node(self, state: dict):
        """Performs all pending tool calls for the answer stage and returns one ToolMessage per call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name2[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])

            # Nebenläufig: Ergebnisse in _glossar / _datasheets sichern
            try:
                tool_name = tool_call.get("name")
                if tool_name == "queryGlossar":
                    # Erwartet ein Dict mit key 'documents' (ChromaDB.query)
                    texts: list[str] = []
                    if isinstance(observation, dict):
                        docs = observation.get("documents")
                        # Chroma liefert List[List[str]] pro Query
                        if isinstance(docs, list):
                            for lst in docs:
                                if isinstance(lst, list):
                                    for s in lst:
                                        if isinstance(s, str) and s.strip():
                                            texts.append(s.strip())
                        # Falls alternative Struktur
                        elif isinstance(docs, str) and docs.strip():
                            texts.append(docs.strip())
                    elif isinstance(observation, list):
                        for s in observation:
                            if isinstance(s, str) and s.strip():
                                texts.append(s.strip())
                    elif isinstance(observation, str) and observation.strip():
                        texts.append(observation.strip())

                    # Deduplizieren und anhängen
                    if texts:
                        existing = set(self._glossar or [])
                        for t in texts:
                            if t not in existing:
                                self._glossar.append(t)
                                existing.add(t)

                elif tool_name == "queryDatasheets":
                    # Erwartet ein DataFrame (siehe ChromaDB.keywordsRetrieval -> load_files())
                    titles: list[str] = []
                    try:
                        import pandas as pd  # type: ignore
                    except Exception:
                        pd = None
                    if 'pd' in locals() and pd is not None and isinstance(observation, pd.DataFrame):
                        df = observation
                        # Bevorzugt document_title, ansonsten versuche aus Metadaten
                        for col in ["document_title", "title", "document_location"]:
                            if col in df.columns:
                                vals = [str(v).strip() for v in df[col].dropna().astype(str).tolist() if str(v).strip()]
                                if vals:
                                    titles.extend(vals)
                                    break
                        if not titles and "content" in df.columns:
                            # Als Fallback: erste Zeile/erste Zeile des Textes als Titelersatz
                            for txt in df["content"].dropna().astype(str).tolist():
                                first = str(txt).strip().splitlines()[0] if str(txt).strip() else ""
                                if first:
                                    titles.append(first)
                    elif isinstance(observation, dict):
                        # Falls dict-Struktur mit Metadaten
                        metas = observation.get("metadatas")
                        if isinstance(metas, list):
                            # metadatas kann List[dict] oder List[List[dict]] sein
                            flat = []
                            for m in metas:
                                if isinstance(m, list):
                                    flat.extend([x for x in m if isinstance(x, dict)])
                                elif isinstance(m, dict):
                                    flat.append(m)
                            for md in flat:
                                title = (md.get("document_title") or md.get("title") or md.get("document_location"))
                                if isinstance(title, str) and title.strip():
                                    titles.append(title.strip())
                    # Deduplizieren und anhängen
                    if titles:
                        existing = set(self._datasheets or [])
                        for t in titles:
                            if t not in existing:
                                self._datasheets.append(t)
                                existing.add(t)
            except Exception:
                # Robust gegen fehlende Felder/Strukturen – keine harten Fehler werfen
                pass

            # Content als String serialisieren, um Modell-API-Anforderungen zu erfüllen
            msg_content = observation if isinstance(observation, (str, bytes)) else str(observation)
            result.append(ToolMessage(content=msg_content, tool_call_id=tool_call["id"]))
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

        self.agent = workflow.compile()

        #workflow.add_edge("tool_node", "sql_agent")
        #workflow.add_edge("sql_agent", "get_SQL_Texts")

        #workflow.add_edge("get_SQL_Texts", "checkContextRelevance")

    def runAgent(self, question, result_dict, key):
        messages = [HumanMessage(content=question)]
        output = ""

        for event in self.agent.invoke(
                {"messages": messages},
                stream_mode="messages"
        ):
            chunk = event[1]
            if hasattr(chunk, "content") and chunk.content:
                # Live-Prints deaktiviert, um Überschneidungen bei Parallelbetrieb zu vermeiden
                output += chunk.content

        # Auch im Result-Dict speichern für parallele Nutzung
        result_dict[key] = output
        return output


import threading


def run_agent_thread(question, result_dict, key):
    agent = Agent()
    try:
        agent.createAgent()
        output = agent.runAgent(question, result_dict, key)
        # Sicherstellen, dass wir das Ergebnis auch bei None-Return nicht überschreiben
        if output is None:
            output = result_dict.get(key, "")
        result_dict[key] = output

        # Prompt speichern (Erfolgspfad)
        p = Prompt()
        p.answer = agent._answer
        p.user_prompt = agent._userPrompt
        p.context = agent._context
        p.instigationids = agent._instigationids
        p.glossar = agent._glossar
        p.datasheets = agent._datasheets

        pm = PromptManager("Samples/Testrun_251202.json")
        pm.add_prompt(p)

        return output
    except Exception as e:
        err_text = f"Fehler in Thread {key}: {e}"
        # Keine Prints hier – Fehler wird dem Caller gemeldet
        result_dict[key] = err_text
        return err_text


if __name__ == "__main__":
    import os, json, threading, sys, time

    # Testmodus: True = nur 2 Fragen, False = alle
    test_mode = False
    test_count = 2

    # Fragen laden
    src_path = os.path.join("Samples", "SilverStandard_Final.json")
    questions: list[str] = []
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data if isinstance(data, list) else []:
                if isinstance(item, dict):
                    up = item.get("user_prompt") or item.get("question") or item.get("prompt")
                    if isinstance(up, str) and up.strip():
                        questions.append(up.strip())
    except Exception:
        questions = ["Bestehen die 5xx den IP67 test?"]

    if test_mode:
        questions = questions[:test_count]

    total = len(questions)
    print(f"[MAIN] Starte {total} Aufgaben mit max. 10 parallel laufenden Threads…")

    # Gemeinsame Zustände
    results: dict[int, str] = {}
    status: list[str] = ["pending"] * total  # pending|running|ok|error
    start_time = time.time()
    lock = threading.Lock()
    stop_reporter = False
    state_path = os.path.join("Samples", "state.json")

    def write_state_file():
        """Schreibt einen kompakten Status-Snapshot als JSON, damit extern der Fortschritt auslesbar ist."""
        try:
            done = sum(1 for s in status if s in ("ok", "error"))
            running = sum(1 for s in status if s == "running")
            pending = sum(1 for s in status if s == "pending")
            data = {
                "total": total,
                "done": done,
                "running": running,
                "queued": pending,
                "ok": sum(1 for s in status if s == "ok"),
                "error": sum(1 for s in status if s == "error"),
                "concurrency_limit": 10,
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def render_progress_line():
        done = sum(1 for s in status if s in ("ok", "error"))
        ok = sum(1 for s in status if s == "ok")
        err = sum(1 for s in status if s == "error")
        run = sum(1 for s in status if s == "running")
        queued = sum(1 for s in status if s == "pending")
        elapsed = time.time() - start_time
        pct = (done / total * 100) if total else 100
        # Ein einfacher Balken (ASCII, 50 Spalten)
        width = 50
        filled = int((done / total) * width) if total else width
        bar = "#" * filled + "-" * (width - filled)
        return f"[MAIN] [{bar}] {done}/{total} ({pct:5.1f}%) | running:{run}/10 queued:{queued} ok:{ok} err:{err} | {elapsed:5.1f}s"

    def reporter():
        while True:
            with lock:
                line = render_progress_line()
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
                write_state_file()
                finished = all(s in ("ok", "error") for s in status)
            if stop_reporter or finished:
                break
            time.sleep(0.5)
        # Endgültige Zeile mit Zeilenumbruch
        with lock:
            sys.stdout.write("\r" + render_progress_line() + "\n")
            sys.stdout.flush()
            write_state_file()

    def thread_wrapper(q, results, i):
        with lock:
            status[i] = "running"
        try:
            result = run_agent_thread(q, results, i)
            results[i] = result
            with lock:
                status[i] = "ok" if not (isinstance(result, str) and result.startswith("Fehler")) else "error"
        except Exception as e:
            with lock:
                results[i] = f"Fehler: {e}"
                status[i] = "error"

    # Reporter-Thread starten
    reporter_thread = threading.Thread(target=reporter, daemon=True)
    reporter_thread.start()

    # Scheduler: Max. 10 Threads gleichzeitig
    max_concurrency = 10
    next_index = 0
    active: dict[int, threading.Thread] = {}

    while True:
        with lock:
            done = sum(1 for s in status if s in ("ok", "error"))
        if done >= total:
            break

        # Starte neue Threads bis Limit erreicht oder keine Aufgaben mehr
        while next_index < total and len(active) < max_concurrency:
            t = threading.Thread(target=thread_wrapper, args=(questions[next_index], results, next_index), daemon=True)
            t.start()
            active[next_index] = t
            next_index += 1

        # Aufräumen: beendete Threads entfernen
        finished_ids = [i for i, th in active.items() if not th.is_alive()]
        for i in finished_ids:
            try:
                active[i].join(timeout=0)
            except Exception:
                pass
            active.pop(i, None)

        time.sleep(0.1)

    # Sicherstellen, dass alle Threads beendet sind
    for th in list(active.values()):
        try:
            th.join()
        except Exception:
            pass

    # Reporter stoppen und zusammenführen
    stop_reporter = True
    reporter_thread.join()

    print("[MAIN] Alle Threads beendet. Ergebnisse:")
    for i, msg in results.items():
        print(f"Antwort {i}: {msg}")
