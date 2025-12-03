from Datasources import ChromaDB, LocalFolderSource
from Functions.classLibrary import Prompt, MetaData, Document, llm_gpt, SilverStandard, PromptManager
from toolsLibrary import *
from PromptLibrary import evalPipeLinePrompt
import numpy as np

def showFiles():
    source=ChromaDB("datasheets")
    a=source.load_files()
    a.to_csv("output.csv", index=False)
    print(a.columns.tolist())
    b=0
    files=source.keywordsRetrieval([""])
    print (files)


def listDB():
    chroma =ChromaDB("datasheets")
    a=chroma.load_files()
    print (chroma.load_files())


def buildGlossarDB():
    source = LocalFolderSource(r"s:\CH_Huba_E\Technologie\GM\Allgemein\Ausbildung\Lexikon\Originale\Documents_docx")
    source.connect()
    chroma = ChromaDB("Glossar")
    docs = source.get_documents(topN=200)
    llm =llm_gpt()
    llm.connect("gpt-5-mini")
    for d in docs:
        d.meta = readMetaData(d)
        chroma.add_document(d)
        print(f"{d.title} ({len(d.content)} Zeichen)")

    df = chroma.load_files()
    source.disconnect()



def buildDatasheetDB():
    source = LocalFolderSource("datasheets")
    source.connect()
    chroma = ChromaDB("datasheets")
    a = chroma.load_files()
    docs = source.get_documents(topN=100)
    llm =llm_gpt()
    llm.connect("gpt-5-mini")


    for d in docs:
        d.meta = readMetaData(d)
        chroma.add_document(d)
        print(f"{d.title} ({len(d.content)} Zeichen)")

    df = chroma.load_files()
    source.disconnect()


def buildSilverstandard():
    ids = np.linspace(1, 200 , 200)
    llm=llm_gpt()
    pm=PromptManager("Samples/SilverStandard_251201")
    llm.connect("gpt-5-mini")
    for id in ids:
        print(id)
        text=get_SQL_Texts(id)
        # Skip if text is empty or whitespace
        if not text or not str(text).strip():
            continue
        prompt=Prompt()
        prompt.user_prompt=f"""You will receive a text describing a technical investigation.

Your task is to create exactly 3 question–answer pairs that help a user later identify this investigation through natural search-style questions.

The questions must follow the style of typical internal lookup queries, for example:

„Bestehen die 5xx den IP67 Test?“

„Was wurde beim 578 für Daikin gemacht?“

„Welche Kleber wurden bei den 400-er Zellen evaluiert?“

„Wie performen 5xx mit Ascenta-Zellen?“

„Gibt es Messungen zum Wika WUC-10?“

„Wurde ein Signal Conditioner auf einem Biegebalken getestet?“

„Welche chinesischen Zellen wurden getestet?“

„Are there any condensation tests done in the 505?“

❗ Strict rules to follow:

Questions must reference one clear technical anchor, such as:

Produktcode (505, 511, 5xx, 711, 2xx, etc.)

Hersteller/Cell Supplier (Ascenta, HXL, Kyocera, …)

Sensor-/Gerätetyp

Material, Kleber, Paste, Membran

Testtyp (IPX7, Vibration, Kondensation, Hysterese …)

This anchor must be taken from facts explicitly in the provided text.

Questions must avoid Instigation IDs unless the text cannot be referenced otherwise.
If possible, use product, sensor, cell supplier, test type, material, component — not Instigation.

Ask only questions that are 100% answerable from the text.
Do NOT ask questions about:
hypothetische Abweichungen („Gab es Abweichungen bei bestimmten Drücken?“)
irgendwelche Qualifikationen ohne Referenz
interne Abläufe, die nicht erwähnt werden
Interpretationen oder Mutmaßungen
Messgenauigkeiten, Fehlergründe, Abweichungsanalysen

The question must be useful as a later search key.
It should help the user find exactly that investigation, even with only a keyword like
“505”, “HXL”, “Kleber”, “WUC-10”, “Ascenta”, “IP67”, etc.

Each question must refer to 1–2 concrete facts from the text.
No overly deep technical dives.

Answers must be short and factual.


Each question must be completely independent.
No question may refer to or depend on another question.
Each question must contain all the necessary context/facts to identify the investigation alone.
Examples of incorrect dependency:
Q1: „Was wurde bei den 505 von Huba gemessen?“
Q2: „Gibt es weitere Tests zu den Sensoren?“ → Q2 hängt von Q1 ab → not allowed
Examples of correct independent questions:
„Was wurde bei den 505 von Huba gemessen?“
„Welche Tests wurden an 511-Sensoren durchgeführt?“
„Gibt es Messungen zum Wika WUC-10?“

Text to use:
{text}

Follow the rules and populate the structured output accordingly.
"""
        # Request structured output with skip option
        prompt.textFormat= SilverStandard
        gen=llm.runQuery(prompt)
        # Skip if the model indicates not enough text or returns no items
        if getattr(gen, "not_enough_text", False) or (hasattr(gen, "items") and len(gen.items) == 0):
            continue
        for item in gen.items:
            p = Prompt(
                sys_prompt="",
                user_prompt=item.question,
                context="",
                answer=item.answer,
                history="",
                ref_Answer=item.answer,
                difficulty=item.difficulty,
                instigationids=[int(id)]
            )
            pm.add_prompt(p)

























def createTextFile():
    ids = np.linspace(1,200 , 200)
    ids_int = ids.astype(int).tolist()

    text=get_SQL_Texts(ids_int)


    with open("allSQLTexts.txt", "w", encoding="utf-8") as f:
        f.write(text['instigation_texts'])

    print("File created.")



import os
import json
import threading
import queue
import time
from tqdm import tqdm


def evaluate_single_item(s, test_map, done, eval_sys_prompt, llm, out_list):
    """Bewertet EIN Item und gibt eval_data zurück."""
    def norm(s):
        return " ".join(s.split()).strip() if isinstance(s, str) else ""

    def get_field(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return ""

    raw_prompt = get_field(s, "user_prompt", "userQuery", "question", "prompt")
    up = norm(raw_prompt)

    ref_answer = get_field(s, "ref_Answer", "reference", "reference_answer", "answer")
    test = test_map.get(up, {"answer": "", "context": ""})

    answer = test["answer"]
    context = test["context"]

    user_prompt = (
        f"Benutzerfrage:\n{raw_prompt}\n\n"
        f"Generierte Antwort:\n{answer}\n\n"
        f"Referenzantwort:\n{ref_answer}\n\n"
        f"Kontext:\n{context}"
    )

    # ---- Bewertung ----
    try:
        if not answer.strip():
            return {
                "user_prompt": raw_prompt,
                "ref_Answer": ref_answer,
                "test_answer": answer,
                "scores": {"ContextRelevance": 0, "Answer": 0, "Usefulness": 0},
                "Comment": "Keine Antwort im Testrun gefunden."
            }

        p = Prompt(sys_prompt=eval_sys_prompt, user_prompt=user_prompt)
        res = llm.getEvaluation(prompt=p)

        return {
            "user_prompt": raw_prompt,
            "ref_Answer": ref_answer,
            "test_answer": answer,
            "scores": {
                "ContextRelevance": int(getattr(res, "RetrievalScore", 0)),
                "Answer": int(getattr(res, "AnswerScore", 0)),
                "Usefulness": int(getattr(res, "UsefulnessScore", 0)),
            },
            "Comment": getattr(res, "Comment", "")
        }

    except Exception as e:
        return {
            "user_prompt": raw_prompt,
            "ref_Answer": ref_answer,
            "test_answer": answer,
            "scores": {"ContextRelevance": 0, "Answer": 0, "Usefulness": 0},
            "Comment": f"LLM-Fehler: {e}"
        }


# --------------------------------------------------------------
#           MULTITHREADED HAUPTFUNKTION
# --------------------------------------------------------------

def evaluateAgentRun(
    run_dir: str = r"Runs\\Run1_021225",
    silver_file: str = "SilverStandard_Final.json",
    testrun_file: str = "Testrun_251202.json",
    out_file: str = "EvalResults_251203.json",
    max_threads: int = 10
):
    import os, json, queue, threading, time
    from tqdm import tqdm

    # ----------------------------
    # Hilfsfunktionen
    # ----------------------------
    def load_json(path, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default

    def norm(s):
        return " ".join(s.split()).strip() if isinstance(s, str) else ""

    def get_field(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return ""

    # ----------------------------
    # Windows-sicheres atomares Schreiben
    # ----------------------------
    def safe_write_json_atomic(path, data, attempts=30, wait=0.1):
        """Schreibe atomar und Windows-safe mit Retry."""
        tmp = path + ".tmp"

        # Schritt 1: sicher Datei schreiben
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Schritt 2: Windows-Locks umgehen
        for _ in range(attempts):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(wait)

        print(f" WARNUNG: Datei konnte nicht ersetzt werden: {path}")

    # ----------------------------
    # Dateien laden
    # ----------------------------
    silver = load_json(os.path.join(run_dir, silver_file), [])
    testrun = load_json(os.path.join(run_dir, testrun_file), [])

    out_path = os.path.join(run_dir, out_file)
    results = load_json(out_path, [])

    done = {norm(get_field(r, "user_prompt")) for r in results}

    test_map = {
        norm(get_field(item, "user_prompt", "userQuery", "question", "prompt")): {
            "answer": get_field(item, "answer", "Antwort", "response", "assistant"),
            "context": item.get("context", "")
        }
        for item in testrun if isinstance(item, dict)
    }

    # ----------------------------
    # LLM Setup
    # ----------------------------
    llm = llm_gpt()
    llm.connect("gpt-5-mini")
    eval_sys_prompt = evalPipeLinePrompt

    # ----------------------------
    # Queue vorbereiten
    # ----------------------------
    q_tasks = queue.Queue()
    q_results = []
    file_lock = threading.Lock()

    for s in silver:
        if not isinstance(s, dict):
            continue
        raw_prompt = get_field(s, "user_prompt", "userQuery", "question", "prompt")
        if norm(raw_prompt) not in done:
            q_tasks.put(s)

    total = q_tasks.qsize()

    # ----------------------------
    # Worker Thread
    # ----------------------------
    def worker():
        while True:
            try:
                s = q_tasks.get_nowait()
            except queue.Empty:
                return

            eval_data = evaluate_single_item(s, test_map, done, eval_sys_prompt, llm, q_results)

            with file_lock:
                results.append(eval_data)
                safe_write_json_atomic(out_path, results)

            q_tasks.task_done()

    # ----------------------------
    # Threads starten
    # ----------------------------
    threads = []
    for _ in range(max_threads):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)

    # ----------------------------
    # Fortschritt
    # ----------------------------
    with tqdm(total=total, desc="Evaluierung", ncols=100) as pbar:
        last_count = 0
        while any(t.is_alive() for t in threads):
            current_count = len(results)
            pbar.update(current_count - last_count)
            last_count = current_count
            time.sleep(0.2)

        current_count = len(results)
        pbar.update(current_count - last_count)

    return out_path




def readMetaData(d: Document) -> MetaData:
    llm = llm_gpt()
    llm.connect("gpt-5-nano")

    prompt = Prompt()
    prompt.sys_prompt = "You read meta-data from a given document and output valid JSON for MetaData."
    prompt.user_prompt = (
        f"Document-Title: {d.title}\n"
        f"Document-Location: {d.location}\n"
        f"Document-Tags: {d.tags}\n"
        f"Document-Created: {d.created}\n"
        f"Document-Content: {d.content}"
    )

    prompt.textFormat = MetaData
    response = llm.runQuery(prompt)

    # Erzwinge Typumwandlung:
    if isinstance(response, MetaData):
        return response

    # Falls nur Text oder JSON kommt
    try:
        import json
        data = json.loads(response) if isinstance(response, str) else response
        return MetaData(**data)
    except Exception as e:
        raise ValueError(f"Fehler beim Umwandeln in MetaData: {e}")

def summarize_eval_results(
    eval_file: str,
    png_file: str,
    txt_file: str,
    show: bool = True,
) -> dict:
    """
    Summarizes and visualizes evaluation scores (1–5).

    - Computes averages for ContextRelevance, Answer, Usefulness.
    - Highlights all score=1 cases (including scores that were 0 in old data, mapped to 1).
    - Produces a bar chart PNG with averages and 1-score counts in red.
    - Writes a text summary with averages, total items, and counts of 1-scores per category.
    """

    import json
    import numpy as np
    import os

    if not os.path.exists(eval_file) or os.path.getsize(eval_file) == 0:
        raise FileNotFoundError(f"Eval file not found or empty: {eval_file}")

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected list of evaluation records.")

    # Helper: clamp scores <=0 to 1, max 5
    def get_score(d, key):
        try:
            v = int(d.get("scores", {}).get(key, 1))
        except Exception:
            v = 1
        if v <= 0:
            v = 1
        return min(5, v)

    ctx, ans, use = [], [], []
    ones_ctx = ones_ans = ones_use = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        c = get_score(item, "ContextRelevance")
        a = get_score(item, "Answer")
        u = get_score(item, "Usefulness")

        ctx.append(c)
        ans.append(a)
        use.append(u)

        if c == 1: ones_ctx += 1
        if a == 1: ones_ans += 1
        if u == 1: ones_use += 1

    total = len(ctx)
    if total == 0:
        raise ValueError("No evaluation records found.")

    def avg(lst): return float(np.mean(lst)) if lst else 0.0
    avg_ctx, avg_ans, avg_use = avg(ctx), avg(ans), avg(use)

    # -------------------------------
    # Plot
    # -------------------------------
    fig = None
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        categories = ["Context", "Answer", "Usefulness"]
        avgs = [avg_ctx, avg_ans, avg_use]
        ones = [ones_ctx, ones_ans, ones_use]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(categories, avgs, color=["#4472C4", "#ED7D31", "#70AD47"])

        ax.set_ylim(0, 5.2)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        ax.set_ylabel("Average score (1–5)")
        ax.set_title(f"Evaluation summary — n={total}")

        for i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)
            # 1-score count in red
            ax.text(bar.get_x() + bar.get_width() / 2, max(0.2, h * 0.1),
                    f"1-scores: {ones[i]}", ha="center", va="bottom",
                    fontsize=9, color="black")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_file, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    except Exception:
        fig = None

    # -------------------------------
    # Text Summary
    # -------------------------------
    summary_lines = [
        f"Evaluation file: {eval_file}",
        f"Total items: {total}",
        "",
        "Averages (1–5):",
        f"  ContextRelevance : {avg_ctx:.2f}",
        f"  Answer           : {avg_ans:.2f}",
        f"  Usefulness       : {avg_use:.2f}",
        "",
        "Count of score=1 per category:",
        f"  ContextRelevance : {ones_ctx}",
        f"  Answer           : {ones_ans}",
        f"  Usefulness       : {ones_use}",
    ]

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return {
        "averages": {
            "ContextRelevance": avg_ctx,
            "Answer": avg_ans,
            "Usefulness": avg_use,
        },
        "one_counts": {
            "ContextRelevance": ones_ctx,
            "Answer": ones_ans,
            "Usefulness": ones_use,
        },
        "total": total,
        "output": {"png": png_file if fig else None, "txt": txt_file},
    }



if __name__ == "__main__":
    evaluateAgentRun()

# === Evaluation Summary & Visualization ===

