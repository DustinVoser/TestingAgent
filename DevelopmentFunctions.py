from Datasources import ChromaDB, LocalFolderSource
from Functions.classLibrary import Prompt, MetaData, Document, llm_gpt
from toolsLibrary import *
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
    chroma = ChromaDB("§")
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






def createTextFile():
    ids = np.linspace(1,200 , 200)
    ids_int = ids.astype(int).tolist()

    text=get_SQL_Texts(ids_int)


    with open("allSQLTexts.txt", "w", encoding="utf-8") as f:
        f.write(text['instigation_texts'])

    print("File created.")




def evaluateAnswers(p:Prompt):
    llm= llm_gpt()
    llm.connect("gpt-5")
    pEval=Prompt()
    pEval.sys_prompt = (
        "Du musst einen KI agent kritisch bewerten\n"
        "Ablauf:\n"
        "1. Der Benutzer stellt eine Frage (Prompt).\n"
        "2. Ein Retrieval-Modul holt passenden Kontext.\n"
        "3. Ein LLM erstellt eine Antwort basierend auf diesem Kontext.\n\n"
        "Vergib für jede Kategorie eine Punktzahl von 0 bis 10:\n"
        "- 10 = perfekt (sehr selten).\n"
        "- 0 = nicht nutzbar.\n\n"
        "Bewertungskategorien:\n"
        "1. Retrieval-Score:\n"
        "   - Enthält der abgerufene Kontext alle notwendigen Informationen, um die Frage korrekt zu beantworten?\n"
        "   - Enthält er zu viele irrelevante Informationen?\n"
        "   - Vergib niedrigere Punkte, wenn unnötiger Kontext enthalten ist.\n\n"
        "2. Answer-Score:\n"
        "   - Enthält die generierte Antwort mindestens die Informationen der Referenzantwort?\n"
        "   - Die Referenzantwort dient nur als Referenz für den Inhalt, nicht den grammatischen Aufbau. \n"
        "   - Relevante und korrekte Zusatzinformationen erhöhen den Score.\n\n"
        "3. Usefulness-Score:\n"
        "   - Ist die Antwort gut strukturiert und leicht verständlich?\n"
        "   - Tabellen, klare Sprache und Erklärungen für komplexe Begriffe erhöhen die Punktzahl.\n"
        "   - Fokus: Form der Antwort, nicht nur der Inhalt.\n\n"
        "4. Comment:\n"
        "   - Kurze Begründung für die vergebenen Scores.\n\n"
    )

    pEval.user_prompt = (
        f"Benutzerfrage:\n{p.user_prompt}\n\n"
        f"Generierte Antwort:\n{p.answer}\n\n"
        f"Referenzantwort:\n{p.ref_Answer}\n\n"
        f"Kontext, auf dem die Antwort basiert:\n{p.context}"
    )
    a=llm.getEvaluation(prompt=pEval)
    return a






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



import os
import win32com.client

def convert_doc_to_docx(source_folder, target_folder):
    """
    Konvertiert alle .doc-Dateien im source_folder zu .docx und speichert sie im target_folder.
    """
    # Prüfen, ob Zielordner existiert, ansonsten erstellen
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Word COM-Objekt starten
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".doc") and not filename.lower().endswith(".docx"):
            doc_path = os.path.join(source_folder, filename)
            docx_filename = os.path.splitext(filename)[0] + ".docx"
            docx_path = os.path.join(target_folder, docx_filename)

            # Datei öffnen und speichern als .docx
            doc = word.Documents.Open(doc_path)
            doc.SaveAs(docx_path, FileFormat=16)  # 16 = wdFormatDocumentDefault (.docx)
            doc.Close()

    word.Quit()
    print("Konvertierung abgeschlossen.")

# Beispielaufruf:
# convert_doc_to_docx(r"C:\Quellordner", r"C:\Zielordner")






if __name__ == "__main__":
    listDB()