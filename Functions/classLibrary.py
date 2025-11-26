from dataclasses import dataclass, Field
from typing import Dict, Literal, List
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pathlib import Path
import json


from openai import OpenAI
from pydantic import BaseModel, Field


from pydantic import BaseModel, Field
from typing import List, Literal

class MetaData(BaseModel):
    """Inhaltliche Metadaten und semantische Klassifikation eines Dokuments."""

    doc_type: Literal[
        "research_paper",
        "manual",
        "email",
        "technical_report",
        "measuring_report"
        "datasheet",
        "drawing",
        "specification",
        "standard",
        "meeting_minutes",
        "presentation",
        "safety_document",
        "maintenance_log",
        "contract",
        "website",
        "other"
    ] = Field(
        "other",
        description=(
            "Inhaltlicher Dokumenttyp:\n"
            "- research_paper: Wissenschaftliche Publikation oder Studie.\n"
            "- manual: Bedienungsanleitung, Nutzerhandbuch oder Leitfaden.\n"
            "- email: Elektronische Korrespondenz, E-Mail.\n"
            "- technical_report: Technischer Bericht oder Analyse\n"
            "- measuring_report: Ein Bericht über eine Messung oder Untersuchung"
            "- datasheet: Datenblatt mit Spezifikationen und Eigenschaften.\n"
            "- drawing: Technische Zeichnung, CAD-Datei oder Schema.\n"
            "- specification: Technische Spezifikation, Anforderungskatalog.\n"
            "- standard: Norm oder Richtlinie (z. B. ISO, DIN).\n"
            "- meeting_minutes: Protokoll von Besprechungen.\n"
            "- presentation: Präsentation, Folien, Slides.\n"
            "- safety_document: Sicherheitsdokument, Gefährdungsbeurteilung.\n"
            "- maintenance_log: Wartungs- oder Instandhaltungsprotokoll.\n"
            "- contract: Vertragliche Vereinbarung oder rechtliches Dokument.\n"
            "- website: Inhalte von Webseiten, Webdokumente.\n"
            "- other: Alle sonstigen Dokumenttypen."
        )
    )
    title: str = Field(
        "",
        description="Titel des Dokuments."
    )

    language: str = Field(
        "",
        description="Sprache des Dokuments (z. B. de, en)."
    )

    documentNumber: str=Field("", description="Eine Dokumentnummer, zb TES9134 oder 25D001 oder RA94557")

    authors: List[str] = Field(
        default_factory=list,
        description="Liste der am Dokument beteiligten Autoren."
    )

    abstract: str = Field(
        "",
        description="Kurze Zusammenfassung des Dokuments."
    )

    keywords: List[str] = Field(
        default_factory=list,
        description="Schlüsselwörter für die semantische Einordnung."
    )

    sender: str = Field(
        "",
        description="Absender (nur relevant für E-Mails oder Schreiben)."
    )

    recipient: List[str] = Field(
        default_factory=list,
        description="Empfänger (nur relevant für E-Mails oder Schreiben)."
    )


    huba_document: bool = Field(
        False,
        description="Markiert Dokumente, die in die Huba-Kategorie fallen."
    )

    linked_Documents: List[str] = Field(
        default_factory=list,
        description="IDs oder Pfade verlinkter Dokumente."
    )



    product_code: str = Field(
        "",
        description="Produktcode zur Zuordnung zu Bauteilen oder Geräten."
    )
    internalProductCode: str = Field(
        "",
        description="Firmeninterne Produktnummer, bestehend aus 3 Ziffern. Bsp. 505, 211, 450 etc. "
    )

    art_no: str = Field(
        "",
        description="Artikelnummer des Produkts."
    )

    body: str = Field(
        "",
        description="Der Dokument Inhalt"
    )

    instigationid: int = Field(
        0,
        description="Bezug zu Instigation/Projekt/Testreihe."
    )



@dataclass
class Document:
    id: str
    title: str
    location: str
    tags: List[str]
    created: str
    content: str
    meta: Optional[MetaData] = None
    chunks: Optional[List[str]] = None

class EvalResult(BaseModel):
    RetrievalScore: int
    AnswerScore: int
    UsefulnessScore: int
    Comment: str

class Prompt:
    def __init__(
        self,
        sys_prompt: str = "",
        user_prompt: str = "",
        context: str = "",
        answer: str = "",
        history: str = "",
        ref_Answer: str = "",
        fewshot_prompts: list[tuple[str, str]] = None,
        textFormat=None



    ):
        self.sys_prompt = sys_prompt
        self.user_prompt = user_prompt
        self.context = context
        self.answer = answer
        self.history = history
        self.ref_Answer = ref_Answer
        self.fewshot_prompts = fewshot_prompts or []
        self.textFormat = textFormat



    def to_dict(self):
        return {
            "sys_prompt": self.sys_prompt,
            "user_prompt": self.user_prompt,
            "context": self.context,
            "answer": self.answer,
            "history": self.history,
        }
    def toConversation(self):
        return[
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": "Userprompt:\n"+self.user_prompt+"\n\nContext:\n" + self.context},
            {"role": "assistant", "content":self.answer},
        ]




    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            sys_prompt=data.get("sys_prompt", ""),
            user_prompt=data.get("user_prompt", ""),
            context=data.get("context", ""),
            answer=data.get("answer", ""),
            history=data.get("history", "")
        )


import json
from typing import List

class PromptManager:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_prompts(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                prompts = [Prompt.from_dict(d) for d in data]
        except FileNotFoundError:
            prompts = []
        return prompts

    def add_prompt(self, prompt: "Prompt"):
        # Aktuelle Prompts laden
        prompts = self.load_prompts()

        # Neuen Prompt hinzufügen
        prompts.append(prompt)

        # In Datei speichern (überschreiben mit allen Prompts inkl. dem Neuen)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in prompts], f, ensure_ascii=False, indent=4)



class llm_gpt:
    def __init__(self):
        self.client = None
        self.model = None

    def connect(self, model: str, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def streamResponse(self, prompt: Prompt, textformat="plain"):
        """
        Generator: Gibt die Antwort stückweise (Delta für Delta) zurück.
        """
        stream = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt.user_prompt}],
            stream=True,
            text_format=textformat
            )

        for event in stream:
            if event.type == "response.output_text.delta":
                delta = event.delta
                yield delta


    async def getResponseStreamedAsync(self, prompt: Prompt, on_delta=None, textformat="plain") -> str:
        """
        Asynchronous: Streamt die Antwort, ruft für jedes Textdelta die Callback-Funktion `on_delta(delta)` auf.
        Rückgabe: gesamte Antwort als String.
        """
        stream = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt.user_prompt}],
            stream=True,
            text_format=textformat
        )

        full_text = ""
        async for event in stream:
            if event.type == "response.output_text.delta":
                delta = event.delta
                full_text += delta
                if on_delta:
                    on_delta(delta)

        return full_text



    def getResponseSimple(self, prompt: Prompt, textformat="plain") -> str:
        """
        Holt die komplette Antwort ohne Streaming.
        """
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system",
                 "content": prompt.sys_prompt},
                {"role": "user", "content": f"\n\n\nKontextinformationen:\n{prompt.context}\n\n\n"},
                {"role": "user", "content": prompt.user_prompt}
            ],
            stream=False,
        )



        return response.output_text

    def runQuery(self, prompt: Prompt) -> EvalResult:
        """
        Holt die Antwort als strukturiertes Objekt mit 3 Scores zurück.
        """
        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": prompt.sys_prompt},
                {"role": "user", "content": prompt.user_prompt}
            ],
            "stream": False
        }

        if prompt.textFormat is not None:
            kwargs["text_format"] = prompt.textFormat

        response = self.client.responses.parse(**kwargs)
        return response.output_parsed



    def getEvaluation(self, prompt: Prompt, textformat=EvalResult) -> EvalResult:
        """
        Holt die Antwort als strukturiertes Objekt mit 3 Scores zurück.
        """
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system",
                 "content": prompt.sys_prompt},
                {"role": "user", "content": prompt.user_prompt}
            ],
            stream=False,
            text_format=textformat
        )
        return response.output_parsed



