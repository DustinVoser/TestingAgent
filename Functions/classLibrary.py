from dataclasses import dataclass, Field
from typing import Dict, Literal, List
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pathlib import Path
import json


from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, conint

from pydantic import BaseModel, Field, validator
from typing import List, Literal



class QAPair(BaseModel):
    question: str = Field(
        ...,
        description="Eine technische Frage basierend auf dem gegebenen Text."
    )
    answer: str = Field(
        ...,
        description="Eine sehr kurze und präzise Antwort auf die Frage."
    )
    difficulty: conint(ge=1, le=3) = Field(
        ...,
        description="Der Schwierigkeitsgrad (1 = leicht, 3 = schwierig)."
    )


class SilverStandard(BaseModel):
    """
    Structured output definition for generating standardized Q&A blocks.
    If there is not enough text to generate meaningful questions, set not_enough_text=True and return an empty list for items.
    Otherwise, return exactly 3 items.
    """

    not_enough_text: bool = Field(
        default=False,
        description="True, wenn der gegebene Text zu kurz/inhaltlich unzureichend ist; in diesem Fall items leer lassen."
    )

    items: List[QAPair] = Field(
        default_factory=list,
        description="Liste mit Frage-Antwort-Paaren. Normalfall: genau 3 Einträge. Bei not_enough_text: 0 Einträge."
    )

    @field_validator("items")
    def validate_items_length(cls, v, info):
        # When not_enough_text is true, items may be empty; otherwise require exactly 3
        data = info.data if hasattr(info, 'data') else {}
        not_enough = data.get('not_enough_text', False)
        if not not_enough and len(v) != 3:
            raise ValueError("Es müssen genau 3 Frage-Antwort-Paare enthalten sein, außer wenn not_enough_text=True.")
        if not_enough and len(v) != 0 and len(v) != 3:
            # be lenient: allow 0 or 3 when flag is present
            raise ValueError("Bei not_enough_text muss items leer sein.")
        return v


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
    RetrievalScore: conint(ge=1, le=5)
    AnswerScore: conint(ge=1, le=5)
    UsefulnessScore: conint(ge=1, le=5)
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
        textFormat=None,
        difficulty: int = 0,
        instigationids: List[int] = [],
        datasheets: List[str] = [],
        glossar: List[str] = []

    ):
        self.sys_prompt = sys_prompt
        self.user_prompt = user_prompt
        self.context = context
        self.answer = answer
        self.history = history
        self.ref_Answer = ref_Answer
        self.fewshot_prompts = fewshot_prompts or []
        self.textFormat = textFormat
        self.difficulty = difficulty
        self.instigationids = instigationids
        self.datasheets = datasheets
        self.glossar = glossar




    def to_dict(self):
        # Ensure JSON-serializable values for complex fields
        # fewshot_prompts may contain tuples; convert to lists for JSON
        fewshots_serializable = [list(fp) for fp in (self.fewshot_prompts or [])]
        # textFormat may be a class or object; store its name/string representation
        if self.textFormat is None:
            text_format_serializable = None
        elif isinstance(self.textFormat, str):
            text_format_serializable = self.textFormat
        elif hasattr(self.textFormat, "__name__"):
            text_format_serializable = self.textFormat.__name__
        else:
            text_format_serializable = type(self.textFormat).__name__
        return {
            "sys_prompt": self.sys_prompt,
            "user_prompt": self.user_prompt,
            "context": self.context,
            "answer": self.answer,
            "history": self.history,
            "ref_Answer": self.ref_Answer,
            "fewshot_prompts": fewshots_serializable,
            "textFormat": text_format_serializable,
            "difficulty": self.difficulty,
            "instigationids": self.instigationids,
            "datasheets": self.datasheets or [],
            "glossar": self.glossar or []
        }
    def toConversation(self):
        return[
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": "Userprompt:\n"+self.user_prompt+"\n\nContext:\n" + self.context},
            {"role": "assistant", "content":self.answer},
        ]




    @classmethod
    def from_dict(cls, data: dict):
        few = data.get("fewshot_prompts") or []
        normalized_few: list[tuple[str, str]] = []
        for fp in few:
            if isinstance(fp, (list, tuple)) and len(fp) == 2:
                normalized_few.append((fp[0], fp[1]))
        text_format = data.get("textFormat", None)
        diff = data.get("difficulty", 0)
        try:
            diff = int(diff)
        except (TypeError, ValueError):
            diff = 0
        return cls(
            sys_prompt=data.get("sys_prompt", ""),
            user_prompt=data.get("user_prompt", ""),
            context=data.get("context", ""),
            answer=data.get("answer", ""),
            history=data.get("history", ""),
            ref_Answer=data.get("ref_Answer", data.get("answer", "")),
            fewshot_prompts=normalized_few,
            textFormat=text_format,
            difficulty=diff,
            instigationids=data.get("instigationids", []),
            datasheets=data.get("datasheets", []),
            glossar=data.get("glossar", [])
        )


class PromptManager:
    def __init__(self, filepath: str):
        # Allow directory or file path; if directory, place default file inside
        p = Path(filepath)
        if p.exists() and p.is_dir():
            p = p / "prompts.json"
        elif p.suffix == "":
            # If no extension is given, assume JSON
            p = p.with_suffix(".json")
        self.filepath = str(p)

    def load_prompts(self):
        try:
            p = Path(self.filepath)
            if not p.exists():
                return []
            # Handle empty files gracefully
            if p.stat().st_size == 0:
                return []
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return [Prompt.from_dict(d) for d in (data or [])]
        except json.JSONDecodeError:
            # Invalid/partial JSON — treat as empty to recover
            return []
        except FileNotFoundError:
            return []

    def add_prompt(self, prompt: "Prompt"):
        # Aktuelle Prompts laden
        prompts = self.load_prompts()

        # Neuen Prompt hinzufügen
        prompts.append(prompt)

        # In Datei speichern (überschreiben mit allen Prompts inkl. dem Neuen)
        p = Path(self.filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump([p_.to_dict() for p_ in prompts], f, ensure_ascii=False, indent=4)



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


