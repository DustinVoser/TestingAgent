from typing import Literal, List
from pydantic import BaseModel, Field


# Define the type alias outside the model
KeywordType = Literal[
    "ProductCode",
    "Ordernumber",
    "Manufacturer",
    "Variant",
    "InstigationNumber",
    "ExternalNumber",
    "Standard",
    "DUT_Configuration",
    "Person",
    "ArtNumber_Internal",
    "ArtNumber_External",
    "SensorRange"
]

class KeywordItem(BaseModel):
    keyword: str = Field(description="Extracted Keyword")
    keyword_type: KeywordType = Field(description="Category of keyword")

class KeywordOutput(BaseModel):
    keywords: List[KeywordItem] = Field(description="List of extracted keywords with types")


class InstigationOutput(BaseModel):
    instigationids: List[int] = Field(
        description="List of the extracted, unique instigation.id values"
    )


class ContextFeedback(BaseModel):
    grade: Literal["relevant", "not relevant"] = Field(
        description="Device if the given context is relevant to answer the users prompt or not",
    )
    feedback: str = Field(
        description="If the data is not relevant, give a feedback why it is not relevant.",
    )





