from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import PromptLibrary
from Datasources import ELabJobsDB, ChromaDB
from Functions.classLibrary import Prompt


@tool
def runQueryTool(prompt: str):
    """This tool can run a limited numer of queries against a measuring data database."""
    elab_DB = ELabJobsDB()
    result=elab_DB.runQuery(prompt)
    return result


@tool
def get_SQL_Texts_Tool(instigationids:list[int]):
    """
     Takes a list with unique instigation id's and queries all summarys from them. Returns a text, containing the texts of the given instigations.
     """
    instigationids = sorted(instigationids, reverse=True)[:12]

    return get_SQL_Texts(instigationids)



def get_SQL_Texts(instigationids:list[int]):


    elab=ELabJobsDB()

    import pandas as pd, numpy as np, warnings

    # --- Hilfsfunktionen ---
    def safe_str(v):
        return "" if v is None else str(v).strip()

    def format_value(v):
        if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d %H:%M")
        if v is None: return ""
        return str(v)

    def indent(text, level):
        return "\n".join(["  " * level + l for l in str(text).split("\n")])

    def parse_workpackages(wps, level=3):
        return "\n".join([indent(f"WorkPackage {format_value(wp.get('WorkPackageId'))}: "
                                 f"JobType {format_value(wp.get('WP_JobTypeid'))}, "
                                 f"Findings: {format_value(wp.get('WP_Findings'))}", level)
                          for wp in (wps or [])])

    def parse_performancetests(ptests, level=2):
        lines = []
        for t in (ptests or []):
            lines.append(
                indent(f"Performance Test: {safe_str(t.get('PT_Title'))} (Norm {format_value(t.get('PT_Normid'))})",
                       level))
            lines.append(indent(f"Description: {safe_str(t.get('PT_Description'))}", level + 1))
            lines.append(indent(f"Test Level: {safe_str(t.get('PT_TestLevel'))}", level + 1))
            lines.append(indent(f"Accuracy: {safe_str(t.get('PT_Accuracy'))}", level + 1))
            if t.get('WorkPackages'): lines.append(parse_workpackages(t['WorkPackages'], level + 1))
        return "\n".join(lines)

    def parse_summary(summaries, level=2):
        lines = []
        for s in (summaries or []):
            lines.append(indent(f"Summary ID {format_value(s.get('SummaryAndNotesId'))}: "
                                f"{safe_str(s.get('SN_PassFail'))} – {safe_str(s.get('SN_Type'))}", level))
            lines.append(indent(safe_str(s.get('SN_Summary')), level + 1))
            lines.append(indent(f"Report: {safe_str(s.get('SN_Report'))}", level + 1))
        return "\n".join(lines)

    def parse_purposeofuse(purposes, level=1):
        out = []
        for p in (purposes or []):
            out.append(indent(f"Purpose of Use: {safe_str(p.get('PurposeOfUse_Title'))} "
                              f"({safe_str(p.get('PurposeOfUse_Number'))})", level))
            out.append(indent(f"Description: {safe_str(p.get('PurposeOfUse_Description'))}", level + 1))
            out.append(indent(f"Due Date: {format_value(p.get('PurposeOfUse_DueDate'))}", level + 1))
            if p.get('PerformanceTests'): out.append(parse_performancetests(p['PerformanceTests'], level + 1))
            if p.get('SummaryAndNotes'): out.append(parse_summary(p['SummaryAndNotes'], level + 1))
        return "\n".join(out)

    def dataframe_to_text(df):
        text_blocks = []
        for _, row in df.iterrows():
            text = [
                f"Instigation {safe_str(row.get('I_Number'))} – {safe_str(row.get('I_Title'))}",
                f"TypeID: {safe_str(row.get('I_InstigationTypeid'))}, "
                f"Creator: {safe_str(row.get('User_FirstName'))}, {safe_str(row.get('User_LastName'))}",
                f"Due: {format_value(row.get('I_DueDate'))}, Classification: {safe_str(row.get('I_Classification'))}",
                f"Description: {safe_str(row.get('I_Description'))}\n"
            ]
            purposes_raw = row.get('PurposeOfUses')
            purposes = eval(purposes_raw) if isinstance(purposes_raw, str) else purposes_raw
            text.append(parse_purposeofuse(purposes))
            text_blocks.append("\n".join(text))
        return "\n\n" + ("-" * 60 + "\n\n").join(text_blocks)

    # --- Daten sammeln ---
    def clean_row(row, cols):
        return {c: (None if pd.isna(row.get(c)) else row.get(c)) for c in cols}

    inst_cols = ["I_Instigationid", "I_DateTimeStamp", "I_InstigationTypeid", "I_Number", "I_Title",
                 "I_Description", "User_FirstName", "User_LastName", "I_DueDate",
                 "I_CustomerReferenceNumber", "I_EndDate", "I_Classification"]
    pou_cols = ["PurposeOfUse_id", "PurposeOfUse_Instigationid", "PurposeOfUse_Title",
                "PurposeOfUse_Description", "PurposeOfUse_DueDate", "PurposeOfUse_Number"]
    pt_cols = ["PT_PerformanceTestid", "PT_Instigationid", "PT_Title", "PT_Normid", "PT_Description",
               "PT_TestLevel", "PT_EstimatedDuration", "PT_Accuracy", "PT_FunctionalStatus", "PT_Criteria",
               "PT_TestDescription", "jl_JobType"]
    san_cols = ["SummaryAndNotesId", "SN_Instigationid", "SN_PurposeOfUseid", "SN_Type", "SN_Summary",
                "SN_DateTimeStamp", "SN_Report", "SN_PassFail", "SN_Notes"]
    wp_cols = ["WorkPackageId", "WP_PurposeOfUseid", "WP_PerformanceTestid", "WP_JobTypeid",
               "WP_Findings", "WP_FunctionalState", "wp_JobType"]

    all_results = []

    if isinstance(instigationids, (np.float64, float, int)):
        instigationids = [instigationids]

    for id in instigationids:
        query=PromptLibrary.queryInstigationTexts.format(instigationid=id)
        df = elab.runQuery(query)
        if df is None or df.empty: continue

        for _, inst_row in df.groupby("I_Instigationid").first().iterrows():
            inst_dict = clean_row(inst_row, inst_cols)
            inst_dict["SummaryAndNotes"] = []
            inst_dict["PurposeOfUses"] = []

            sn_inst = df[df["SN_PurposeOfUseid"].isna()].drop_duplicates(subset=["SummaryAndNotesId"])
            inst_dict["SummaryAndNotes"] = [clean_row(r, san_cols) for _, r in sn_inst.iterrows() if
                                            pd.notna(r.get("SummaryAndNotesId"))]

            pou_rows = df[df["PurposeOfUse_id"].notna()]
            for _, pou_group in pou_rows.groupby("PurposeOfUse_id"):
                pou_row = pou_group.iloc[0]
                pou_dict = clean_row(pou_row, pou_cols)
                pou_dict["SummaryAndNotes"] = [clean_row(r, san_cols) for _, r in
                                               pou_group.drop_duplicates(subset=["SummaryAndNotesId"]).iterrows() if
                                               pd.notna(r.get("SummaryAndNotesId"))]

                pt_list = []
                for pt_id, pt_group in pou_group.groupby("PT_PerformanceTestid"):
                    if pd.isna(pt_id): continue
                    pt_row = pt_group.iloc[0]
                    pt_dict = clean_row(pt_row, pt_cols)
                    pt_dict["WorkPackages"] = [clean_row(wp, wp_cols) for _, wp in
                                               pt_group.drop_duplicates(subset=["WorkPackageId"]).iterrows() if
                                               pd.notna(wp.get("WorkPackageId"))]
                    pt_list.append(pt_dict)
                pou_dict["PerformanceTests"] = pt_list
                inst_dict["PurposeOfUses"].append(pou_dict)

            all_results.append(inst_dict)

    # --- Ergebnis als Text ---
    if not all_results: return {"instigation_texts": ""}
    result_df = pd.DataFrame(all_results)
    return {"instigation_texts":dataframe_to_text(result_df)}


@tool
def triggerDatasheetsRetrieval(keywords:list[str]):
    """ Runs a RAG"""

    vector = ChromaDB("normDB")
    results=vector.keywordsRetrieval(keywords, n_results=2)

    return results

@tool
def queryGlossar(prompt: str):
    """You can lookup a word or definition by passing a semantic close query"""
    vector = ChromaDB("glossar")
    p=Prompt()
    p.user_prompt=prompt
    results = vector.semanticRetrieval(p, n_results=3)
    return results


@tool
def queryDatasheets(productCodes: List[str]):
    """Takes a list of product codes (505,711, 200 etc...) and returns available datasheets."""
    vector = ChromaDB("datasheets")
    results = vector.keywordsRetrieval(productCodes, n_results=2)

    return results