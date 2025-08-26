import asyncio
import json
import os
import re
import unicodedata
from typing import TypedDict

import pandas as pd
from conflux import HandlerChain, Message, handler
from conflux.handlers import OpenAiLLM
import requests
from pydantic import BaseModel

# Explicit OpenAI API key. Replace the placeholder with your key if you don't want to use env vars.
# WARNING: Avoid committing real keys to source control.
OPENAI_API_KEY = ""  # e.g., "sk-..."; leave empty to rely on environment
if OPENAI_API_KEY and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Local LLM (Ollama) configuration
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "0") == "1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

with open("./data/cdisc_json/sdtmig.json", "r") as file:
    sdtmig = json.load(file)

with open("./data/cdisc_json/sdtmct.json", "r") as file:
    sdtmct = json.load(file)

# codelists = sdtmct["codelists"]

sdtm_categories = sdtmig["classes"]


class Term(TypedDict):
    name: str
    code: str


class Codelist(TypedDict):
    name: str
    code: str
    terms: dict[str, Term]


codelists: dict[str, Codelist] = {}
for codelist in sdtmct["codelists"]:
    clist = Codelist(
        name=codelist["submissionValue"],
        code=codelist["conceptId"],
        terms={},
    )
    if "terms" not in codelist:
        continue
    terms = {}
    for term in codelist["terms"]:
        if "submissionValue" not in term or "conceptId" not in term:
            continue
        terms[term["conceptId"]] = Term(
            name=term["submissionValue"], code=term["conceptId"]
        )
    clist["terms"] = terms
    codelists[codelist["conceptId"]] = clist


class Variable(TypedDict):
    name: str
    role: str
    core: str
    codelists: list[str]


class Dataset(TypedDict):
    name: str
    variables: dict[str, Variable]


datasets: dict[str, Dataset] = {}

for category in sdtm_categories:
    if "datasets" not in category:
        continue
    cg_datasets = category["datasets"]
    for dataset in cg_datasets:
        ds = Dataset(name=dataset["name"], variables={})
        if "datasetVariables" not in dataset:
            continue
        dataset_variables = dataset["datasetVariables"]
        for variable in dataset_variables:
            if (
                "name" not in variable
                or "role" not in variable
                or "core" not in variable
            ):
                continue

            var = Variable(
                name=variable["name"],
                role=variable["role"],
                core=variable["core"],
                codelists=[],
            )
            var_codelists = variable.get("_links", {}).get("codelist", [])
            if var_codelists:
                for clist in var_codelists:
                    var["codelists"].append(
                        clist["href"].split("/")[-1]
                    )  # Extract the codelist name from the URL

            ds["variables"][var["name"]] = var
        datasets[ds["name"]] = ds


class CodelistAssociation(BaseModel):
    raw_data_code: str
    codelist_term: str


class AllCodelistAssociation(BaseModel):
    associations: list[CodelistAssociation]


class SDTMTargetVariable(BaseModel):
    target_variable: str


def _normalize_text(value: str) -> str:
    """Lowercase and remove spaces/underscores/hyphens and punctuation for robust matching."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    # Drop any non a-z or 0-9 characters (including spaces, underscores, hyphens, punctuation)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


##
## Deterministic pre-match helper (kept separate from the chain)
##
def _prematch_against_terms(raw_values: list[str], terms: list[str]) -> tuple[list[CodelistAssociation], list[str]]:
    term_set_exact = set(terms)
    term_map_lower: dict[str, str] = {}
    term_map_normalized: dict[str, str] = {}

    for term in terms:
        lower = term.lower()
        norm = _normalize_text(term)
        if lower not in term_map_lower:
            term_map_lower[lower] = term
        if norm not in term_map_normalized:
            term_map_normalized[norm] = term

    prematched: list[CodelistAssociation] = []
    unmatched: list[str] = []

    for raw in raw_values:
        if raw is None:
            continue
        raw_str = str(raw)
        if raw_str in term_set_exact:
            prematched.append(CodelistAssociation(raw_data_code=raw_str, codelist_term=raw_str))
            continue
        lower = raw_str.lower()
        if lower in term_map_lower:
            prematched.append(CodelistAssociation(raw_data_code=raw_str, codelist_term=term_map_lower[lower]))
            continue
        norm = _normalize_text(raw_str)
        if norm in term_map_normalized:
            prematched.append(CodelistAssociation(raw_data_code=raw_str, codelist_term=term_map_normalized[norm]))
            continue
        unmatched.append(raw_str)

    return prematched, unmatched


def _extract_json_object(text: str) -> dict:
    """Extract a JSON object from model output, tolerant to extra text/markdown fences."""
    if not text:
        return {}
    # Try fenced block first
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Fallback: take substring from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _ollama_generate(prompt: str) -> str:
    """Call Ollama's generate API and return the response text."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        print(f"[Local LLM] Error calling Ollama: {e}")
        return ""


def build_target_variable_prompt(raw_column_name: str) -> str:
    """Construct the same prompt as get_sdtm_target_variables without using the handler wrapper."""
    vs_columns = [var for var in datasets["VS"]["variables"].keys()]
    prompt = (
        "Your task is to find the SDTM target variables for the given column in raw clinical data.\n\n"
        f"Raw data column name: {raw_column_name}\n"
        "Here are the SDTM target variables you need to consider:\n"
        f"{', '.join(vs_columns)}\n\n"
        "Respond with a structured json output in the following format:\n````json\n{\n"
        "    \"target_variable\": \"<sdtm_target_variable_your_answer>\"\n}\n````\n"
    )
    return prompt


@handler
async def get_sdtm_target_variables(msg: Message, csd: HandlerChain) -> str:
    vs_columns = [var for var in datasets["VS"]["variables"].keys()]

    prompt = """Your task is to find the SDTM target variables for the given column in raw clinical data.

Raw data column name: {raw_data_column}
Here are the SDTM target variables you need to consider:
{target_variables}

Respond with a structured json output in the following format:
```json
{{
    "target_variable": "<sdtm_target_variable_your_answer>"
}}
""".format(
        raw_data_column=msg,
        target_variables=", ".join(vs_columns),
    )
    return prompt


@handler
async def get_sdtm_target_codelists(
    msg: Message, csd: HandlerChain
) -> Message:
    try:
        sdtm_target: SDTMTargetVariable = msg.info["structure"]
        print(f"{sdtm_target=}")
        target_variable = sdtm_target.target_variable
    except (KeyError, TypeError):
        # Fallback for when structure is not available
        print("No structure found in message, using default target variable")
        target_variable = "VSLOC"  # Default fallback
    
    try:
        sdtm_col_codelist = datasets["VS"]["variables"][target_variable]["codelists"]
        codelist_terms = [
            c["name"] for c in codelists[sdtm_col_codelist[0]]["terms"].values()
        ]
    except (KeyError, IndexError):
        # Fallback to all VS terms if specific target fails
        print(f"Failed to get codelist for {target_variable}, using all VS terms")
        vs_codelist_ids: set[str] = set()
        for var in datasets["VS"]["variables"].values():
            for cid in var["codelists"]:
                vs_codelist_ids.add(cid)
        codelist_terms: list[str] = []
        for cid in vs_codelist_ids:
            cl = codelists.get(cid)
            if not cl:
                continue
            codelist_terms.extend([t["name"] for t in cl["terms"].values()])
    
    csd.variables["codelist_terms"] = codelist_terms
    return Message("", info={"codelist_terms": codelist_terms})


@handler
async def get_codelist_terms(msg: Message, csd: HandlerChain) -> str:
    prompt = """Your task is to find the association between the raw clinical data points to their corresponding codelist terms according to CDISC standard.

Here are the raw data points you are assigned to find the association for:
{raw_data_points}

Here are the codelist terms you need to consider:
{codelist_terms}

Important constraints:
- Return exactly one association per raw data point; do not add extra associations.
- Keep the same order as the raw data points provided.
- Use only one of the provided codelist terms for each association. Use the term with exact spelling and case as provided.
- Do not invent new terms or synonyms.
- If no codelist term clearly applies, set "codelist_term" to "NO_MATCH" for that raw data point.
- The length of the "associations" array MUST equal the number of raw data points provided.

Respond with a structured json output in the following format:
```json
{{
    "associations": [
        {{
            "raw_data_code": "<raw_data_code_given>",
            "codelist_term": "<codelist_term_your_answer>"
        }},
        ...
    ]
}}
```
""".format(
        raw_data_points=", ".join(csd.variables["raw_data_points"]),
        codelist_terms="\n".join(csd.variables["codelist_terms"]),
    )
    return prompt


@handler
async def local_llm_target_variable(msg: Message, csd: HandlerChain) -> Message:
    # Accept prompt from previous handler and call local LLM
    prompt = str(msg)
    text = _ollama_generate(prompt)
    data = _extract_json_object(text)
    try:
        model = SDTMTargetVariable.model_validate(data)
    except Exception:
        # try to find key casing variations
        tv = data.get("target_variable") or data.get("TargetVariable") or ""
        model = SDTMTargetVariable(target_variable=tv)
    return Message("", info={"structure": model})


@handler
async def local_llm_associations(msg: Message, csd: HandlerChain) -> Message:
    prompt = str(msg)
    text = _ollama_generate(prompt)
    data = _extract_json_object(text)
    items = data.get("associations", []) if isinstance(data, dict) else []
    assocs: list[CodelistAssociation] = []
    for it in items:
        try:
            assocs.append(CodelistAssociation(raw_data_code=str(it.get("raw_data_code", "")),
                                              codelist_term=str(it.get("codelist_term", ""))))
        except Exception:
            pass
    return Message("", info={"structure": AllCodelistAssociation(associations=assocs)})


# Use the appropriate chain based on the LLM choice
if USE_LOCAL_LLM:
    chain = (
        get_sdtm_target_variables
        >> local_llm_target_variable
        >> get_sdtm_target_codelists
        >> get_codelist_terms
        >> local_llm_associations
    )
else:
    chain = (
        get_sdtm_target_variables
        >> OpenAiLLM(structure=SDTMTargetVariable)
        >> get_sdtm_target_codelists
        >> get_codelist_terms
        >> OpenAiLLM(structure=AllCodelistAssociation)
    )


async def get_association(raw_data: pd.DataFrame, colname: str):
    raw_data_points = list(raw_data[colname].dropna().unique())

    # Gather all VS codelist terms for deterministic pre-match (domain fixed to VS per current code)
    vs_codelist_ids: set[str] = set()
    for var in datasets["VS"]["variables"].values():
        for cid in var["codelists"]:
            vs_codelist_ids.add(cid)
    all_vs_terms: list[str] = []
    for cid in vs_codelist_ids:
        cl = codelists.get(cid)
        if not cl:
            continue
        all_vs_terms.extend([t["name"] for t in cl["terms"].values()])

    prematched, unmatched = _prematch_against_terms(raw_data_points, all_vs_terms)

    # If everything matched deterministically, skip LLM
    if not unmatched:
        return AllCodelistAssociation(associations=prematched), None

    # If using local LLM, bypass chain and call Ollama directly for both steps
    if USE_LOCAL_LLM:
        # Target variable via local model
        tv_prompt = build_target_variable_prompt(colname)
        tv_resp = _ollama_generate(tv_prompt)
        tv_data = _extract_json_object(tv_resp)
        try:
            sdtm_target_variable = SDTMTargetVariable.model_validate(tv_data).target_variable
        except Exception:
            sdtm_target_variable = tv_data.get("target_variable", "")

        # Build codelist terms for that target (fallback to all VS terms if missing)
        try:
            sdtm_col_codelist = datasets["VS"]["variables"][sdtm_target_variable]["codelists"]
            codelist_terms = [c["name"] for c in codelists[sdtm_col_codelist[0]]["terms"].values()]
        except Exception:
            codelist_terms = all_vs_terms

        # Association prompt via local model
        assoc_prompt = (
            "Your task is to find the association between the raw clinical data points to their corresponding codelist terms according to CDISC standard.\n\n"
            f"Here are the raw data points you are assigned to find the association for:\n{', '.join(unmatched)}\n\n"
            f"Here are the codelist terms you need to consider:\n{'\n'.join(codelist_terms)}\n\n"
            "Important constraints:\n"
            "- Return exactly one association per raw data point; do not add extra associations.\n"
            "- Keep the same order as the raw data points provided.\n"
            "- Use only one of the provided codelist terms for each association. Use the term with exact spelling and case as provided.\n"
            "- Do not invent new terms or synonyms. Match to the CLOSEST provided term only.\n"
            "- Treat raw values case-insensitively; ignore extra spaces.\n"
            "- If the raw value is plural and the corresponding singular exists in the codelist, choose the singular term (e.g., 'ears' -> 'Ear').\n"
            "- If no codelist term clearly applies, set \"codelist_term\" to \"NO_MATCH\" for that raw data point.\n"
            "- The length of the \"associations\" array MUST equal the number of raw data points provided.\n\n"
            "Example:\n"
            "Raw: ears\n"
            "Codelist contains: Ear, Nose, Throat\n"
            "Then map: ears -> Ear\n\n"
            "Respond with a structured json output in the following format:\n```json\n{\n    \"associations\": [\n        {\n            \"raw_data_code\": \"<raw_data_code_given>\",\n            \"codelist_term\": \"<codelist_term_your_answer>\"\n        },\n        ...\n    ]\n}\n```\n"
        )
        assoc_resp = _ollama_generate(assoc_prompt)
        assoc_data = _extract_json_object(assoc_resp)
        items = assoc_data.get("associations", []) if isinstance(assoc_data, dict) else []

        # Strict post-processing: only keep associations for the provided unmatched list
        # and only allow codelist terms from the provided set (case sensitive).
        valid_terms_set = set(codelist_terms)
        lower_term_map = {t.lower(): t for t in codelist_terms}
        assoc_map: dict[str, str] = {}
        for it in items:
            try:
                raw_code = str(it.get("raw_data_code", "")).strip()
                term = str(it.get("codelist_term", "")).strip()
            except Exception:
                continue
            if raw_code not in unmatched:
                continue
            # Coerce to a valid CT term
            if term in valid_terms_set:
                assoc_map[raw_code] = term
            elif term.lower() in lower_term_map:
                assoc_map[raw_code] = lower_term_map[term.lower()]
            else:
                assoc_map[raw_code] = "NO_MATCH"

        # Ensure one association per unmatched input, preserve order
        assocs: list[CodelistAssociation] = []
        for raw_code in unmatched:
            ct = assoc_map.get(raw_code, "NO_MATCH")
            assocs.append(CodelistAssociation(raw_data_code=raw_code, codelist_term=ct))

        llm_struct = AllCodelistAssociation(associations=assocs)

        merged = AllCodelistAssociation(associations=[*prematched, *llm_struct.associations])
        return merged, sdtm_target_variable

    # Otherwise use the cloud chain for unmatched values
    res = await chain.run(
        Message(colname), variables={"raw_data_points": unmatched}
    )
    llm_struct: AllCodelistAssociation = res.info["structure"]

    target_res = await (get_sdtm_target_variables >> OpenAiLLM(structure=SDTMTargetVariable)).run(Message(colname))
    sdtm_target_variable = target_res.info["structure"].target_variable

    merged = AllCodelistAssociation(associations=[*prematched, *llm_struct.associations])
    return merged, sdtm_target_variable


df = pd.read_csv("./data/raw/VS.csv")

df["TEMP_VSLOC"] = df["TEMP_VSLOC"].map({"ORAL CAVITY": "Oral Cavity", "EAR": "ears"}) # change collected values intentionally
print(df["TEMP_VSLOC"].unique())

res, sdtm_variable = asyncio.run(get_association(df, "TEMP_VSLOC"))
for assoc in res.associations:
    print(f"Raw Data Code: {assoc.raw_data_code}, Codelist Term: {assoc.codelist_term}")

# Create output CSV with collected values, codelist values, SDTM domain and variable
output_data = []
sdtm_domain = "VS"  # Fixed to VS domain as per current implementation
for assoc in res.associations:
    output_data.append({
        "collected_value": assoc.raw_data_code,
        "codelist_value": assoc.codelist_term,
        "sdtm_domain": sdtm_domain,
        "sdtm_variable": sdtm_variable or "UNKNOWN"
    })

output_df = pd.DataFrame(output_data)
output_df.to_csv("codelist_mapping_output.csv", index=False)
print(f"\nOutput saved to codelist_mapping_output.csv with {len(output_data)} mappings")
print(f"SDTM Domain: {sdtm_domain}, Variable: {sdtm_variable or 'UNKNOWN'}")
