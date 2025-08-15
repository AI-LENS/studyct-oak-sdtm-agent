import asyncio
import json
import os
import re
import unicodedata
from typing import TypedDict

import pandas as pd
from conflux import HandlerChain, Message, handler
from conflux.handlers import OpenAiLLM
from pydantic import BaseModel

# Explicit OpenAI API key. Replace the placeholder with your key if you don't want to use env vars.
# WARNING: Avoid committing real keys to source control.
OPENAI_API_KEY = ""  # e.g., "sk-..."; leave empty to rely on environment
if OPENAI_API_KEY and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
    sdtm_target: SDTMTargetVariable = msg.info["structure"]
    print(f"{sdtm_target=}")
    target_variable = sdtm_target.target_variable
    sdtm_col_codelist = datasets["VS"]["variables"][target_variable]["codelists"]
    # print(raw_collections)
    codelist_terms = [
        c["name"] for c in codelists[sdtm_col_codelist[0]]["terms"].values()
    ]
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

    # Run the existing chain only on unmatched values
    res = await chain.run(
        Message(colname), variables={"raw_data_points": unmatched}
    )
    llm_struct: AllCodelistAssociation = res.info["structure"]
    
    # Get the SDTM target variable from the chain result
    sdtm_target_variable = None
    # We need to run a separate call to get the target variable since we modified the flow
    target_res = await (get_sdtm_target_variables >> OpenAiLLM(structure=SDTMTargetVariable)).run(Message(colname))
    sdtm_target_variable = target_res.info["structure"].target_variable

    # Merge results (deterministic first)
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
