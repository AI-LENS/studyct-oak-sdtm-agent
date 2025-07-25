import asyncio
import json
from typing import TypedDict

import pandas as pd
from conflux import HandlerChain, Message, handler
from conflux.handlers import OpenAiLLM
from pydantic import BaseModel

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
    res = await chain.run(
        Message(colname), variables={"raw_data_points": raw_data_points}
    )
    return res.info["structure"]


df = pd.read_csv("./data/raw/VS.csv")

df["TEMP_VSLOC"] = df["TEMP_VSLOC"].map({"ORAL CAVITY": "Oral Cavity", "EAR": "ears"}) # change collected values intentionally
print(df["TEMP_VSLOC"].unique())

res: AllCodelistAssociation = asyncio.run(get_association(df, "TEMP_VSLOC"))
for assoc in res.associations:
    print(f"Raw Data Code: {assoc.raw_data_code}, Codelist Term: {assoc.codelist_term}")
