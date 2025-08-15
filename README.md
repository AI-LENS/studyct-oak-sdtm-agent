# Study CT creation

This project demonstrates an automated approach to map raw clinical data values to their corresponding CDISC SDTM codelist terms using AI-powered analysis required for sdtm.oak library in R.

## Overview

The system performs a two-step AI-assisted mapping process:

1. **SDTM Variable Mapping**: Given a raw data column name, an LLM identifies the corresponding SDTM target variable from the available VS (Vital Signs) dataset variables.

2. **Codelist Association**: Using the identified SDTM variable's associated codelists from CDISC SDTMIG standards, another LLM maps raw data values to their appropriate standardized codelist terms.

## Key Components

- **CDISC Standards Integration**: Loads SDTM Implementation Guide (SDTMIG) and Controlled Terminology (SDTMCT) JSON files to access standardized variables and codelists
- **AI-Powered Mapping**: Uses OpenAI LLMs with structured output to perform intelligent mapping between raw and standardized data
- **Validation Testing**: Intentionally modifies raw values (e.g., "ORAL CAVITY" → "Oral Cavity", "EAR" → "ears") to demonstrate the system's ability to correctly associate non-exact matches with their proper CDISC codelist terms

## Workflow

**Input**: Raw clinical data DataFrame and column name

### Step 1: SDTM Variable Identification

- The `get_sdtm_target_variables` handler receives the raw data column name (e.g., "TEMP_VSLOC")
- An OpenAI LLM analyzes the column name against available VS dataset variables and returns the most appropriate SDTM target variable

### Step 2: Codelist Extraction

- The `get_sdtm_target_codelists` handler looks up the identified SDTM variable's associated codelists from CDISC standards
- Extracts all available codelist terms for that specific variable and stores them for the next step

### Step 3: Value Association Mapping

- The `get_codelist_terms` handler receives raw data points and available codelist terms from previous steps
- Another OpenAI LLM intelligently maps each raw data value to its corresponding standardized codelist term (handles non-exact matches)

**Output**: Structured `AllCodelistAssociation` object containing mappings between raw data codes and their standardized CDISC codelist terms

### Implementation Details

The entire workflow is orchestrated using the Conflux framework's handler chain pattern:

```
chain = (
    get_sdtm_target_variables
    >> OpenAiLLM(structure=SDTMTargetVariable)
    >> get_sdtm_target_codelists
    >> get_codelist_terms
    >> OpenAiLLM(structure=AllCodelistAssociation)
)
```

This approach enables automated standardization of clinical data according to CDISC guidelines, reducing manual mapping effort while maintaining compliance with regulatory standards.
