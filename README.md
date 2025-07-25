# SDTM Codelist Association Demo

This project demonstrates an automated approach to map raw clinical data values to their corresponding CDISC SDTM codelist terms using AI-powered analysis.

## Overview

The system performs a two-step AI-assisted mapping process:

1. **SDTM Variable Mapping**: Given a raw data column name, an LLM identifies the corresponding SDTM target variable from the available VS (Vital Signs) dataset variables.

2. **Codelist Association**: Using the identified SDTM variable's associated codelists from CDISC SDTMIG standards, another LLM maps raw data values to their appropriate standardized codelist terms.

## Key Components

- **CDISC Standards Integration**: Loads SDTM Implementation Guide (SDTMIG) and Controlled Terminology (SDTMCT) JSON files to access standardized variables and codelists
- **AI-Powered Mapping**: Uses OpenAI LLMs with structured output to perform intelligent mapping between raw and standardized data
- **Validation Testing**: Intentionally modifies raw values (e.g., "ORAL CAVITY" → "Oral Cavity", "EAR" → "ears") to demonstrate the system's ability to correctly associate non-exact matches with their proper CDISC codelist terms

## Workflow

Input: Raw clinical data DataFrame and column name →

1. LLM identifies corresponding SDTM variable →
2. Extract associated codelists from CDISC standards →
3. LLM maps raw values to codelist terms →
   Output: Structured associations between raw data codes and standardized terms

This approach enables automated standardization of clinical data according to CDISC guidelines, reducing manual mapping effort while maintaining compliance with regulatory standards.
