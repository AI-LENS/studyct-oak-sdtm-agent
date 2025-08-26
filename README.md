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

**Output**: Structured `AllCodelistAssociation` object containing mappings between raw data codes and their standardized CDISC codelist terms. Additionally, the script writes a CSV `codelist_mapping_output.csv` in the project root with columns: `collected_value`, `codelist_value`, `sdtm_domain`, `sdtm_variable`.

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

## Setup

1. Install uv

```
pipx install uv
# or
python -m pip install uv
```

2. Configure API key (choose one)

- Option A (simple): edit `main.py` and set `OPENAI_API_KEY = "sk-xxx"` near the top.
- Option B (env var): set it in your shell before running:

```
export OPENAI_API_KEY=sk-xxx
```

3. Run

```
uv run main.py
```

### Models

This app supports both cloud (OpenAI) and local (Ollama) models.

## Local LLM (Ollama) Setup

The project can run fully locally using Ollama and an instruction-tuned model that fits on machines with ~8 GB RAM.

### macOS

1. Install Ollama (choose one):

   - Using Homebrew:

     ```bash
     brew install ollama
     ```

   - Using the official install script:

     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```

2. Start the Ollama server (if not auto-started):

   ```bash
   ollama serve
   ```

3. Pull the recommended model (fits ~8 GB RAM):

   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

### Windows

1. Install Ollama via the official Windows installer (from the Ollama website).

2. Ensure the Ollama service is running. If needed, start it from PowerShell:

   ```powershell
   Start-Process -FilePath "C:\\Program Files\\Ollama\\ollama.exe" -ArgumentList "serve"
   ```

3. Pull the recommended model:

   ```powershell
   ollama pull qwen2.5:3b-instruct
   ```

Notes:
- If you already run another model server on a different port, set `OLLAMA_BASE_URL` accordingly (default is `http://127.0.0.1:11434`).
- Qwen 2.5 3B Instruct is instruction-tuned and performs well for this task on CPU.

## Usage

### Cloud (OpenAI)

1. Set the OpenAI API key (either export it or set it in `main.py`):

   ```bash
   export OPENAI_API_KEY=sk-xxx
   ```

2. Run:

   ```bash
   uv run python -u main.py
   ```

### Local (Ollama)

Make sure the Ollama server is running and the model is pulled (see setup above).

- macOS / Linux:

  ```bash
  USE_LOCAL_LLM=1 OLLAMA_MODEL=qwen2.5:3b-instruct uv run python -u main.py
  ```

- Windows PowerShell:

  ```powershell
  $env:USE_LOCAL_LLM = "1"
  $env:OLLAMA_MODEL = "qwen2.5:3b-instruct"
  uv run python -u main.py
  ```

- Windows cmd:

  ```cmd
  set USE_LOCAL_LLM=1
  set OLLAMA_MODEL=qwen2.5:3b-instruct
  uv run python -u main.py
  ```

Optional environment variables:
- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)

## Outputs

- `codelist_mapping_output.csv`: final mapping with columns `collected_value`, `codelist_value`, `sdtm_domain`, `sdtm_variable`.

## Notes on Matching Behavior

- Prematching is strict and lightweight: direct match → case-insensitive → space-insensitive.
- Only unmatched values are sent to the LLM.
- Local LLM responses are post-processed to ensure one mapping per input and only valid CT terms are accepted.
