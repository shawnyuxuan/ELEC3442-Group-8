# ELEC3442 - Adaptive Schedule Design

## Overview
This project explores an adaptive scheduling workflow based on sleep data.

The current prototype does the following:
1. Extract sleep records from Apple Health export data and group them into sleep sessions.
2. Train a local K-means model on sleep-session features.
3. Use mock sleep input plus the trained local model to predict a daily sleep-state cluster.
4. Feed the predicted sleep-state context, environment, and daily schedule into an external LLM to generate schedule adjustment suggestions.

The main design goal is privacy: raw personal health data stays local, while the LLM only receives abstracted context such as cluster meaning, environment, and schedule.

## Current Workflow
1. Data pre-processing
   - `data-preprocessing.ipynb`: original exploratory notebook.
   - `src/data-preprocessing.py`: reusable script for extracting one user's sleep records from `export.xml`.
   - Output example: `data_jayden.xml` and `dates_jayden.json`.
2. Routine learning
   - `routine-learning.ipynb`: exploratory clustering notebook.
   - `src/routine-learning.py`: reusable training script.
   - Current default setup uses `3` clusters, not `5`.
   - Output example: `output/jayden_model.pkl`.
3. Edge-side behavior simulation
   - `src/sensehat-behavior.py` loads the local clustering model, predicts a cluster from mock sleep features, reads sensor values, and prepares cluster metadata plus a mock schedule.
4. LLM schedule optimization
   - `src/llm-interaction.py` reuses the mock inference path from `sensehat-behavior.py`, builds structured LLM input, renders external prompt templates, and can invoke Qwen through the DashScope-compatible API.
   - Prompt templates are stored in `prompts/system_instruction.txt` and `prompts/user_prompt_template.txt`.

## What Is Working Now
- Sleep-data extraction from Apple Health export
- Session grouping by day / weekday
- Local K-means model training
- Jayden-specific dataset export
- Jayden-specific local model export
- Mock sleep input -> local model cluster prediction
- Mock cluster/environment/schedule -> Qwen schedule suggestion
- Externalized prompt templates

## Repository Structure
- `src/data-preprocessing.py`: reusable preprocessing script for filtered user exports
- `src/routine-learning.py`: reusable model training script
- `src/sensehat-behavior.py`: mock Sense HAT side behavior and local cluster prediction
- `src/llm-interaction.py`: structured LLM input builder and Qwen invocation entrypoint
- `prompts/system_instruction.txt`: system prompt template
- `prompts/user_prompt_template.txt`: user prompt template
- `data-preprocessing.ipynb`: preprocessing notebook
- `routine-learning.ipynb`: clustering notebook
- `data.xml`, `dates.json`: original sample processed dataset
- `data_jayden.xml`, `dates_jayden.json`: Jayden-specific processed dataset
- `output/local_model.pkl`, `output/jayden_model.pkl`: trained local models

## Quick Start
Create the local environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Prepare Jayden-specific processed sleep data:

```bash
.venv/bin/python src/data-preprocessing.py
```

Train the local model (uses `data.xml` and `dates.json` by default; edit the `data_path`, `dates_path`, and `output_path` constants at the top of `src/routine-learning.py` to point to a different dataset):

```bash
.venv/bin/python src/routine-learning.py
```

Run the local mock prediction path (auto-discovers the model from the `output/` directory):

```bash
.venv/bin/python src/sensehat-behavior.py
```

Inspect the LLM prompt and invoke Qwen (set `INVOKE_LLM = False` at the top of the script to skip the API call and only print the rendered prompt):

```bash
.venv/bin/python src/llm-interaction.py
```

## Environment Variables
Create a local `.env` file. Example fields:

```env
QWEN_API_KEY=your_qwen_api_key_here
LLM_PROVIDER=qwen
LLM_MODEL=qwen-max
```

`src/llm-interaction.py` currently reads:
- `QWEN_API_KEY`
- `Qwen_API_KEY`
- `DASHSCOPE_API_KEY`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `QWEN_BASE_URL` (optional)
- `SLEEP_MODEL_PATH` (optional)

## Current Limitations
- `sensehat-behavior.py` still uses mock sleep features for testing.
- The schedule is still mock data, not synced from a real calendar or TODO source.
- Sense HAT on non-Raspberry Pi devices falls back to emulator or dummy sensor values.
- Early historical sleep sessions with dominant `AsleepUnspecified` are excluded from model training.
- LLM output is prompt-constrained but not yet post-validated into a strict machine-readable schema.

## Next Steps
- Support multiple mock sleep scenarios for systematic testing
- Connect real schedule / calendar data
- Replace mock sleep input with real edge-side input when hardware integration is ready
- Add stronger output validation for LLM responses
- Refine cluster semantics and prompt quality further
