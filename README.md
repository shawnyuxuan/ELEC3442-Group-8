# ELEC3442 - Adaptive Schedule Design

## Overview
This project explores an adaptive scheduling workflow based on sleep data.

The current prototype does the following:
1. Extract sleep records from Apple Health export data and group them into sleep sessions.
2. Train a local K-means model on sleep-session features.
3. Receive a daily sleep report through a listener endpoint.
4. Use the trained local model to predict a daily sleep-state cluster.
5. Read the real daily calendar schedule, combine it with sleep-state context and environment data, and send structured context to an external LLM.
6. Convert the LLM response into structured calendar operations and apply them back to the user's calendar.

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
3. Daily report ingestion
   - `src/daily_analysis.py` exposes `/report`, validates incoming sleep payloads, stores them in `daily_data/sleep_data.csv`, and forwards them into the worker pipeline.
4. Edge-side behavior simulation
   - `src/sensehat_behavior.py` loads the local clustering model, predicts a cluster from sleep features, reads sensor values, and provides cluster metadata.
5. LLM schedule optimization
   - `src/llm_interaction.py` builds structured LLM input, renders prompt templates, invokes Qwen through the DashScope-compatible API, validates the JSON response, and converts it into a typed contract.
   - Prompt templates are stored in `prompts/system_instruction.txt` and `prompts/user_prompt_template.txt`.
   - Structured output is defined in `src/llm_contract.py`.
6. Calendar application
   - `src/calendar_interaction.py` reads and updates CalDAV calendar events.
   - `calendar-agent.py` wires the listener, local model inference, LLM generation, and calendar update into one pipeline.

## What Is Working Now
- Sleep-data extraction from Apple Health export
- Session grouping by day / weekday
- Local K-means model training
- Jayden-specific dataset export
- Jayden-specific local model export
- Mock sleep input -> local model cluster prediction
- Structured Qwen JSON output -> typed contract objects
- Real calendar read from iCloud CalDAV
- Real calendar update from structured LLM operations
- `POST /report` -> local model -> Qwen -> calendar update end-to-end pipeline
- Externalized prompt templates

## Repository Structure
- `src/data-preprocessing.py`: reusable preprocessing script for filtered user exports
- `src/routine-learning.py`: reusable model training script
- `src/daily_analysis.py`: `/report` listener for incoming daily sleep reports
- `src/sensehat_behavior.py`: Sense HAT side behavior and local cluster prediction
- `src/llm_interaction.py`: structured LLM input builder, JSON validation, and Qwen invocation entrypoint
- `src/llm_contract.py`: typed contract for structured LLM output
- `src/calendar_interaction.py`: CalDAV calendar read/write integration
- `src/schedule_rules.py`: event classification rules used when turning calendar events into schedule items
- `calendar-agent.py`: end-to-end pipeline entrypoint
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
.venv/bin/python src/sensehat_behavior.py
```

Inspect the LLM prompt and invoke Qwen (set `INVOKE_LLM = False` at the top of the script to skip the API call and only print the rendered prompt):

```bash
.venv/bin/python src/llm_interaction.py
```

Run the full pipeline:

```bash
.venv/bin/python calendar-agent.py
```

Send a test sleep report:

```bash
curl -X POST http://127.0.0.1:5888/report \
  -H "Content-Type: application/json" \
  -d '{"date":"2026-04-14","core":4.5,"deep":1.3,"rem":1.2}'
```

## Environment Variables
Create a local `.env` file. Example fields:

```env
QWEN_API_KEY=your_qwen_api_key_here
LLM_PROVIDER=qwen
LLM_MODEL=qwen3.6-plus
CALENDAR_NAME=HKU Schedule
```

`src/llm_interaction.py` currently reads:
- `QWEN_API_KEY`
- `Qwen_API_KEY`
- `DASHSCOPE_API_KEY`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `QWEN_BASE_URL` (optional)
- `SLEEP_MODEL_PATH` (optional)

The calendar pipeline currently reads:
- `CALENDAR_NAME`
- `CALDAV_USERNAME`
- `CALDAV_PASSWORD`
- `CALDAV_URL`

## Current Limitations
- `src/sensehat_behavior.py` still uses mock sleep features for testing.
- Sense HAT on non-Raspberry Pi devices falls back to emulator or dummy sensor values.
- Early historical sleep sessions with dominant `AsleepUnspecified` are excluded from model training.
- Calendar event intensity and movability are still inferred by simple keyword rules.
- CalDAV reads/writes can still intermittently time out, so the pipeline includes retry logic.

## Next Steps
- Support multiple mock sleep scenarios for systematic testing
- Replace mock sleep input with real edge-side input when hardware integration is ready
- Add richer calendar event classification rules
- Refine cluster semantics and prompt quality further
