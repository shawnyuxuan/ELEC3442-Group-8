import importlib.util
import datetime
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from src.llm_contract import ScheduleRecommendation


load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen3.6-plus")
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_TEMPLATE_PATH = PROMPTS_DIR / "system_instruction.txt"
USER_TEMPLATE_PATH = PROMPTS_DIR / "user_prompt_template.txt"
SLEEP_MODEL_PATH = os.getenv("SLEEP_MODEL_PATH")
PRINT_STRUCTURED_INPUT = True
INVOKE_LLM = True


# These data classes define the shared structure passed into the prompt layer.
# The goal is to keep the LLM input explicit instead of passing around loose dicts.
@dataclass
class EnvironmentContext:
    temperature_c: float
    humidity_percent: float
    pressure_hpa: float


@dataclass
class ScheduleItem:
    event_id: str | None
    start: str
    end: str
    task: str
    description: str
    intensity: str
    movable: bool


@dataclass
class SleepAssessment:
    cluster_id: int
    cluster_label: str
    summary: str
    likely_risks: list[str]
    recommended_work_style: str


@dataclass
class UserConstraints:
    preserve_fixed_events: bool
    avoid_medical_claims: bool
    max_schedule_changes: int
    preferred_output_language: str


@dataclass
class LLMInput:
    schedule_date: str
    user_name: str
    provider: str
    model: str
    sleep_assessment: SleepAssessment
    environment: EnvironmentContext
    schedule: list[ScheduleItem]
    constraints: UserConstraints


def build_static_mock_input(provider: str, model: str) -> LLMInput:
    return LLMInput(
        schedule_date=datetime.date.today().isoformat(),
        user_name="Jayden",
        provider=provider,
        model=model,
        sleep_assessment=SleepAssessment(
            cluster_id=1,
            cluster_label="Brain Fog Risk",
            summary=(
                "The user likely slept late and had a shorter, less restorative night. "
                "Morning cognitive sharpness may be lower than usual."
            ),
            likely_risks=[
                "slower task switching in the morning",
                "reduced focus on deep work before noon",
                "higher mental fatigue during long meetings",
            ],
            recommended_work_style=(
                "Start with light or administrative work, then shift deep work to the afternoon."
            ),
        ),
        environment=EnvironmentContext(
            temperature_c=20.0,
            humidity_percent=60.0,
            pressure_hpa=1013.2,
        ),
        schedule=[
            ScheduleItem(None, "09:00", "10:00", "Team Meeting", "Weekly team sync covering priorities, blockers, and coordination.", "medium", False),
            ScheduleItem(None, "10:00", "12:00", "Deep Work Session", "Focused individual work requiring sustained concentration and minimal interruptions.", "high", True),
            ScheduleItem(None, "12:00", "13:00", "Lunch Break", "Midday meal and recovery break.", "low", False),
            ScheduleItem(None, "13:00", "15:00", "Project A", "High-priority project execution block with deliverable progress expected.", "high", True),
            ScheduleItem(None, "15:00", "16:00", "Project B", "Medium-intensity project work such as follow-ups, implementation, or coordination.", "medium", True),
            ScheduleItem(None, "16:00", "17:00", "Wrap-up and Planning for Tomorrow", "Admin wrap-up, status notes, and planning for the next day.", "low", True),
        ],
        constraints=UserConstraints(
            preserve_fixed_events=True,
            avoid_medical_claims=True,
            max_schedule_changes=3,
            preferred_output_language="English",
        ),
    )


def load_sensehat_module():
    # sensehat-behavior.py owns the mock sleep input, model prediction, and cluster meaning.
    # Load it dynamically so the LLM layer can reuse that logic without duplicating it.
    module_path = Path(__file__).with_name("sensehat_behavior.py")
    spec = importlib.util.spec_from_file_location("sensehat_behavior", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load sensehat module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_input_from_sensehat_mock(provider: str, model: str, sleep_model_path: str | None) -> LLMInput:
    sensehat_module = load_sensehat_module()
    controller = sensehat_module.SenseHatController(model_path=sleep_model_path)
    sleep_data = controller.fetch_sleep_data()
    cluster = int(controller.predict_sleep_quality(sleep_data))
    cluster_profile = sensehat_module.get_cluster_profile(cluster)
    if cluster_profile is None:
        raise RuntimeError(f"Unsupported cluster profile: {cluster}")

    schedule = [
        ScheduleItem(
            item.get("event_id"),
            item["start"],
            item["end"],
            item["task"],
            item.get("description", ""),
            item["intensity"],
            item["movable"],
        )
        for item in sensehat_module.get_mock_schedule()
    ]

    return LLMInput(
        schedule_date=datetime.date.today().isoformat(),
        user_name="Jayden",
        provider=provider,
        model=model,
        sleep_assessment=SleepAssessment(
            cluster_id=cluster,
            cluster_label=cluster_profile["label"],
            summary=cluster_profile["summary"],
            likely_risks=list(cluster_profile["likely_risks"]),
            recommended_work_style=cluster_profile["recommended_work_style"],
        ),
        environment=EnvironmentContext(
            temperature_c=float(controller.sensor.get_temperature()),
            humidity_percent=float(controller.sensor.get_humidity()),
            pressure_hpa=float(controller.sensor.get_pressure()),
        ),
        schedule=schedule,
        constraints=UserConstraints(
            preserve_fixed_events=True,
            avoid_medical_claims=True,
            max_schedule_changes=3,
            preferred_output_language="English",
        ),
    )


def load_template(template_path: Path) -> str:
    with open(template_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_system_instruction(template_path: Path) -> str:
    return load_template(template_path)


def build_user_prompt(llm_input: LLMInput, template_path: Path) -> str:
    # The text template stays outside Python so prompt editing does not require code changes.
    schedule_lines = []
    for item in llm_input.schedule:
        movable = "movable" if item.movable else "fixed"
        event_id = "null" if item.event_id is None else json.dumps(item.event_id)
        schedule_lines.append(
            (
                f"- event_id={event_id} | {item.start}-{item.end} | {item.task} "
                f"| description={item.description} | intensity={item.intensity} | {movable}"
            )
        )

    template = load_template(template_path)
    return template.format(
        schedule_date=llm_input.schedule_date,
        user_name=llm_input.user_name,
        provider=llm_input.provider,
        model=llm_input.model,
        cluster_id=llm_input.sleep_assessment.cluster_id,
        cluster_label=llm_input.sleep_assessment.cluster_label,
        summary=llm_input.sleep_assessment.summary,
        likely_risks=", ".join(llm_input.sleep_assessment.likely_risks),
        recommended_work_style=llm_input.sleep_assessment.recommended_work_style,
        temperature_c=llm_input.environment.temperature_c,
        humidity_percent=llm_input.environment.humidity_percent,
        pressure_hpa=llm_input.environment.pressure_hpa,
        schedule_lines="\n".join(schedule_lines),
        preserve_fixed_events=llm_input.constraints.preserve_fixed_events,
        avoid_medical_claims=llm_input.constraints.avoid_medical_claims,
        max_schedule_changes=llm_input.constraints.max_schedule_changes,
        preferred_output_language=llm_input.constraints.preferred_output_language,
    )


def _validate_text_list(value: object, field_name: str):
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"Field '{field_name}' must be a list of strings.")


def _parse_hhmm(value: str, field_name: str) -> int:
    try:
        hours_text, minutes_text = value.split(":")
        hours = int(hours_text)
        minutes = int(minutes_text)
    except ValueError as exc:
        raise RuntimeError(f"Field '{field_name}' must use HH:MM format.") from exc

    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise RuntimeError(f"Field '{field_name}' must use a valid HH:MM time.")

    return hours * 60 + minutes


def _validate_event_object(event: object, field_name: str):
    if not isinstance(event, dict):
        raise RuntimeError(f"Field '{field_name}' must be an object.")

    required_fields = ["event_id", "title", "start", "end", "description"]
    for key in required_fields:
        if key not in event:
            raise RuntimeError(f"Field '{field_name}.{key}' is required.")

    if event["event_id"] is not None and not isinstance(event["event_id"], str):
        raise RuntimeError(f"Field '{field_name}.event_id' must be null or a string.")

    for key in ["title", "start", "end", "description"]:
        if not isinstance(event[key], str):
            raise RuntimeError(f"Field '{field_name}.{key}' must be a string.")


def validate_llm_output(payload: object) -> dict[str, object]:
    # Validate the JSON shape before calendar_interaction consumes it.
    if not isinstance(payload, dict):
        raise RuntimeError("LLM output must be a JSON object.")

    required_top_level = [
        "date",
        "daily_summary",
        "adjustment_principles",
        "calendar_operations",
        "optimized_schedule",
    ]
    for field_name in required_top_level:
        if field_name not in payload:
            raise RuntimeError(f"Missing required top-level field: {field_name}")

    if not isinstance(payload["date"], str):
        raise RuntimeError("Field 'date' must be a string.")
    if not isinstance(payload["daily_summary"], str):
        raise RuntimeError("Field 'daily_summary' must be a string.")

    _validate_text_list(payload["adjustment_principles"], "adjustment_principles")

    if not isinstance(payload["calendar_operations"], list):
        raise RuntimeError("Field 'calendar_operations' must be a list.")

    allowed_actions = {"move", "update_description"}
    for index, operation in enumerate(payload["calendar_operations"]):
        if not isinstance(operation, dict):
            raise RuntimeError(f"calendar_operations[{index}] must be an object.")
        for field_name in ["action", "target", "updated", "reason"]:
            if field_name not in operation:
                raise RuntimeError(f"calendar_operations[{index}].{field_name} is required.")
        if operation["action"] not in allowed_actions:
            raise RuntimeError(
                f"calendar_operations[{index}].action must be one of {sorted(allowed_actions)}."
            )
        _validate_event_object(operation["target"], f"calendar_operations[{index}].target")
        _validate_event_object(operation["updated"], f"calendar_operations[{index}].updated")
        if not isinstance(operation["reason"], str):
            raise RuntimeError(f"calendar_operations[{index}].reason must be a string.")

    if not isinstance(payload["optimized_schedule"], list):
        raise RuntimeError("Field 'optimized_schedule' must be a list.")

    for index, item in enumerate(payload["optimized_schedule"]):
        _validate_event_object(item, f"optimized_schedule[{index}]")
        for field_name in ["intensity", "movable"]:
            if field_name not in item:
                raise RuntimeError(f"optimized_schedule[{index}].{field_name} is required.")
        if item["intensity"] not in {"low", "medium", "high"}:
            raise RuntimeError(
                f"optimized_schedule[{index}].intensity must be one of low, medium, high."
            )
        if not isinstance(item["movable"], bool):
            raise RuntimeError(f"optimized_schedule[{index}].movable must be a boolean.")

    return payload


def validate_schedule_consistency(payload: dict[str, object], llm_input: LLMInput):
    # Check that the final schedule can actually be applied back to the calendar safely.
    optimized_schedule = payload["optimized_schedule"]
    assert isinstance(optimized_schedule, list)

    original_schedule = llm_input.schedule
    if len(optimized_schedule) != len(original_schedule):
        raise RuntimeError("optimized_schedule must contain every original event exactly once.")

    original_start_times = [_parse_hhmm(item.start, f"original_schedule[{index}].start") for index, item in enumerate(original_schedule)]
    original_end_times = [_parse_hhmm(item.end, f"original_schedule[{index}].end") for index, item in enumerate(original_schedule)]
    latest_original_end = max(original_end_times)

    def signature_from_schedule_item(item: ScheduleItem):
        return (item.task, item.description, item.intensity, item.movable)

    def signature_from_output_item(item: dict[str, object], index: int):
        return (
            item["title"],
            item["description"],
            item["intensity"],
            item["movable"],
        )

    original_signatures = sorted(signature_from_schedule_item(item) for item in original_schedule)
    output_signatures = sorted(signature_from_output_item(item, index) for index, item in enumerate(optimized_schedule))
    if original_signatures != output_signatures:
        raise RuntimeError("optimized_schedule must contain the same events as the original schedule.")

    previous_end = None
    for index, item in enumerate(optimized_schedule):
        assert isinstance(item, dict)
        start_minutes = _parse_hhmm(item["start"], f"optimized_schedule[{index}].start")
        end_minutes = _parse_hhmm(item["end"], f"optimized_schedule[{index}].end")
        if end_minutes <= start_minutes:
            raise RuntimeError(f"optimized_schedule[{index}] must end after it starts.")
        if previous_end is not None and start_minutes < previous_end:
            raise RuntimeError("optimized_schedule must not contain overlapping events.")
        if end_minutes > latest_original_end:
            raise RuntimeError("optimized_schedule must not extend beyond the original latest end time.")
        previous_end = end_minutes

    original_fixed = {
        (item.task, item.description): (item.start, item.end)
        for item in original_schedule
        if not item.movable
    }
    for index, item in enumerate(optimized_schedule):
        assert isinstance(item, dict)
        key = (item["title"], item["description"])
        if key in original_fixed:
            original_start, original_end = original_fixed[key]
            if item["start"] != original_start or item["end"] != original_end:
                raise RuntimeError(
                    f"optimized_schedule[{index}] changes a fixed event, which is not allowed."
                )


def resolve_api_key(provider: str) -> str | None:
    if provider.lower() != "qwen":
        return None

    return (
        os.getenv("QWEN_API_KEY")
        or os.getenv("Qwen_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )


def call_qwen_chat(
    model: str,
    system_instruction: str,
    user_prompt: str,
    llm_input: LLMInput,
) -> ScheduleRecommendation:
    api_key = resolve_api_key("qwen")
    if not api_key:
        raise RuntimeError(
            "Qwen API key not found. Set one of QWEN_API_KEY, Qwen_API_KEY, or DASHSCOPE_API_KEY."
        )

    base_url = os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).rstrip("/")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    max_attempts = 3
    retry_delay_seconds = 2
    last_error: Exception | None = None

    print(f"Qwen base URL: {base_url}")
    print(f"Qwen model: {model}")

    # Retry a few times because the DashScope-compatible endpoint has been intermittently closing connections.
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
        except (APIConnectionError, OpenAIError) as exc:
            last_error = exc
            print(f"Qwen request attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            break

        try:
            message_content = completion.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise RuntimeError("Qwen returned an unexpected completion payload.") from exc
        except APIStatusError as exc:
            body_preview = str(exc.body)[:500] if getattr(exc, "body", None) else "<empty>"
            message = (
                f"Qwen returned HTTP {exc.status_code} from base URL {base_url} "
                f"for model '{model}'. Response body: {body_preview or '<empty>'}"
            )
            if exc.status_code < 500 or attempt == max_attempts:
                raise RuntimeError(message)

            print(f"{message}. Retrying...")
            time.sleep(retry_delay_seconds)
            continue
        except OpenAIError as exc:
            last_error = exc
            print(f"Qwen stream attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            break

        if not isinstance(message_content, str) or not message_content.strip():
            last_error = RuntimeError(f"Qwen returned an empty JSON response for model '{model}'.")
            print(f"Qwen response attempt {attempt}/{max_attempts} failed: {last_error}")
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            break

        try:
            parsed = json.loads(message_content)
        except json.JSONDecodeError as exc:
            last_error = RuntimeError(f"Qwen returned invalid JSON: {message_content}")
            print(f"Qwen response attempt {attempt}/{max_attempts} failed: {last_error}")
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            raise last_error from exc

        try:
            validated = validate_llm_output(parsed)
            validate_schedule_consistency(validated, llm_input)
        except RuntimeError as exc:
            last_error = exc
            print(f"Qwen response attempt {attempt}/{max_attempts} failed validation: {exc}")
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            break

        return ScheduleRecommendation.from_dict(validated)

    raise RuntimeError(
        f"Qwen request failed after {max_attempts} attempts to base URL {base_url} for model '{model}': {last_error}"
    ) from last_error


def main() -> int:
    try:
        llm_input = build_input_from_sensehat_mock(
            provider=DEFAULT_PROVIDER,
            model=DEFAULT_MODEL,
            sleep_model_path=SLEEP_MODEL_PATH,
        )
    except Exception as exc:
        print(f"Warning: failed to build input from sensehat mock pipeline ({exc}). Falling back to static mock input.")
        llm_input = build_static_mock_input(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)

    system_instruction = build_system_instruction(SYSTEM_TEMPLATE_PATH)
    user_prompt = build_user_prompt(llm_input, USER_TEMPLATE_PATH)

    if PRINT_STRUCTURED_INPUT:
        print("Structured mock input:")
        print(json.dumps(asdict(llm_input), indent=2, ensure_ascii=False))
        print()

    print("System instruction:")
    print(system_instruction)
    print()
    print("User prompt:")
    print(user_prompt)
    print()

    if not INVOKE_LLM:
        print("LLM API calling is disabled. Set INVOKE_LLM = True to send the current mock input.")
        return 0

    if DEFAULT_PROVIDER.lower() != "qwen":
        raise RuntimeError(f"Provider '{DEFAULT_PROVIDER}' is not implemented yet.")

    print("LLM response:")
    try:
        llm_output = call_qwen_chat(
            model=llm_input.model,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            llm_input=llm_input,
        )
        print(json.dumps(llm_output.to_dict(), indent=2, ensure_ascii=False))
    except RuntimeError as exc:
        print(f"Qwen request failed: {exc}")
        return 1
    return 0

def generate_schedule_recommendation(llm_input: LLMInput | None = None) -> ScheduleRecommendation:
    if llm_input is None:
        try:
            llm_input = build_input_from_sensehat_mock(
                provider=DEFAULT_PROVIDER,
                model=DEFAULT_MODEL,
                sleep_model_path=SLEEP_MODEL_PATH,
            )
        except Exception:
            llm_input = build_static_mock_input(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)

    system_instruction = build_system_instruction(SYSTEM_TEMPLATE_PATH)
    user_prompt = build_user_prompt(llm_input, USER_TEMPLATE_PATH)
    return call_qwen_chat(
        model=llm_input.model,
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        llm_input=llm_input,
    )


def mock_pipeline():
    # Mock pipeline for exporting to calendar-agent without running the full LLM interaction.
    llm_input = build_static_mock_input(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)
    system_instruction = build_system_instruction(SYSTEM_TEMPLATE_PATH)
    user_prompt = build_user_prompt(llm_input, USER_TEMPLATE_PATH)
    return system_instruction, user_prompt

if __name__ == "__main__":
    raise SystemExit(main())
