import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError


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
    start: str
    end: str
    task: str
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
    user_name: str
    provider: str
    model: str
    sleep_assessment: SleepAssessment
    environment: EnvironmentContext
    schedule: list[ScheduleItem]
    constraints: UserConstraints


def build_static_mock_input(provider: str, model: str) -> LLMInput:
    return LLMInput(
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
            ScheduleItem("09:00", "10:00", "Team Meeting", "medium", False),
            ScheduleItem("10:00", "12:00", "Deep Work Session", "high", True),
            ScheduleItem("12:00", "13:00", "Lunch Break", "low", False),
            ScheduleItem("13:00", "15:00", "Project A", "high", True),
            ScheduleItem("15:00", "16:00", "Project B", "medium", True),
            ScheduleItem("16:00", "17:00", "Wrap-up and Planning for Tomorrow", "low", True),
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
    module_path = Path(__file__).with_name("sensehat-behavior.py")
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
            item["start"],
            item["end"],
            item["task"],
            item["intensity"],
            item["movable"],
        )
        for item in sensehat_module.get_mock_schedule()
    ]

    return LLMInput(
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
        schedule_lines.append(
            f"- {item.start}-{item.end} | {item.task} | intensity={item.intensity} | {movable}"
        )

    template = load_template(template_path)
    return template.format(
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


def resolve_api_key(provider: str) -> str | None:
    if provider.lower() != "qwen":
        return None

    return (
        os.getenv("QWEN_API_KEY")
        or os.getenv("Qwen_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )


def call_qwen_chat(model: str, system_instruction: str, user_prompt: str) -> str:
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
                extra_body={"enable_thinking": True},
                stream=True,
            )
        except (APIConnectionError, OpenAIError) as exc:
            last_error = exc
            print(f"Qwen request attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            break

        answer_parts: list[str] = []
        reasoning_parts: list[str] = []

        try:
            for chunk in completion:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)

                if reasoning_content:
                    reasoning_parts.append(reasoning_content)
                if content:
                    answer_parts.append(content)
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

        if answer_parts:
            return "".join(answer_parts).strip()

        if reasoning_parts:
            return "".join(reasoning_parts).strip()

        raise RuntimeError(f"Qwen returned an empty streamed response for model '{model}'.")

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
        print(call_qwen_chat(model=DEFAULT_MODEL, system_instruction=system_instruction, user_prompt=user_prompt))
    except RuntimeError as exc:
        print(f"Qwen request failed: {exc}")
        return 1
    return 0

def mock_pipeline():
    # Mock pipeline for exporting to calendar-agent without running the full LLM interaction.
    llm_input = build_static_mock_input(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)
    system_instruction = build_system_instruction(SYSTEM_TEMPLATE_PATH)
    user_prompt = build_user_prompt(llm_input, USER_TEMPLATE_PATH)
    return system_instruction, user_prompt

if __name__ == "__main__":
    raise SystemExit(main())
