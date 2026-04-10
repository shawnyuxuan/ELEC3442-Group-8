import argparse
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
import requests


load_dotenv()

DEFAULT_PROVIDER = "qwen"
DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_SYSTEM_TEMPLATE_PATH = PROMPTS_DIR / "system_instruction.txt"
DEFAULT_USER_TEMPLATE_PATH = PROMPTS_DIR / "user_prompt_template.txt"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare structured mock input and prompt for schedule optimization."
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER),
        help=f"LLM provider label. Default: {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        help=f"Model name label. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--sleep-model-path",
        default=os.getenv("SLEEP_MODEL_PATH"),
        help="Path to the local sleep clustering model used by sensehat-behavior.py.",
    )
    parser.add_argument(
        "--system-template",
        default=str(DEFAULT_SYSTEM_TEMPLATE_PATH),
        help="Path to the system instruction template text file.",
    )
    parser.add_argument(
        "--user-template",
        default=str(DEFAULT_USER_TEMPLATE_PATH),
        help="Path to the user prompt template text file.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the structured mock input as JSON before the rendered prompt.",
    )
    parser.add_argument(
        "--invoke",
        action="store_true",
        help="Actually call the configured LLM API instead of only printing the prepared prompt.",
    )
    return parser.parse_args()


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


def load_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_system_instruction(template_path: str) -> str:
    return load_template(template_path)


def build_user_prompt(llm_input: LLMInput, template_path: str) -> str:
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
    endpoint = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Qwen response format: {body}") from exc


def main() -> int:
    args = parse_args()
    try:
        llm_input = build_input_from_sensehat_mock(
            provider=args.provider,
            model=args.model,
            sleep_model_path=args.sleep_model_path,
        )
    except Exception as exc:
        print(f"Warning: failed to build input from sensehat mock pipeline ({exc}). Falling back to static mock input.")
        llm_input = build_static_mock_input(provider=args.provider, model=args.model)
    system_instruction = build_system_instruction(args.system_template)
    user_prompt = build_user_prompt(llm_input, args.user_template)

    if args.print_json:
        print("Structured mock input:")
        print(json.dumps(asdict(llm_input), indent=2, ensure_ascii=False))
        print()

    print("System instruction:")
    print(system_instruction)
    print()
    print("User prompt:")
    print(user_prompt)
    print()

    if not args.invoke:
        print("LLM API calling is disabled. Use --invoke to send this mock input to the configured provider.")
        return 0

    if args.provider.lower() != "qwen":
        raise RuntimeError(f"Provider '{args.provider}' is not implemented yet.")

    print("LLM response:")
    print(call_qwen_chat(model=args.model, system_instruction=system_instruction, user_prompt=user_prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
