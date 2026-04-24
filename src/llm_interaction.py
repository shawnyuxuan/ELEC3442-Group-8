import datetime
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from src.llm_contract import ScheduleRecommendation
from src.sensehat_behavior import SenseHatController, get_cluster_profile

load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen3.6-plus")
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_TEMPLATE_PATH = PROMPTS_DIR / "system_instruction.txt"
USER_TEMPLATE_PATH = PROMPTS_DIR / "user_prompt_template.txt"


# ============================================================================
# Data Structures: LLMInput represents the complete input to the LLM.
# ============================================================================

@dataclass
class EnvironmentContext:
    """Current environment conditions."""
    temperature_c: float
    humidity_percent: float
    pressure_hpa: float


@dataclass
class ScheduleItem:
    """A single calendar event."""
    event_id: str | None
    start: str
    end: str
    task: str
    description: str
    intensity: str
    movable: bool


@dataclass
class SleepAssessment:
    """Sleep quality analysis result."""
    cluster_id: int
    cluster_label: str
    summary: str
    likely_risks: list[str]
    recommended_work_style: str


@dataclass
class UserConstraints:
    """Constraints on schedule optimization."""
    preserve_fixed_events: bool
    avoid_medical_claims: bool
    max_schedule_changes: int
    preferred_output_language: str


@dataclass
class LLMInput:
    """Complete input passed to the LLM for schedule optimization."""
    schedule_date: str
    user_name: str
    provider: str
    model: str
    sleep_assessment: SleepAssessment
    environment: EnvironmentContext
    schedule: list[ScheduleItem]
    constraints: UserConstraints




# ============================================================================
# Template Loading
# ============================================================================

def load_template(template_path: Path) -> str:
    """Load template text from file."""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_system_instruction() -> str:
    """Load system instruction from template file."""
    return load_template(SYSTEM_TEMPLATE_PATH)


def build_user_prompt(llm_input: LLMInput) -> str:
    """Format user prompt from LLMInput using template."""
    # Convert schedule items to formatted lines
    schedule_lines = []
    for item in llm_input.schedule:
        movable_label = "movable" if item.movable else "fixed"
        event_id_str = "null" if item.event_id is None else json.dumps(item.event_id)
        schedule_lines.append(
            f"- event_id={event_id_str} | {item.start}-{item.end} | {item.task} "
            f"| description={item.description} | intensity={item.intensity} | {movable_label}"
        )

    # Build event reference table for LLM to use in calendar_operations
    event_reference_table = []
    for idx, item in enumerate(llm_input.schedule, 1):
        event_obj = {
            "event_id": item.event_id,
            "title": item.task,
            "start": item.start,
            "end": item.end,
            "description": item.description,
        }
        event_reference_table.append(
            f"Event #{idx}: {json.dumps(event_obj, ensure_ascii=False)}"
        )

    template = load_template(USER_TEMPLATE_PATH)
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
        num_events=len(llm_input.schedule),
        schedule_lines="\n".join(schedule_lines),
        event_reference_table="\n".join(event_reference_table),
        preserve_fixed_events=llm_input.constraints.preserve_fixed_events,
        avoid_medical_claims=llm_input.constraints.avoid_medical_claims,
        max_schedule_changes=llm_input.constraints.max_schedule_changes,
        preferred_output_language=llm_input.constraints.preferred_output_language,
    )


def build_static_mock_input(provider: str, model: str) -> LLMInput:
    """Build a static mock LLMInput for testing."""
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


def build_input_from_sensehat_mock(provider: str, model: str) -> LLMInput:
    """Build LLMInput from SenseHat sensor data (for local device testing)."""
    sleep_model_path = os.getenv("SLEEP_MODEL_PATH")
    controller = SenseHatController(model_path=sleep_model_path)
    sleep_data = controller.fetch_sleep_data()
    cluster = int(controller.predict_sleep_quality(sleep_data))
    
    cluster_profile = get_cluster_profile(cluster)
    if cluster_profile is None:
        raise RuntimeError(f"Unsupported sleep cluster: {cluster}")

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
        for item in SenseHatController.get_mock_schedule()
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





# ============================================================================
# Output Validation and Processing
# ============================================================================

def parse_hhmm(value: str) -> int:
    """Parse HH:MM format to total minutes."""
    try:
        hours, minutes = value.split(":")
        h, m = int(hours), int(minutes)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError()
        return h * 60 + m
    except (ValueError, AttributeError):
        raise RuntimeError(f"Invalid time format: {value}. Expected HH:MM.")


def validate_event_object(event: dict, context: str) -> dict:
    """Validate event object has required fields."""
    if not isinstance(event, dict):
        raise RuntimeError(f"{context} must be an object, got {type(event).__name__}")
    
    required = ["event_id", "title", "start", "end", "description"]
    for key in required:
        if key not in event:
            raise RuntimeError(f"{context}.{key} is required")
    
    if event["event_id"] is not None and not isinstance(event["event_id"], str):
        raise RuntimeError(f"{context}.event_id must be null or string")
    
    for key in ["title", "start", "end", "description"]:
        if not isinstance(event[key], str):
            raise RuntimeError(f"{context}.{key} must be string, got {type(event[key]).__name__}")
    
    return event


def validate_json_structure(payload: dict) -> dict:
    """Validate top-level JSON structure from LLM."""
    required = ["date", "daily_summary", "adjustment_principles", "calendar_operations", "optimized_schedule"]
    for field in required:
        if field not in payload:
            raise RuntimeError(f"Missing required field: {field}")
    
    if not isinstance(payload["date"], str):
        raise RuntimeError("Field 'date' must be string")
    
    if not isinstance(payload["daily_summary"], str):
        raise RuntimeError("Field 'daily_summary' must be string")
    
    if not isinstance(payload["adjustment_principles"], list):
        raise RuntimeError("Field 'adjustment_principles' must be list")
    
    if not all(isinstance(item, str) for item in payload["adjustment_principles"]):
        raise RuntimeError("All items in 'adjustment_principles' must be strings")
    
    if not isinstance(payload["calendar_operations"], list):
        raise RuntimeError("Field 'calendar_operations' must be list")
    
    if not isinstance(payload["optimized_schedule"], list):
        raise RuntimeError("Field 'optimized_schedule' must be list")
    
    return payload


def validate_calendar_operations(operations: list, llm_input: LLMInput) -> list:
    """Validate calendar_operations array."""
    original_events = [
        {
            "event_id": item.event_id,
            "title": item.task,
            "start": item.start,
            "end": item.end,
            "description": item.description,
            "intensity": item.intensity,
            "movable": item.movable,
        }
        for item in llm_input.schedule
    ]
    
    if len(operations) != len(original_events):
        raise RuntimeError(
            f"calendar_operations must contain exactly {len(original_events)} items "
            f"(one per original event), got {len(operations)}"
        )
    
    allowed_actions = {"move", "update_description", "no_update"}
    
    for idx, (op, original) in enumerate(zip(operations, original_events)):
        if not isinstance(op, dict):
            raise RuntimeError(f"calendar_operations[{idx}] must be object, got {type(op).__name__}")
        
        if "action" not in op:
            raise RuntimeError(f"calendar_operations[{idx}] missing 'action' field")
        
        if "reason" not in op:
            raise RuntimeError(f"calendar_operations[{idx}] missing 'reason' field")
        
        action = op["action"]
        if action not in allowed_actions:
            raise RuntimeError(f"calendar_operations[{idx}] action '{action}' must be one of {allowed_actions}")
        
        if action == "no_update":
            # For no_update, target and updated must match original - allow some flexibility here
            # since no_update means no change and using the original event is semantically correct
            if "target" not in op or op["target"] is None or not isinstance(op["target"], dict):
                if "target" in op and op["target"] is not None:
                    print(f"[INFO] calendar_operations[{idx}].target was {type(op['target']).__name__}, "
                          f"replacing with original event for no_update operation")
                op["target"] = dict(original)
            
            if "updated" not in op or op["updated"] is None or not isinstance(op["updated"], dict):
                if "updated" in op and op["updated"] is not None:
                    print(f"[INFO] calendar_operations[{idx}].updated was {type(op['updated']).__name__}, "
                          f"replacing with original event for no_update operation")
                op["updated"] = dict(original)
            continue
        
        # For move and update_description, both target and updated should be provided
        # and MUST be objects, not strings or other types
        if "target" not in op or op["target"] is None:
            raise RuntimeError(
                f"calendar_operations[{idx}] (action={action}) missing required 'target' field. "
                f"target must be a complete object with fields: event_id, title, start, end, description"
            )
        
        if "updated" not in op or op["updated"] is None:
            raise RuntimeError(
                f"calendar_operations[{idx}] (action={action}) missing required 'updated' field. "
                f"updated must be a complete object with fields: event_id, title, start, end, description"
            )
        
        # Strictly validate that target and updated are objects, not strings
        if not isinstance(op["target"], dict):
            raise RuntimeError(
                f"calendar_operations[{idx}].target must be a JSON object (dict), "
                f"not {type(op['target']).__name__}: {repr(op['target'])[:100]}. "
                f"Ensure target has fields: event_id, title, start, end, description"
            )
        
        if not isinstance(op["updated"], dict):
            raise RuntimeError(
                f"calendar_operations[{idx}].updated must be a JSON object (dict), "
                f"not {type(op['updated']).__name__}: {repr(op['updated'])[:100]}. "
                f"Ensure updated has fields: event_id, title, start, end, description"
            )
        
        validate_event_object(op["target"], f"calendar_operations[{idx}].target")
        validate_event_object(op["updated"], f"calendar_operations[{idx}].updated")
    
    # Final safety check: ensure no None values remain
    for idx, op in enumerate(operations):
        if op.get("target") is None:
            raise RuntimeError(f"calendar_operations[{idx}].target is still null after validation")
        if op.get("updated") is None:
            raise RuntimeError(f"calendar_operations[{idx}].updated is still null after validation")
    
    return operations


def validate_optimized_schedule(schedule: list, llm_input: LLMInput) -> list:
    """Validate optimized_schedule array."""
    original_events = llm_input.schedule
    
    if len(schedule) != len(original_events):
        raise RuntimeError(
            f"optimized_schedule must contain exactly {len(original_events)} items, got {len(schedule)}"
        )
    
    # Validate each item has required fields
    for idx, item in enumerate(schedule):
        validate_event_object(item, f"optimized_schedule[{idx}]")
        
        if "intensity" not in item or item["intensity"] not in {"low", "medium", "high"}:
            raise RuntimeError(f"optimized_schedule[{idx}].intensity must be low/medium/high")
        
        if "movable" not in item or not isinstance(item["movable"], bool):
            raise RuntimeError(f"optimized_schedule[{idx}].movable must be boolean")
    
    # Check all original events are present by title
    original_titles = sorted(item.task for item in original_events)
    output_titles = sorted(item["title"] for item in schedule)
    if original_titles != output_titles:
        raise RuntimeError(
            f"optimized_schedule must contain all original events. "
            f"Expected: {original_titles}, got: {output_titles}"
        )
    
    if not original_events:
        # No original schedule to compare against; nothing else to validate.
        return schedule
    
    # Check for overlaps and valid times
    sorted_items = sorted(schedule, key=lambda x: (parse_hhmm(x["start"]), parse_hhmm(x["end"])))
    prev_end = None
    
    for idx, item in enumerate(sorted_items):
        try:
            start_min = parse_hhmm(item["start"])
            end_min = parse_hhmm(item["end"])
        except RuntimeError as e:
            raise RuntimeError(f"optimized_schedule[{idx}]: {e}")
        
        if end_min <= start_min:
            raise RuntimeError(f"optimized_schedule[{idx}]: end time must be after start time")
        
        if prev_end is not None and start_min < prev_end:
            raise RuntimeError(f"optimized_schedule: overlapping events detected")
        
        prev_end = end_min
    
    # Check latest end time doesn't exceed original
    original_end_times = [parse_hhmm(item.end) for item in original_events]
    latest_original_end = max(original_end_times)
    latest_output_end = max(parse_hhmm(item["end"]) for item in schedule)
    
    if latest_output_end > latest_original_end:
        raise RuntimeError(
            f"optimized_schedule must not extend beyond original latest end time "
            f"({latest_original_end} minutes)"
        )
    
    # Check fixed events weren't changed
    fixed_events = {
        item.task: (item.start, item.end)
        for item in original_events
        if not item.movable
    }
    
    for idx, item in enumerate(schedule):
        title = item["title"]
        if title in fixed_events:
            orig_start, orig_end = fixed_events[title]
            if item["start"] != orig_start or item["end"] != orig_end:
                raise RuntimeError(
                    f"optimized_schedule[{idx}]: '{title}' is marked as fixed "
                    f"but times were changed"
                )
    
    return schedule


# ============================================================================
# LLM API Call
# ============================================================================

def resolve_api_key() -> str:
    """Get Qwen API key from environment."""
    api_key = (
        os.getenv("QWEN_API_KEY")
        or os.getenv("Qwen_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Qwen API key not found. Set QWEN_API_KEY, Qwen_API_KEY, or DASHSCOPE_API_KEY."
        )
    return api_key


def call_llm(
    system_instruction: str,
    user_prompt: str,
    llm_input: LLMInput,
) -> dict:
    """
    Call Qwen LLM API with retry logic.
    
    Args:
        system_instruction: System instruction text
        user_prompt: Formatted user prompt
        llm_input: Original LLMInput for validation context
    
    Returns:
        Parsed JSON response from LLM
        
    Raises:
        RuntimeError: If API call fails or response is invalid
    """
    api_key = resolve_api_key()
    base_url = os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).rstrip("/")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    max_attempts = 3
    retry_delay = 2
    model_name = getattr(llm_input, "model", None) or DEFAULT_MODEL
    
    print(f"[LLM] Calling {DEFAULT_PROVIDER} model '{model_name}' at {base_url}")
    
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            
            message_content = completion.choices[0].message.content
            if not isinstance(message_content, str) or not message_content.strip():
                raise RuntimeError("LLM returned empty response")
            
            # Parse JSON
            try:
                payload = json.loads(message_content)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"LLM returned invalid JSON: {str(e)[:100]}")
            
            print(f"[LLM] ✓ Response received and parsed")
            if isinstance(payload, dict):
                top_level_keys = sorted(payload.keys())
                print(
                    f"[LLM] Response summary: top_level_key_count={len(top_level_keys)}, "
                    f"top_level_keys={top_level_keys}"
                )
            else:
                print(f"[LLM] Response summary: payload_type={type(payload).__name__}")
            return payload
            
        except (APIConnectionError, APIStatusError, OpenAIError) as e:
            last_error = e
            error_msg = str(e)
            if isinstance(e, APIStatusError):
                error_msg = f"HTTP {e.status_code}: {error_msg[:200]}"
            
            print(f"[LLM] Attempt {attempt}/{max_attempts} failed: {error_msg}")
            
            if attempt < max_attempts:
                time.sleep(retry_delay)
                continue
            break
    
    raise RuntimeError(
        f"LLM API call failed after {max_attempts} attempts: {last_error}"
    ) from last_error

def generate_voice_response(voice_prompt: str, llm_input: LLMInput) -> str:
    """Takes STT voice input and generates a conversational schedule response using Qwen/LLM."""
    api_key = resolve_api_key()
    base_url = os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).rstrip("/")
    client = OpenAI(api_key=api_key, base_url=base_url)
    model_name = getattr(llm_input, "model", None) or DEFAULT_MODEL

    system = "You are a voice assistant managing a user's calendar. Provide clear, concise, and helpful audio-friendly spoken answers about their schedule or sleep data."
    
    schedule_lines = []
    for item in llm_input.schedule:
        schedule_lines.append(f"- {item.start}-{item.end} | {item.task} | {item.description}")
        
    user_context = (
        f"Sleep State: {llm_input.sleep_assessment.cluster_label}\n"
        f"Sleep Summary: {llm_input.sleep_assessment.summary}\n"
        "Today's Schedule:\n" + "\n".join(schedule_lines)
    )
    
    prompt = f"Given this context:\n{user_context}\n\nUser asked: {voice_prompt}"
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )
    
    return str(completion.choices[0].message.content).strip()

# ============================================================================
# Main Entry Points
# ============================================================================

def generate_schedule_recommendation(llm_input: LLMInput | None = None) -> ScheduleRecommendation:
    """
    Generate a schedule recommendation using the LLM.
    
    This is the main entry point from calendar-agent.py.
    It expects llm_input to be fully constructed with:
    - Current sleep assessment
    - Environment conditions
    - Original schedule
    - User constraints
    
    Args:
        llm_input: Complete LLMInput (if None, uses mock data for testing)
    
    Returns:
        ScheduleRecommendation with optimized schedule and calendar operations
        
    Raises:
        RuntimeError: If LLM call fails or output validation fails
    """
    # If no input provided, try to build from SenseHat or fall back to mock
    if llm_input is None:
        try:
            llm_input = build_input_from_sensehat_mock(DEFAULT_PROVIDER, DEFAULT_MODEL)
            print("[Input] Built from SenseHat sensor data")
        except Exception as e:
            print(f"[Input] Failed to build from SenseHat ({e}). Using mock data.")
            llm_input = build_static_mock_input(DEFAULT_PROVIDER, DEFAULT_MODEL)
    
    # Build prompts
    system_instruction = build_system_instruction()
    user_prompt = build_user_prompt(llm_input)
    
    print(f"[Prompt] Generated for {llm_input.user_name} on {llm_input.schedule_date}")
    
    # Call LLM
    llm_response = call_llm(system_instruction, user_prompt, llm_input)
    
    # Validate structure
    try:
        print("\n[Validation] Checking JSON structure...")
        validate_json_structure(llm_response)
        print("[Validation] ✓ JSON structure valid")
        
        print("[Validation] Checking calendar_operations...")
        validate_calendar_operations(llm_response["calendar_operations"], llm_input)
        print(f"[Validation] ✓ calendar_operations valid ({len(llm_response['calendar_operations'])} items)")
        
        print("[Validation] Checking optimized_schedule...")
        validate_optimized_schedule(llm_response["optimized_schedule"], llm_input)
        print(f"[Validation] ✓ optimized_schedule valid ({len(llm_response['optimized_schedule'])} items)")
        
    except RuntimeError as e:
        print(f"\n[Validation] ✗ FAILED: {e}")
        print(f"\n[Validation] Problem context:")
        print(f"[Validation] LLM response was:\n{json.dumps(llm_response, indent=2, ensure_ascii=False)}")
        raise
    except Exception as e:
        # Catch any other errors during validation
        print(f"\n[Validation] ✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        print(f"\n[Validation] Full traceback:")
        import traceback
        traceback.print_exc()
        print(f"\n[Validation] LLM response was:\n{json.dumps(llm_response, indent=2, ensure_ascii=False)}")
        raise RuntimeError(f"Validation failed with {type(e).__name__}: {e}") from e
    
    print("\n[Validation] ✓ All checks passed")
    
    # Convert to ScheduleRecommendation
    recommendation = ScheduleRecommendation.from_dict(llm_response)
    return recommendation


# ============================================================================
# Debug/Test Entry Point
# ============================================================================

def main() -> int:
    """
    Standalone debug entry point for testing LLM interaction.
    
    This script allows testing the full LLM pipeline without running
    the calendar-agent. Set environment variables:
    - QWEN_API_KEY: Your Qwen API key
    - LLM_MODEL: Qwen model (default: qwen3.6-plus)
    
    Returns:
        0 on success, 1 on failure
    """
    print("=" * 70)
    print("LLM Interaction Debug - Schedule Recommendation Pipeline")
    print("=" * 70)
    print()
    
    # Build input
    try:
        llm_input = build_input_from_sensehat_mock(DEFAULT_PROVIDER, DEFAULT_MODEL)
        print("[Input] Built from SenseHat sensor data")
    except Exception as e:
        print(f"[Input] SenseHat not available ({e}), using mock data")
        llm_input = build_static_mock_input(DEFAULT_PROVIDER, DEFAULT_MODEL)
    
    print()
    print("Input structured data (LLMInput):")
    print(json.dumps(asdict(llm_input), indent=2, ensure_ascii=False))
    print()
    
    # Build prompts
    system_instruction = build_system_instruction()
    user_prompt = build_user_prompt(llm_input)
    
    print("-" * 70)
    print("System Instruction:")
    print("-" * 70)
    print(system_instruction)
    print()
    
    print("-" * 70)
    print("User Prompt:")
    print("-" * 70)
    print(user_prompt)
    print()
    
    # Call LLM
    try:
        print("=" * 70)
        print("Calling LLM API...")
        print("=" * 70)
        print()
        
        recommendation = generate_schedule_recommendation(llm_input)
        
        print()
        print("=" * 70)
        print("Schedule Recommendation (ScheduleRecommendation):")
        print("=" * 70)
        print(json.dumps(recommendation.to_dict(), indent=2, ensure_ascii=False))
        print()
        
        return 0
        
    except RuntimeError as e:
        print()
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

