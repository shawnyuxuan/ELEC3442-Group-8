# Calendar Agent - Multi-threaded Sleep & Schedule Optimization Pipeline
#
# Data Flow:
#   1. SleepReportListener → sleep_report_queue
#   2. llm_worker: reads sleep data → predicts sleep quality → builds LLMInput → calls LLM → ScheduleRecommendation
#   3. calendar_worker: reads ScheduleRecommendation → applies calendar operations
#
# Three independent threads:
#   - listener-thread: HTTP endpoint accepting daily sleep reports
#   - llm-worker: processes sleep data with ML model + LLM
#   - calendar-worker: modifies calendar based on LLM recommendations

import datetime
import math
import os
import re
import multiprocessing
import traceback
import pandas as pd
import queue
import signal
import threading
import time
import requests
from typing import Any

from src.calendar_interaction import CalendarController, DEFAULT_CALENDAR_NAME
from src.daily_analysis import SleepReportListener
from src.calendar_feedback import CalendarFeedbackListener
from src.llm_contract import CalendarOperation, ScheduleRecommendation
from src.llm_interaction import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    EnvironmentContext,
    LLMInput,
    ScheduleItem,
    SleepAssessment,
    UserConstraints,
    generate_schedule_adjustment,
    generate_schedule_recommendation,
    generate_voice_response,
)
from src.schedule_rules import schedule_item_from_calendar_event
from src.sensehat_behavior import SenseHatController, get_cluster_profile, get_mock_schedule
from src.speech_to_text import VoiceAssistant, build_voice_feedback_payload

QUEUE_MAXSIZE = 100
WORKER_POLL_TIMEOUT_SECONDS = 1
CALENDAR_READ_MAX_ATTEMPTS = 3
CALENDAR_READ_RETRY_SECONDS = 2
SENTINEL = object()
USER_NAME = os.getenv("USER_NAME", "Jayden")
DEFAULT_WAKE_WORDS = tuple(
    token.strip().lower()
    for token in os.getenv("VOICE_WAKE_WORDS", "hey calendar").split(",")
    if token.strip()
)
DEFAULT_VOICE_CREATE_DURATION_MINUTES = max(15, int(os.getenv("VOICE_CREATE_DURATION_MINUTES", "60")))


def _to_local_datetime(value):
    if isinstance(value, datetime.datetime) and value.tzinfo is not None:
        local_tz = datetime.datetime.now().astimezone().tzinfo
        return value.astimezone(local_tz)
    return value


def normalize_date_key(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d/%m/%y",
    ):
        try:
            parsed = datetime.datetime.strptime(text, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue

    if "T" in text:
        try:
            parsed = datetime.datetime.fromisoformat(text)
            return parsed.date().isoformat()
        except ValueError:
            pass

    return text[:10]

def log(stage: str, message: str):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{now}] [{stage}] {message}")

def normalize_report_date(date_text: str) -> datetime.datetime:
    # Accept common date formats from client reports and default to today if parsing fails.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.datetime.strptime(date_text, fmt)
            return datetime.datetime.combine(parsed.date(), datetime.time.min)
        except ValueError:
            continue
    return datetime.datetime.combine(datetime.datetime.now().date(), datetime.time.min)

def build_mock_schedule_items() -> list[ScheduleItem]:
    return [
        ScheduleItem(
            item.get("event_id"),
            item["start"],
            item["end"],
            item["task"],
            str(item.get("description", "")),
            item["intensity"],
            item["movable"],
        )
        for item in get_mock_schedule()
    ]


def build_schedule_from_calendar(calendar_name: str, date_text: str) -> list[ScheduleItem]:
    date_start = normalize_report_date(date_text)
    last_error: Exception | None = None

    for attempt in range(1, CALENDAR_READ_MAX_ATTEMPTS + 1):
        try:
            controller = CalendarController(calendar_name=calendar_name)
            events = controller.fetch_events(date_start)
            break
        except Exception as exc:
            last_error = exc
            if attempt < CALENDAR_READ_MAX_ATTEMPTS:
                log(
                    "calendar-read",
                    (
                        f"failed to read calendar '{calendar_name}' for {date_text} "
                        f"(attempt {attempt}/{CALENDAR_READ_MAX_ATTEMPTS}): {exc}"
                    ),
                )
                time.sleep(CALENDAR_READ_RETRY_SECONDS)
                continue
            raise RuntimeError(
                f"Failed to read calendar '{calendar_name}' for {date_text} after "
                f"{CALENDAR_READ_MAX_ATTEMPTS} attempts: {exc}"
            ) from exc

    schedule = []
    for event in events:
        schedule.append(schedule_item_from_calendar_event(event))

    return schedule


def _parse_voice_date(transcript: str, fallback_date: str) -> str:
    lowered = transcript.lower()
    if "tomorrow" in lowered:
        return (datetime.date.fromisoformat(fallback_date) + datetime.timedelta(days=1)).isoformat()
    if "today" in lowered:
        return fallback_date

    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", transcript)
    if date_match:
        return date_match.group(1)

    return fallback_date


def _parse_voice_time(transcript: str) -> datetime.time | None:
    lowered = transcript.lower()
    patterns = [
        r"\b(?:at|around|for)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue

        hour_text = match.group(1)
        minute_text = match.group(2) or "00"
        meridiem = match.group(3)

        hour = int(hour_text)
        minute = int(minute_text)

        if meridiem:
            if hour == 12:
                hour = 0
            if meridiem == "pm":
                hour += 12

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return datetime.time(hour=hour, minute=minute)

    return None


def _parse_voice_duration_minutes(transcript: str) -> int:
    lowered = transcript.lower()
    if match := re.search(r"\bfor\s*(\d{1,2})\s*hours?\b", lowered):
        return max(15, int(match.group(1)) * 60)
    if match := re.search(r"\bfor\s*(\d{1,3})\s*minutes?\b", lowered):
        return max(15, int(match.group(1)))
    return DEFAULT_VOICE_CREATE_DURATION_MINUTES


def _parse_voice_event_title(transcript: str) -> str:
    lowered = transcript.lower()
    cleaned = re.sub(r"\b(hey\s+calendar|calendar|please|could you|can you)\b", "", lowered)
    cleaned = re.sub(r"\b(add|create|new|schedule|event|meeting|task)\b", "", cleaned)
    cleaned = re.sub(r"\b(at|around|for|today|tomorrow|the|a|an)\b", "", cleaned)
    cleaned = re.sub(r"\b\d{1,2}(:\d{2})?\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        return cleaned[:80].title()
    return "Voice Created Event"


def parse_voice_move_request(transcript: str, fallback_date: str) -> dict | None:
    lowered = transcript.lower()
    if "move" not in lowered:
        return None

    # Examples supported:
    # - move the schedule at 15 to 16
    # - move deep work at 15:00 to 16:30
    match = re.search(
        r"\bmove\s+(?P<title>.*?)\s*at\s*(?P<from>\d{1,2}(?::\d{2})?)\s+to\s*(?P<to>\d{1,2}(?::\d{2})?)\b",
        lowered,
    )
    if not match:
        return None

    def _to_hhmm(value: str) -> str:
        if ":" in value:
            hour, minute = value.split(":", 1)
        else:
            hour, minute = value, "00"
        return f"{int(hour):02d}:{int(minute):02d}"

    title = re.sub(r"\b(the|schedule|event|task|meeting)\b", "", match.group("title") or "")
    title = re.sub(r"\s+", " ", title).strip()

    return {
        "date": _parse_voice_date(transcript, fallback_date),
        "from_hhmm": _to_hhmm(match.group("from")),
        "to_hhmm": _to_hhmm(match.group("to")),
        "title_hint": title,
    }


def move_calendar_event_from_voice(calendar_name: str, transcript: str, fallback_date: str) -> dict:
    move_request = parse_voice_move_request(transcript, fallback_date)
    if move_request is None:
        raise ValueError("Unable to parse a move-event request from the voice transcript")

    date_start = normalize_report_date(move_request["date"])
    controller = CalendarController(calendar_name=calendar_name)
    events = controller.fetch_events(date_start)

    source_event = None
    for event in events:
        vevent = event.vobject_instance.vevent
        title = str(vevent.summary.value)
        start_hhmm = _to_local_datetime(vevent.dtstart.value).strftime("%H:%M")

        if start_hhmm != move_request["from_hhmm"]:
            continue

        title_hint = move_request["title_hint"]
        if title_hint and title_hint not in title.lower():
            continue

        source_event = event
        break

    if source_event is None:
        raise RuntimeError(
            f"No matching event found at {move_request['from_hhmm']} on {move_request['date']}"
        )

    vevent = source_event.vobject_instance.vevent
    original_start = vevent.dtstart.value
    original_end = vevent.dtend.value
    duration = original_end - original_start

    to_hour, to_minute = [int(token) for token in move_request["to_hhmm"].split(":")]
    local_tz = datetime.datetime.now().astimezone().tzinfo
    local_original_start = _to_local_datetime(original_start)
    new_start_local = datetime.datetime.combine(
        local_original_start.date(),
        datetime.time(to_hour, to_minute),
        local_tz,
    )

    if isinstance(original_start, datetime.datetime) and original_start.tzinfo is not None:
        new_start = new_start_local.astimezone(original_start.tzinfo)
    else:
        new_start = datetime.datetime.combine(
            local_original_start.date(),
            datetime.time(to_hour, to_minute),
        )
    new_end = new_start + duration

    controller.update_event(
        event=source_event,
        new_start_time=new_start,
        new_end_time=new_end,
    )

    return {
        "date": move_request["date"],
        "title": str(vevent.summary.value),
        "from": move_request["from_hhmm"],
        "to": move_request["to_hhmm"],
    }


def parse_voice_create_request(transcript: str, fallback_date: str) -> dict | None:
    lowered = transcript.lower()
    if not any(keyword in lowered for keyword in ("add", "create", "new schedule", "new event")):
        return None

    event_date = _parse_voice_date(transcript, fallback_date)
    event_time = _parse_voice_time(transcript)
    if event_time is None:
        return None

    duration_minutes = _parse_voice_duration_minutes(transcript)
    local_tz = datetime.datetime.now().astimezone().tzinfo
    start_dt = datetime.datetime.combine(datetime.date.fromisoformat(event_date), event_time, local_tz)
    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)

    return {
        "date": event_date,
        "title": _parse_voice_event_title(transcript),
        "start_time": start_dt,
        "end_time": end_dt,
        "description": f"Created from voice command: {transcript}",
    }


def create_calendar_event_from_voice(calendar_name: str, transcript: str, fallback_date: str) -> dict:
    create_request = parse_voice_create_request(transcript, fallback_date)
    if create_request is None:
        raise ValueError("Unable to parse a create-event request from the voice transcript")

    controller = CalendarController(calendar_name=calendar_name)
    controller.create_event(
        event_title=create_request["title"],
        start_time=create_request["start_time"],
        end_time=create_request["end_time"],
        description=create_request["description"],
    )

    return create_request


def listen_calendar_feedback(host: str, port: int, feedback_queue: queue.Queue):
    def on_feedback(feedback_data: dict[str, Any]):
        try:
            feedback_queue.put(dict(feedback_data), timeout=2)
            log(
                "feedback-listener",
                f"enqueued voice feedback date={feedback_data.get('date')} queue={feedback_queue.qsize()}",
            )
        except queue.Full:
            log("feedback-listener", "feedback queue is full; feedback dropped")

    listener = CalendarFeedbackListener(host=host, port=port, on_feedback=on_feedback)
    listener.start(debug=False, threaded=True, use_reloader=False)


def build_sleep_features_from_report(sleep_report: dict[str, Any]) -> pd.DataFrame:
    date_start = normalize_report_date(str(sleep_report.get("date", "")))
    weekday = date_start.weekday()

    core_minutes = max(0.0, float(sleep_report.get("core", 0)) * 60.0)
    deep_minutes = max(0.0, float(sleep_report.get("deep", 0)) * 60.0)
    rem_minutes = max(0.0, float(sleep_report.get("rem", 0)) * 60.0)

    # Without explicit start-time in the report, use a stable default bedtime phase.
    start_sin = math.sin((23.0 / 24.0) * 2.0 * math.pi)

    return pd.DataFrame(
        {
            "start_sin": [start_sin],
            "CORE": [core_minutes],
            "DEEP": [deep_minutes],
            "REM": [rem_minutes],
            "AWAKE": [10.0],
            "UNSPECIFIED": [0.0],
            "weekday": [weekday],
        }
    )


def llm_worker(
    sleep_report_queue: queue.Queue,
    calendar_ops_queue: queue.Queue,
    stop_event: threading.Event,
    calendar_name: str,
    shared_state: dict[str, Any],
):
    """
    LLM Worker Thread - Orchestrates the sleep analysis → LLM → recommendation pipeline.
    
    Data flow:
      sleep_report → sleep_features + sensor_data → LLMInput → LLM API → ScheduleRecommendation
    """
    thread_name = threading.current_thread().name
    log(thread_name, "started. Waiting for sleep reports...")

    sensehat_controller: SenseHatController | None = None

    def get_sensehat_controller(force_refresh: bool = False) -> SenseHatController:
        nonlocal sensehat_controller
        if force_refresh or sensehat_controller is None:
            model_path = os.getenv("SLEEP_MODEL_PATH")
            sensehat_controller = SenseHatController(model_path=model_path)
        print(sensehat_controller.__class__)
        return sensehat_controller

    while True:
        # Stop when signaled AND queue is empty
        if stop_event.is_set() and sleep_report_queue.empty():
            break

        # Wait for sleep report from listener
        try:
            payload = sleep_report_queue.get(timeout=WORKER_POLL_TIMEOUT_SECONDS)
        except queue.Empty:
            continue

        # Sentinel signals graceful shutdown
        if payload is SENTINEL:
            sleep_report_queue.task_done()
            log(thread_name, "received sentinel, exiting")
            break

        # Process sleep report through pipeline
        date_text = str(payload.get("date", "")) or datetime.datetime.now().date().isoformat()
        cluster_id = None
        cluster_label = "Unknown"
        
        try:
            # ========== STAGE 1: Sleep Quality Prediction ==========
            log(thread_name, f"[1/4] Processing sleep report for {date_text}")
            
            # Extract sleep features from report and predict cluster
            sleep_features = build_sleep_features_from_report(payload)
            sensehat_ctl = get_sensehat_controller()
            cluster_id = int(sensehat_ctl.predict_sleep_quality(sleep_features))
            cluster_profile = get_cluster_profile(cluster_id) or {}
            cluster_profile["cluster_id"] = cluster_id
            cluster_label = str(cluster_profile.get("label", "Unknown"))
            
            log(thread_name, f"[1/4] ✓ Predicted cluster={cluster_id} ({cluster_label})")
            
            # Display on SenseHat LED
            sensehat_ctl.led_display(cluster_id)
            print(f"\n--- Sleep Analysis ---\nCluster: {cluster_id} ({cluster_label})\n"
                  f"Summary: {cluster_profile.get('summary', 'N/A')}\n"
                  f"Risks: {', '.join(cluster_profile.get('likely_risks', []))}\n"
                  f"Work Style: {cluster_profile.get('recommended_work_style', 'N/A')}\n")
            
            # ========== STAGE 2: Gather Real-Time Data ==========
            log(thread_name, f"[2/4] Gathering sensor and calendar data")
            
            # Read real-time sensor data
            environment_data = {
                "temperature_c": float(sensehat_ctl.sensor.get_temperature()),
                "humidity_percent": float(sensehat_ctl.sensor.get_humidity()),
                "pressure_hpa": float(sensehat_ctl.sensor.get_pressure()),
            }
            
            print(environment_data)
            # Read calendar events for today
            schedule_items = build_schedule_from_calendar(calendar_name, date_text)
            
            log(thread_name, f"[2/4] ✓ Gathered: temp={environment_data['temperature_c']:.1f}°C, "
                f"schedule={len(schedule_items)} events")
            
            # ========== STAGE 3: Build LLMInput ==========
            log(thread_name, f"[3/4] Building LLMInput contract")
            
            llm_input = LLMInput(
                schedule_date=date_text,
                user_name=USER_NAME,
                provider=DEFAULT_PROVIDER,
                model=DEFAULT_MODEL,
                sleep_assessment=SleepAssessment(
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    summary=str(cluster_profile.get("summary", "")),
                    likely_risks=[str(item) for item in cluster_profile.get("likely_risks", [])],
                    recommended_work_style=str(cluster_profile.get("recommended_work_style", "")),
                ),
                environment=EnvironmentContext(
                    temperature_c=environment_data["temperature_c"],
                    humidity_percent=environment_data["humidity_percent"],
                    pressure_hpa=environment_data["pressure_hpa"],
                ),
                schedule=schedule_items,
                constraints=UserConstraints(
                    preserve_fixed_events=True,
                    avoid_medical_claims=True,
                    max_schedule_changes=3,
                    preferred_output_language="English",
                ),
            )

            # Keep the latest real-time context for feedback adjustments.
            shared_state["latest_llm_input"] = llm_input
            
            log(thread_name, f"[3/4] ✓ LLMInput built: user={llm_input.user_name}, "
                f"schedule={len(llm_input.schedule)} items")
            
            # ========== STAGE 4: Call LLM for Recommendations ==========
            log(thread_name, f"[4/4] Calling LLM for schedule optimization")
            
            try:
                recommendation = generate_schedule_recommendation(llm_input)
                
                log(thread_name, f"[4/4] ✓ LLM returned recommendation: "
                    f"{len(recommendation.calendar_operations)} operations, "
                    f"{len(recommendation.optimized_schedule)} schedule items")
                
                # Enqueue recommendation for calendar processing
                calendar_ops_queue.put(recommendation, timeout=2)
                
                log(thread_name, f"✓ COMPLETE: date={date_text} cluster={cluster_id}({cluster_label}) "
                    f"sleep_q={sleep_report_queue.qsize()} ops_q={calendar_ops_queue.qsize()}")
                    
            except RuntimeError as llm_error:
                log(thread_name, f"✗ LLM error: {llm_error}")
                raise
                
        except Exception as exc:
            # Reset SenseHat controller on error (it may be in bad state)
            sensehat_controller = None
            log(thread_name, f"✗ FAILED: date={date_text} cluster={cluster_label} error={exc}")
            
        finally:
            sleep_report_queue.task_done()

    log(thread_name, "stopped")


def feedback_worker(
    feedback_queue: queue.Queue,
    calendar_ops_queue: queue.Queue,
    stop_event: threading.Event,
    shared_state: dict[str, Any],
):
    """
    Feedback worker - revises the latest applied recommendation from a voice transcript.
    """
    thread_name = threading.current_thread().name
    log(thread_name, "started. Waiting for voice feedback...")

    while True:
        if stop_event.is_set() and feedback_queue.empty():
            break

        try:
            feedback = feedback_queue.get(timeout=WORKER_POLL_TIMEOUT_SECONDS)
        except queue.Empty:
            continue

        if feedback is SENTINEL:
            feedback_queue.task_done()
            log(thread_name, "received sentinel, exiting")
            break

        try:
            transcript = str(feedback.get("transcript", "")).strip()
            if not transcript:
                log(thread_name, "skipping empty transcript feedback")
                continue

            latest_recommendation = shared_state.get("latest_recommendation")
            fallback_date = (
                latest_recommendation.date
                if isinstance(latest_recommendation, ScheduleRecommendation)
                else str(feedback.get("date") or datetime.date.today().isoformat())
            )

            if "move" in transcript.lower():
                try:
                    moved = move_calendar_event_from_voice(
                        calendar_name=os.getenv("CALENDAR_NAME", DEFAULT_CALENDAR_NAME),
                        transcript=transcript,
                        fallback_date=fallback_date,
                    )
                    msg = f"✓ moved event on {moved['date']} {moved['title']} {moved['from']} -> {moved['to']}"
                    log(thread_name, msg)
                    threading.Thread(target=broadcast_voice_update, args=(f"I have successfully moved {moved['title']} to {moved['to']}",), daemon=True).start()
                except Exception as exc:
                    log(thread_name, f"✗ voice move failed: {exc}")
                    threading.Thread(target=broadcast_voice_update, args=("I could not move the event. Please check the logs.",), daemon=True).start()
                continue

            lowered_transcript = transcript.lower()
            if any(keyword in lowered_transcript for keyword in ("add", "create", "new schedule", "new event")):
                try:
                    created = create_calendar_event_from_voice(
                        calendar_name=os.getenv("CALENDAR_NAME", DEFAULT_CALENDAR_NAME),
                        transcript=transcript,
                        fallback_date=fallback_date,
                    )
                    msg = f"✓ created voice event on {created['date']} {created['title']} {created['start_time'].strftime('%H:%M')}"
                    log(thread_name, msg)
                    threading.Thread(target=broadcast_voice_update, args=(f"I have scheduled {created['title']} at {created['start_time'].strftime('%H:%M')}",), daemon=True).start()
                except Exception as exc:
                    log(thread_name, f"✗ voice create failed: {exc}")
                    threading.Thread(target=broadcast_voice_update, args=("I could not create the event. Please check the logs.",), daemon=True).start()
                continue

            base_recommendation = shared_state.get("latest_recommendation")
            if not isinstance(base_recommendation, ScheduleRecommendation):
                log(thread_name, "skipping feedback because no base recommendation is available")
                continue

            context_input = None
            latest_llm_input = shared_state.get("latest_llm_input")
            if isinstance(latest_llm_input, LLMInput):
                latest_date = normalize_date_key(latest_llm_input.schedule_date)
                recommendation_date = normalize_date_key(base_recommendation.date)
                if latest_date == recommendation_date:
                    context_input = latest_llm_input
                else:
                    log(
                        thread_name,
                        (
                            "feedback date/context date mismatch "
                            f"({recommendation_date} vs {latest_date}); "
                            "falling back to local context build"
                        ),
                    )

            log(thread_name, f"processing feedback for {feedback.get('date')}: {transcript}")
            adjusted_recommendation = generate_schedule_adjustment(
                base_recommendation,
                transcript,
                context_input=context_input,
            )

            calendar_ops_queue.put(adjusted_recommendation, timeout=2)
            log(
                thread_name,
                (
                    f"✓ queued adjusted recommendation: date={adjusted_recommendation.date} "
                    f"ops={len(adjusted_recommendation.calendar_operations)}"
                ),
            )

        except Exception as exc:
            log(thread_name, f"✗ FAILED: error processing feedback: {exc}")
        finally:
            feedback_queue.task_done()

    log(thread_name, "stopped")


def broadcast_voice_update(text: str):
    """
    Sends an HTTP POST to Pi 2 (the voice agent) to trigger TTS.
    """
    pi2_url = os.getenv("PI2_VOICE_AGENT_URL", "http://192.168.1.101:5890/speak")
    try:
        requests.post(pi2_url, json={"text": text}, timeout=5)
    except Exception as exc:
        log("broadcast", f"failed to contact Pi 2 voice agent: {exc}")


def calendar_worker(
    calendar_ops_queue: queue.Queue,
    stop_event: threading.Event,
    calendar_name: str,
    shared_state: dict[str, Any],
):
    """
    Calendar Worker Thread - Applies schedule recommendations to the calendar.
    
    Data flow:
      ScheduleRecommendation → preflight check → apply operations → calendar sync
    """
    thread_name = threading.current_thread().name
    log(thread_name, f"started. Listening for recommendations on calendar '{calendar_name}'")

    controller: CalendarController | None = None

    def get_controller(force_refresh: bool = False) -> CalendarController:
        nonlocal controller
        if force_refresh or controller is None:
            controller = CalendarController(calendar_name=calendar_name)
        return controller

    def is_retryable_calendar_error(error: Exception) -> bool:
        message = str(error).lower()
        return "keepalive timeout" in message or "read timed out" in message

    def format_operation_label(operation: CalendarOperation) -> str:
        if operation.action == "no_update":
            return f"no_update: {operation.reason}"

        target = operation.target
        updated = operation.updated
        action_str = f"[{operation.action}]"
        
        if target and updated:
            return (f"{action_str} {target.title} "
                    f"{target.start}-{target.end} → {updated.start}-{updated.end}")
        return f"{action_str} unknown operation"

    def rollback_operations(date_start: datetime.datetime, inverse_operations: list[CalendarOperation]):
        nonlocal controller
        if not inverse_operations:
            return

        log(thread_name, f"↩ Rolling back {len(inverse_operations)} operations")
        for inverse_op in reversed(inverse_operations):
            op_label = format_operation_label(inverse_op)
            for attempt in range(1, 4):
                try:
                    active_ctl = get_controller(force_refresh=(attempt > 1))
                    active_ctl.apply_calendar_operation(date_start, inverse_op)
                    log(thread_name, f"  ↩ Rolled back: {op_label}")
                    break
                except Exception as exc:
                    controller = None
                    if is_retryable_calendar_error(exc) and attempt < 3:
                        log(thread_name, f"  ↻ Rollback retry {attempt}/3 for {op_label}")
                        continue
                    log(thread_name, f"  ✗ Rollback failed for {op_label}: {exc}")
                    break

    while True:
        # Stop when signaled AND queue is empty
        if stop_event.is_set() and calendar_ops_queue.empty():
            break

        # Wait for recommendation from LLM worker
        try:
            recommendation = calendar_ops_queue.get(timeout=WORKER_POLL_TIMEOUT_SECONDS)
        except queue.Empty:
            continue

        # Sentinel signals graceful shutdown
        if recommendation is SENTINEL:
            calendar_ops_queue.task_done()
            log(thread_name, "received sentinel, exiting")
            break

        # Apply recommendation to calendar
        try:
            if not isinstance(recommendation, ScheduleRecommendation):
                log(thread_name, f"✗ Invalid payload type: {type(recommendation).__name__}")
                continue

            # Do not block wake-word interaction if current recommendation fails to apply.
            shared_state["voice_listener_enabled"] = True

            # Parse date from recommendation
            date_start = datetime.datetime.combine(
                datetime.datetime.strptime(recommendation.date, "%Y-%m-%d").date(),
                datetime.time.min,
            )
            
            log(thread_name, f"📅 Processing recommendation for {recommendation.date}")
            log(thread_name, f"   {len(recommendation.calendar_operations)} operations to apply")

            # Preflight check: verify operations can be applied
            active_controller = get_controller(force_refresh=True)
            preflight_results = active_controller.preflight_operations(
                date_start,
                recommendation.calendar_operations,
            )
            
            # Log preflight results
            for result, operation in zip(preflight_results, recommendation.calendar_operations):
                status = result["status"]
                match status:
                    case "target": status_icon = "✓"
                    case "updated": status_icon = "~"
                    case "no_update": status_icon = "~"
                    case _: status_icon = "✗"
                log(thread_name, f"   {status_icon} [{status}] {format_operation_label(operation)}")

            # Apply each operation with retry logic
            applied_count = 0
            skipped_count = 0
            inverse_ops: list[CalendarOperation] = []
            
            for idx, operation in enumerate(recommendation.calendar_operations):
                op_label = format_operation_label(operation)
                preflight_status = preflight_results[idx]["status"]
                
                # Skip if no update needed
                if preflight_status == "no_update":
                    skipped_count += 1
                    continue
                
                # Apply operation with retries
                for attempt in range(1, 4):
                    try:
                        active_controller = get_controller(force_refresh=(attempt > 1))
                        active_controller.apply_calendar_operation(date_start, operation)
                        
                        # Collect inverse operation for potential rollback
                        match preflight_status:
                            case "target":
                                inverse_op = preflight_results[idx].get("inverse_operation")
                                if inverse_op:
                                    inverse_ops.append(inverse_op)
                                applied_count += 1
                                log(thread_name, f"   ✓ Applied: {op_label}")
                                break
                            case "no_update":
                                skipped_count += 1
                                continue
                            case "updated":
                                applied_count += 1
                                break
                            case _:
                                log(thread_name, f"Unknown preflight status '{preflight_status}' for operation: {op_label}. Retrial {attempt}/3")
                                continue                        
                        
                        
                    except Exception as exc:
                        controller = None
                        
                        if is_retryable_calendar_error(exc) and attempt < 3:
                            log(thread_name, f"   ↻ Retry {attempt}/3 for {op_label}")
                            time.sleep(1)
                            continue
                        
                        # Apply failed - rollback previous operations
                        log(thread_name, f"   ✗ Failed to apply: {op_label}")
                        rollback_operations(date_start, inverse_ops)
                        raise

            # Log summary
            total_ops = len(recommendation.calendar_operations)
            if applied_count == 0 and skipped_count == total_ops:
                summary_msg = "No calendar changes were needed today."
                log(thread_name, f"✓ COMPLETE: No calendar changes needed for {recommendation.date}")
            else:
                summary_msg = f"I have applied {applied_count} updates to your schedule based on your sleep analysis."
                log(thread_name, f"✓ COMPLETE: Applied {applied_count} ops, "
                    f"skipped {skipped_count} no_updates on {recommendation.date}")

            threading.Thread(target=broadcast_voice_update, args=(summary_msg,), daemon=True).start()

            shared_state["latest_recommendation"] = recommendation
            shared_state["voice_listener_enabled"] = True

        except Exception as exc:
            log(thread_name, f"✗ FAILED: Error processing recommendation: {exc}")
            traceback.print_exc()
            shared_state["voice_listener_enabled"] = True
            
        finally:
            calendar_ops_queue.task_done()

    log(thread_name, "stopped")


def listen_daily_analysis(host: str, port: int, calendar_ops_queue: queue.Queue, sleep_report_queue: queue.Queue, calendar_name: str):
    def on_report(sleep_data: dict[str, Any]):
        try:
            sleep_report_queue.put(dict(sleep_data), timeout=2)
            log("listener", f"enqueued sleep report date={sleep_data.get('date')} queue={sleep_report_queue.qsize()}")
        except queue.Full:
            log("listener", "sleep report queue is full; report dropped")

    def on_voice(voice_input: Any) -> str:
        voice_payload = voice_input if isinstance(voice_input, dict) else build_voice_feedback_payload(transcript=str(voice_input or ""))
        transcript = str(voice_payload.get("transcript") or "").strip()
        if not transcript:
            raise RuntimeError("Voice transcript is empty")

        log("listener-voice", f"Processing voice command: {transcript}")
        date_text = datetime.datetime.now().date().isoformat()
        lowered_transcript = transcript.lower()

        # Fast path: explicit create requests should directly create events instead of
        # forcing a schedule-optimization response over an empty/non-empty schedule.
        if any(keyword in lowered_transcript for keyword in ("add", "create", "new schedule", "new event")):
            try:
                created = create_calendar_event_from_voice(
                    calendar_name=os.getenv("CALENDAR_NAME", DEFAULT_CALENDAR_NAME),
                    transcript=transcript,
                    fallback_date=date_text,
                )
                created_time = created["start_time"].strftime("%H:%M")
                log("listener-voice", f"Voice create handled directly: {created['title']} at {created_time}")
                return f"I have scheduled {created['title']} at {created_time}"
            except Exception as create_error:
                log("listener-voice", f"Voice create parse/apply failed, falling back to LLM flow: {create_error}")
        
        try:
            # Build current schedule
            schedule_items = build_schedule_from_calendar(calendar_name, date_text)
            
            # Build sensor + sleep mock (using SenseHat directly)
            model_path = os.getenv("SLEEP_MODEL_PATH")
            sensehat_ctl = SenseHatController(model_path=model_path)
            
            try:
                # Provide a generic/default sleep mock if we don't have current sleep data
                sleep_features = sensehat_ctl.fetch_sleep_data()
            except Exception:
                sleep_features = pd.DataFrame({
                    "start_sin": [math.sin((23.0 / 24.0) * 2.0 * math.pi)],
                    "CORE": [180.0],
                    "DEEP": [60.0],
                    "REM": [90.0],
                    "AWAKE": [10.0],
                    "UNSPECIFIED": [0.0],
                    "weekday": [datetime.datetime.now().weekday()],
                })
            
            cluster_id = int(sensehat_ctl.predict_sleep_quality(sleep_features))
            cluster_profile = get_cluster_profile(cluster_id) or {}
            
            environment_data = {
                "temperature_c": float(sensehat_ctl.sensor.get_temperature()),
                "humidity_percent": float(sensehat_ctl.sensor.get_humidity()),
                "pressure_hpa": float(sensehat_ctl.sensor.get_pressure()),
            }
            
            llm_input = LLMInput(
                schedule_date=date_text,
                user_name=USER_NAME,
                provider=DEFAULT_PROVIDER,
                model=DEFAULT_MODEL,
                sleep_assessment=SleepAssessment(
                    cluster_id=cluster_id,
                    cluster_label=str(cluster_profile.get("label", "Unknown")),
                    summary=str(cluster_profile.get("summary", "")),
                    likely_risks=[str(item) for item in cluster_profile.get("likely_risks", [])],
                    recommended_work_style=str(cluster_profile.get("recommended_work_style", "")),
                ),
                environment=EnvironmentContext(
                    temperature_c=environment_data["temperature_c"],
                    humidity_percent=environment_data["humidity_percent"],
                    pressure_hpa=environment_data["pressure_hpa"],
                ),
                schedule=schedule_items,
                constraints=UserConstraints(
                    preserve_fixed_events=True,
                    avoid_medical_claims=True,
                    max_schedule_changes=3,
                    preferred_output_language="English",
                ),
            )
            try:
                recommendation = generate_voice_response(voice_payload, llm_input)
                calendar_ops_queue.put(recommendation, timeout=2)
                response_text = recommendation.adjustment_principles[0] if recommendation.adjustment_principles else "I have processed your command, but I don't have a specific recommendation to share."
                
                log("listener-voice", "Voice command handled successfully")
                return response_text
            except RuntimeError as llm_error:
                log("listener-voice", f"LLM error while processing voice command: {llm_error}")
                return "Sorry, I had trouble understanding your command. Please try again."
        except Exception as exc:
            log("listener-voice", f"Failed: {exc}")
            raise

    listener = SleepReportListener(host=host, port=port, on_report=on_report)
    listener.on_voice = on_voice
    listener.start(debug=False, threaded=True, use_reloader=False)


def start_pipeline(host: str = "0.0.0.0", port: int = 5888):
    sleep_report_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    calendar_ops_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    feedback_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()
    shared_state: dict[str, Any] = {
        "latest_recommendation": None,
        "latest_llm_input": None,
        "voice_listener_enabled": False,
        "wake_words": DEFAULT_WAKE_WORDS,
    }

    calendar_name = os.getenv("CALENDAR_NAME", DEFAULT_CALENDAR_NAME)
    feedback_port = int(os.getenv("CALENDAR_FEEDBACK_PORT", "5889"))

    threads = [
        threading.Thread(
            target=listen_daily_analysis,
            name="listener-thread",
            args=(host, port, calendar_ops_queue, sleep_report_queue, calendar_name),
            daemon=True,
        ),
        threading.Thread(
            target=llm_worker,
            name="llm-worker",
            args=(sleep_report_queue, calendar_ops_queue, stop_event, calendar_name, shared_state),
            daemon=True,
        ),
        threading.Thread(
            target=calendar_worker,
            name="calendar-worker",
            args=(calendar_ops_queue, stop_event, calendar_name, shared_state),
            daemon=True,
        ),
        threading.Thread(
            target=listen_calendar_feedback,
            name="feedback-listener-thread",
            args=(host, feedback_port, feedback_queue),
            daemon=True,
        ),
        threading.Thread(
            target=feedback_worker,
            name="feedback-worker",
            args=(feedback_queue, calendar_ops_queue, stop_event, shared_state),
            daemon=True,
        ),
    ]

    def shutdown_handler(signum, _frame):
        log("main", f"received signal={signum}, shutting down")
        stop_event.set()
        for target_queue in (sleep_report_queue, calendar_ops_queue):
            try:
                target_queue.put_nowait(SENTINEL)
            except queue.Full:
                pass

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    for worker_thread in threads:
        worker_thread.start()

    log("main", f"pipeline started at http://{host}:{port}/report")
    log("main", f"feedback listener started at http://{host}:{feedback_port}/feedback")

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        stop_event.set()
        for target_queue in (sleep_report_queue, calendar_ops_queue):
            try:
                target_queue.put_nowait(SENTINEL)
            except queue.Full:
                pass

        for worker_thread in threads[1:]:
            worker_thread.join(timeout=5)

        log("main", "pipeline stopped")


if __name__ == "__main__":
    start_pipeline()
        
