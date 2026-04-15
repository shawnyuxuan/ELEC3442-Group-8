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
import traceback
import pandas as pd
import queue
import signal
import threading
import time
from typing import Any

from src.calendar_interaction import CalendarController, DEFAULT_CALENDAR_NAME
from src.daily_analysis import SleepReportListener
from src.llm_contract import CalendarOperation, ScheduleRecommendation
from src.llm_interaction import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    EnvironmentContext,
    LLMInput,
    ScheduleItem,
    SleepAssessment,
    UserConstraints,
    generate_schedule_recommendation,
)
from src.schedule_rules import schedule_item_from_calendar_event
from src.sensehat_behavior import SenseHatController, get_cluster_profile, get_mock_schedule

QUEUE_MAXSIZE = 100
WORKER_POLL_TIMEOUT_SECONDS = 1
CALENDAR_READ_MAX_ATTEMPTS = 3
CALENDAR_READ_RETRY_SECONDS = 2
SENTINEL = object()
USER_NAME = os.getenv("USER_NAME", "Jayden")

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


def calendar_worker(calendar_ops_queue: queue.Queue, stop_event: threading.Event, calendar_name: str):
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
                status_icon = "✓" if status == "ok" else "~" if status == "no_update" else "✗"
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
                        if preflight_status == "created" or preflight_status == "modified":
                            inverse_op = preflight_results[idx].get("inverse_operation")
                            if inverse_op:
                                inverse_ops.append(inverse_op)
                        
                        applied_count += 1
                        log(thread_name, f"   ✓ Applied: {op_label}")
                        break
                        
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
                log(thread_name, f"✓ COMPLETE: No calendar changes needed for {recommendation.date}")
            else:
                log(thread_name, f"✓ COMPLETE: Applied {applied_count} ops, "
                    f"skipped {skipped_count} no_updates on {recommendation.date}")

        except Exception as exc:
            log(thread_name, f"✗ FAILED: Error processing recommendation: {exc}")
            traceback.print_exc()
            
        finally:
            calendar_ops_queue.task_done()

    log(thread_name, "stopped")


def listen_daily_analysis(host: str, port: int, sleep_report_queue: queue.Queue):
    def on_report(sleep_data: dict[str, Any]):
        try:
            sleep_report_queue.put(dict(sleep_data), timeout=2)
            log("listener", f"enqueued sleep report date={sleep_data.get('date')} queue={sleep_report_queue.qsize()}")
        except queue.Full:
            log("listener", "sleep report queue is full; report dropped")

    listener = SleepReportListener(host=host, port=port, on_report=on_report)
    listener.start(debug=False, threaded=True, use_reloader=False)


def start_pipeline(host: str = "0.0.0.0", port: int = 5888):
    sleep_report_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    calendar_ops_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()

    calendar_name = os.getenv("CALENDAR_NAME", DEFAULT_CALENDAR_NAME)

    threads = [
        threading.Thread(
            target=listen_daily_analysis,
            name="listener-thread",
            args=(host, port, sleep_report_queue),
            daemon=True,
        ),
        threading.Thread(
            target=llm_worker,
            name="llm-worker",
            args=(sleep_report_queue, calendar_ops_queue, stop_event, calendar_name),
            daemon=True,
        ),
        threading.Thread(
            target=calendar_worker,
            name="calendar-worker",
            args=(calendar_ops_queue, stop_event, calendar_name),
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
        
