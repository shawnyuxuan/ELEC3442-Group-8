# The agent file creates a pipeline that
# 1. listens for daily sleep reports and processes them using the daily-analysis module.
# 2. uses the processed data to interact with LLM for insights and recommendations.
# 3. analyze the LLM feedback and make modifications to the user's calendar for better sleep hygiene.

import datetime
import math
import os
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

# Build the same structured LLM input contract used by llm_interaction.py so the
# calendar pipeline and standalone script share one output format.
def build_llm_input_from_report(
    sleep_report: dict[str, Any],
    cluster_profile: dict[str, Any],
    environment: dict[str, float],
    calendar_name: str,
) -> LLMInput:
    date_text = str(sleep_report.get("date", "")) or datetime.datetime.now().date().isoformat()
    schedule = build_schedule_from_calendar(calendar_name=calendar_name, date_text=date_text)

    return LLMInput(
        schedule_date=date_text,
        user_name="Jayden",
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL,
        sleep_assessment=SleepAssessment(
            cluster_id=int(cluster_profile.get("cluster_id", -1)),
            cluster_label=str(cluster_profile.get("label", "Unknown")),
            summary=str(cluster_profile.get("summary", "")),
            likely_risks=[str(item) for item in cluster_profile.get("likely_risks", [])],
            recommended_work_style=str(cluster_profile.get("recommended_work_style", "")),
        ),
        environment=EnvironmentContext(
            temperature_c=float(environment["temperature_c"]),
            humidity_percent=float(environment["humidity_percent"]),
            pressure_hpa=float(environment["pressure_hpa"]),
        ),
        schedule=schedule,
        constraints=UserConstraints(
            preserve_fixed_events=True,
            avoid_medical_claims=True,
            max_schedule_changes=3,
            preferred_output_language="English",
        ),
    )


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
    thread_name = threading.current_thread().name
    log(thread_name, "started")

    sensehat_controller: SenseHatController | None = None

    def get_sensehat_controller(force_refresh: bool = False) -> SenseHatController:
        nonlocal sensehat_controller
        if force_refresh or sensehat_controller is None:
            model_path = os.getenv("SLEEP_MODEL_PATH")
            sensehat_controller = SenseHatController(model_path=model_path)
        return sensehat_controller

    while True:
        if stop_event.is_set() and sleep_report_queue.empty():
            break

        try:
            payload = sleep_report_queue.get(timeout=WORKER_POLL_TIMEOUT_SECONDS)
        except queue.Empty:
            continue

        if payload is SENTINEL:
            sleep_report_queue.task_done()
            log(thread_name, "received sentinel and exiting")
            break

        try:
            features = build_sleep_features_from_report(payload)
            controller = get_sensehat_controller()

            cluster = int(controller.predict_sleep_quality(features))
            cluster_profile = get_cluster_profile(cluster) or {}
            cluster_profile["cluster_id"] = cluster
            cluster_label = str(cluster_profile.get("label", "Unknown"))
            #FIXME: remove print and directly log to sensehat in the future after confirming the pipeline is stable.
            print(f"""
--- Sleep Report Analysis ---
Predicted Sleep Quality Cluster: {cluster} ({cluster_label})
Summary: {cluster_profile.get("summary", "N/A")}
Likely Risks: {', '.join(cluster_profile.get("likely_risks", []))}
Recommended Work Style: {cluster_profile.get("recommended_work_style", "N/A")}
--- Ready for LED Display and LLM Input Construction ---
""")
            controller.led_display(cluster)
            environment = {
                "temperature_c": float(controller.sensor.get_temperature()),
                "humidity_percent": float(controller.sensor.get_humidity()),
                "pressure_hpa": float(controller.sensor.get_pressure()),
            }
            llm_input = build_llm_input_from_report(
                sleep_report=payload,
                cluster_profile=cluster_profile,
                environment=environment,
                calendar_name=calendar_name,
            )
            recommendation = generate_schedule_recommendation(llm_input)

            calendar_ops_queue.put(recommendation, timeout=2)

            log(
                thread_name,
                (
                    f"processed report date={payload.get('date')} cluster={cluster}({cluster_label}) "
                    f"sleep_queue={sleep_report_queue.qsize()} ops_queue={calendar_ops_queue.qsize()} "
                    f"llm_ops={len(recommendation.calendar_operations)}"
                ),
            )
        except Exception as exc:
            sensehat_controller = None
            log(thread_name, f"failed to process report: {exc}")
        finally:
            sleep_report_queue.task_done()

    log(thread_name, "stopped")


def calendar_worker(calendar_ops_queue: queue.Queue, stop_event: threading.Event, calendar_name: str):
    thread_name = threading.current_thread().name
    log(thread_name, f"starting with calendar={calendar_name}")

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
            return f"no_update ({operation.reason})"

        target = operation.target
        updated = operation.updated
        target_title = target.title if target is not None else "<missing-target>"
        target_start = target.start if target is not None else "??:??"
        target_end = target.end if target is not None else "??:??"
        updated_start = updated.start if updated is not None else "??:??"
        updated_end = updated.end if updated is not None else "??:??"
        return f"{target_title} {target_start}-{target_end} -> {updated_start}-{updated_end}"

    def rollback_operations(
        date_start: datetime.datetime,
        inverse_operations: list[CalendarOperation],
    ):
        nonlocal controller
        if not inverse_operations:
            return

        log(thread_name, f"starting rollback for {len(inverse_operations)} operations")
        for inverse_operation in reversed(inverse_operations):
            operation_label = format_operation_label(inverse_operation)
            for attempt in range(1, 4):
                try:
                    active_controller = get_controller(force_refresh=(attempt > 1))
                    active_controller.apply_calendar_operation(date_start, inverse_operation)
                    log(thread_name, f"rolled back operation {operation_label}")
                    break
                except Exception as exc:
                    controller = None
                    if is_retryable_calendar_error(exc) and attempt < 3:
                        log(
                            thread_name,
                            (
                                f"rollback retry {attempt}/3 for {operation_label}: {exc}"
                            ),
                        )
                        continue
                    log(thread_name, f"rollback failed for {operation_label}: {exc}")
                    break

    while True:
        if stop_event.is_set() and calendar_ops_queue.empty():
            break

        try:
            recommendation = calendar_ops_queue.get(timeout=WORKER_POLL_TIMEOUT_SECONDS)
        except queue.Empty:
            continue

        if recommendation is SENTINEL:
            calendar_ops_queue.task_done()
            log(thread_name, "received sentinel and exiting")
            break

        try:
            if not isinstance(recommendation, ScheduleRecommendation):
                log(thread_name, f"unsupported recommendation payload: {type(recommendation).__name__}")
                continue

            date_start = datetime.datetime.combine(
                datetime.datetime.strptime(recommendation.date, "%Y-%m-%d").date(),
                datetime.time.min,
            )
            attempts = 3
            applied_operations = 0
            skipped_no_update_operations = 0
            inverse_operations: list[CalendarOperation] = []

            active_controller = get_controller(force_refresh=True)
            preflight = active_controller.preflight_operations(
                date_start,
                recommendation.calendar_operations,
            )
            for resolved, operation in zip(preflight, recommendation.calendar_operations):
                operation_label = format_operation_label(operation)
                log(
                    thread_name,
                    f"preflight {resolved['status']} for {operation_label}",
                )

            for index, operation in enumerate(recommendation.calendar_operations):
                operation_label = format_operation_label(operation)
                for attempt in range(1, attempts + 1):
                    try:
                        active_controller = get_controller(force_refresh=(attempt > 1))
                        active_controller.apply_calendar_operation(date_start, operation)
                        if preflight[index]["status"] == "no_update":
                            skipped_no_update_operations += 1
                            log(thread_name, f"no calendar update required for {operation_label}")
                        elif preflight[index]["status"] != "updated":
                            inverse_operation = preflight[index]["inverse_operation"]
                            if inverse_operation is not None:
                                inverse_operations.append(inverse_operation)
                            applied_operations += 1
                            log(thread_name, f"applied operation {operation_label}")
                        else:
                            log(thread_name, f"operation already applied {operation_label}")
                        break
                    except Exception as exc:
                        controller = None
                        if is_retryable_calendar_error(exc) and attempt < attempts:
                            log(
                                thread_name,
                                (
                                    f"calendar operation retry {attempt}/{attempts} for "
                                    f"{operation_label}: {exc}"
                                ),
                            )
                            continue
                        rollback_operations(date_start, inverse_operations)
                        raise

            if recommendation.calendar_operations and skipped_no_update_operations == len(recommendation.calendar_operations):
                log(
                    thread_name,
                    f"recommendation for {recommendation.date} required no calendar changes",
                )
            elif recommendation.calendar_operations:
                log(
                    thread_name,
                    (
                        f"applied {applied_operations} operations, skipped "
                        f"{skipped_no_update_operations} no_update operations on {recommendation.date}"
                    ),
                )
            else:
                log(
                    thread_name,
                    f"recommendation for {recommendation.date} required no calendar changes",
                )
        except Exception as exc:
            log(thread_name, f"calendar update failed: {exc}")
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
        
