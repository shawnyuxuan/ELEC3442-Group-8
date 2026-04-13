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

from src.calendar_interaction import CalendarController
from src.daily_analysis import SleepReportListener
from src.sensehat_behavior import SenseHatController, get_cluster_profile

QUEUE_MAXSIZE = 100
WORKER_POLL_TIMEOUT_SECONDS = 1
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

# TODO: The LLM output should be structured and this method should be refactored to parse that structured output 
# instead of using a mock implementation and generating the calendar operation directly here.
def build_calendar_ops_from_llm(
    sleep_report: dict[str, Any],
    cluster: int,
    prompt: str,
    environment: dict[str, float],
    cluster_label: str,
) -> list[dict[str, Any]]:
    date_text = str(sleep_report.get("date", ""))
    total_sleep = sleep_report.get("total")
    recommendation_summary = (
        f"Sleep total: {total_sleep}h; cluster={cluster} ({cluster_label}). "
        f"Env: {environment['temperature_c']:.1f}C/{environment['humidity_percent']:.1f}%/{environment['pressure_hpa']:.1f}hPa. "
        f"Sense prompt: {prompt[:120].replace(chr(10), ' ')}"
    )

    return [
        {
            "action": "update_existing",
            "target_date": date_text,
            "new_description": recommendation_summary,
            "meta": {
                "cluster": cluster,
                "cluster_label": cluster_label,
                "environment": environment,
            },
        }
    ]


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
            cluster_label = str(cluster_profile.get("label", "Unknown"))

            controller.led_display(cluster)
            environment = {
                "temperature_c": float(controller.sensor.get_temperature()),
                "humidity_percent": float(controller.sensor.get_humidity()),
                "pressure_hpa": float(controller.sensor.get_pressure()),
            }
            prompt = controller.generate_prompt(cluster)

            calendar_ops = build_calendar_ops_from_llm(
                sleep_report=payload,
                cluster=cluster,
                prompt=prompt,
                environment=environment,
                cluster_label=cluster_label,
            )

            for operation in calendar_ops:
                calendar_ops_queue.put(operation, timeout=2)

            log(
                thread_name,
                (
                    f"processed report date={payload.get('date')} cluster={cluster}({cluster_label}) "
                    f"sleep_queue={sleep_report_queue.qsize()} ops_queue={calendar_ops_queue.qsize()}"
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

    def is_keepalive_timeout(error: Exception) -> bool:
        return "keepalive timeout" in str(error).lower()

    while True:
        if stop_event.is_set() and calendar_ops_queue.empty():
            break

        try:
            operation = calendar_ops_queue.get(timeout=WORKER_POLL_TIMEOUT_SECONDS)
        except queue.Empty:
            continue

        if operation is SENTINEL:
            calendar_ops_queue.task_done()
            log(thread_name, "received sentinel and exiting")
            break

        try:
            if operation.get("action") != "update_existing":
                log(thread_name, f"unsupported action: {operation.get('action')}")
                continue

            date_start = normalize_report_date(str(operation.get("target_date", "")))

            attempts = 2
            for attempt in range(1, attempts + 1):
                try:
                    active_controller = get_controller(force_refresh=(attempt > 1))
                    events = active_controller.fetch_events(date_start)

                    if not events:
                        log(thread_name, f"no existing events found on {date_start.date()}, skipping")
                        break

                    # Current phase only updates existing events, no new event creation.
                    active_controller.update_event(
                        event=events[0],
                        new_description=str(operation.get("new_description", "")),
                    )
                    log(thread_name, f"updated first event on {date_start.date()}")
                    break
                except Exception as exc:
                    controller = None
                    if is_keepalive_timeout(exc) and attempt < attempts:
                        log(thread_name, f"calendar session expired, rebuilding client and retrying: {exc}")
                        continue
                    raise
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

    calendar_name = os.getenv("CALENDAR_NAME", "Home")

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
            args=(sleep_report_queue, calendar_ops_queue, stop_event),
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
        