import json
import os
from typing import Callable

from flask import Flask, jsonify, request


class CalendarFeedbackListener:
    def __init__(
        self,
        host="0.0.0.0",
        port=5889,
        data_dir=None,
        data_file="calendar_feedback.jsonl",
        on_feedback: Callable[[dict], None] | None = None,
    ):
        self.host = host
        self.port = port
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "daily_data")
        self.data_file_path = os.path.join(self.data_dir, data_file)
        self.on_feedback = on_feedback

        os.makedirs(self.data_dir, exist_ok=True)

        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule("/feedback", view_func=self.receive_feedback, methods=["POST"])

    def process_feedback_payload(self, payload):
        date = payload.get("date")
        if not date:
            raise ValueError("Missing required field: date")

        transcript = str(payload.get("transcript", "")).strip()
        if not transcript:
            raise ValueError("Missing required field: transcript")

        source = str(payload.get("source", "microphone")).strip() or "microphone"
        language = str(payload.get("language", "en-US")).strip() or "en-US"
        captured_at = str(payload.get("captured_at", "")).strip()

        context = payload.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError("context must be an object when provided")

        normalized = {
            "date": date,
            "transcript": transcript,
            "source": source,
            "language": language,
            "captured_at": captured_at,
            "context": context or {},
        }

        print(f"--- 收到 {date} 的语音反馈 ---")
        print(f"来源: {source}")
        print(f"语言: {language}")
        print(f"转写: {transcript}")

        return normalized

    def _append_to_jsonl(self, feedback_data):
        with open(self.data_file_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")

    def receive_feedback(self):
        payload = request.get_json(silent=True)
        if not payload:
            return "No data", 400

        try:
            feedback_data = self.process_feedback_payload(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        self._append_to_jsonl(feedback_data)

        if self.on_feedback:
            try:
                self.on_feedback(dict(feedback_data))
            except Exception as exc:
                print(f"[feedback-listener] failed to dispatch feedback callback: {exc}")

        return jsonify({"status": "success"}), 200

    def start(self, debug=False, threaded=True, use_reloader=False):
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=threaded,
            use_reloader=use_reloader,
        )


_default_listener = CalendarFeedbackListener()
app = _default_listener.app


if __name__ == "__main__":
    _default_listener.start()
