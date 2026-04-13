from flask import Flask, request, jsonify
import os
from typing import Callable


class SleepReportListener:
    def __init__(
        self,
        host="0.0.0.0",
        port=5888,
        data_dir=None,
        data_file="sleep_data.csv",
        on_report: Callable[[dict], None] | None = None,
    ):
        self.host = host
        self.port = port
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "daily_data")
        self.data_file_path = os.path.join(self.data_dir, data_file)
        self.on_report = on_report

        os.makedirs(self.data_dir, exist_ok=True)

        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule("/report", view_func=self.receive_sleep_data, methods=["POST"])

    def _to_float(self, value, field_name):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for '{field_name}': {value}") from exc

    def process_sleep_payload(self, payload):
        date = payload.get("date")
        if not date:
            raise ValueError("Missing required field: date")

        core = self._to_float(payload.get("core", 0), "core")
        deep = self._to_float(payload.get("deep", 0), "deep")
        rem = self._to_float(payload.get("rem", 0), "rem")
        total = round(core + deep + rem, 2)

        print(f"--- 收到 {date} 的睡眠报告 ---")
        print(f"浅睡(Core): {core}h")
        print(f"深睡(Deep): {deep}h")
        print(f"REM: {rem}h")
        print(f"总计时长: {total}h")

        return {
            "date": date,
            "core": core,
            "deep": deep,
            "rem": rem,
            "total": total,
        }

    def _append_to_csv(self, sleep_data):
        with open(self.data_file_path, "a", encoding="utf-8") as file:
            file.write(
                f"{sleep_data['date']},{sleep_data['core']},{sleep_data['deep']},{sleep_data['rem']},{sleep_data['total']}\n"
            )

    def receive_sleep_data(self):
        payload = request.get_json(silent=True)
        if not payload:
            return "No data", 400

        try:
            sleep_data = self.process_sleep_payload(payload)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        self._append_to_csv(sleep_data)

        if self.on_report:
            try:
                self.on_report(dict(sleep_data))
            except Exception as exc:
                print(f"[listener] failed to dispatch sleep report callback: {exc}")

        return jsonify({"status": "success", "total_calculated": sleep_data["total"]}), 200

    def start(self, debug=False, threaded=True, use_reloader=False):
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=threaded,
            use_reloader=use_reloader,
        )


# Keep global app for backward compatibility with existing imports.
_default_listener = SleepReportListener()
app = _default_listener.app


if __name__ == "__main__":
    _default_listener.start()