from flask import Flask, request, jsonify
import json
from datetime import datetime
import os

app = Flask(__name__)
data_path = os.path.join(os.path.dirname(__file__), "..", "daily_data")
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
@app.route('/report', methods=['POST'])
def receive_sleep_data():
    data = request.json
    if not data:
        return "No data", 400

    # 提取各阶段数值
    date = data.get("date")
    core = data.get("core", 0)
    deep = data.get("deep", 0)
    rem = data.get("rem", 0)
    total = round(float(core) + float(deep) + float(rem), 2)

    print(f"--- 收到 {date} 的睡眠报告 ---")
    print(f"浅睡(Core): {core}h")
    print(f"深睡(Deep): {deep}h")
    print(f"REM: {rem}h")
    print(f"总计时长: {total}h")

    # 保存到本地文件 (CSV格式，方便后续分析)
    with open(os.path.join(data_path, "sleep_data.csv"), "a") as f:
        f.write(f"{date},{core},{deep},{rem},{total}\n")

    return jsonify({"status": "success", "total_calculated": total}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5888)