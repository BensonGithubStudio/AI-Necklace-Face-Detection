from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import base64
import os
from datetime import datetime
import subprocess  # 用於執行外部程式

app = Flask(__name__)
CORS(app)  # 啟用 CORS

# 設定靜態檔案目錄 (包含 ObjectDetect.html)
STATIC_DIR = os.path.join(os.getcwd(), "website")
os.makedirs(STATIC_DIR, exist_ok=True)

# 根路徑處理器
@app.route("/")
def serve_object_detect():
    # 提供 ObjectDetect.html
    return send_from_directory(STATIC_DIR, "ObjectDetect.html")

# 設定圖片存儲目錄
SAVE_DIR = "test"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    data = request.json
    if not data or "image" not in data:
        return "No image data received", 400

    # 提取 base64 圖片數據
    image_data = data["image"].split(",")[1]
    image_binary = base64.b64decode(image_data)

    # 使用時間戳命名文件
    filename = datetime.now().strftime("photo_%Y-%m-%d_%H-%M-%S.png")
    filepath = os.path.join(SAVE_DIR, filename)

    # 將圖片保存到本地
    with open(filepath, "wb") as f:
        f.write(image_binary)

    # 呼叫另一個 Python 程式
    try:
        print(f"Running script/detection.py with {filepath}")
        subprocess.run(["python", "script/detection.py", filepath], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return jsonify({"message": "Image saved, but processing failed", "filepath": filepath})

    return jsonify({"message": "Image saved", "filepath": filepath})

if __name__ == "__main__":
    app.run(debug=True)