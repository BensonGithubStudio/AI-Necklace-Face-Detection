from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 啟用 CORS

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

    return jsonify({"message": "Image saved", "filepath": filepath})

if __name__ == "__main__":
    app.run(debug=True)