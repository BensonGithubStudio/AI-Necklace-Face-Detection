import sys
import cv2
import numpy as np
from ultralytics import YOLO

def main(image_path):
    # 檢查路徑是否存在
    try:
        # 載入 YOLO 模型
        model = YOLO("runs/detect/train/weights/best.pt")

        # 推論圖片
        results = model(image_path)

        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            return

        # 建立一個文字結果清單
        detection_texts = []

        # 獲取圖片尺寸
        image_height, image_width = image.shape[:2]

        # 遍歷檢測結果，繪製方框和標籤
        for result in results:
            for box in result.boxes.data:  # 獲取每個檢測框
                x1, y1, x2, y2, score, cls = box[:6]
                if score < 0.1:  # 設定較低信心閾值
                    label = "no detection"
                else:
                    label = model.names[int(cls)]

                label_text = f"{label}: {score:.2f}" if score >= 0.1 else "no detection"
                detection_texts.append(
                    f"recognition frame: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | {label_text}"
                )
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 確保檢測文字清單非空
        if not detection_texts:
            detection_texts.append("no object!")

        # 在新視窗顯示所有檢測標籤文字
        text_image = 255 * np.ones((400, 800, 3), dtype=np.uint8)
        y_offset = 20
        for text in detection_texts:
            print(f"Adding text: {text}")  # 調試用
            cv2.putText(text_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 20

        # 顯示圖片與文字視窗
        cv2.imshow("Detection", image)
        cv2.imshow("Detection Texts", text_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path>")
    else:
        image_path = sys.argv[1]
        main(image_path)
