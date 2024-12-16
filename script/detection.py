import sys
import cv2
import numpy as np
from ultralytics import YOLO

def resize_image(image, max_width=800, max_height=600):
    """縮放圖片以限制最大尺寸"""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)  # 確保縮放比例不會放大
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def wrap_text(text, max_width, font, font_scale, thickness):
    """將文字分行以適應指定的寬度"""
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        # 測量該行的寬度
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def main(image_path):
    try:
        # 確保所有舊視窗被關閉
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # 等待一個短暫時間確保視窗關閉
        
        # 載入 YOLO 模型
        model = YOLO("runs/detect/train/weights/best.pt")

        # 推論圖片
        results = model(image_path)

        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            return

        # 縮放圖片限制最大大小
        image = resize_image(image)

        # 建立一個文字結果清單
        detection_texts = []

        # 遍歷檢測結果，繪製方框和標籤
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, score, cls = box[:6]
                if score < 0.1:
                    label = "no detection"
                else:
                    label = model.names[int(cls)]

                label_text = f"{label}: {score:.2f}" if score >= 0.1 else "no detection"
                detection_texts.append(
                    f"recognition frame: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | {label_text}"
                )
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if not detection_texts:
            detection_texts.append("no object!")

        # 建立一個空白的文字結果圖像
        max_text_width = image.shape[1] - 20  # 限制文字的寬度
        text_image_height = 300
        text_image = 255 * np.ones((text_image_height, image.shape[1], 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        y_offset = 40  # 初始文字的垂直偏移
        line_spacing = 40  # 行間距
        for text in detection_texts:
            wrapped_lines = wrap_text(text, max_text_width, font, font_scale, thickness)
            for line in wrapped_lines:
                cv2.putText(
                    text_image,
                    line,
                    (10, y_offset),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness
                )
                y_offset += line_spacing
                if y_offset > text_image_height - 10:  # 超過文字區域高度則停止
                    break

        # 合併圖片與文字圖像（垂直方向）
        combined_image = np.vstack((image, text_image))

        # 使用統一視窗名稱
        window_name = "Detection Results"

        # 顯示合併結果
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, combined_image)
        cv2.waitKey(0)  # 等待用戶輸入
        cv2.destroyAllWindows()  # 最後關閉視窗

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path>")
    else:
        image_path = sys.argv[1]
        main(image_path)
