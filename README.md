# AI-Necklace-Face-Detection
Use "Label Studio" and "Face-api.js" to detection user's necklace and face.


## 模型訓練

``` bash
pip install label-studio
label-studio start
```

``` bash
yolo task=detect mode=train model=yolov8n.pt data=AI-Necklace-Face-Detection/data.yaml epochs=30 imgsz=580
```

## 安裝 python 服務

``` bash
pip install opencv-python
pip install opencv-contrib-python
pip install ultralytics
pip install flask
pip install flask-cors
```

## 啟動 python 環境

``` bash
cd AI-Necklace-Face-Detection
python server.py
```

## 啟動 Face-api.js 環境

``` bash
cd face-api.js/examples/examples-browser
npm i
npm start
```
