import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image

# モデルをロード
@st.cache_resource
def load_model():
    return YOLO("./best_1024_E6_150epoch.pt")  # モデルパスを調整

# 抵抗値マッピング
resistance_map = {
    "0": '100', "1": '150',"2": '220',"3": '330',"4": '470',"5": '680',"6": "1k","7": "1.5k",
    "8": "2.2k","9": "3.3k","10": "4.7k","11": "6.8k","12": "10k","13": "15k","14": "22k",
    "15": "33k","16": "47k","17": "68k","18": "100k"
}

st.title("リアルタイム炭素被膜抵抗識別")

# モデルをロード
model = load_model()

# カメラ入力
frame = st.camera_input("カメラで抵抗を映してください")
if frame is not None:
    image = Image.open(frame)
    img_array = np.array(image)
    results = model.predict(source=img_array, imgsz=1024, conf=0.3)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            resistance = resistance_map.get(str(cls + 23), f"不明 (タグ: {cls + 23})")
            st.write(f"抵抗: {resistance}, スコア: {score:.2f}, 位置: [{x1}, {y1}, {x2}, {y2}]")
