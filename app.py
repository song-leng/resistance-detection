import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# モデルをロード
@st.cache_resource
def load_model():
    try:
        model = YOLO("./best.pt")  # ローカルパスを調整
        st.write("モデルをロードしました！")
        return model
    except Exception as e:
        st.error(f"モデルロードエラー: {str(e)}")
        return None

# メインアプリ
st.title("リアルタイム炭素被膜抵抗識別")

# モデルをロード
model = load_model()
if model is None:
    st.stop()

# カメラ入力
frame = st.camera_input("カメラで抵抗を映してください")

if frame is not None:
    # フレームをOpenCV形式に変換
    image = Image.open(frame)
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # StreamlitはRGB、OpenCVはBGR

    # YOLOで物体検出
    try:
        results = model.predict(source=img_array, imgsz=640, conf=0.3)  # 入力サイズを640に調整
        annotated_frame = results[0].plot()  # 検出結果を描画

        # BGRをRGBに変換してStreamlitで表示
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="検出結果", channels="RGB", use_column_width=True)
    except Exception as e:
        st.error(f"推論エラー: {str(e)}")
else:
    st.write("カメラからフレームを取得してください")