import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import json

# 加载模型
model = tf.keras.models.load_model("cnn_cloud3.h5")

# 类别标签
classes = {
    'Ci': 'cirrus',
    'Cs': 'cirrostratus',
    'Cc': 'cirrocumulus',
    'Ac': 'altocumulus',
    'As': 'altostratus',
    'Cu': 'cumulus',
    'Cb': 'cumulonimbus',
    'Ns': 'nimbostratus',
    'Sc': 'stratocumulus',
    'St': 'stratus',
    'Ct': 'contrail'
}
class_keys = list(classes.keys())

# 图片预处理
def preprocess(img):
    img = img.resize((224, 224)).convert('RGB')  # 修改为你的输入尺寸
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# 收藏功能
def load_favorites():
    if os.path.exists("favorites.json"):
        with open("favorites.json", "r") as f:
            return json.load(f)
    return []

def save_favorites(data):
    with open("favorites.json", "w") as f:
        json.dump(data, f)

# UI 部分
st.set_page_config(page_title="云朵识别", layout="centered")
st.title("☁️ 云朵识别小工具")

uploaded = st.file_uploader("上传一张云朵图片", type=['jpg', 'png'])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="上传的图片", use_column_width=True)

    if st.button("开始识别"):
        input_tensor = preprocess(image)
        pred = model.predict(input_tensor)[0]
        idx = int(np.argmax(pred))
        label_key = class_keys[idx]
        label = classes[label_key]
        confidence = float(pred[idx])

        st.success(f"识别结果：{label}（{label_key}）")
        st.write(f"置信度：{confidence*100:.2f}%")

        if st.button("⭐ 收藏这张图片"):
            favs = load_favorites()
            favs.append({
                "label": label,
                "code": label_key,
                "confidence": round(confidence, 2),
                "image_name": uploaded.name
            })
            save_favorites(favs)
            st.toast("✅ 已添加到收藏夹")

# 收藏夹
with st.expander("📂 查看收藏夹"):
    favorites = load_favorites()
    if favorites:
        for fav in favorites:
            st.write(f"🌤️ {fav['label']} ({fav['code']}) - {fav['confidence']*100:.1f}% - {fav['image_name']}")
    else:
        st.write("暂无收藏")
