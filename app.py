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

def preprocess(img):
    img = img.resize((100, 100)).convert('RGB')  # 将尺寸修改为 (100, 100)
    arr = np.array(img) / 255.0  # 归一化
    return np.expand_dims(arr, axis=0)  # 添加批次维度

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
st.set_page_config(page_title="云朵识别", page_icon="☁️", layout="wide")
st.title("☁️ 云朵识别小工具")

uploaded = st.file_uploader("上传一张云朵图片", type=['jpg', 'png'])

st.markdown("""
<style>
.big-font {
    font-size:30px;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">欢迎使用云朵识别工具！</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("上传一张云朵图片", type=['jpg', 'png'])

with col2:
    if uploaded:
        st.image(uploaded, caption="上传的图片", use_container_width=True)

with st.spinner('正在识别...'):
    pred = model.predict(input_tensor)[0]




if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="上传的图片", use_container_width=True)

    if st.button("开始识别"):
        input_tensor = preprocess(image)
        
    st.toast("✅ 已添加到收藏夹")
        
        # 打印输入张量的形状，查看是否与模型的要求匹配
    st.write(f"Input tensor shape: {input_tensor.shape}")  # 打印输入张量的形状

    if st.button("⭐ 收藏这张图片"):
        favs = load_favorites()
        favs.append({
           "label": label,
           "code": label_key,
           "confidence": round(confidence, 2),
           "image_name": uploaded.name
    })
    save_favorites(favs)

        
# 进行预测
try:
    pred = model.predict(input_tensor)[0]  # 获取预测结果
    idx = int(np.argmax(pred))  # 获取最大概率的索引
    label_key = class_keys[idx]  # 获取类别的键
    label = classes[label_key]  # 获取类别名称
    confidence = float(pred[idx])  # 获取置信度

    # 显示预测结果和置信度
    st.success(f"识别结果：{label}（{label_key}）")
    st.write(f"置信度：{confidence*100:.2f}%")
except Exception as e:
    # 捕获错误并显示
    st.error(f"发生错误: {str(e)}")


with st.expander("📂 查看收藏夹"):
    favorites = load_favorites()
    if favorites:
        for fav in favorites:
            st.write(f"🌤️ {fav['label']} ({fav['code']}) - {fav['confidence']*100:.1f}% - {fav['image_name']}")
    else:
        st.write("暂无收藏")
