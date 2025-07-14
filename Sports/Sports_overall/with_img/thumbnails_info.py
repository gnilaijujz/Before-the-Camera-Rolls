import os
import cv2
import pandas as pd
import requests
import pytesseract
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# 可选: 使用 DeepFace 检测人脸
# from deepface import DeepFace

# 设置你的 API keys（用于其它扩展功能，如分析视频信息）
API_KEYS = ['AIzaSyC4FzQ3jVTBKanstgyKdRFj4qp0CDOrIaQ',
    'AIzaSyDMVEJwXu4Rs3amiDIk44QyppzWO_NF0u4',
    'AIzaSyC33-8fcs9okEn8qJIbM3K_oQ4GsJhG75o']

# ============ 封面分析函数 ============

def get_thumbnail_url(video_id):
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    return False

def detect_face_opencv(img_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def detect_text(img_path):
    img = cv2.imread(img_path)
    text = pytesseract.image_to_string(img)
    return len(text.strip()) > 10

def get_dominant_color(img_path, k=3):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))  # 降低计算量
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img)
    dominant = kmeans.cluster_centers_[0].astype(int)
    return f"rgb({dominant[2]},{dominant[1]},{dominant[0]})"  # 转成RGB格式

# ============ 主程序 ============

# 读取视频ID
df = pd.read_csv("sports_2024-2025_with_popularity.csv")
video_ids = df["video_id"].dropna().unique()

# 创建结果表
results = []

# 创建保存目录
os.makedirs("thumbnails", exist_ok=True)

# 分析每个视频
for vid in tqdm(video_ids):
    try:
        url = get_thumbnail_url(vid)
        save_path = f"thumbnails/{vid}.jpg"

        if not os.path.exists(save_path):
            download_image(url, save_path)

        has_face = detect_face_opencv(save_path)
        has_text = detect_text(save_path)
        dominant_color = get_dominant_color(save_path)

        results.append({
            "video_id": vid,
            "has_face": has_face,
            "has_text": has_text,
            "dominant_color": dominant_color
        })
    except Exception as e:
        print(f"Error processing {vid}: {e}")

# 保存为 DataFrame
result_df = pd.DataFrame(results)

# 合并原始表
merged_df = df.merge(result_df, on="video_id", how="left")
merged_df.to_csv("sports_with_thumbnails_info.csv", index=False)

print("✅ 封面分析完成，已保存为 sports_with_thumbnails_info.csv")
