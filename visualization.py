import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 读取 CSV 文件
df = pd.read_csv('sports_videos.csv')

# 🧹 如果存在缺失值，先清理（保险处理）
df = df.dropna(subset=['duration_seconds'])

# 🎯 1. 时长分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(df['duration_seconds'], bins=30, kde=True, color='skyblue')
plt.title('Video Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of Videos')
plt.grid(True)
plt.tight_layout()

# ✅ 保存为图片
plt.savefig('video_duration_distribution.png')
print("✅ 已保存：video_duration_distribution.png")
plt.close()  # 关闭当前图

# 🎯 2. 按时长分类（柱状图）
bins = [0, 60, 180, 600, 1800, 3600, 10000]
labels = ['<1min', '1-3min', '3-10min', '10-30min', '30-60min', '60min+']
df['duration_category'] = pd.cut(df['duration_seconds'], bins=bins, labels=labels)

duration_counts = df['duration_category'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(x=duration_counts.index, y=duration_counts.values, palette='Set2')
plt.title('Video Duration Categories')
plt.xlabel('Duration Range')
plt.ylabel('Number of Videos')
plt.grid(True, axis='y')
plt.tight_layout()

# ✅ 保存为图片
plt.savefig('video_duration_categories.png')
print("✅ 已保存：video_duration_categories.png")
plt.close()
