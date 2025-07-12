import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.style.use('ggplot')  # 使用ggplot风格，美化图表

# 读取数据
df = pd.read_csv("D:\下载\sports_2024-2025_with_popularity.csv")

# 将秒转换为分钟
df['duration_minutes'] = df['duration_seconds'] / 60

# 2) 时长分布直方图
plt.figure(figsize=(10, 6))
ax = sns.histplot(df['duration_minutes'], bins=30, kde=True, color='steelblue')
plt.title('视频时长分布', fontsize=16, fontweight='bold')
plt.xlabel('时长（分钟）', fontsize=12)
plt.ylabel('视频数量', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3) 使用KMeans进行聚类
# 准备数据，只使用时长特征进行聚类
X = df[['duration_minutes']].values
# 由于KMeans对尺度敏感，我们进行标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用KMeans进行聚类，分为3类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 获取聚类中心（转换回原始尺度）
centers = scaler.inverse_transform(kmeans.cluster_centers_)
# 按照聚类中心的大小排序类别
center_order = np.argsort(centers.flatten())
mapping = {center_order[0]: 0, center_order[1]: 1, center_order[2]: 2}
df['cluster_ordered'] = df['cluster'].map(mapping)

# 为每个类别设置名称
cluster_names = ['短视频', '中等视频', '长视频']
df['cluster_name'] = df['cluster_ordered'].map({0: cluster_names[0], 
                                               1: cluster_names[1], 
                                               2: cluster_names[2]})

# 统计每个聚类的数量和时长范围
cluster_stats = df.groupby('cluster_name').agg(
    视频数量=('cluster_name', 'count'),
    最小时长_分钟=('duration_minutes', 'min'),
    最大时长_分钟=('duration_minutes', 'max'),
    平均时长_分钟=('duration_minutes', 'mean'),
    中位时长_分钟=('duration_minutes', 'median'),
)

# 按照短视频、中等视频、长视频的顺序排序
cluster_stats = cluster_stats.reindex(cluster_names)

print("\nKMeans聚类结果 - 按视频时长将视频分为三类：")
print(cluster_stats)

# 为了更直观地展示，创建标签
category_labels = [f"{name}\n({cluster_stats.loc[name, '最小时长_分钟']:.1f}~{cluster_stats.loc[name, '最大时长_分钟']:.1f}分钟)\n平均: {cluster_stats.loc[name, '平均时长_分钟']:.1f}分钟" 
                  for name in cluster_names]

# 绘制分类柱状图，使用更好看的配色
plt.figure(figsize=(12, 8))
colors = ['#3274A1', '#E1812C', '#3A923A']
bars = plt.bar(category_labels, cluster_stats['视频数量'], color=colors, width=0.6)

# 添加标题和标签
plt.title('视频按时长的KMeans聚类结果', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('视频时长分类', fontsize=14, labelpad=10)
plt.ylabel('视频数量', fontsize=14, labelpad=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# 在柱状图上标注具体数量及百分比
total_videos = cluster_stats['视频数量'].sum()
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = 100 * height / total_videos
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}个 ({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加图例说明
plt.figtext(0.5, 0.01, 
           '注: 基于视频时长特征使用KMeans算法进行聚类，将视频分为三类', 
           ha='center', fontsize=10, style='italic')

# 美化图表
plt.xticks(fontsize=11)
plt.yticks(fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 为底部注释留出空间

plt.show()

# 4) 可视化聚类结果 - 散点图
plt.figure(figsize=(12, 6))
# 绘制散点图，x轴为视频索引，y轴为视频时长，颜色区分聚类
scatter = plt.scatter(range(len(df)), df['duration_minutes'], 
                     c=df['cluster_ordered'], cmap='viridis', 
                     alpha=0.6, s=30, edgecolors='w')

# 添加聚类中心线
for i, center in enumerate(centers):
    plt.axhline(y=center[0], color=plt.cm.viridis(i/3), linestyle='--', 
               linewidth=2, label=f'聚类中心 {cluster_names[mapping[i]]}: {center[0]:.1f}分钟')

plt.title('视频时长聚类分布', fontsize=16, fontweight='bold')
plt.xlabel('视频索引', fontsize=14)
plt.ylabel('视频时长（分钟）', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5) 可视化聚类结果 - 时长分布图
plt.figure(figsize=(12, 6))

# 为每个聚类创建子图
for i, name in enumerate(cluster_names):
    subset = df[df['cluster_name'] == name]
    sns.kdeplot(subset['duration_minutes'], label=name, fill=True, alpha=0.3)

plt.title('各聚类视频时长分布', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('密度', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='视频类别')
plt.tight_layout()
plt.show()
