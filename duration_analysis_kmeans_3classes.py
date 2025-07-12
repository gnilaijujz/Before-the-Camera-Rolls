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

# 1) 时长分布直方图 - 总体分布
plt.figure(figsize=(10, 6))
ax = sns.histplot(df['duration_minutes'], bins=30, kde=True, color='steelblue')
plt.title('视频时长总体分布', fontsize=16, fontweight='bold')
plt.xlabel('时长（分钟）', fontsize=12)
plt.ylabel('视频数量', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2) 使用KMeans进行聚类，分为3类
# 准备数据，只使用时长特征进行聚类
X = df[['duration_minutes']].values
# 由于KMeans对尺度敏感，我们进行标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用KMeans进行聚类，分为3类
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 获取聚类中心（转换回原始尺度）
centers = scaler.inverse_transform(kmeans.cluster_centers_)
# 按照聚类中心的大小排序类别
center_order = np.argsort(centers.flatten())
mapping = {center_order[i]: i for i in range(n_clusters)}
df['cluster_ordered'] = df['cluster'].map(mapping)

# 为每个类别设置名称
cluster_names = ['短视频', '中长视频', '长视频']
df['cluster_name'] = df['cluster_ordered'].map({i: cluster_names[i] for i in range(n_clusters)})

# 统计每个聚类的数量和时长范围
cluster_stats = df.groupby('cluster_name').agg(
    视频数量=('cluster_name', 'count'),
    最小时长_分钟=('duration_minutes', 'min'),
    最大时长_分钟=('duration_minutes', 'max'),
    平均时长_分钟=('duration_minutes', 'mean'),
    中位时长_分钟=('duration_minutes', 'median'),
)

# 按照短、中长、长视频的顺序排序
cluster_stats = cluster_stats.reindex(cluster_names)

print("\nKMeans聚类结果 - 按视频时长将视频分为三类：")
print(cluster_stats)

# 3) 绘制聚类结果柱状图
# 为了更直观地展示，创建标签
category_labels = [f"{name}\n({cluster_stats.loc[name, '最小时长_分钟']:.1f}~{cluster_stats.loc[name, '最大时长_分钟']:.1f}分钟)\n平均: {cluster_stats.loc[name, '平均时长_分钟']:.1f}分钟" 
                  for name in cluster_names]

# 绘制分类柱状图，使用更好看的配色
plt.figure(figsize=(12, 8))
colors = ['#3274A1', '#E1812C', '#3A923A']
bars = plt.bar(category_labels, cluster_stats['视频数量'], color=colors, width=0.6)

# 添加标题和标签
plt.title('视频按时长的KMeans聚类结果（3类）', fontsize=16, fontweight='bold', pad=20)
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
    cluster_idx = center_order[i]
    plt.axhline(y=center[0], color=plt.cm.viridis(i/(n_clusters-1)), linestyle='--', 
               linewidth=2, label=f'聚类中心 {cluster_names[mapping[cluster_idx]]}: {center[0]:.1f}分钟')

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

# 6) 数量最多的类别的细分分析
# 找出数量最多的类别
largest_cluster = cluster_stats['视频数量'].idxmax()
largest_cluster_data = df[df['cluster_name'] == largest_cluster]

print(f"\n数量最多的类别是：{largest_cluster}，共有 {len(largest_cluster_data)} 个视频")
print(f"该类别的时长范围：{largest_cluster_data['duration_minutes'].min():.2f}分钟 - {largest_cluster_data['duration_minutes'].max():.2f}分钟")

# 绘制这个类别的细分直方图
plt.figure(figsize=(10, 6))
bin_width = (largest_cluster_data['duration_minutes'].max() - largest_cluster_data['duration_minutes'].min()) / 20
bins = int(20)  # 或者使用更多的bins来获得更细致的分布

# 创建直方图并添加KDE曲线
ax = sns.histplot(largest_cluster_data['duration_minutes'], bins=bins, kde=True, 
                 color='coral', edgecolor='black', alpha=0.7)

# 添加垂直线标记平均值和中位数
mean_duration = largest_cluster_data['duration_minutes'].mean()
median_duration = largest_cluster_data['duration_minutes'].median()
plt.axvline(x=mean_duration, color='red', linestyle='--', linewidth=1.5, 
           label=f'平均时长: {mean_duration:.2f}分钟')
plt.axvline(x=median_duration, color='blue', linestyle='--', linewidth=1.5, 
           label=f'中位时长: {median_duration:.2f}分钟')

# 添加标题和标签
plt.title(f'{largest_cluster}类别的时长细分分布', fontsize=16, fontweight='bold')
plt.xlabel('时长（分钟）', fontsize=14)
plt.ylabel('视频数量', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 添加统计信息文本框
stats_text = f"统计信息:\n"              f"总数量: {len(largest_cluster_data)}个\n"              f"最短时长: {largest_cluster_data['duration_minutes'].min():.2f}分钟\n"              f"最长时长: {largest_cluster_data['duration_minutes'].max():.2f}分钟\n"              f"平均时长: {mean_duration:.2f}分钟\n"              f"中位时长: {median_duration:.2f}分钟\n"              f"标准差: {largest_cluster_data['duration_minutes'].std():.2f}分钟"

# 添加文本框
plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
            va='top', fontsize=10)

plt.tight_layout()
plt.show()
