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

# 2) 使用KMeans进行聚类，分为3类初始分类
# 准备数据，只使用时长特征进行聚类
X = df[['duration_minutes']].values
# 由于KMeans对尺度敏感，我们进行标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用KMeans进行聚类，分为3类（超短、短、长）
n_clusters_initial = 3
kmeans_initial = KMeans(n_clusters=n_clusters_initial, random_state=42, n_init=10)
df['initial_cluster'] = kmeans_initial.fit_predict(X_scaled)

# 获取聚类中心（转换回原始尺度）
centers_initial = scaler.inverse_transform(kmeans_initial.cluster_centers_)
# 按照聚类中心的大小排序类别
center_order_initial = np.argsort(centers_initial.flatten())

# 创建初始分类的映射
initial_mapping = {
    center_order_initial[0]: '超短视频',
    center_order_initial[1]: '短视频原类',
    center_order_initial[2]: '长视频'
}
df['initial_category'] = df['initial_cluster'].map(initial_mapping)

print("\n初始聚类结果 - 将视频分为三类：")
initial_stats = df.groupby('initial_category').agg(
    视频数量=('initial_category', 'count'),
    最小时长_分钟=('duration_minutes', 'min'),
    最大时长_分钟=('duration_minutes', 'max'),
    平均时长_分钟=('duration_minutes', 'mean'),
    中位时长_分钟=('duration_minutes', 'median'),
)
print(initial_stats)

# 3) 对"短视频原类"再次聚类为2类
# 筛选出短视频组的数据
short_videos = df[df['initial_category'] == '短视频原类']
if len(short_videos) > 0:
    # 准备短视频的时长数据
    X_short = short_videos[['duration_minutes']].values
    # 标准化处理
    scaler_short = StandardScaler()
    X_short_scaled = scaler_short.fit_transform(X_short)

    # 对短视频进行再聚类，分为2类
    n_clusters_short = 2
    kmeans_short = KMeans(n_clusters=n_clusters_short, random_state=42, n_init=10)
    short_videos_clusters = kmeans_short.fit_predict(X_short_scaled)

    # 获取聚类中心并排序
    centers_short = scaler_short.inverse_transform(kmeans_short.cluster_centers_)
    center_order_short = np.argsort(centers_short.flatten())

    # 在原始数据框中创建子类别
    df.loc[df['initial_category'] == '短视频原类', 'short_video_subclass'] = np.nan
    df.loc[short_videos.index, 'short_video_subclass'] = short_videos_clusters

    # 映射为短视频和中长视频
    mapping_short = {
        center_order_short[0]: '短视频', 
        center_order_short[1]: '中长视频'
    }
    df.loc[short_videos.index, 'short_subclass_name'] = df.loc[short_videos.index, 'short_video_subclass'].map(mapping_short)

# 4) 创建最终的四类分类
df['final_category'] = df['initial_category']
# 用细分的短视频和中长视频替换原来的'短视频原类'
df.loc[df['initial_category'] == '短视频原类', 'final_category'] = df.loc[df['initial_category'] == '短视频原类', 'short_subclass_name']

# 5) 统计最终分类的数量和时长范围
category_stats = df.groupby('final_category').agg(
    视频数量=('final_category', 'count'),
    最小时长_分钟=('duration_minutes', 'min'),
    最大时长_分钟=('duration_minutes', 'max'),
    平均时长_分钟=('duration_minutes', 'mean'),
    中位时长_分钟=('duration_minutes', 'median'),
)

# 按照超短、短、中长、长视频的顺序排序
category_order = ['超短视频', '短视频', '中长视频', '长视频']
category_stats = category_stats.reindex(category_order)

print("\n最终分类结果 - 将视频分为四类：")
print(category_stats)

# 6) 绘制分类结果柱状图
# 为了更直观地展示，创建标签
category_labels = [f"{name}\n({category_stats.loc[name, '最小时长_分钟']:.1f}~{category_stats.loc[name, '最大时长_分钟']:.1f}分钟)\n平均: {category_stats.loc[name, '平均时长_分钟']:.1f}分钟" 
                  for name in category_order]

# 绘制分类柱状图，使用更好看的配色
plt.figure(figsize=(14, 8))
colors = ['#3274A1', '#E1812C', '#3A923A', '#D64550']
bars = plt.bar(category_labels, category_stats['视频数量'], color=colors, width=0.6)

# 添加标题和标签
plt.title('视频时长分类结果（四类）', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('视频时长分类', fontsize=14, labelpad=10)
plt.ylabel('视频数量', fontsize=14, labelpad=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# 在柱状图上标注具体数量及百分比
total_videos = category_stats['视频数量'].sum()
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = 100 * height / total_videos
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}个 ({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 美化图表
plt.xticks(fontsize=11)
plt.yticks(fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 为底部注释留出空间

plt.show()

# 7) 可视化聚类结果 - 散点图
plt.figure(figsize=(12, 6))
# 创建颜色映射
color_map = {'超短视频': 0, '短视频': 1, '中长视频': 2, '长视频': 3}
colors = df['final_category'].map(color_map)

# 绘制散点图，x轴为视频索引，y轴为视频时长，颜色区分聚类
scatter = plt.scatter(range(len(df)), df['duration_minutes'], 
                     c=colors, cmap='viridis', 
                     alpha=0.6, s=30, edgecolors='w')

# 添加类别分界线
for i, name in enumerate(category_order):
    if i < len(category_order) - 1:  # 不为最后一类添加上限线
        threshold = category_stats.loc[name, '最大时长_分钟']
        plt.axhline(y=threshold, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        plt.text(len(df)*0.02, threshold+1, f"{name}上限: {threshold:.1f}分钟", 
                fontsize=10, color='red')

plt.title('视频时长分布', fontsize=16, fontweight='bold')
plt.xlabel('视频索引', fontsize=14)
plt.ylabel('视频时长（分钟）', fontsize=14)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0/3), markersize=10, label='超短视频'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(1/3), markersize=10, label='短视频'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(2/3), markersize=10, label='中长视频'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(3/3), markersize=10, label='长视频')
], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 8) 可视化聚类结果 - 时长分布图
plt.figure(figsize=(12, 6))

# 为每个聚类创建子图
for i, name in enumerate(category_order):
    subset = df[df['final_category'] == name]
    if len(subset) > 0:
        sns.kdeplot(subset['duration_minutes'], label=name, fill=True, alpha=0.3)

plt.title('各类别视频时长分布', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('密度', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='视频类别')
plt.tight_layout()
plt.show()

# 9) 对新的短视频类别进行细分分析
short_video_category = '短视频'
short_video_data = df[df['final_category'] == short_video_category]

if len(short_video_data) > 0:
    print(f"\n{short_video_category}类别的详细分析，共有 {len(short_video_data)} 个视频")
    print(f"该类别的时长范围：{short_video_data['duration_minutes'].min():.2f}分钟 - {short_video_data['duration_minutes'].max():.2f}分钟")

    # 绘制短视频类别的细分直方图
    plt.figure(figsize=(10, 6))
    bins = 20  # 或者根据数据范围调整bins数量

    # 创建直方图并添加KDE曲线
    ax = sns.histplot(short_video_data['duration_minutes'], bins=bins, kde=True, 
                     color='#E1812C', edgecolor='black', alpha=0.7)

    # 添加垂直线标记平均值和中位数
    mean_duration = short_video_data['duration_minutes'].mean()
    median_duration = short_video_data['duration_minutes'].median()
    plt.axvline(x=mean_duration, color='red', linestyle='--', linewidth=1.5, 
               label=f'平均时长: {mean_duration:.2f}分钟')
    plt.axvline(x=median_duration, color='blue', linestyle='--', linewidth=1.5, 
               label=f'中位时长: {median_duration:.2f}分钟')

    # 添加标题和标签
    plt.title(f'{short_video_category}类别的时长细分分布', fontsize=16, fontweight='bold')
    plt.xlabel('时长（分钟）', fontsize=14)
    plt.ylabel('视频数量', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 添加统计信息文本框
    stats_text = f"统计信息:\n"                  f"总数量: {len(short_video_data)}个\n"                  f"最短时长: {short_video_data['duration_minutes'].min():.2f}分钟\n"                  f"最长时长: {short_video_data['duration_minutes'].max():.2f}分钟\n"                  f"平均时长: {mean_duration:.2f}分钟\n"                  f"中位时长: {median_duration:.2f}分钟\n"                  f"标准差: {short_video_data['duration_minutes'].std():.2f}分钟"

    # 添加文本框
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
                va='top', fontsize=10)

    plt.tight_layout()
    plt.show()

# 10) 对中长视频类别进行细分分析
medium_category = '中长视频'
medium_data = df[df['final_category'] == medium_category]

if len(medium_data) > 0:
    print(f"\n{medium_category}类别的详细分析，共有 {len(medium_data)} 个视频")
    print(f"该类别的时长范围：{medium_data['duration_minutes'].min():.2f}分钟 - {medium_data['duration_minutes'].max():.2f}分钟")

    # 绘制中长视频类别的细分直方图
    plt.figure(figsize=(10, 6))
    bins = 20  # 或者根据数据范围调整bins数量

    # 创建直方图并添加KDE曲线
    ax = sns.histplot(medium_data['duration_minutes'], bins=bins, kde=True, 
                     color='#3A923A', edgecolor='black', alpha=0.7)

    # 添加垂直线标记平均值和中位数
    mean_duration = medium_data['duration_minutes'].mean()
    median_duration = medium_data['duration_minutes'].median()
    plt.axvline(x=mean_duration, color='red', linestyle='--', linewidth=1.5, 
               label=f'平均时长: {mean_duration:.2f}分钟')
    plt.axvline(x=median_duration, color='blue', linestyle='--', linewidth=1.5, 
               label=f'中位时长: {median_duration:.2f}分钟')

    # 添加标题和标签
    plt.title(f'{medium_category}类别的时长细分分布', fontsize=16, fontweight='bold')
    plt.xlabel('时长（分钟）', fontsize=14)
    plt.ylabel('视频数量', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 添加统计信息文本框
    stats_text = f"统计信息:\n"                  f"总数量: {len(medium_data)}个\n"                  f"最短时长: {medium_data['duration_minutes'].min():.2f}分钟\n"                  f"最长时长: {medium_data['duration_minutes'].max():.2f}分钟\n"                  f"平均时长: {mean_duration:.2f}分钟\n"                  f"中位时长: {median_duration:.2f}分钟\n"                  f"标准差: {medium_data['duration_minutes'].std():.2f}分钟"

    # 添加文本框
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
                va='top', fontsize=10)

    plt.tight_layout()
    plt.show()

# 11) 分析全局时长分布，将四类视频在同一图上展示
plt.figure(figsize=(12, 6))

# 设置x轴的范围限制，避免极端值影响可视化
max_duration_to_show = min(df['duration_minutes'].max(), 60)  # 限制为60分钟以内的视频

# 创建子图并添加KDE曲线
for name, color in zip(category_order, colors):
    subset = df[df['final_category'] == name]
    if len(subset) > 0:
        # 限制x轴范围，使图形更加清晰
        sns.kdeplot(subset['duration_minutes'].clip(upper=max_duration_to_show), 
                   label=f"{name} (n={len(subset)})", color=color, fill=True, alpha=0.3)

plt.title('四类视频时长分布对比', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('密度', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='视频类别')

# 添加每个类别的中位数垂直线
for name, color in zip(category_order, colors):
    subset = df[df['final_category'] == name]
    if len(subset) > 0:
        median_val = subset['duration_minutes'].median()
        if median_val <= max_duration_to_show:
            plt.axvline(x=median_val, color=color, linestyle='--', alpha=0.7,
                       label=f"{name}中位数: {median_val:.1f}分钟")

plt.xlim(0, max_duration_to_show)
plt.tight_layout()
plt.show()

# 12) 饼图展示各类别占比
plt.figure(figsize=(10, 8))
sizes = category_stats['视频数量']
labels = [f"{name} ({sizes[i]}个, {sizes[i]/sum(sizes)*100:.1f}%)" for i, name in enumerate(category_order)]
explode = (0.1, 0.05, 0.02, 0)  # 突出显示超短视频

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=140, textprops={'fontsize': 12})
plt.axis('equal')  # 确保饼图是圆的
plt.title('视频时长分类占比', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
