import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# 2) 根据固定阈值分类视频
# 设置时长阈值
short_threshold = 30  # 短视频上限：30分钟
medium_threshold = 60  # 中长视频上限：60分钟

# 创建类别
def categorize_video(duration):
    if duration < short_threshold:
        return '短视频'
    elif duration < medium_threshold:
        return '中长视频'
    else:
        return '长视频'

# 应用分类函数
df['video_category'] = df['duration_minutes'].apply(categorize_video)

# 3) 统计各类别的数量和时长范围
category_stats = df.groupby('video_category').agg(
    视频数量=('video_category', 'count'),
    最小时长_分钟=('duration_minutes', 'min'),
    最大时长_分钟=('duration_minutes', 'max'),
    平均时长_分钟=('duration_minutes', 'mean'),
    中位时长_分钟=('duration_minutes', 'median'),
)

# 按照短、中长、长视频的顺序排序
category_order = ['短视频', '中长视频', '长视频']
category_stats = category_stats.reindex(category_order)

print("\n固定阈值分类结果 - 将视频分为三类：")
print(f"短视频: < {short_threshold}分钟")
print(f"中长视频: {short_threshold}-{medium_threshold}分钟")
print(f"长视频: > {medium_threshold}分钟")
print(category_stats)

# 4) 绘制分类结果柱状图
# 为了更直观地展示，创建标签
category_labels = [
    f"短视频\n(< {short_threshold}分钟)\n平均: {category_stats.loc['短视频', '平均时长_分钟']:.1f}分钟", 
    f"中长视频\n({short_threshold}-{medium_threshold}分钟)\n平均: {category_stats.loc['中长视频', '平均时长_分钟']:.1f}分钟",
    f"长视频\n(> {medium_threshold}分钟)\n平均: {category_stats.loc['长视频', '平均时长_分钟']:.1f}分钟"
]

# 绘制分类柱状图，使用更好看的配色
plt.figure(figsize=(12, 8))
colors = ['#3274A1', '#E1812C', '#3A923A']
bars = plt.bar(category_labels, category_stats['视频数量'], color=colors, width=0.6)

# 添加标题和标签
plt.title('视频时长分类结果（三类）', fontsize=16, fontweight='bold', pad=20)
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

# 5) 可视化聚类结果 - 散点图
plt.figure(figsize=(12, 6))
# 创建颜色映射
color_map = {'短视频': 0, '中长视频': 1, '长视频': 2}
colors = df['video_category'].map(color_map)

# 绘制散点图，x轴为视频索引，y轴为视频时长，颜色区分聚类
scatter = plt.scatter(range(len(df)), df['duration_minutes'], 
                     c=colors, cmap='viridis', 
                     alpha=0.6, s=30, edgecolors='w')

# 添加类别分界线
plt.axhline(y=short_threshold, color='red', linestyle='--', 
           linewidth=1.5, alpha=0.7)
plt.text(len(df)*0.02, short_threshold+1, f"短视频上限: {short_threshold}分钟", 
        fontsize=10, color='red')

plt.axhline(y=medium_threshold, color='red', linestyle='--', 
           linewidth=1.5, alpha=0.7)
plt.text(len(df)*0.02, medium_threshold+1, f"中长视频上限: {medium_threshold}分钟", 
        fontsize=10, color='red')

plt.title('视频时长分布', fontsize=16, fontweight='bold')
plt.xlabel('视频索引', fontsize=14)
plt.ylabel('视频时长（分钟）', fontsize=14)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0/2), markersize=10, label='短视频'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(1/2), markersize=10, label='中长视频'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(2/2), markersize=10, label='长视频')
], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6) 可视化聚类结果 - 时长分布图
plt.figure(figsize=(12, 6))

# 为每个聚类创建子图
for i, name in enumerate(category_order):
    subset = df[df['video_category'] == name]
    sns.kdeplot(subset['duration_minutes'], label=name, fill=True, alpha=0.3)

plt.title('各类别视频时长分布', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('密度', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='视频类别')
plt.tight_layout()
plt.show()

# 7) 对短视频类别进行细分分析
short_video_category = '短视频'
short_video_data = df[df['video_category'] == short_video_category]

print(f"\n{short_video_category}类别的详细分析，共有 {len(short_video_data)} 个视频")
print(f"该类别的时长范围：{short_video_data['duration_minutes'].min():.2f}分钟 - {short_video_data['duration_minutes'].max():.2f}分钟")

# 绘制短视频类别的细分直方图
plt.figure(figsize=(10, 6))
bins = 30  # 或者根据数据范围调整bins数量

# 创建直方图并添加KDE曲线
ax = sns.histplot(short_video_data['duration_minutes'], bins=bins, kde=True, 
                 color='#3274A1', edgecolor='black', alpha=0.7)

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
stats_text = f"统计信息:\n"              f"总数量: {len(short_video_data)}个\n"              f"最短时长: {short_video_data['duration_minutes'].min():.2f}分钟\n"              f"最长时长: {short_video_data['duration_minutes'].max():.2f}分钟\n"              f"平均时长: {mean_duration:.2f}分钟\n"              f"中位时长: {median_duration:.2f}分钟\n"              f"标准差: {short_video_data['duration_minutes'].std():.2f}分钟"

# 添加文本框
plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
            va='top', fontsize=10)

plt.tight_layout()
plt.show()

# 8) 对中长视频类别进行细分分析
medium_category = '中长视频'
medium_data = df[df['video_category'] == medium_category]

print(f"\n{medium_category}类别的详细分析，共有 {len(medium_data)} 个视频")
print(f"该类别的时长范围：{medium_data['duration_minutes'].min():.2f}分钟 - {medium_data['duration_minutes'].max():.2f}分钟")

# 绘制中长视频类别的细分直方图
plt.figure(figsize=(10, 6))
bins = int(medium_threshold - short_threshold)  # 每分钟一个bin

# 创建直方图并添加KDE曲线
ax = sns.histplot(medium_data['duration_minutes'], bins=bins, kde=True, 
                 color='#E1812C', edgecolor='black', alpha=0.7)

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
stats_text = f"统计信息:\n"              f"总数量: {len(medium_data)}个\n"              f"最短时长: {medium_data['duration_minutes'].min():.2f}分钟\n"              f"最长时长: {medium_data['duration_minutes'].max():.2f}分钟\n"              f"平均时长: {mean_duration:.2f}分钟\n"              f"中位时长: {median_duration:.2f}分钟\n"              f"标准差: {medium_data['duration_minutes'].std():.2f}分钟"

# 添加文本框
plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
            va='top', fontsize=10)

plt.tight_layout()
plt.show()