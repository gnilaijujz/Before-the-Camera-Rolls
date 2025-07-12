import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.style.use('ggplot')  # 使用ggplot风格，美化图表

# 读取数据
df = pd.read_csv("D:\下载\sports_2024-2025_with_popularity.csv")

# 将秒转换为分钟
df['duration_minutes'] = df['duration_seconds'] / 60

# 1) 按5分钟为间隔创建时长分组
# 改进的分组函数
def group_by_5min(duration):
    # 数据验证
    if pd.isna(duration) or duration <= 0:
        return None  # 返回None，后续会被过滤
    
    # 向上取整到最近的5分钟倍数
    upper_bound = np.ceil(duration / 5) * 5
    lower_bound = max(0, upper_bound - 5)  # 确保下限不为负
    return f"{int(lower_bound)}-{int(upper_bound)}分钟"

# 应用分组函数
df['duration_group'] = df['duration_minutes'].apply(group_by_5min)

# 过滤掉无效分组的数据
df = df.dropna(subset=['duration_group'])

# 计算分组统计（使用英文列名）
group_stats = df.groupby('duration_group').agg(
    video_count=('duration_group', 'count'),
    avg_popularity=('popularity_normalized', 'mean'),
    median_popularity=('popularity_normalized', 'median'),
    min_popularity=('popularity_normalized', 'min'),
    max_popularity=('popularity_normalized', 'max'),
    std_popularity=('popularity_normalized', 'std')
).reset_index()

# 安全排序
group_stats['lower_bound'] = group_stats['duration_group'].apply(lambda x: int(x.split('-')[0]))
group_stats = group_stats.sort_values('lower_bound')

# 重命名列名为中文（用于显示）
group_stats.columns = ['时长分组', '视频数量', '平均热度', '中位热度', '最小热度', '最大热度', '热度标准差', 'lower_bound']

# 安全排序
group_stats['lower_bound'] = group_stats['duration_group'].apply(lambda x: int(x.split('-')[0]))
group_stats = group_stats.sort_values('lower_bound')

# 打印描述性统计数据
print("按5分钟间隔的视频时长分组统计：")
print(group_stats[['duration_group', '视频数量', '平均热度', '中位热度']])

# 3) 可视化 - 时长分布柱状图（显示每5分钟一组的频数分布）
plt.figure(figsize=(14, 8))

# 设置颜色渐变
cmap = plt.cm.viridis
max_count = group_stats['视频数量'].max()
colors = [cmap(i / max_count) for i in group_stats['视频数量']]

# 绘制柱状图
bars = plt.bar(group_stats['duration_group'], group_stats['视频数量'], color=colors, width=0.8)

# 在柱子上显示数量
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10)

# 美化图表
plt.title('视频时长分布（5分钟间隔）', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('时长分组（分钟）', fontsize=14, labelpad=10)
plt.ylabel('视频数量', fontsize=14, labelpad=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

# 添加数据标签，用百分比表示
total_videos = len(df)
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = 100 * height / total_videos
    if percentage >= 2:  # 只显示占比较大的标签，避免拥挤
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{percentage:.1f}%',
                ha='center', va='center', fontsize=9, 
                color='white', fontweight='bold')

plt.show()

# 4) 可视化 - 时长组与热度关系图（双Y轴）
plt.figure(figsize=(14, 8))

# 选择合适的视图范围，截断极端值以便更好可视化
max_group_index = 18  # 只显示前18个组（即0-90分钟），避免长尾干扰可视化
selected_groups = group_stats.iloc[:max_group_index]

# 创建渐变色彩
mean_pop = selected_groups['平均热度']
normalized_pop = (mean_pop - mean_pop.min()) / (mean_pop.max() - mean_pop.min())
colors = plt.cm.plasma(normalized_pop)

# 使用双轴图展示数量和热度
fig, ax1 = plt.subplots(figsize=(14, 8))

# 柱状图显示数量
bars = ax1.bar(selected_groups['duration_group'], selected_groups['视频数量'], 
              alpha=0.7, color=colors, width=0.7)

# 在柱子上显示数量
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10)

# 配置第一个Y轴
ax1.set_xlabel('时长分组（分钟）', fontsize=14, labelpad=10)
ax1.set_ylabel('视频数量', fontsize=14, labelpad=10)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

# 添加第二个Y轴
ax2 = ax1.twinx()
line = ax2.plot(selected_groups['duration_group'], selected_groups['平均热度'], 'o-', 
               color='red', markersize=8, linewidth=2, label='平均热度')
ax2.fill_between(selected_groups['duration_group'], 
                selected_groups['平均热度'] - selected_groups['热度标准差'],
                selected_groups['平均热度'] + selected_groups['热度标准差'],
                color='red', alpha=0.2)

# 配置第二个Y轴
ax2.set_ylabel('热度（归一化）', fontsize=14, labelpad=10, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.grid(False)

# 添加标题和调整布局
plt.title('视频时长与热度关系（按5分钟分组）', fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

# 添加图例
lines, labels = ax2.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper right', frameon=True)

plt.show()

# 5) 热力图可视化 - 时长与热度分布
plt.figure(figsize=(15, 8))

# 筛选数据，限制在更合理的范围内
max_duration_to_show = 90  # 只显示90分钟以内的视频
filtered_df = df[df['duration_minutes'] <= max_duration_to_show]

# 使用seaborn的kdeplot创建平滑热图
ax = sns.kdeplot(
    data=filtered_df,
    x="duration_minutes",
    y="popularity_normalized",
    cmap="rocket",
    fill=True,
    thresh=0.05,
    levels=15,
    alpha=0.7
)

# 叠加散点图
sns.scatterplot(
    data=filtered_df, 
    x="duration_minutes", 
    y="popularity_normalized",
    s=30, 
    color="white",
    alpha=0.4,
    ax=ax
)

# 设置5分钟间隔的x轴刻度
plt.xticks(range(0, max_duration_to_show + 5, 5))

# 添加5分钟间隔的垂直线
for i in range(5, max_duration_to_show + 5, 5):
    plt.axvline(x=i, color='white', linestyle='--', alpha=0.3)

# 添加标题和标签
plt.title('视频时长与热度分布热图', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('视频时长（分钟）', fontsize=14, labelpad=10)
plt.ylabel('热度（归一化）', fontsize=14, labelpad=10)

# 添加颜色条
cbar = plt.colorbar(ax.collections[0], label='密度')
cbar.set_label('密度', rotation=270, labelpad=20, fontsize=12)

plt.tight_layout()
plt.show()

# 6) 热度前10名的时长组柱状图
top_heat_groups = group_stats.sort_values('平均热度', ascending=False).head(10).copy()
top_heat_groups['排名'] = range(1, 11)

plt.figure(figsize=(14, 8))

# 创建柱状图
bars = plt.bar(top_heat_groups['duration_group'], top_heat_groups['平均热度'], 
              color=plt.cm.plasma(np.linspace(0, 0.8, len(top_heat_groups))), alpha=0.8)

# 在柱子上显示排名和热度
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f"Top {i+1}\n{height:.4f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加视频数量标注
for i, bar in enumerate(bars):
    count = top_heat_groups.iloc[i]['视频数量']
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
            f"{count}个视频",
            ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')

plt.title('热度最高的10个时长组', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('时长分组（分钟）', fontsize=14, labelpad=10)
plt.ylabel('平均热度（归一化）', fontsize=14, labelpad=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
