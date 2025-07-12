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
# 创建分组函数，每5分钟一组
def group_by_5min(duration):
    # 向上取整到最近的5分钟倍数
    upper_bound = np.ceil(duration / 5) * 5
    # 创建组名，例如 "0-5分钟"
    lower_bound = upper_bound - 5
    return f"{int(lower_bound)}-{int(upper_bound)}分钟"

# 应用分组函数
df['duration_group'] = df['duration_minutes'].apply(group_by_5min)

# 2) 计算每个时长组的描述性统计数据
group_stats = df.groupby('duration_group').agg(
    视频数量=('duration_group', 'count'),
    平均热度=('popularity_normalized', 'mean'),
    中位热度=('popularity_normalized', 'median'),
    最小热度=('popularity_normalized', 'min'),
    最大热度=('popularity_normalized', 'max'),
    热度标准差=('popularity_normalized', 'std')
).reset_index()

# 确保按时长组排序
# 提取组的下限用于排序
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

# 4) 可视化 - 时长组与热度关系图
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

# 5) 热力图可视化 - 5分钟间隔的热度分布
plt.figure(figsize=(16, 10))

# 准备热图数据
# 我们将只选择前18个组，以便更好地可视化
heatmap_data = group_stats.iloc[:18].copy()

# 创建一个新列用于标记每个组的热度水平
heatmap_data['热度级别'] = pd.qcut(heatmap_data['平均热度'], 5, 
                              labels=['很低', '较低', '中等', '较高', '很高'])

# 创建热图数据矩阵
matrix_data = pd.DataFrame({
    'duration_group': heatmap_data['duration_group'],
    '视频数量': heatmap_data['视频数量'],
    '平均热度': heatmap_data['平均热度'],
    '热度级别': heatmap_data['热度级别']
})

# 将数据重塑为宽格式
matrix = matrix_data.pivot_table(index='热度级别', columns='duration_group', 
                               values='视频数量', aggfunc='sum', fill_value=0)

# 创建自定义颜色映射
cmap = LinearSegmentedColormap.from_list('custom_heat', 
                                       ['#FFFFFF', '#FFFFCC', '#FFEDA0', 
                                        '#FED976', '#FEB24C', '#FD8D3C', 
                                        '#FC4E2A', '#E31A1C', '#B10026'])

# 绘制热图
ax = sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap, linewidths=0.5, 
                linecolor='gray', cbar_kws={'label': '视频数量'})

# 添加标题和标签
plt.title('视频时长与热度分布热力图（按5分钟分组）', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('时长分组（分钟）', fontsize=14, labelpad=10)
plt.ylabel('热度级别', fontsize=14, labelpad=10)

# 旋转x轴标签
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 6) 高级热力图 - 散点密度图
plt.figure(figsize=(16, 9))

# 筛选数据，限制在合理范围内以获得更好的可视化效果
max_duration_to_show = 90  # 只显示90分钟以内的视频
filtered_df = df[df['duration_minutes'] <= max_duration_to_show]

# 创建一个类别型的颜色映射，基于5分钟分组
duration_categories = pd.cut(filtered_df['duration_minutes'], 
                            bins=range(0, max_duration_to_show + 5, 5),
                            labels=[f"{i}-{i+5}" for i in range(0, max_duration_to_show, 5)])

# 计算每个时长组的平均热度
mean_pop_by_group = filtered_df.groupby(duration_categories)['popularity_normalized'].mean()

# 将分类映射到颜色
norm = plt.Normalize(min(mean_pop_by_group), max(mean_pop_by_group))
colors = plt.cm.plasma(norm(mean_pop_by_group))
color_dict = {cat: colors[i] for i, cat in enumerate(mean_pop_by_group.index)}

# 为每个点分配颜色
point_colors = [color_dict.get(cat, 'gray') for cat in duration_categories]

# 创建散点图
scatter = plt.scatter(filtered_df['duration_minutes'], filtered_df['popularity_normalized'], 
                    c=point_colors, alpha=0.7, s=50, edgecolors='none')

# 添加5分钟间隔的垂直线
for i in range(5, max_duration_to_show + 5, 5):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.3)

# 设置刻度和标签
plt.xticks(range(0, max_duration_to_show + 5, 5))
plt.xlabel('视频时长（分钟）', fontsize=14, labelpad=10)
plt.ylabel('热度（归一化）', fontsize=14, labelpad=10)
plt.title('视频时长与热度关系散点图（按5分钟间隔着色）', fontsize=18, fontweight='bold', pad=20)

# 添加平均热度趋势线
x_vals = [np.mean([i, i+5]) for i in range(0, max_duration_to_show, 5)]
plt.plot(x_vals, mean_pop_by_group, 'o-', color='red', linewidth=3, markersize=8, 
        label='各组平均热度')

# 添加图例
plt.legend(loc='upper right')

# 调整布局
plt.tight_layout()
plt.show()

# 7) 双变量分布热图 - 更精细的热力图
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

# 8) 创建组合图表 - 热度按时长变化趋势
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 2]})

# 上半部分：视频数量分布
bars = ax1.bar(group_stats.iloc[:18]['duration_group'], group_stats.iloc[:18]['视频数量'], 
              color='skyblue', alpha=0.8, width=0.7)

# 在柱子上显示数量和百分比
total_videos = group_stats['视频数量'].sum()
for bar in bars:
    height = bar.get_height()
    percentage = 100 * height / total_videos
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=9)

ax1.set_title('视频时长分布与热度关系（按5分钟分组）', fontsize=20, fontweight='bold', pad=20)
ax1.set_ylabel('视频数量', fontsize=14, labelpad=10)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
ax1.tick_params(axis='x', labelrotation=45, labelsize=10)

# 下半部分：热度趋势与热力图
# 创建热度颜色映射
duration_groups = group_stats.iloc[:18]['duration_group']
heat_colors = plt.cm.plasma(np.linspace(0, 1, len(duration_groups)))

# 绘制热度趋势线
ax2.plot(duration_groups, group_stats.iloc[:18]['平均热度'], 'o-', 
        color='red', markersize=10, linewidth=2.5, label='平均热度')

# 绘制热度标准差范围
ax2.fill_between(duration_groups,
                group_stats.iloc[:18]['平均热度'] - group_stats.iloc[:18]['热度标准差'],
                group_stats.iloc[:18]['平均热度'] + group_stats.iloc[:18]['热度标准差'],
                color='red', alpha=0.2, label='标准差范围')

# 添加最高热度和最低热度
ax2.plot(duration_groups, group_stats.iloc[:18]['最大热度'], '--', 
        color='orange', alpha=0.7, linewidth=1.5, label='最高热度')
ax2.plot(duration_groups, group_stats.iloc[:18]['最小热度'], '--', 
        color='green', alpha=0.7, linewidth=1.5, label='最低热度')

# 突出显示热度最高的时长组
max_heat_idx = group_stats.iloc[:18]['平均热度'].idxmax()
max_heat_group = group_stats.iloc[max_heat_idx]
ax2.plot(max_heat_group['duration_group'], max_heat_group['平均热度'], 'o',
        markersize=15, markerfacecolor='gold', markeredgecolor='black', markeredgewidth=2,
        label=f"热度最高: {max_heat_group['duration_group']} ({max_heat_group['平均热度']:.4f})")

# 添加热度峰值标注
ax2.annotate(f"热度峰值: {max_heat_group['平均热度']:.4f}",
            xy=(max_heat_group['duration_group'], max_heat_group['平均热度']),
            xytext=(0, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'),
            fontsize=12, ha='center', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))

# 设置标签和网格
ax2.set_xlabel('时长分组（分钟）', fontsize=14, labelpad=10)
ax2.set_ylabel('热度（归一化）', fontsize=14, labelpad=10)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right', frameon=True)
ax2.tick_params(axis='x', labelrotation=45, labelsize=10)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.show()

# 9) 表格可视化：热度前10名的时长组
top_heat_groups = group_stats.sort_values('平均热度', ascending=False).head(10).copy()
top_heat_groups['排名'] = range(1, 11)

# 重新排列列顺序
top_heat_groups = top_heat_groups[['排名', 'duration_group', '视频数量', '平均热度', '中位热度', '热度标准差']]

print("\n热度最高的前10个时长组：")
print(top_heat_groups)

# 10) 绘制热度前10名的时长组柱状图
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

plt.title('热度最高的前10个时长组', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('时长分组（分钟）', fontsize=14, labelpad=10)
plt.ylabel('平均热度（归一化）', fontsize=14, labelpad=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')

# 添加视频数量标注
for i, (_, row) in enumerate(top_heat_groups.iterrows()):
    plt.annotate(f"{row['视频数量']}个视频",
                xy=(i, row['平均热度'] / 2),
                ha='center', va='center',
                color='white', fontweight='bold')

plt.tight_layout()
plt.show()
