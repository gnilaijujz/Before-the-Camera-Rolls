import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取数据
df = pd.read_csv(r"D:\下载\sports_2024-2025_with_popularity.csv")
df['duration_min'] = df['duration_seconds'] / 60

# 1. 更精细的时长分类 (6个类别)
duration_bins = [0, 3, 5, 10, 20, 30, np.inf]
duration_labels = ['极短视频(0-3分钟)', '短视频(3-5分钟)', '中短视频(5-10分钟)', 
                  '中视频(10-20分钟)', '中长视频(20-30分钟)', '长视频(30分钟以上)']

df['duration_group'] = pd.cut(
    df['duration_min'],
    bins=duration_bins,
    labels=duration_labels,
    right=False
)

# 2. 直接使用热度均值，而不是分类
# 计算每个时长组的平均热度
duration_heat_mean = df.groupby('duration_group')['popularity_normalized'].mean().reset_index()

# 打印每个时长组的平均热度
print("各时长组的平均热度:")
print(duration_heat_mean)

# 3. 创建热力图，颜色直接代表热度值
plt.figure(figsize=(12, 8))

# 将数据重塑为矩阵形式，用于热力图
# 创建包含所有时长组的DataFrame
heat_matrix = pd.DataFrame(index=duration_labels)

# 添加平均热度列
heat_matrix['平均热度'] = duration_heat_mean.set_index('duration_group')['popularity_normalized']

# 创建自定义颜色映射 - 深色代表高热度
colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
cmap = LinearSegmentedColormap.from_list("custom_blues", colors[::-1])  # 反转颜色列表，使深色代表高值

# 绘制热力图
ax = sns.heatmap(
    heat_matrix,
    cmap=cmap,
    annot=True,
    fmt=".4f",  # 显示4位小数
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': '平均热度值（越深=越高热度）'}
)

# 美化图表
ax.set_title("各时长段视频的平均热度热力图", pad=20, fontsize=18, fontweight='bold')
ax.set_xlabel("热度指标", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分类", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_avg_heat_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 创建更详细的分析 - 按时长组的热度分布热力图
# 将热度进行细分
heat_bins = 5  # 将热度分为5个等级
df['heat_percentile'] = pd.qcut(
    df['popularity_normalized'],
    q=heat_bins,
    labels=[f'P{i+1}' for i in range(heat_bins)]
)

# 计算每个时长组在各热度百分位的占比
heat_distribution = pd.crosstab(
    df['duration_group'], 
    df['heat_percentile'],
    normalize='index'  # 按行归一化
)

# 绘制热力图
plt.figure(figsize=(14, 8))

# 使用与热度值相对应的颜色映射 - 深色表示高热度
cmap_heat = LinearSegmentedColormap.from_list(
    "heat_colors",
    ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"][::-1]
)

# 绘制热力图
ax = sns.heatmap(
    heat_distribution,
    cmap=cmap_heat,
    annot=True,
    fmt=".1%",
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': '在该时长组中的占比（越深=越高热度）'}
)

# 美化图表
ax.set_title("各时长视频的热度分布热力图", pad=20, fontsize=18, fontweight='bold')
ax.set_xlabel("热度百分位（P1=最低20%, P5=最高20%）", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分类", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_distribution_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 计算并可视化每个时长组的热度统计信息
# 计算每个时长组的热度统计
heat_stats = df.groupby('duration_group')['popularity_normalized'].agg(
    ['mean', 'median', 'std', 'min', 'max']
).reset_index()

print("\n各时长组的热度统计:")
print(heat_stats)

# 创建热力图，同时显示多个热度统计指标
plt.figure(figsize=(16, 10))

# 将数据重新组织为热力图格式
heat_stats_matrix = heat_stats.set_index('duration_group')

# 确保列的顺序
columns_order = ['mean', 'median', 'std', 'min', 'max']
heat_stats_matrix = heat_stats_matrix[columns_order]

# 重命名列以便更易理解
heat_stats_matrix.columns = ['平均热度', '中位热度', '标准差', '最低热度', '最高热度']

# 绘制热力图 - 每个指标使用不同的颜色映射
ax = sns.heatmap(
    heat_stats_matrix,
    cmap=cmap_heat,  # 深色表示高热度
    annot=True,
    fmt=".4f",
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': '热度值（越深=越高热度）'}
)

# 美化图表
ax.set_title("各时长段视频的热度统计指标热力图", pad=20, fontsize=18, fontweight='bold')
ax.set_xlabel("热度统计指标", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分类", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_statistics_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 创建热度分位点的热力图
# 计算每个时长组的热度分位数
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # 10%, 25%, 50%, 75%, 90%分位数

# 创建一个空DataFrame来存储结果
heat_percentiles = pd.DataFrame(index=duration_labels)

# 计算每个分位点
for p in percentiles:
    col_name = f'P{int(p*100)}' # 如P10, P25等
    heat_percentiles[col_name] = df.groupby('duration_group')['popularity_normalized'].quantile(p)

print("\n各时长组的热度分位点:")
print(heat_percentiles)

# 绘制热力图
plt.figure(figsize=(14, 8))

# 绘制热力图
ax = sns.heatmap(
    heat_percentiles,
    cmap=cmap_heat,  # 深色表示高热度
    annot=True,
    fmt=".4f",
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': '热度分位点值（越深=越高热度）'}
)

# 美化图表
ax.set_title("各时长组的热度分位点热力图", pad=20, fontsize=18, fontweight='bold')
ax.set_xlabel("热度分位点", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分类", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_percentiles_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 创建热度排名视图 - 按平均热度给时长组排序
# 按平均热度排序时长组
duration_rank = heat_stats.sort_values('mean', ascending=False).reset_index()
duration_rank['排名'] = range(1, len(duration_rank) + 1)

# 绘制横向条形图
plt.figure(figsize=(12, 8))
bars = plt.barh(duration_rank['duration_group'], duration_rank['mean'], color=plt.cm.Blues(np.linspace(0.3, 0.9, len(duration_rank))))

# 在条形图上添加热度值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', 
            ha='left', va='center', fontsize=10)

# 在Y轴左侧添加排名标签
for i, (_, row) in enumerate(duration_rank.iterrows()):
    plt.text(-0.01, i, f"#{int(row['排名'])}", 
            ha='right', va='center', fontsize=12, fontweight='bold')

plt.title('各时长组的平均热度排名', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('平均热度值', fontsize=14, labelpad=10)
plt.ylabel('视频时长分类', fontsize=14, labelpad=10)
plt.xlim(0, duration_rank['mean'].max() * 1.1)  # 为标签留出空间
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('duration_heat_ranking.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 使用色带热力图 - 将热度分为10个等级，颜色越深代表热度越高
# 创建热度分组
heat_deciles = 10  # 将热度分为10等分
df['heat_decile'] = pd.qcut(
    df['popularity_normalized'],
    q=heat_deciles,
    labels=[f'D{i+1}' for i in range(heat_deciles)]
)

# 计算每个时长组在各热度十分位的占比
heat_decile_distribution = pd.crosstab(
    df['duration_group'], 
    df['heat_decile'],
    normalize='index'  # 按行归一化
)

# 绘制热力图
plt.figure(figsize=(16, 8))

# 创建更细致的颜色渐变
heat_colors = ["#f7fbff", "#e3eef9", "#d0e1f2", "#b7d4e8", "#9dc6e0", 
              "#74add1", "#4a98c9", "#2886bc", "#0472b4", "#084594"][::-1]  # 反转使深色代表高热度
cmap_detailed = LinearSegmentedColormap.from_list("detailed_heat", heat_colors)

# 绘制热力图
ax = sns.heatmap(
    heat_decile_distribution,
    cmap=cmap_detailed,
    annot=True,
    fmt=".1%",
    linewidths=0.8,
    linecolor='white',
    cbar_kws={'label': '在该时长组中的占比（颜色越深=热度越高）'}
)

# 美化图表
ax.set_title("各时长视频的热度分布细化热力图", pad=20, fontsize=18, fontweight='bold')
ax.set_xlabel("热度十分位（D1=最低10%, D10=最高10%）", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分类", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_decile_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 添加热度分布偏移图 - 显示与全局平均热度的偏差
# 计算全局平均热度
global_mean_heat = df['popularity_normalized'].mean()

# 计算每个时长组的平均热度与全局平均热度的差异
heat_diff = pd.DataFrame(index=duration_labels)
heat_diff['热度差异'] = heat_stats.set_index('duration_group')['mean'] - global_mean_heat

print("\n各时长组与全局平均热度的差异:")
print(heat_diff)

# 绘制水平条形图
plt.figure(figsize=(12, 8))

# 根据正负值设置颜色
colors = ['#08519c' if x > 0 else '#d73027' for x in heat_diff['热度差异']]

bars = plt.barh(heat_diff.index, heat_diff['热度差异'], color=colors)

# 在条形图上添加值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    x_pos = width + 0.001 if width > 0 else width - 0.003
    ha = 'left' if width > 0 else 'right'
    plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', 
            ha=ha, va='center', fontsize=10,
            color='black')

# 添加垂直线表示全局平均
plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
plt.text(0, -0.5, f'全局平均: {global_mean_heat:.4f}', 
        ha='center', va='top', fontsize=10, rotation=90)

plt.title('各时长组热度与全局平均的差异', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('热度差异（正=高于平均，负=低于平均）', fontsize=14, labelpad=10)
plt.ylabel('视频时长分类', fontsize=14, labelpad=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('duration_heat_difference.png', dpi=300, bbox_inches='tight')
plt.show()
