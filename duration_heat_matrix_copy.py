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

# 1. 定义更精细的时长分类
duration_bins = [0, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, np.inf]
duration_labels = [
    '0-3分钟', '3-5分钟', '5-10分钟', '10-15分钟', 
    '15-20分钟', '20-25分钟', '25-30分钟', '30-40分钟',
    '40-50分钟', '50-60分钟', '60+分钟'
]

df['duration_group'] = pd.cut(
    df['duration_min'],
    bins=duration_bins,
    labels=duration_labels,
    right=False
)

# 2. 将热度分为十个等级
heat_quantiles = 10
df['heat_group'] = pd.qcut(
    df['popularity_normalized'],
    q=heat_quantiles,
    labels=[f'H{i+1}' for i in range(heat_quantiles)]
)

# 3. 计算每个时长-热度组合的平均热度
heat_matrix_raw = df.groupby(['duration_group', 'heat_group'])['popularity_normalized'].mean().unstack()

# 填充缺失值（如果有）
heat_matrix = heat_matrix_raw.fillna(0)

# 确保热度列的顺序正确
heat_matrix = heat_matrix[[f'H{i+1}' for i in range(heat_quantiles)]]

# 4. 首先创建矩形热力图 - 6×10规格 - 颜色直接代表热度值
plt.figure(figsize=(16, 10))

# 创建自定义颜色映射 - 从浅蓝到深蓝
blues = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
custom_cmap = LinearSegmentedColormap.from_list("custom_blues", blues)

# 绘制热力图
ax = sns.heatmap(
    heat_matrix,
    cmap=custom_cmap,
    annot=True,
    fmt=".4f",
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '平均热度值（越深=越高热度）'}
)

# 美化图表
ax.set_title("视频时长与热度关系矩形热力图", pad=20, fontsize=20, fontweight='bold')
ax.set_xlabel("热度分组（H1=最低10%, H10=最高10%）", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分组", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_rectangle_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 创建热度分布矩形热力图 - 基于视频数量占比
# 计算每个时长-热度组合的视频数量
count_matrix = pd.crosstab(
    df['duration_group'], 
    df['heat_group'],
    normalize='index'  # 按行归一化，显示每个时长组中不同热度的占比
)

# 确保热度列的顺序正确
count_matrix = count_matrix[[f'H{i+1}' for i in range(heat_quantiles)]]

plt.figure(figsize=(16, 10))

# 绘制热力图
ax = sns.heatmap(
    count_matrix,
    cmap=custom_cmap,
    annot=True,
    fmt=".1%",
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '视频占比（越深=越高热度）'}
)

# 美化图表
ax.set_title("视频时长与热度分布矩形热力图", pad=20, fontsize=20, fontweight='bold')
ax.set_xlabel("热度分组（H1=最低10%, H10=最高10%）", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分组", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_distribution_rectangle.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 使用绝对数值的热力图 - 直观显示各时长和热度组合的视频数量
# 计算每个时长-热度组合的视频数量（绝对值）
abs_count_matrix = pd.crosstab(
    df['duration_group'], 
    df['heat_group']
)

# 确保热度列的顺序正确
abs_count_matrix = abs_count_matrix[[f'H{i+1}' for i in range(heat_quantiles)]]

plt.figure(figsize=(16, 10))

# 创建自定义颜色映射 - 从浅蓝到深蓝
count_cmap = LinearSegmentedColormap.from_list("count_blues", blues)

# 绘制热力图
ax = sns.heatmap(
    abs_count_matrix,
    cmap=count_cmap,
    annot=True,
    fmt="d",  # 整数格式
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '视频数量（越深=越多视频）'}
)

# 美化图表
ax.set_title("视频时长与热度分布矩形热力图 (视频数量)", pad=20, fontsize=20, fontweight='bold')
ax.set_xlabel("热度分组（H1=最低10%, H10=最高10%）", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分组", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_count_rectangle.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 创建高级混合热力图 - 颜色表示热度，文本显示视频数量和百分比
plt.figure(figsize=(18, 12))

# 创建带注释的热力图
ax = sns.heatmap(
    heat_matrix,
    cmap=custom_cmap,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '平均热度值（越深=越高热度）'}
)

# 添加详细文本注释 - 包含热度值、视频数和占比
for i, duration in enumerate(heat_matrix.index):
    for j, heat_level in enumerate(heat_matrix.columns):
        # 获取相关数据
        heat_value = heat_matrix.loc[duration, heat_level]
        count = abs_count_matrix.loc[duration, heat_level]
        percentage = count_matrix.loc[duration, heat_level]

        # 创建注释文本
        text = f"{heat_value:.4f}\n{count} 个视频\n({percentage:.1%})"

        # 根据背景色选择文本颜色
        text_color = 'white' if heat_value > heat_matrix.values.mean() else 'black'

        # 添加注释
        ax.text(j + 0.5, i + 0.5, text,
               ha='center', va='center', color=text_color, fontsize=9)

# 美化图表
ax.set_title("视频时长与热度关系综合热力图", pad=20, fontsize=20, fontweight='bold')
ax.set_xlabel("热度分组（H1=最低10%, H10=最高10%）", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分组", fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_comprehensive_rectangle.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 热度空间分布图 - 每个时长组的热度分布（高级版本）
plt.figure(figsize=(20, 12))

# 创建子图网格
duration_groups = heat_matrix.index.tolist()
n_groups = len(duration_groups)
n_cols = 3
n_rows = (n_groups + n_cols - 1) // n_cols  # 向上取整

# 创建热度值的颜色映射
min_heat = df['popularity_normalized'].min()
max_heat = df['popularity_normalized'].max()
norm = plt.Normalize(min_heat, max_heat)

for i, duration in enumerate(duration_groups):
    # 创建子图
    ax = plt.subplot(n_rows, n_cols, i+1)

    # 获取该时长组的数据
    group_data = df[df['duration_group'] == duration]

    # 绘制热度分布直方图
    hist, bins, _ = plt.hist(
        group_data['popularity_normalized'], 
        bins=20, 
        alpha=0.7,
        color=plt.cm.Blues(norm(group_data['popularity_normalized'].mean()))
    )

    # 添加平均值线
    mean_val = group_data['popularity_normalized'].mean()
    plt.axvline(x=mean_val, color='red', linestyle='--', linewidth=1.5)

    # 添加注释
    plt.text(
        mean_val, 
        hist.max() * 0.8, 
        f"平均: {mean_val:.4f}", 
        rotation=90, 
        va='top', 
        ha='right',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    # 设置子图标题和标签
    plt.title(f"{duration}", fontsize=12)

    # 只在底部子图添加x轴标签
    if i >= n_groups - n_cols:
        plt.xlabel('热度值', fontsize=10)
    else:
        plt.xticks([])

    # 只在左侧子图添加y轴标签
    if i % n_cols == 0:
        plt.ylabel('视频数量', fontsize=10)
    else:
        plt.yticks([])

    plt.grid(True, alpha=0.3)

plt.suptitle('各时长组的热度分布图', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('duration_heat_distribution_grid.png', dpi=300, bbox_inches='tight')
plt.show()
