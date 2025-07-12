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

# 2. 更精细的热度分类 (6个等级)
df['pop_quantile'] = pd.qcut(
    df['popularity_normalized'],
    q=6,
    labels=['Q1(0-17%)', 'Q2(17-33%)', 'Q3(33-50%)', 'Q4(50-67%)', 'Q5(67-83%)', 'Q6(83-100%)']
)

# 将热度量化分类映射到更友好的标签
heat_map = {
    'Q1(0-17%)': '极低热度', 
    'Q2(17-33%)': '低热度',
    'Q3(33-50%)': '中低热度', 
    'Q4(50-67%)': '中高热度',
    'Q5(67-83%)': '高热度', 
    'Q6(83-100%)': '极高热度'
}
df['heat_group'] = df['pop_quantile'].map(heat_map)

# 3. 构造「占比」透视表：行=时长组，列=热度档，值=各档视频数占本行总数的比例
pivot = df.pivot_table(
    index='duration_group',
    columns='heat_group',
    values='popularity_normalized',
    aggfunc='count'
).fillna(0)

# 计算每行占比
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

# 也计算每列占比（可选用）
pivot_col_pct = pivot.div(pivot.sum(axis=0), axis=1)

# 确保热度列顺序正确
heat_order = ['极低热度', '低热度', '中低热度', '中高热度', '高热度', '极高热度']
pivot_pct = pivot_pct[heat_order]

# 4. 创建更美观的热力图（横向版）
plt.figure(figsize=(14, 8))

# 创建自定义颜色映射 - 使用渐变色从浅到深
colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
cmap = LinearSegmentedColormap.from_list("custom_blues", colors)

# 绘制热力图
ax = sns.heatmap(
    pivot_pct,
    cmap=cmap,
    annot=True,
    fmt=".1%",
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': '占该时长段视频的百分比', 'format': mtick.PercentFormatter(1.0)}
)

# 美化图表
ax.set_title("各时长段视频的热度分布热力图", pad=20, fontsize=18, fontweight='bold')
ax.set_xlabel("热度分类", fontsize=14, labelpad=10)
ax.set_ylabel("视频时长分类", fontsize=14, labelpad=10)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)

# 添加网格线
for i in range(len(pivot_pct.index)):
    ax.axhline(y=i, color='white', linewidth=1.5)

for j in range(len(pivot_pct.columns)):
    ax.axvline(x=j, color='white', linewidth=1.5)

plt.tight_layout()
plt.savefig('duration_heat_heatmap_horizontal.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 创建垂直版热力图（更适合展示更多分类）
plt.figure(figsize=(10, 12))

# 创建自定义颜色映射 - 使用火焰色反转
colors_flare = ["#f0f0f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]
cmap_flare = LinearSegmentedColormap.from_list("custom_flare", colors_flare)

# 转置透视表用于垂直显示
pivot_pct_T = pivot_pct.T

# 绘制热力图
ax = sns.heatmap(
    pivot_pct_T,
    cmap=cmap_flare,
    annot=True,
    fmt=".1%",
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': '占该时长段视频的百分比', 'format': mtick.PercentFormatter(1.0)}
)

# 美化图表
ax.set_title("热度分类 vs 视频时长分布热力图", pad=20, fontsize=18, fontweight='bold')
ax.set_ylabel("热度分类", fontsize=14, labelpad=10)
ax.set_xlabel("视频时长分类", fontsize=14, labelpad=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)
ax.tick_params(axis='x', rotation=45, labelsize=10)

# 添加网格线
for i in range(len(pivot_pct_T.index)):
    ax.axhline(y=i, color='white', linewidth=1.5)

for j in range(len(pivot_pct_T.columns)):
    ax.axvline(x=j, color='white', linewidth=1.5)

plt.tight_layout()
plt.savefig('duration_heat_heatmap_vertical.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 热力图 - 时长与热度的具体分布（不同视角）
# 创建热度阈值对应的实际值
quantiles = df['popularity_normalized'].quantile([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]).round(4)
print("热度分位数阈值:")
for i, q in enumerate(quantiles):
    if i < len(quantiles)-1:
        print(f"{heat_order[i]}: {q} - {quantiles[i+1]}")

# 创建直方图热力图（显示实际分布密度）
plt.figure(figsize=(14, 10))

# 限制最大时长以获得更好的可视化效果
max_duration = 60  # 最多显示60分钟
filtered_df = df[df['duration_min'] <= max_duration]

# 创建2D直方图
h = plt.hist2d(filtered_df['duration_min'], filtered_df['popularity_normalized'], 
             bins=[30, 30], cmap='viridis', alpha=0.8)

# 添加颜色条
cbar = plt.colorbar(h[3])
cbar.set_label('视频数量', rotation=270, labelpad=20, fontsize=12)

# 添加时长分类的垂直线
for bin_edge in duration_bins[1:-1]:
    if bin_edge <= max_duration:
        plt.axvline(x=bin_edge, color='red', linestyle='--', alpha=0.7)

# 添加热度分类的水平线
for i in range(1, len(quantiles)-1):
    plt.axhline(y=quantiles[i], color='white', linestyle='--', alpha=0.7)

# 添加类别标签
for i in range(len(duration_bins)-1):
    if duration_bins[i] < max_duration and duration_bins[i+1] <= max_duration:
        mid_x = (duration_bins[i] + min(duration_bins[i+1], max_duration)) / 2
        plt.text(mid_x, filtered_df['popularity_normalized'].max() * 0.95, 
                duration_labels[i].split('(')[0], 
                ha='center', color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

# 添加热度标签
for i in range(len(quantiles)-1):
    mid_y = (quantiles[i] + quantiles[i+1]) / 2
    plt.text(1, mid_y, heat_order[i], color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

plt.title('视频时长与热度分布热力图', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('视频时长（分钟）', fontsize=14, labelpad=10)
plt.ylabel('热度（归一化）', fontsize=14, labelpad=10)
plt.grid(False)
plt.tight_layout()
plt.savefig('duration_heat_density.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 按热度分组的堆叠柱状图
plt.figure(figsize=(16, 8))

# 创建堆叠柱状图数据
stack_data = pivot.copy()

# 绘制堆叠柱状图
ax = stack_data.plot(
    kind='bar', 
    stacked=True, 
    figsize=(16, 8),
    colormap='viridis',
    width=0.7
)

# 在每个段上添加百分比标签
for i, (name, values) in enumerate(pivot_pct.iterrows()):
    total = 0
    for j, value in enumerate(values):
        if value >= 0.05:  # 只显示占比5%以上的标签
            plt.text(i, total + value/2, f'{value:.1%}', 
                    ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold')
        total += value

plt.title('各时长段视频的热度占比分布', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('视频时长分类', fontsize=14, labelpad=10)
plt.ylabel('视频数量占比', fontsize=14, labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.legend(title='热度分类', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('duration_heat_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 按时长分组的堆叠柱状图（展示每个热度组的时长分布）
plt.figure(figsize=(16, 8))

# 计算各热度组中的时长分布（列归一化）
pivot_heat_normalized = pivot.div(pivot.sum(axis=0), axis=1).T

# 绘制堆叠柱状图
ax = pivot_heat_normalized.plot(
    kind='bar', 
    stacked=True, 
    figsize=(16, 8),
    colormap='plasma',
    width=0.7
)

# 在每个段上添加百分比标签
for i, (name, values) in enumerate(pivot_heat_normalized.iterrows()):
    total = 0
    for j, value in enumerate(values):
        if value >= 0.05:  # 只显示占比5%以上的标签
            plt.text(i, total + value/2, f'{value:.1%}', 
                    ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold')
        total += value

plt.title('各热度级别的视频时长分布', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('热度分类', fontsize=14, labelpad=10)
plt.ylabel('视频数量占比', fontsize=14, labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.legend(title='视频时长', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('duration_heat_stacked_by_heat.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 添加气泡图来直观展示数量分布
plt.figure(figsize=(14, 10))

# 为了气泡图，我们需要原始计数
bubble_data = pivot.copy()

# 创建归一化后的气泡大小，使气泡适当大小
max_count = bubble_data.values.max()
bubble_sizes = bubble_data / max_count * 1000  # 将最大气泡尺寸设为1000

# 创建X和Y坐标
x_coords = []
y_coords = []
sizes = []
colors = []
annotations = []

# 为每个单元格准备数据
for i, row_label in enumerate(bubble_data.index):
    for j, col_label in enumerate(bubble_data.columns):
        x_coords.append(j)
        y_coords.append(i)
        sizes.append(bubble_sizes.iloc[i, j])
        colors.append(j / len(bubble_data.columns))  # 基于列索引的颜色
        annotations.append(str(int(bubble_data.iloc[i, j])))

# 绘制气泡图
scatter = plt.scatter(x_coords, y_coords, s=sizes, c=colors, cmap='viridis', 
                     alpha=0.7, edgecolors='white')

# 添加文本标注（视频数量）
for i, txt in enumerate(annotations):
    if int(txt) > 0:  # 只标注非零值
        plt.annotate(txt, (x_coords[i], y_coords[i]), 
                    ha='center', va='center', color='white',
                    fontweight='bold')

# 设置轴标签
plt.yticks(range(len(bubble_data.index)), bubble_data.index)
plt.xticks(range(len(bubble_data.columns)), bubble_data.columns, rotation=45, ha='right')

plt.grid(False)
plt.colorbar(scatter, label='热度级别')
plt.title('视频时长与热度分布气泡图', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('热度分类', fontsize=14, labelpad=10)
plt.ylabel('视频时长分类', fontsize=14, labelpad=10)
plt.tight_layout()
plt.savefig('duration_heat_bubble.png', dpi=300, bbox_inches='tight')
plt.show()
