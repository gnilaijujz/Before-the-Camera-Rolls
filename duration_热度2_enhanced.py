import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# 设置更专业的绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']  # 中文字体优先级
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['savefig.dpi'] = 300  # 高分辨率保存
plt.rcParams['figure.dpi'] = 100   # 显示分辨率

# 1. 读取并预处理
df = pd.read_csv(r"D:\下载\sports_2024-2025_with_popularity.csv")
df['duration_min'] = df['duration_seconds'] / 60

print(f"数据集中共有 {len(df)} 条视频记录")
print(f"热度值范围: {df['popularity_normalized'].min():.4f} - {df['popularity_normalized'].max():.4f}")
print(f"视频时长范围: {df['duration_min'].min():.1f} - {df['duration_min'].max():.1f} 分钟")

# 2. 业务时长分组
bins = [0, 3, 5, 10, 20, 30, np.inf]
labels = [
    '0-3分钟', '3-5分钟', '5-10分钟',
    '10-20分钟', '20-30分钟', '30+分钟'
]

# 保留Interval对象以便访问left和right属性
df['dur_group_interval'] = pd.cut(df['duration_min'], bins=bins, right=False)
# 同时保存字符串标签版本用于显示
df['dur_group'] = pd.cut(df['duration_min'], bins=bins, labels=labels, right=False)

# 打印各时长组的视频数量和占比
dur_group_counts = df['dur_group'].value_counts().sort_index()
dur_group_percent = dur_group_counts / len(df) * 100
print("\n各时长组的视频数量和占比:")
for group, count in dur_group_counts.items():
    percent = dur_group_percent[group]
    print(f"{group}: {count}个视频 ({percent:.1f}%)")

# 3. 对每个组再等分成 6 段，标号 0～5
def assign_subbin(x):
    grp_interval = x['dur_group_interval']
    if pd.isna(grp_interval):
        return np.nan

    low, high = grp_interval.left, grp_interval.right
    # 对于最后一个开区间 (30+, inf) 把 inf 临时设为 max+0.001
    if np.isinf(high):
        high = df.loc[df['dur_group'] == x['dur_group'], 'duration_min'].max() + 1e-3

    span = high - low
    # 计算该视频在组内所处的段号
    idx = int(np.floor((x['duration_min'] - low) / span * 6))
    return min(max(idx, 0), 5)

df['sub_bin'] = df.apply(assign_subbin, axis=1)

# 为每个子区间计算实际的时长范围，用于标注
def get_subbin_ranges(dur_group_interval, n_bins=6):
    """计算指定时长组内每个子区间的实际时长范围"""
    low, high = dur_group_interval.left, dur_group_interval.right
    if np.isinf(high):
        high = df.loc[df['dur_group_interval'] == dur_group_interval, 'duration_min'].max() + 1e-3

    ranges = []
    span = high - low
    for i in range(n_bins):
        bin_low = low + i * span / n_bins
        bin_high = low + (i + 1) * span / n_bins
        ranges.append((bin_low, bin_high))
    return ranges

# 获取每个时长组的子区间范围
subbin_ranges = {}
for dur_group in df['dur_group_interval'].dropna().unique():
    subbin_ranges[dur_group] = get_subbin_ranges(dur_group)

# 4. 计算「组 × 段」的平均热度
pivot = (
    df
    .groupby(['dur_group', 'sub_bin'])['popularity_normalized']
    .mean()
    .unstack(fill_value=0)
)

# 计算每个单元格中的视频数量，用于标注
count_pivot = (
    df
    .groupby(['dur_group', 'sub_bin']).size()
    .unstack(fill_value=0)
)

# 5. 创建自定义颜色映射 - 更吸引人的红黄渐变
colors = ["#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
custom_cmap = LinearSegmentedColormap.from_list("custom_heat", colors)

# 6. 绘制增强版热力图
plt.figure(figsize=(14, 10))

# 绘制主热力图
ax = sns.heatmap(
    pivot,
    cmap=custom_cmap,
    annot=False,  # 先不添加注释，后面自定义添加
    fmt=".2f",
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': '平均热度值', 'shrink': 0.8}
)

# 添加自定义注释 - 同时显示热度值和视频数量
for i, dur_group in enumerate(pivot.index):
    for j, sub_bin in enumerate(pivot.columns):
        heat_value = pivot.iloc[i, j]
        count = count_pivot.iloc[i, j]

        # 确定文本颜色 - 深色背景用白色文本
        text_color = 'white' if heat_value > pivot.values.mean() else 'black'

        # 添加热度值和视频数量注释
        ax.text(j + 0.5, i + 0.5, f"{heat_value:.3f}\n({count}个)",
               ha='center', va='center', fontsize=9, 
               color=text_color, fontweight='bold')

# 7. 美化图表
title_text = "视频时长精细分析：各时长组内子区间的热度分布"
subtitle_text = "每个时长组平均分为6个子区间，颜色越深表示热度越高"

plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
plt.title(subtitle_text, fontsize=12, pad=10)

# 设置更详细的x轴标签 - 显示子区间序号
subbin_labels = [f"{i+1}段" for i in range(6)]
ax.set_xticklabels(subbin_labels, rotation=0, fontsize=10)

# 设置y轴标签 - 时长组
ax.set_yticklabels(pivot.index, rotation=0, fontsize=10)

# 添加轴标签和说明
ax.set_xlabel("时长组内的子区间序号", fontsize=12, labelpad=10)
ax.set_ylabel("视频时长分组", fontsize=12, labelpad=10)

# 8. 添加图表说明文本
explanation_text = """说明：
1. 每个时长组(如0-3分钟)被等分为6个子区间
2. 颜色表示子区间内视频的平均热度
3. 单元格数值格式：热度值(视频数量)
"""

plt.figtext(0.02, 0.02, explanation_text, fontsize=10, 
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# 9. 添加每个子区间的实际时长范围说明（在右侧）
subbin_text = "子区间实际时长范围:\n"
for i, dur_group in enumerate(df['dur_group'].unique()):
    # 获取对应的interval对象
    interval = df.loc[df['dur_group'] == dur_group, 'dur_group_interval'].iloc[0]

    ranges = subbin_ranges[interval]
    subbin_text += f"\n{dur_group}:\n"
    for j, (low, high) in enumerate(ranges):
        # 处理无穷大
        high_str = f"{high:.1f}" if high < 1000 else "最大值"
        subbin_text += f"  {j+1}段: {low:.1f}-{high_str}分钟\n"

plt.figtext(0.85, 0.5, subbin_text, fontsize=9, va='center',
           bbox=dict(facecolor='#f9f9f9', alpha=0.9, boxstyle='round,pad=0.5'))

# 10. 调整布局并保存
plt.tight_layout(rect=[0, 0.03, 0.82, 0.95])  # 为右侧说明文本留出空间
plt.savefig('duration_heat_subintervals_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 额外创建一个热度条形图比较 - 不同时长组的整体热度对比
plt.figure(figsize=(12, 6))

# 计算每个时长组的平均热度
duration_heat_mean = df.groupby('dur_group')['popularity_normalized'].mean().reset_index()
duration_heat_mean = duration_heat_mean.sort_values('popularity_normalized', ascending=False)

# 为条形图设置渐变色
bars = plt.bar(
    duration_heat_mean['dur_group'], 
    duration_heat_mean['popularity_normalized'], 
    color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(duration_heat_mean)))
)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, height + 0.0005,
        f'{height:.4f}', ha='center', va='bottom', 
        fontsize=11, fontweight='bold'
    )

plt.title('各时长组的平均热度比较', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('视频时长分组', fontsize=12, labelpad=10)
plt.ylabel('平均热度值', fontsize=12, labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('duration_heat_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show()
