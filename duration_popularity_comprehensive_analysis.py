import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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

# 1) 使用固定阈值定义视频类别
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

# 2) 统计各类别的数量和热度特征
category_stats = df.groupby('video_category').agg(
    视频数量=('video_category', 'count'),
    平均热度=('popularity_normalized', 'mean'),
    中位热度=('popularity_normalized', 'median'),
    最小热度=('popularity_normalized', 'min'),
    最大热度=('popularity_normalized', 'max'),
    热度标准差=('popularity_normalized', 'std')
).reset_index()

# 按照短、中长、长视频的顺序排序
category_order = ['短视频', '中长视频', '长视频']
category_stats = category_stats.set_index('video_category').reindex(category_order).reset_index()

print("\n固定阈值分类结果及热度统计：")
print(f"短视频: < {short_threshold}分钟")
print(f"中长视频: {short_threshold}-{medium_threshold}分钟")
print(f"长视频: > {medium_threshold}分钟")
print(category_stats)

# 3) 热度分布箱线图 - 按视频类别
plt.figure(figsize=(12, 7))
sns.boxplot(x='video_category', y='popularity_normalized', data=df, 
          order=category_order, palette=['#3274A1', '#E1812C', '#3A923A'])

# 添加小提琴图层
sns.stripplot(x='video_category', y='popularity_normalized', data=df,
             order=category_order, size=4, color='.3', alpha=0.3)

# 添加标题和标签
plt.title('各类别视频热度分布箱线图', fontsize=16, fontweight='bold')
plt.xlabel('视频类别', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')

# 在每个类别旁添加统计信息
for i, cat in enumerate(category_order):
    cat_data = df[df['video_category'] == cat]
    plt.text(i, -0.05, 
            f"平均: {cat_data['popularity_normalized'].mean():.4f}\n"
            f"中位: {cat_data['popularity_normalized'].median():.4f}\n"
            f"数量: {len(cat_data)}",
            ha='center', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()

# 4) 热度分布直方图 - 按类别着色
plt.figure(figsize=(14, 8))

# 使用不同颜色绘制各类别的直方图
colors = {'短视频': '#3274A1', '中长视频': '#E1812C', '长视频': '#3A923A'}
for category in category_order:
    subset = df[df['video_category'] == category]
    sns.histplot(subset['popularity_normalized'], kde=True, 
                label=category, color=colors[category], alpha=0.5, bins=30)

plt.title('各类别视频热度分布直方图', fontsize=16, fontweight='bold')
plt.xlabel('热度（归一化）', fontsize=14)
plt.ylabel('频数', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# 5) 视频时长与热度的散点图和回归线（按类别）
plt.figure(figsize=(14, 8))

# 限制时长以获得更好的可视化效果
max_duration_to_show = 120  # 最多显示120分钟

# 按类别添加回归线
for category in category_order:
    subset = df[(df['video_category'] == category) & 
               (df['duration_minutes'] <= max_duration_to_show)]

    # 散点图和回归线
    sns.regplot(x='duration_minutes', y='popularity_normalized', data=subset,
               scatter_kws={'alpha': 0.5, 's': 30, 'label': category}, 
               line_kws={'label': f"{category}趋势线"},
               color=colors[category])

# 添加垂直线标记类别阈值
plt.axvline(x=short_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'短视频上限: {short_threshold}分钟')
plt.axvline(x=medium_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'中长视频上限: {medium_threshold}分钟')

plt.title('视频时长与热度关系（按类别）', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# 6) 热度核密度估计 - 类别对比
plt.figure(figsize=(14, 7))

# 使用KDE图比较各类别热度分布
for category in category_order:
    subset = df[df['video_category'] == category]
    sns.kdeplot(data=subset['popularity_normalized'], 
               label=f"{category} (n={len(subset)})",
               color=colors[category], fill=True, alpha=0.3)

# 添加全局平均值的垂直线
mean_popularity = df['popularity_normalized'].mean()
plt.axvline(x=mean_popularity, color='red', linestyle='--', 
           label=f'全局平均热度: {mean_popularity:.4f}')

plt.title('各类别视频热度分布密度图', fontsize=16, fontweight='bold')
plt.xlabel('热度（归一化）', fontsize=14)
plt.ylabel('密度', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# 7) 时长区间热度分析 - 5分钟间隔
# 创建5分钟间隔的时长组
def group_by_5min(duration):
    upper_bound = np.ceil(duration / 5) * 5
    lower_bound = upper_bound - 5
    return f"{int(lower_bound)}-{int(upper_bound)}"

df['duration_group_5min'] = df['duration_minutes'].apply(group_by_5min)

# 计算每个5分钟时长组的热度统计
duration_heat_stats = df.groupby('duration_group_5min').agg(
    视频数量=('popularity_normalized', 'count'),
    平均热度=('popularity_normalized', 'mean'),
    中位热度=('popularity_normalized', 'median'),
    热度标准差=('popularity_normalized', 'std')
).reset_index()

# 提取组别的下限值，用于排序
def safe_extract_lower(group_name):
    try:
        return int(group_name.split('-')[0])
    except:
        return 999999

duration_heat_stats['lower_bound'] = duration_heat_stats['duration_group_5min'].apply(safe_extract_lower)
duration_heat_stats = duration_heat_stats.sort_values('lower_bound')

# 只选择前20个组（0-100分钟）
top_20_groups = duration_heat_stats.iloc[:20].copy()

# 创建双轴图
fig, ax1 = plt.subplots(figsize=(16, 8))

# 视频数量柱状图
bars = ax1.bar(top_20_groups['duration_group_5min'], top_20_groups['视频数量'],
              alpha=0.6, color='skyblue', label='视频数量')

# 在柱子上标注数量
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('视频时长分组（分钟）', fontsize=14)
ax1.set_ylabel('视频数量', fontsize=14)
ax1.tick_params(axis='y')
ax1.set_xticklabels(top_20_groups['duration_group_5min'], rotation=45, ha='right')

# 第二个Y轴 - 热度
ax2 = ax1.twinx()
heat_line = ax2.plot(top_20_groups['duration_group_5min'], top_20_groups['平均热度'],
                    'ro-', linewidth=2, label='平均热度')
heat_fill = ax2.fill_between(top_20_groups['duration_group_5min'],
                           top_20_groups['平均热度'] - top_20_groups['热度标准差'],
                           top_20_groups['平均热度'] + top_20_groups['热度标准差'],
                           color='red', alpha=0.2, label='±1个标准差')

ax2.set_ylabel('平均热度', color='r', fontsize=14)
ax2.tick_params(axis='y', labelcolor='r')

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('视频时长与热度关系（5分钟分组）', fontsize=16, fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.show()

# 8) 热门Top 10视频时长分析
top_videos = df.sort_values('popularity_normalized', ascending=False).head(10).copy()
top_videos['rank'] = range(1, 11)

plt.figure(figsize=(14, 8))

# 创建横向柱状图
bars = plt.barh(top_videos['rank'], top_videos['duration_minutes'],
              color=plt.cm.plasma(np.linspace(0, 0.8, 10)), alpha=0.8)

# 添加热度和时长标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    heat = top_videos.iloc[i]['popularity_normalized']
    plt.text(width + 1, bar.get_y() + bar.get_height()/2,
            f"热度: {heat:.4f} | {width:.1f}分钟",
            ha='left', va='center', fontsize=10)

plt.title('热门Top10视频的时长分析', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度排名', fontsize=14)
plt.gca().invert_yaxis()  # 倒置Y轴，使排名1在顶部
plt.grid(True, linestyle='--', alpha=0.5, axis='x')
plt.tight_layout()
plt.show()

# 9) 热力图 - 时长与热度的二维分布
plt.figure(figsize=(16, 10))

# 限制视频时长范围，以便更好地可视化
filtered_df = df[df['duration_minutes'] <= 120].copy()

# 创建二维热图
heatmap = plt.hist2d(filtered_df['duration_minutes'], filtered_df['popularity_normalized'],
                   bins=[24, 20], cmap='YlOrRd', alpha=0.8)

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('视频数量', rotation=270, labelpad=20, fontsize=12)

# 添加类别分界线
plt.axvline(x=short_threshold, color='black', linestyle='-', linewidth=1.5)
plt.axvline(x=medium_threshold, color='black', linestyle='-', linewidth=1.5)

# 添加类别标签
plt.text(short_threshold/2, filtered_df['popularity_normalized'].max() * 0.9,
        '短视频', ha='center', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.text((short_threshold + medium_threshold)/2, filtered_df['popularity_normalized'].max() * 0.9,
        '中长视频', ha='center', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.text(medium_threshold + 20, filtered_df['popularity_normalized'].max() * 0.9,
        '长视频', ha='center', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.title('视频时长与热度二维分布热图', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()

# 10) 分类热度的小提琴图对比
plt.figure(figsize=(14, 8))
sns.violinplot(x='video_category', y='popularity_normalized', data=df,
              order=category_order, palette=colors, inner='quartile')

# 添加数据点
sns.swarmplot(x='video_category', y='popularity_normalized', data=df,
             order=category_order, color='black', alpha=0.3, size=3)

plt.title('各类别视频热度分布小提琴图', fontsize=16, fontweight='bold')
plt.xlabel('视频类别', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
plt.show()

# 11) 热度与视频类别的关系总结
mean_heat_by_category = df.groupby('video_category')['popularity_normalized'].mean().reindex(category_order)
median_heat_by_category = df.groupby('video_category')['popularity_normalized'].median().reindex(category_order)

plt.figure(figsize=(12, 7))

# 绘制均值和中位数的柱状图
x = np.arange(len(category_order))
width = 0.35

bar1 = plt.bar(x - width/2, mean_heat_by_category, width, 
              label='平均热度', color=[colors[cat] for cat in category_order], alpha=0.7)
bar2 = plt.bar(x + width/2, median_heat_by_category, width, 
              label='中位热度', color=[colors[cat] for cat in category_order], alpha=0.4, 
              hatch='///')

# 添加数据标签
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

plt.title('各类别视频热度比较', fontsize=16, fontweight='bold')
plt.xlabel('视频类别', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.xticks(x, category_order)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
plt.show()

# 12) 视频时长分布
plt.figure(figsize=(14, 8))

# 绘制每个类别的时长分布
for category in category_order:
    subset = df[df['video_category'] == category]
    sns.kdeplot(subset['duration_minutes'], label=f"{category} (n={len(subset)})",
               color=colors[category], fill=True, alpha=0.3)

# 添加垂直线标记类别边界
plt.axvline(x=short_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'短视频上限: {short_threshold}分钟')
plt.axvline(x=medium_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'中长视频上限: {medium_threshold}分钟')

plt.title('各类别视频时长分布', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('密度', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
