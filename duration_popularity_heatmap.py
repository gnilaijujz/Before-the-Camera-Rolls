import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.style.use('ggplot')  # 使用ggplot风格，美化图表

# 读取数据
df = pd.read_csv("D:\下载\sports_2024-2025_with_popularity.csv")

# 将秒转换为分钟
df['duration_minutes'] = df['duration_seconds'] / 60

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

# 定义更好看的颜色映射
colors = {'短视频': '#3274A1', '中长视频': '#E1812C', '长视频': '#3A923A'}

# 1) 视频长度与热度的散点图（按类别着色）
plt.figure(figsize=(12, 8))

# 设置x轴范围，避免极端值
max_duration_to_show = min(df['duration_minutes'].max(), 120)  # 限制在120分钟内

# 创建散点图，按类别着色
for category in ['短视频', '中长视频', '长视频']:
    subset = df[df['video_category'] == category]
    plt.scatter(subset['duration_minutes'].clip(upper=max_duration_to_show), 
               subset['popularity_normalized'], 
               alpha=0.6, label=category, color=colors[category], 
               edgecolors='w', s=50)

# 添加垂直线标记类别阈值
plt.axvline(x=short_threshold, color='black', linestyle='--', alpha=0.5, 
           label=f'短视频上限: {short_threshold}分钟')
plt.axvline(x=medium_threshold, color='black', linestyle='--', alpha=0.5,
           label=f'中长视频上限: {medium_threshold}分钟')

plt.title('视频时长与热度关系散点图', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# 2) 视频长度与热度的关系（热力图）- 使用核密度估计
plt.figure(figsize=(12, 8))

# 准备数据，剔除极端值以获得更好的可视化效果
x = df['duration_minutes'].clip(upper=max_duration_to_show)
y = df['popularity_normalized']
xy = np.vstack([x, y])

# 计算核密度估计
z = gaussian_kde(xy)(xy)

# 创建排序，以确保高密度点绘制在上面
idx = z.argsort()
x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

# 创建一个自定义的渐变色彩映射
custom_cmap = LinearSegmentedColormap.from_list("custom", 
                                              ["#f0f0f0", "#c6dbef", "#4292c6", "#08306b"])

# 绘制散点图，使用核密度估计值确定颜色
plt.scatter(x, y, c=z, s=50, alpha=0.8, cmap=custom_cmap, edgecolors='k', linewidths=0.3)

# 添加垂直线标记类别阈值
plt.axvline(x=short_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7, 
           label=f'短视频上限: {short_threshold}分钟')
plt.axvline(x=medium_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'中长视频上限: {medium_threshold}分钟')

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('密度', rotation=270, labelpad=20, fontsize=12)

plt.title('视频时长与热度关系热力图', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3) 2D热力图（使用六边形分箱）
plt.figure(figsize=(12, 8))

# 创建六边形分箱图 (hexbin)
hb = plt.hexbin(df['duration_minutes'].clip(upper=max_duration_to_show), 
               df['popularity_normalized'], 
               gridsize=30, cmap='Blues', mincnt=1)

# 添加垂直线标记类别阈值
plt.axvline(x=short_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
           label=f'短视频上限: {short_threshold}分钟')
plt.axvline(x=medium_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'中长视频上限: {medium_threshold}分钟')

# 添加颜色条
cb = plt.colorbar(hb)
cb.set_label('视频数量', rotation=270, labelpad=20, fontsize=12)

plt.title('视频时长与热度关系六边形分箱图', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 4) 热度分布箱线图 - 按视频类别比较
plt.figure(figsize=(10, 6))
sns.boxplot(x='video_category', y='popularity_normalized', data=df, 
           order=['短视频', '中长视频', '长视频'], palette=colors)

plt.title('各类别视频热度分布箱线图', fontsize=16, fontweight='bold')
plt.xlabel('视频类别', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.show()

# 5) 带有回归线的热度与时长关系图
plt.figure(figsize=(12, 8))

# 绘制散点图和回归线
sns.regplot(x='duration_minutes', y='popularity_normalized', data=df.loc[df['duration_minutes'] <= max_duration_to_show], 
           scatter_kws={'alpha': 0.5, 's': 30, 'color': 'steelblue'}, 
           line_kws={'color': 'red', 'linewidth': 2})

# 添加垂直线标记类别阈值
plt.axvline(x=short_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axvline(x=medium_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title('视频时长与热度关系及趋势线', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加标注来解释回归线
# 计算相关系数
correlation = df.loc[df['duration_minutes'] <= max_duration_to_show, ['duration_minutes', 'popularity_normalized']].corr().iloc[0,1]
plt.annotate(f"相关系数: {correlation:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            va='top', fontsize=12)

plt.tight_layout()
plt.show()

# 6) 热度与时长的2D密度分布热力图（使用KDE）
plt.figure(figsize=(10, 8))

# 使用seaborn的kdeplot创建2D密度图
ax = sns.kdeplot(x=df['duration_minutes'].clip(upper=max_duration_to_show), 
                y=df['popularity_normalized'], 
                cmap="Blues", fill=True, thresh=0, levels=15)

# 添加散点图用小点表示原始数据
plt.scatter(df['duration_minutes'].clip(upper=max_duration_to_show), 
           df['popularity_normalized'], 
           c='black', s=5, alpha=0.1)

# 添加垂直线标记类别阈值
plt.axvline(x=short_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axvline(x=medium_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

plt.title('视频时长与热度关系密度热力图', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)

# 添加标记表示短视频、中长视频和长视频区域
plt.text(short_threshold/2, 0.95, '短视频', ha='center', fontsize=12, 
        bbox=dict(boxstyle="round,pad=0.3", fc=colors['短视频'], ec="gray", alpha=0.3))
plt.text((short_threshold + medium_threshold)/2, 0.95, '中长视频', ha='center', fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc=colors['中长视频'], ec="gray", alpha=0.3))
plt.text(medium_threshold + 10, 0.95, '长视频', ha='center', fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc=colors['长视频'], ec="gray", alpha=0.3))

plt.tight_layout()
plt.show()

# 7) 计算并展示各类别的平均热度
avg_popularity = df.groupby('video_category')['popularity_normalized'].mean().reindex(['短视频', '中长视频', '长视频'])
print("\n各类别视频的平均热度:")
for category, avg in avg_popularity.items():
    print(f"{category}: {avg:.4f}")

# 绘制平均热度柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(avg_popularity.index, avg_popularity.values, color=[colors[cat] for cat in avg_popularity.index], width=0.6)

# 在柱状图上标注具体数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('各类别视频的平均热度比较', fontsize=16, fontweight='bold')
plt.xlabel('视频类别', fontsize=14)
plt.ylabel('平均热度（归一化）', fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(avg_popularity.values) * 1.2)  # 设置y轴上限为最大值的1.2倍
plt.tight_layout()
plt.show()

# 8) 创建一个热度-时长的热图，按照类别划分区域
plt.figure(figsize=(14, 9))

# 创建分箱数据进行热图绘制
# 限制时长范围以获得更好的可视化效果
max_duration = min(df['duration_minutes'].max(), 120)
x_bins = np.linspace(0, max_duration, 40)  # 时长分箱
y_bins = np.linspace(0, df['popularity_normalized'].max(), 40)  # 热度分箱

# 创建二维直方图数据
H, xedges, yedges = np.histogram2d(
    df['duration_minutes'].clip(upper=max_duration),
    df['popularity_normalized'],
    bins=[x_bins, y_bins]
)

# 转置H以匹配imshow期望的方向
H = H.T

# 使用一个更好的颜色映射
plt.imshow(H, interpolation='gaussian', origin='lower', aspect='auto',
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
          cmap='viridis')

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('视频数量', rotation=270, labelpad=20, fontsize=12)

# 添加垂直线标记类别阈值
plt.axvline(x=short_threshold, color='red', linestyle='-', linewidth=2, alpha=0.7)
plt.axvline(x=medium_threshold, color='red', linestyle='-', linewidth=2, alpha=0.7)

# 为不同区域添加标签
plt.text(short_threshold/2, df['popularity_normalized'].max()*0.9, '短视频', 
        ha='center', fontsize=14, color='white', weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc='black', ec="white", alpha=0.5))
plt.text((short_threshold + medium_threshold)/2, df['popularity_normalized'].max()*0.9, '中长视频', 
        ha='center', fontsize=14, color='white', weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc='black', ec="white", alpha=0.5))
plt.text(medium_threshold + 15, df['popularity_normalized'].max()*0.9, '长视频', 
        ha='center', fontsize=14, color='white', weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc='black', ec="white", alpha=0.5))

plt.title('视频时长与热度关系热力图', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()
