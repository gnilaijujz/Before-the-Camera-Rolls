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

# 1) 定义视频类别（使用固定阈值）
short_threshold = 30  # 短视频上限：30分钟
medium_threshold = 60  # 中长视频上限：60分钟

def categorize_video(duration):
    if duration < short_threshold:
        return '短视频'
    elif duration < medium_threshold:
        return '中长视频'
    else:
        return '长视频'

df['video_category'] = df['duration_minutes'].apply(categorize_video)

# 2) 热度阈值划分 - 将视频分为高热度和低热度
# 计算热度的分位数
heat_percentiles = df['popularity_normalized'].quantile([0.5, 0.75, 0.9])
low_heat_threshold = heat_percentiles[0.5]  # 使用中位数作为低热度阈值
medium_heat_threshold = heat_percentiles[0.75]  # 使用75%分位数作为中等热度阈值
high_heat_threshold = heat_percentiles[0.9]  # 使用90%分位数作为高热度阈值

def categorize_heat(heat):
    if heat < low_heat_threshold:
        return '低热度'
    elif heat < medium_heat_threshold:
        return '中等热度'
    elif heat < high_heat_threshold:
        return '高热度'
    else:
        return '超高热度'

df['heat_category'] = df['popularity_normalized'].apply(categorize_heat)

# 打印热度阈值
print("热度划分阈值:")
print(f"低热度: < {low_heat_threshold:.4f} (0-50%分位数)")
print(f"中等热度: {low_heat_threshold:.4f} - {medium_heat_threshold:.4f} (50-75%分位数)")
print(f"高热度: {medium_heat_threshold:.4f} - {high_heat_threshold:.4f} (75-90%分位数)")
print(f"超高热度: > {high_heat_threshold:.4f} (90-100%分位数)")

# 3) 统计各个类别和热度组合的数量
category_heat_counts = df.groupby(['video_category', 'heat_category']).size().unstack(fill_value=0)
print("\n各类别视频的热度分布:")
print(category_heat_counts)

# 4) 高对比度热力图 - 标准化版本
plt.figure(figsize=(16, 10))

# 限制视频时长范围，以便更好地可视化
max_duration_to_show = 90  # 最多显示90分钟
filtered_df = df[df['duration_minutes'] <= max_duration_to_show].copy()

# 定义自定义颜色映射，使高热度区域更加突出
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_heat", 
    ["#FFFFFF", "#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", 
     "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
)

# 创建二维热图，使用更多的bins以获得更高的分辨率
heatmap = plt.hist2d(
    filtered_df['duration_minutes'], 
    filtered_df['popularity_normalized'],
    bins=[45, 40],  # 更多的bins获得更细致的分布
    cmap=custom_cmap, 
    norm=plt.Normalize(0, 15)  # 调整颜色归一化以突出高热度区域
)

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('视频数量', rotation=270, labelpad=20, fontsize=12)

# 添加类别分界线
plt.axvline(x=short_threshold, color='black', linestyle='-', linewidth=2)
plt.axvline(x=medium_threshold, color='black', linestyle='-', linewidth=2)

# 添加热度阈值水平线
plt.axhline(y=low_heat_threshold, color='darkgreen', linestyle='--', linewidth=1.5)
plt.axhline(y=medium_heat_threshold, color='blue', linestyle='--', linewidth=1.5)
plt.axhline(y=high_heat_threshold, color='purple', linestyle='--', linewidth=1.5)

# 添加类别标签
plt.text(short_threshold/2, filtered_df['popularity_normalized'].max() * 0.95,
        '短视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.text((short_threshold + medium_threshold)/2, filtered_df['popularity_normalized'].max() * 0.95,
        '中长视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.text(medium_threshold + 10, filtered_df['popularity_normalized'].max() * 0.95,
        '长视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# 添加热度区域标签
plt.text(1, high_heat_threshold + 0.02, '超高热度 (90-100%分位)',
        ha='left', fontsize=12, color='purple',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
plt.text(1, medium_heat_threshold + 0.02, '高热度 (75-90%分位)',
        ha='left', fontsize=12, color='blue',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
plt.text(1, low_heat_threshold + 0.02, '中等热度 (50-75%分位)',
        ha='left', fontsize=12, color='darkgreen',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
plt.text(1, low_heat_threshold - 0.02, '低热度 (0-50%分位)',
        ha='left', va='top', fontsize=12, color='black',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title('视频时长与热度二维分布热图（高对比度）', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()

# 5) 聚焦短视频和中长视频的高热度区域 - 对数缩放版本
plt.figure(figsize=(16, 10))

# 筛选短视频和中长视频
short_medium_df = filtered_df[filtered_df['duration_minutes'] < medium_threshold].copy()

# 使用对数归一化创建热图，突出稀疏区域
from matplotlib.colors import LogNorm

heatmap = plt.hist2d(
    short_medium_df['duration_minutes'], 
    short_medium_df['popularity_normalized'],
    bins=[30, 40],  
    cmap='inferno',  # 使用更鲜明的颜色图
    norm=LogNorm()  # 使用对数归一化突出稀疏区域
)

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('视频数量 (对数缩放)', rotation=270, labelpad=20, fontsize=12)

# 添加类别分界线
plt.axvline(x=short_threshold, color='white', linestyle='-', linewidth=2)

# 添加热度阈值水平线
plt.axhline(y=low_heat_threshold, color='cyan', linestyle='--', linewidth=1.5)
plt.axhline(y=medium_heat_threshold, color='lime', linestyle='--', linewidth=1.5)
plt.axhline(y=high_heat_threshold, color='yellow', linestyle='--', linewidth=1.5)

# 添加类别标签
plt.text(short_threshold/2, short_medium_df['popularity_normalized'].max() * 0.95,
        '短视频', ha='center', fontsize=14, weight='bold', color='white',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))
plt.text(short_threshold + (medium_threshold-short_threshold)/2, short_medium_df['popularity_normalized'].max() * 0.95,
        '中长视频', ha='center', fontsize=14, weight='bold', color='white',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))

# 添加热度区域标签
plt.text(1, high_heat_threshold + 0.02, '超高热度 (90-100%分位)',
        ha='left', fontsize=12, color='yellow',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
plt.text(1, medium_heat_threshold + 0.02, '高热度 (75-90%分位)',
        ha='left', fontsize=12, color='lime',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
plt.text(1, low_heat_threshold + 0.02, '中等热度 (50-75%分位)',
        ha='left', fontsize=12, color='cyan',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
plt.text(1, low_heat_threshold - 0.02, '低热度 (0-50%分位)',
        ha='left', va='top', fontsize=12, color='white',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))

plt.title('短视频和中长视频热度分布（对数缩放）', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()

# 6) 热度密度等高线图 - 聚焦短视频和中长视频
plt.figure(figsize=(16, 10))

# 筛选短视频和中长视频
short_medium_df = filtered_df[filtered_df['duration_minutes'] < medium_threshold].copy()

# 使用KDE图创建平滑的密度等高线
sns.kdeplot(
    data=short_medium_df,
    x='duration_minutes',
    y='popularity_normalized',
    cmap="rocket_r",  # 使用反转的火箭配色，使高密度区域更明亮
    fill=True,  # 填充等高线
    thresh=0.05,  # 阈值设置
    levels=15,  # 等高线数量
    alpha=0.75  # 透明度
)

# 叠加散点图，按热度类别着色
colors = {'低热度': 'gray', '中等热度': 'green', '高热度': 'blue', '超高热度': 'red'}
for heat_cat, color in colors.items():
    subset = short_medium_df[short_medium_df['heat_category'] == heat_cat]
    plt.scatter(subset['duration_minutes'], subset['popularity_normalized'],
               c=color, s=20, alpha=0.5, label=heat_cat)

# 添加类别分界线
plt.axvline(x=short_threshold, color='black', linestyle='-', linewidth=2)

# 添加热度阈值水平线
plt.axhline(y=low_heat_threshold, color='darkgreen', linestyle='--', linewidth=1.5)
plt.axhline(y=medium_heat_threshold, color='blue', linestyle='--', linewidth=1.5)
plt.axhline(y=high_heat_threshold, color='purple', linestyle='--', linewidth=1.5)

# 添加类别标签
plt.text(short_threshold/2, short_medium_df['popularity_normalized'].max() * 0.95,
        '短视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.text(short_threshold + (medium_threshold-short_threshold)/2, short_medium_df['popularity_normalized'].max() * 0.95,
        '中长视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.title('短视频和中长视频热度密度分布', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.legend(title='热度类别')
plt.grid(False)
plt.tight_layout()
plt.show()

# 7) 热度饼图 - 按视频类别和热度类别
plt.figure(figsize=(18, 8))

# 创建子图
plt.subplot(1, 3, 1)
short_heat = df[df['video_category'] == '短视频']['heat_category'].value_counts()
plt.pie(short_heat, labels=short_heat.index, autopct='%1.1f%%', 
       colors=['gray', 'green', 'blue', 'red'], shadow=True, startangle=90)
plt.title('短视频热度分布', fontsize=14, fontweight='bold')

plt.subplot(1, 3, 2)
medium_heat = df[df['video_category'] == '中长视频']['heat_category'].value_counts()
plt.pie(medium_heat, labels=medium_heat.index, autopct='%1.1f%%', 
       colors=['gray', 'green', 'blue', 'red'], shadow=True, startangle=90)
plt.title('中长视频热度分布', fontsize=14, fontweight='bold')

plt.subplot(1, 3, 3)
long_heat = df[df['video_category'] == '长视频']['heat_category'].value_counts()
plt.pie(long_heat, labels=long_heat.index, autopct='%1.1f%%', 
       colors=['gray', 'green', 'blue', 'red'], shadow=True, startangle=90)
plt.title('长视频热度分布', fontsize=14, fontweight='bold')

plt.suptitle('各类别视频的热度占比', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 8) 3D热力图 - 聚焦短视频和中长视频
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

plt.figure(figsize=(16, 12))
ax = plt.axes(projection='3d')

# 筛选短视频和中长视频，同时排除极端热度值
short_medium_df = filtered_df[filtered_df['duration_minutes'] < medium_threshold].copy()
q_low = short_medium_df['popularity_normalized'].quantile(0.01)
q_high = short_medium_df['popularity_normalized'].quantile(0.99)
filtered_heat_df = short_medium_df[(short_medium_df['popularity_normalized'] >= q_low) & 
                                  (short_medium_df['popularity_normalized'] <= q_high)]

# 提取数据
x = filtered_heat_df['duration_minutes']
y = filtered_heat_df['popularity_normalized']

# 创建网格点
xi = np.linspace(0, medium_threshold, 100)
yi = np.linspace(q_low, q_high, 100)
xi, yi = np.meshgrid(xi, yi)

# 计算KDE
positions = np.vstack([xi.ravel(), yi.ravel()])
values = np.vstack([x, y])
kernel = gaussian_kde(values)
zi = np.reshape(kernel(positions), xi.shape)

# 绘制3D表面
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', linewidth=0, antialiased=True, alpha=0.7)

# 添加颜色条
cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('密度', rotation=270, labelpad=20, fontsize=12)

# 添加类别分界线
xline = np.linspace(short_threshold, short_threshold, 100)
yline = np.linspace(q_low, q_high, 100)
zline = kernel(np.vstack([xline, yline]))
ax.plot(xline, yline, zline, 'r-', linewidth=2)

# 设置视角
ax.view_init(elev=30, azim=45)  # 调整仰角和方位角

# 添加标签
ax.set_xlabel('视频时长（分钟）', fontsize=12, labelpad=10)
ax.set_ylabel('热度（归一化）', fontsize=12, labelpad=10)
ax.set_zlabel('密度', fontsize=12, labelpad=10)
ax.set_title('短视频和中长视频热度分布 3D视图', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# 9) 细分时长热度分析 - 每5分钟一组
def group_by_5min(duration):
    upper_bound = np.ceil(duration / 5) * 5
    lower_bound = upper_bound - 5
    return f"{int(lower_bound)}-{int(upper_bound)}"

# 应用分组函数并限制在60分钟以内
short_medium_df = df[df['duration_minutes'] <= medium_threshold].copy()
short_medium_df['duration_group_5min'] = short_medium_df['duration_minutes'].apply(group_by_5min)

# 计算每个时长组在各热度类别的占比
heat_by_duration = pd.crosstab(
    short_medium_df['duration_group_5min'], 
    short_medium_df['heat_category'],
    normalize='index'
) * 100  # 转换为百分比

# 安全排序
def safe_extract_lower(group_name):
    try:
        return int(group_name.split('-')[0])
    except:
        return 999999

heat_by_duration['sort_key'] = heat_by_duration.index.map(safe_extract_lower)
heat_by_duration = heat_by_duration.sort_values('sort_key').drop('sort_key', axis=1)

# 绘制堆叠条形图
plt.figure(figsize=(16, 8))
heat_by_duration[['低热度', '中等热度', '高热度', '超高热度']].plot(
    kind='bar', 
    stacked=True, 
    color=['gray', 'green', 'blue', 'red'],
    width=0.8,
    ax=plt.gca()
)

plt.title('不同时长组的热度分布占比', fontsize=16, fontweight='bold')
plt.xlabel('视频时长分组（分钟）', fontsize=14)
plt.ylabel('占比（%）', fontsize=14)
plt.axvline(x=short_threshold/5, color='black', linestyle='--', linewidth=1.5)
plt.legend(title='热度类别')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 10) 细粒度热力图 - 高热度区域放大
plt.figure(figsize=(16, 12))

# 只选择中等热度以上的视频
high_heat_df = filtered_df[filtered_df['popularity_normalized'] >= low_heat_threshold].copy()

# 创建更细致的热力图
hexbin = plt.hexbin(
    high_heat_df['duration_minutes'], 
    high_heat_df['popularity_normalized'],
    gridsize=30,  # 六边形网格大小
    cmap='YlOrRd',  # 颜色映射
    mincnt=1,  # 最小计数
    bins='log'  # 使用对数缩放
)

# 添加颜色条
cbar = plt.colorbar(hexbin)
cbar.set_label('视频数量 (对数缩放)', rotation=270, labelpad=20, fontsize=12)

# 添加类别分界线
plt.axvline(x=short_threshold, color='black', linestyle='-', linewidth=2)
plt.axvline(x=medium_threshold, color='black', linestyle='-', linewidth=2)

# 添加热度阈值水平线
plt.axhline(y=medium_heat_threshold, color='blue', linestyle='--', linewidth=1.5)
plt.axhline(y=high_heat_threshold, color='purple', linestyle='--', linewidth=1.5)

# 添加类别标签
plt.text(short_threshold/2, high_heat_df['popularity_normalized'].max() * 0.95,
        '短视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.text((short_threshold + medium_threshold)/2, high_heat_df['popularity_normalized'].max() * 0.95,
        '中长视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.text(medium_threshold + 10, high_heat_df['popularity_normalized'].max() * 0.95,
        '长视频', ha='center', fontsize=14, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# 添加热度区域标签
plt.text(1, high_heat_threshold + 0.02, '超高热度 (90-100%分位)',
        ha='left', fontsize=12, color='purple',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
plt.text(1, medium_heat_threshold + 0.02, '高热度 (75-90%分位)',
        ha='left', fontsize=12, color='blue',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title('中高热度视频分布六边形热图', fontsize=16, fontweight='bold')
plt.xlabel('视频时长（分钟）', fontsize=14)
plt.ylabel('热度（归一化）', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()
