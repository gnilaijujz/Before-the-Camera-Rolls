import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取数据
df = pd.read_csv(r"D:\下载\sports_2024-2025_with_popularity.csv")
df['duration_min'] = df['duration_seconds'] / 60

# 1. 定义6个时长分类
duration_bins = [0, 3, 5, 10, 20, 30, np.inf]
duration_labels = [
    '0-3分钟', '3-5分钟', '5-10分钟', 
    '10-20分钟', '20-30分钟', '30+分钟'
]

df['duration_group'] = pd.cut(
    df['duration_min'],
    bins=duration_bins,
    labels=duration_labels,
    right=False
)

# 2. 将热度分为5个等级
heat_quantiles = 5
df['heat_group'] = pd.qcut(
    df['popularity_normalized'],
    q=heat_quantiles,
    labels=['低热度', '中低热度', '中高热度', '高热度', '极高热度']
)

# 3. 计算每个时长-热度组合的视频数量占总体比例
# 总视频数
total_videos = len(df)

# 创建交叉表
count_matrix = pd.crosstab(
    df['duration_group'], 
    df['heat_group']
)

# 计算每个单元格占总体视频数的百分比
percentage_matrix = count_matrix / total_videos

# 确保热度列的顺序正确
heat_order = ['低热度', '中低热度', '中高热度', '高热度', '极高热度']
percentage_matrix = percentage_matrix[heat_order]

# 4. 使用指定的颜色系统
# 定义热度对应的颜色
heat_colors = {
    '低热度': '#FDE929',    # 浅黄色
    '中低热度': '#E98D42',  # 橙色
    '中高热度': '#63ACBE',  # 青色
    '高热度': '#315B8F',    # 藏青
    '极高热度': '#440154'   # 深紫
}

# 5. 创建6x5的矩形热力图，统一使用指定的颜色系统
plt.figure(figsize=(16, 14))

# 自定义绘制热力图
for i, duration in enumerate(percentage_matrix.index):
    for j, heat_level in enumerate(percentage_matrix.columns):
        # 获取当前单元格的值
        value = percentage_matrix.loc[duration, heat_level]

        # 获取热度对应的颜色
        base_color = heat_colors[heat_level]

        # 将RGB颜色转换为RGBA格式
        rgb_color = mcolors.to_rgb(base_color)

        # 根据值调整颜色的透明度来表示深浅
        # 值越大，透明度越低（颜色越深）
        alpha = 0.2 + 0.8 * (value / percentage_matrix.values.max())
        color = (*rgb_color, alpha)

        # 绘制矩形
        rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
        plt.gca().add_patch(rect)

        # 添加文本标签
        # 根据背景色亮度决定文本颜色
        text_color = 'white' if heat_level in ['高热度', '极高热度'] else 'black'

        plt.text(j + 0.5, i + 0.5, f"{value:.2%}\n({count_matrix.loc[duration, heat_level]}个)",
                 ha='center', va='center', 
                 color=text_color,
                 fontsize=10, fontweight='bold')

# 设置坐标轴范围和标签
plt.xlim(0, len(percentage_matrix.columns))
plt.ylim(0, len(percentage_matrix.index))
plt.xticks(np.arange(len(percentage_matrix.columns)) + 0.5, percentage_matrix.columns)
plt.yticks(np.arange(len(percentage_matrix.index)) + 0.5, percentage_matrix.index)

# 美化图表
plt.title("视频时长与热度关系矩形热力图\n(统一使用指定颜色系统，透明度表示占比)", 
          pad=20, fontsize=20, fontweight='bold')
plt.xlabel("热度分组", fontsize=14, labelpad=10)
plt.ylabel("视频时长分组", fontsize=14, labelpad=10)

# 添加图例
legend_elements = []
for heat_level, color in heat_colors.items():
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white',
                                         label=heat_level))

plt.legend(handles=legend_elements, title="热度分类", 
           loc="upper right", bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig('duration_heat_6x5_unified_colors.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 创建增强版热力图 - 使用纯色并调整透明度
plt.figure(figsize=(16, 14))

# 自定义绘制热力图 - 使用纯色版本
for i, duration in enumerate(percentage_matrix.index):
    for j, heat_level in enumerate(percentage_matrix.columns):
        # 获取当前单元格的值
        value = percentage_matrix.loc[duration, heat_level]

        # 获取热度对应的颜色
        base_color = heat_colors[heat_level]

        # 绘制矩形 - 使用原始颜色，通过alpha调整深浅
        rect = plt.Rectangle((j, i), 1, 1, 
                            facecolor=base_color, 
                            edgecolor='white', 
                            linewidth=1,
                            alpha=0.2 + 0.8 * (value / percentage_matrix.values.max()))
        plt.gca().add_patch(rect)

        # 添加文本标签
        # 根据热度级别决定文本颜色
        text_color = 'white' if heat_level in ['高热度', '极高热度'] else 'black'

        plt.text(j + 0.5, i + 0.5, f"{value:.2%}\n({count_matrix.loc[duration, heat_level]}个)",
                 ha='center', va='center', 
                 color=text_color,
                 fontsize=10, fontweight='bold')

# 设置坐标轴范围和标签
plt.xlim(0, len(percentage_matrix.columns))
plt.ylim(0, len(percentage_matrix.index))
plt.xticks(np.arange(len(percentage_matrix.columns)) + 0.5, percentage_matrix.columns)
plt.yticks(np.arange(len(percentage_matrix.index)) + 0.5, percentage_matrix.index)

# 美化图表
plt.title("视频时长与热度关系矩形热力图 (增强版)\n(原始颜色表示热度分类，透明度表示占比)", 
          pad=20, fontsize=20, fontweight='bold')
plt.xlabel("热度分组", fontsize=14, labelpad=10)
plt.ylabel("视频时长分组", fontsize=14, labelpad=10)

# 添加图例
legend_elements = []
for heat_level, color in heat_colors.items():
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white',
                                         label=heat_level))

plt.legend(handles=legend_elements, title="热度分类", 
           loc="upper right", bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig('duration_heat_6x5_enhanced_colors.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 创建表格式热力图 - 使用指定颜色系统
plt.figure(figsize=(18, 12))

# 表格尺寸
n_rows, n_cols = len(percentage_matrix), len(percentage_matrix.columns)
cell_width, cell_height = 1.0, 1.0

# 绘制表格
ax = plt.gca()
ax.set_axis_off()

# 表头颜色
header_color = '#E6E6E6'  # 浅灰色

# 绘制表头
for j, heat_level in enumerate(percentage_matrix.columns):
    # 绘制表头单元格
    rect = plt.Rectangle((j * cell_width, -cell_height), cell_width, cell_height, 
                         facecolor=heat_colors[heat_level], edgecolor='black', linewidth=1,
                         alpha=0.7)  # 使用70%不透明度
    ax.add_patch(rect)

    # 添加表头文本 - 根据背景色决定文本颜色
    text_color = 'white' if heat_level in ['高热度', '极高热度'] else 'black'
    plt.text((j + 0.5) * cell_width, -cell_height / 2, heat_level,
             ha='center', va='center', fontsize=12, fontweight='bold',
             color=text_color)

# 绘制行标题
for i, duration in enumerate(percentage_matrix.index):
    # 绘制行标题单元格
    rect = plt.Rectangle((-cell_width, i * cell_height), cell_width, cell_height, 
                         facecolor=header_color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    # 添加行标题文本
    plt.text(-cell_width / 2, (i + 0.5) * cell_height, duration,
             ha='center', va='center', fontsize=12, fontweight='bold')

# 绘制数据单元格
for i, duration in enumerate(percentage_matrix.index):
    for j, heat_level in enumerate(percentage_matrix.columns):
        # 获取数据
        value = percentage_matrix.loc[duration, heat_level]
        count = count_matrix.loc[duration, heat_level]
        max_value = percentage_matrix.values.max()

        # 获取热度颜色
        base_color = heat_colors[heat_level]

        # 绘制单元格 - 透明度根据值变化
        rect = plt.Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                            facecolor=base_color, edgecolor='black', linewidth=1,
                            alpha=0.2 + 0.8 * (value / max_value))
        ax.add_patch(rect)

        # 添加文本 - 根据热度级别决定文本颜色
        text_color = 'white' if heat_level in ['高热度', '极高热度'] else 'black'
        plt.text((j + 0.5) * cell_width, (i + 0.5) * cell_height,
                 f"{value:.2%}\n({count}个)",
                 ha='center', va='center',
                 color=text_color,
                 fontsize=11, fontweight='bold')

# 设置坐标轴范围
plt.xlim(-cell_width, n_cols * cell_width)
plt.ylim(-cell_height, n_rows * cell_height)

# 添加图例
legend_elements = []
for heat_level, color in heat_colors.items():
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black',
                                        label=heat_level, alpha=0.7))

plt.legend(handles=legend_elements, title="热度分类", 
           loc="upper right", bbox_to_anchor=(1.1, 1))

# 添加标题
plt.title("视频时长与热度关系表格热力图\n(颜色=热度分类，透明度=占比)",
          pad=20, fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig('duration_heat_6x5_table_colored.png', dpi=300, bbox_inches='tight')
plt.show()
