import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
from matplotlib import cm

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

# 2. 将热度分为6个等级
heat_quantiles = 6
df['heat_group'] = pd.qcut(
    df['popularity_normalized'],
    q=heat_quantiles,
    labels=[f'H{i+1}' for i in range(heat_quantiles)]
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
percentage_matrix = percentage_matrix[[f'H{i+1}' for i in range(heat_quantiles)]]

# 4. 创建6x6的矩形热力图，不同时长用不同颜色，深浅表示热度占比
plt.figure(figsize=(16, 14))

# 为每个时长组定义基础颜色
base_colors = {
    '0-3分钟': 'Blues',
    '3-5分钟': 'Greens',
    '5-10分钟': 'Oranges',
    '10-20分钟': 'Purples',
    '20-30分钟': 'Reds',
    '30+分钟': 'YlOrBr'
}

# 自定义绘制热力图函数，每行使用不同的颜色映射
for i, duration in enumerate(percentage_matrix.index):
    for j, heat_level in enumerate(percentage_matrix.columns):
        # 获取当前单元格的值
        value = percentage_matrix.loc[duration, heat_level]

        # 获取当前时长组的颜色映射
        cmap_name = base_colors[duration]
        cmap = plt.cm.get_cmap(cmap_name)

        # 根据值计算颜色深度 - 在每个颜色族中，值越大颜色越深
        # 为了增强效果，使用非线性映射
        color_intensity = np.power(value / percentage_matrix.values.max(), 0.5)
        color = cmap(0.3 + 0.7 * color_intensity)  # 从30%到100%的颜色深度

        # 绘制矩形
        rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
        plt.gca().add_patch(rect)

        # 添加文本标签
        plt.text(j + 0.5, i + 0.5, f"{value:.2%}\n({count_matrix.loc[duration, heat_level]}个)",
                 ha='center', va='center', 
                 color='white' if color_intensity > 0.5 else 'black',
                 fontsize=10, fontweight='bold')

# 设置坐标轴范围和标签
plt.xlim(0, len(percentage_matrix.columns))
plt.ylim(0, len(percentage_matrix.index))
plt.xticks(np.arange(len(percentage_matrix.columns)) + 0.5, percentage_matrix.columns)
plt.yticks(np.arange(len(percentage_matrix.index)) + 0.5, percentage_matrix.index)

# 美化图表
plt.title("视频时长与热度关系矩形热力图\n(不同颜色=不同时长，深浅=占总视频比例)", 
          pad=20, fontsize=20, fontweight='bold')
plt.xlabel("热度分组（H1=最低热度, H6=最高热度）", fontsize=14, labelpad=10)
plt.ylabel("视频时长分组", fontsize=14, labelpad=10)

# 添加图例
legend_elements = []
for duration, cmap_name in base_colors.items():
    cmap = plt.cm.get_cmap(cmap_name)
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0.7), edgecolor='white',
                                        label=duration))

plt.legend(handles=legend_elements, title="时长分组", 
           loc="upper right", bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig('duration_heat_6x6_colored.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 创建增强版热力图 - 添加行列汇总信息
plt.figure(figsize=(18, 16))

# 计算行和列的汇总数据
row_sums = percentage_matrix.sum(axis=1)
col_sums = percentage_matrix.sum(axis=0)

# 创建包含汇总信息的扩展矩阵
extended_matrix = pd.DataFrame(
    np.zeros((len(percentage_matrix) + 1, len(percentage_matrix.columns) + 1)),
    index=list(percentage_matrix.index) + ['所有时长'],
    columns=list(percentage_matrix.columns) + ['所有热度']
)

# 填充原始数据
for i, duration in enumerate(percentage_matrix.index):
    for j, heat_level in enumerate(percentage_matrix.columns):
        extended_matrix.loc[duration, heat_level] = percentage_matrix.loc[duration, heat_level]

# 填充行汇总
for i, duration in enumerate(percentage_matrix.index):
    extended_matrix.loc[duration, '所有热度'] = row_sums[duration]

# 填充列汇总
for j, heat_level in enumerate(percentage_matrix.columns):
    extended_matrix.loc['所有时长', heat_level] = col_sums[heat_level]

# 填充总计
extended_matrix.loc['所有时长', '所有热度'] = row_sums.sum()

# 为每个时长组和汇总行定义基础颜色
extended_base_colors = {
    '0-3分钟': 'Blues',
    '3-5分钟': 'Greens',
    '5-10分钟': 'Oranges',
    '10-20分钟': 'Purples',
    '20-30分钟': 'Reds',
    '30+分钟': 'YlOrBr',
    '所有时长': 'Greys'
}

# 自定义绘制热力图函数，每行使用不同的颜色映射
for i, duration in enumerate(extended_matrix.index):
    for j, heat_level in enumerate(extended_matrix.columns):
        # 获取当前单元格的值
        value = extended_matrix.loc[duration, heat_level]

        # 获取当前行的颜色映射
        cmap_name = extended_base_colors[duration]
        cmap = plt.cm.get_cmap(cmap_name)

        # 计算颜色深度 - 根据是否是汇总行/列使用不同逻辑
        if duration == '所有时长' or heat_level == '所有热度':
            # 汇总单元格使用基于数值的线性映射
            max_value = max(row_sums.max(), col_sums.max())
            color_intensity = value / max_value
            color = cmap(0.3 + 0.7 * color_intensity)
        else:
            # 普通单元格使用之前的非线性映射
            max_value = percentage_matrix.values.max()
            color_intensity = np.power(value / max_value, 0.5)
            color = cmap(0.3 + 0.7 * color_intensity)

        # 绘制矩形
        rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
        plt.gca().add_patch(rect)

        # 添加文本标签
        if duration == '所有时长' or heat_level == '所有热度':
            # 汇总单元格显示百分比
            text = f"{value:.1%}"
        else:
            # 普通单元格显示百分比和数量
            count = count_matrix.loc[duration, heat_level]
            text = f"{value:.2%}\n({count}个)"

        plt.text(j + 0.5, i + 0.5, text,
                 ha='center', va='center', 
                 color='white' if color_intensity > 0.5 else 'black',
                 fontsize=10, fontweight='bold')

# 设置坐标轴范围和标签
plt.xlim(0, len(extended_matrix.columns))
plt.ylim(0, len(extended_matrix.index))
plt.xticks(np.arange(len(extended_matrix.columns)) + 0.5, extended_matrix.columns)
plt.yticks(np.arange(len(extended_matrix.index)) + 0.5, extended_matrix.index)

# 添加分隔线
plt.axvline(x=len(extended_matrix.columns)-1, color='black', linewidth=2, linestyle='-')
plt.axhline(y=len(extended_matrix.index)-1, color='black', linewidth=2, linestyle='-')

# 美化图表
plt.title("视频时长与热度关系矩形热力图 (带汇总)\n(不同颜色=不同时长，深浅=占总视频比例)", 
          pad=20, fontsize=20, fontweight='bold')
plt.xlabel("热度分组", fontsize=14, labelpad=10)
plt.ylabel("视频时长分组", fontsize=14, labelpad=10)

# 添加图例
legend_elements = []
for duration, cmap_name in extended_base_colors.items():
    if duration != '所有时长':  # 排除汇总行
        cmap = plt.cm.get_cmap(cmap_name)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0.7), edgecolor='white',
                                            label=duration))

plt.legend(handles=legend_elements, title="时长分组", 
           loc="upper right", bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig('duration_heat_6x6_colored_with_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 创建表格式热力图 - 更清晰地呈现数据
plt.figure(figsize=(18, 12))

# 定义颜色映射函数 - 根据时长组获取颜色
def get_cell_color(duration, heat_level, value, max_value):
    cmap_name = base_colors[duration]
    cmap = plt.cm.get_cmap(cmap_name)
    # 使用非线性映射增强对比度
    color_intensity = np.power(value / max_value, 0.5)
    return cmap(0.3 + 0.7 * color_intensity)

# 创建表格式热力图
ax = plt.gca()
ax.set_axis_off()

# 表格尺寸
n_rows, n_cols = len(percentage_matrix), len(percentage_matrix.columns)
cell_width, cell_height = 1.0, 1.0

# 最大值用于颜色归一化
max_value = percentage_matrix.values.max()

# 绘制表头
header_color = '#E6E6E6'  # 浅灰色
for j, heat_level in enumerate(percentage_matrix.columns):
    # 绘制表头单元格
    rect = plt.Rectangle((j * cell_width, -cell_height), cell_width, cell_height, 
                         facecolor=header_color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    # 添加表头文本
    plt.text((j + 0.5) * cell_width, -cell_height / 2, heat_level,
             ha='center', va='center', fontsize=12, fontweight='bold')

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

        # 获取颜色
        color = get_cell_color(duration, heat_level, value, max_value)

        # 绘制单元格
        rect = plt.Rectangle((j * cell_width, i * cell_height), cell_width, cell_height,
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # 添加文本
        plt.text((j + 0.5) * cell_width, (i + 0.5) * cell_height,
                 f"{value:.2%}\n({count}个)",
                 ha='center', va='center',
                 color='white' if np.power(value / max_value, 0.5) > 0.5 else 'black',
                 fontsize=11, fontweight='bold')

# 设置坐标轴范围
plt.xlim(-cell_width, n_cols * cell_width)
plt.ylim(-cell_height, n_rows * cell_height)

# 添加图例
legend_elements = []
for duration, cmap_name in base_colors.items():
    cmap = plt.cm.get_cmap(cmap_name)
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0.7), edgecolor='black',
                                        label=duration))

plt.legend(handles=legend_elements, title="时长分组", 
           loc="upper right", bbox_to_anchor=(1.1, 1))

# 添加标题
plt.title("视频时长与热度关系表格热力图\n(不同颜色=不同时长，深浅=占总视频比例)",
          pad=20, fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig('duration_heat_6x6_table.png', dpi=300, bbox_inches='tight')
plt.show()
