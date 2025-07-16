


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def classify_and_plot_duration(file_path=r"D:\下载\sports_2024-2025_with_popularity (1).csv.xlsx"):
    """
    Read video data, classify by duration, and generate visualization charts

    Parameters:
        file_path: Excel file path containing video duration data
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在!")
        return

    print(f"正在处理文件: {file_path}")

    # 读取数据
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 清洗并转换为分钟
    print(f"原始数据行数: {len(df)}")
    df = df.dropna(subset=['duration_seconds'])
    df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
    df = df.dropna(subset=['duration_seconds'])
    print(f"清洗后数据行数: {len(df)}")

    df['duration_min'] = df['duration_seconds'] / 60  # 转换为分钟

    # 分类函数（单位：分钟）
    def classify_duration(minutes):
        if minutes < 4:
            return 'Short (0-4 min)'
        elif minutes < 20:
            return 'Medium (4-20 min)'
        else:
            return 'Long (20+ min)'

    # 应用分类
    df['duration_category'] = df['duration_min'].apply(classify_duration)

    # 定义类别顺序
    category_order = ['Short (0-4 min)', 'Medium (4-20 min)', 'Long (20+ min)']

    # 统计数量和基本数据
    duration_counts = df['duration_category'].value_counts()
    # 重新索引以确保正确的顺序
    duration_counts = duration_counts.reindex(category_order)

    # 使用指定的三种颜色
    # R056G136B192（最深）- 蓝灰，理性柔和
    # R170G207B229（中）- 浅蓝，温和通透
    # R210G227B243（浅）- 雪蓝，极简风格

    specific_blues = [
        (56/255, 136/255, 192/255),   # 蓝灰，理性柔和（最深）
        (170/255, 207/255, 229/255),  # 浅蓝，温和通透（中）
        (210/255, 227/255, 243/255)   # 雪蓝，极简风格（浅）
    ]

    # 直接按顺序分配颜色
    color_map = {}
    for i, category in enumerate(category_order):
        if i < len(specific_blues):
            color_map[category] = specific_blues[i]

    # 获取颜色列表
    ordered_colors = [color_map[cat] for cat in category_order]

    duration_stats = df.groupby('duration_category')['duration_min'].agg(['count', 'min', 'max', 'mean', 'median'])

    # 打印详细统计信息和颜色分配
    print("Video Duration Classification Statistics:")
    print("-" * 60)
    total_videos = len(df)
    for category in category_order:
        if category in duration_counts.index:
            count = duration_counts[category]
            percentage = count / total_videos * 100
            stats = duration_stats.loc[category]
            color = color_map[category]
            rgb_values = f"R: {int(color[0]*255)}, G: {int(color[1]*255)}, B: {int(color[2]*255)}"
            print(f"{category}:")
            print(f"  Count: {count} videos ({percentage:.1f}%)")
            print(f"  Average duration: {stats['mean']:.1f} minutes")
            print(f"  Color: {rgb_values}")
    print("-" * 60)

    # 创建一个包含两个子图的图表
    plt.figure(figsize=(16, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 柱状图 (左侧)
    plt.subplot(1, 2, 1)
    bars = plt.bar(duration_counts.index, duration_counts.values, color=ordered_colors, 
                  width=0.7, edgecolor='#90A4AE', linewidth=1)
    plt.title('Video Duration Distribution', fontsize=16, fontweight='bold', color='#455A64')
    plt.xlabel('Duration Category', fontsize=12)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5, color='#CFD8DC')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 设置背景色为白色
    ax = plt.gca()
    ax.set_facecolor('white')

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', 
                fontsize=10, color='#455A64')

    # 饼图 (右侧)
    plt.subplot(1, 2, 2)
    wedges, texts, autotexts = plt.pie(
        duration_counts.values, 
        labels=None,
        autopct='%1.1f%%', 
        startangle=90, 
        shadow=False, 
        colors=ordered_colors, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )

    # 自定义饼图文本
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('#455A64')

    plt.title('Duration Distribution (%)', fontsize=16, fontweight='bold', color='#455A64')

    # 将图例放在饼图的下方
    legend_labels = [f"{cat} ({int(duration_counts[cat])})" for cat in category_order]
    plt.legend(wedges, legend_labels, title="Duration Categories",
               loc="center", bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=3)

    # 添加整个图表的主标题
    plt.suptitle('Analysis of Video Content Duration', fontsize=18, fontweight='bold', y=0.98, color='#455A64')

    # 调整布局并显示
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # 保存图表
    output_path = "video_duration_distribution_final.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")

    # 显示图表
    plt.show()

    return df

# 如果直接运行此脚本，则执行分类和绘图
if __name__ == "__main__":
    classify_and_plot_duration()

