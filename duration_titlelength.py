# duration_title_analysis.py

import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# 设置字体

df = pd.read_csv(r"D:\下载\sports_2024-2025.csv")
df['duration_min'] = df['duration_seconds'] / 60

# Optional: For raincloud plot, install ptitprince
try:
    import ptitprince as pt
    HAS_PTITPRINCE = True
except ImportError:
    pt = None
    HAS_PTITPRINCE = False

# Blue color palette
color_palette = {
    'deep_sea_blue': (8/255, 51/255, 110/255),
    'indigo_blue': (16/255, 92/255, 164/255),
    'blue_gray': (56/255, 136/255, 192/255),
    'light_blue': (108/255, 176/255, 217/255),
    'pale_blue': (190/255, 221/255, 240/255)
}

# Create blue gradient colormap
blue_cmap = LinearSegmentedColormap.from_list(
    'blue_gradient', 
    [color_palette['pale_blue'], color_palette['light_blue'], 
     color_palette['blue_gray'], color_palette['indigo_blue'], 
     color_palette['deep_sea_blue']], 
    N=100
)

def plot_raincloud_alternative(df, x_col, y_col, title="Alternative Raincloud Plot"):
    """
    使用seaborn和matplotlib创建雨云图的替代实现
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取唯一的类别
    categories = df[x_col].dropna().unique()
    n_cats = len(categories)
    
    # 设置颜色
    colors = plt.cm.Set2(np.linspace(0, 1, n_cats))
    
    # 为每个类别绘制雨云图组件
    for i, category in enumerate(categories):
        data = df[df[x_col] == category][y_col].dropna()
        
        if len(data) == 0:
            continue
            
        pos = i
        
        # 1. 小提琴图（分布密度）
        parts = ax.violinplot([data], positions=[pos - 0.2], widths=0.3, 
                             showmeans=False, showextrema=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        
        # 2. 箱线图（统计摘要）
        bp = ax.boxplot([data], positions=[pos], widths=0.1, 
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=colors[i], alpha=0.8),
                       medianprops=dict(color='black', linewidth=2))
        
        # 3. 散点图（原始数据点）
        y_data = data.values
        x_data = np.random.normal(pos + 0.2, 0.03, len(y_data))
        ax.scatter(x_data, y_data, alpha=0.5, s=15, color=colors[i], 
                  edgecolor='white', linewidth=0.5)
    
    # 设置坐标轴
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def analyze_duration_and_title_length(
    file_path: str = 'sports_2024-2025.csv',
    duration_col: str = 'duration_seconds',
    title_col: str = 'title'
):
    """
    1. 把 duration（秒）转换成分钟
    2. 计算标题词长（空格拆分与 jieba 分词两种方式）
    3. 输出描述统计、相关性
    4. 绘制散点图、箱线图、雨云图、KDE、Hexbin 与密度图
    """
    # 1. 读取并清洗
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[duration_col, title_col])
    df['duration_min'] = pd.to_numeric(df[duration_col], errors='coerce') / 60
    df = df.dropna(subset=['duration_min'])

    # 2. 计算标题词长
    df['word_count_space'] = df[title_col].astype(str).apply(lambda s: len(s.split()))
    df['word_count_jieba'] = df[title_col].astype(str).apply(lambda s: len(list(jieba.cut(s))))

    # 3. 描述统计 & 相关性
    stats = df[['duration_min','word_count_space','word_count_jieba']].describe().T
    stats = stats[['mean','50%','std','min','max']].rename(columns={'50%':'median'})
    print("\n=== 描述性统计 ===")
    print(stats)
    corr = df[['duration_min','word_count_space','word_count_jieba']].corr()
    print("\n=== 相关性矩阵 ===")
    print(corr)

    # 4. 准备绘图风格
    sns.set(style='whitegrid', font_scale=1.1)

    # 5.1 散点图：时长 vs 标题词数
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='duration_min', y='word_count_space', data=df, alpha=0.6)
    plt.title('Duration (min) vs Title Word Count')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Title Word Count (split by space)')
    plt.tight_layout()
    plt.show()

    # 5.2 箱型图：按时长分类
    bins = [0, 5, 20, np.inf]
    labels = ['Short (<5min)', 'Medium (5–20min)', 'Long (>20min)']
    df['duration_cat'] = pd.cut(df['duration_min'], bins=bins, labels=labels)

    plt.figure(figsize=(8,6))
    sns.boxplot(x='duration_cat', y='word_count_space', data=df, palette='Set2')
    plt.title('Title Word Count by Duration Category')
    plt.xlabel('Duration Category')
    plt.ylabel('Title Word Count')
    plt.tight_layout()
    plt.show()

    # 5.3 密度图：标题词长分布
    plt.figure(figsize=(8,6))
    sns.kdeplot(df['word_count_space'], fill=True, label='split by space')
    sns.kdeplot(df['word_count_jieba'], fill=True, label='jieba')
    plt.title('Density of Title Word Count')
    plt.xlabel('Word Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5.4 联合 KDE 等高线
    plt.figure(figsize=(8,6))
    sns.kdeplot(x='duration_min', y='word_count_space', data=df,
                fill=False, levels=5, cmap='Reds')
    plt.scatter(df['duration_min'], df['word_count_space'], s=5, alpha=0.3)
    plt.title('2D KDE Contour of Duration vs Word Count')
    plt.xlabel('Duration (min)')
    plt.ylabel('Title Word Count')
    plt.tight_layout()
    plt.show()

    # 5.5 Joint KDE
    sns.jointplot(x='duration_min', y='word_count_space', data=df,
                  kind='kde', fill=True, cmap='Blues', height=8)
    plt.tight_layout()
    plt.show()

    # 5.6 Hexbin 图
    plt.figure(figsize=(8,6))
    hb = plt.hexbin(df['duration_min'], df['word_count_space'],
                    gridsize=30, cmap='YlGnBu', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.title('Hexbin of Duration vs Title Word Count')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Title Word Count')
    plt.tight_layout()
    plt.show()

    # 5.7 时长与词长一维密度
    plt.figure(figsize=(8,6))
    sns.kdeplot(df['duration_min'], fill=True, label='Duration (min)')
    sns.kdeplot(df['word_count_space'], fill=True, label='Title Word Count')
    plt.title('Density Plot of Duration and Title Word Count')
    plt.xlabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5.8 雨云图：结合小提琴、箱线与散点
    print("\n=== 创建雨云图 ===")
    if HAS_PTITPRINCE and pt is not None:
        try:
            plt.figure(figsize=(10, 6))
            pt.RainCloud(
                x='duration_cat', y='word_count_space', data=df,
                palette=['#4C72B0','#55A868','#C44E52'],
                width_viol=0.6, orient='h', move=0.2
            )
            plt.title('Raincloud Plot of Title Word Count by Duration Category')
            plt.xlabel('Title Word Count')
            plt.ylabel('Duration Category')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error with ptitprince: {e}")
            print("Using alternative raincloud plot...")
            plot_raincloud_alternative(df, 'duration_cat', 'word_count_space',
                                     'Alternative Raincloud Plot')
            plt.show()
    else:
        print("ptitprince not available, using alternative visualization...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左侧：小提琴图
        sns.violinplot(x='duration_cat', y='word_count_space', data=df, 
                      palette='Set2', ax=axes[0])
        axes[0].set_title('Violin Plot of Title Word Count by Duration Category')
        
        # 右侧：箱线图 + 散点图
        sns.boxplot(x='duration_cat', y='word_count_space', data=df, 
                   palette='Set2', ax=axes[1])
        sns.stripplot(x='duration_cat', y='word_count_space', data=df, 
                     size=3, alpha=0.5, ax=axes[1])
        axes[1].set_title('Box Plot + Strip Plot of Title Word Count by Duration Category')
        
        plt.tight_layout()
        plt.show()

    return df

if __name__ == '__main__':
    file_path = r"D:\下载\sports_2024-2025_with_popularity.csv"
    try:
        result_df = analyze_duration_and_title_length(file_path)
        print("Analysis completed successfully!")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
