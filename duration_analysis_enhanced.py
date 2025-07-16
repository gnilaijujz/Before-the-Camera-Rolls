# duration_analysis_聚类.py - 视频时长与互动指标聚类分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap
import os
import warnings
from datetime import datetime

# 忽略警告信息
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    加载并清洗数据

    Args:
        file_path: "D:\下载\sports_2024-2025.csv"

    Returns:
        清洗后的DataFrame
    """
    print(f"读取数据文件: {file_path}")

    # 确保文件存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    # 读取数据
    try:
        df = pd.read_csv(file_path)
        print(f"原始数据形状: {df.shape}")
    except Exception as e:
        raise Exception(f"读取CSV文件失败: {str(e)}")

    # 检查必要列是否存在
    required_cols = ['duration_seconds', 'view_count', 'like_count', 'comment_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV文件中缺少以下必需列: {missing_cols}")

    # 处理缺失值
    before_rows = df.shape[0]
    df = df.dropna(subset=required_cols)
    after_rows = df.shape[0]
    print(f"删除缺失值后的数据形状: {df.shape} (删除了 {before_rows - after_rows} 行)")

    # 计算时长（分钟）
    df['duration_minutes'] = df['duration_seconds'] / 60

    # 处理异常值 - 去除视频时长和观看量为0的数据
    df = df[(df['duration_seconds'] > 0) & (df['view_count'] > 0)]
    print(f"处理异常值后的数据形状: {df.shape}")

    # 计算基础互动率指标
    df['like_rate'] = df['like_count'] / df['view_count']
    df['comment_rate'] = df['comment_count'] / df['view_count']
    df['eng_rate'] = df['like_rate'] + df['comment_rate']

    return df

def transform_features(df):
    """
    特征变换: Log变换 → 分位数变换 → 稳健缩放

    Args:
        df: 输入DataFrame

    Returns:
        变换后的DataFrame和变换后的特征矩阵
    """
    print("执行特征变换...")

    # 1. Log变换 - 处理长尾分布
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    df[['view_log', 'like_log', 'comment_log']] = log_transformer.transform(
        df[['view_count', 'like_count', 'comment_count']]
    )

    # 2. 分位数变换 - 转换为正态分布
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    df[['view_q', 'like_q', 'comment_q', 'duration_q']] = quantile_transformer.fit_transform(
        df[['view_log', 'like_log', 'comment_log', 'duration_minutes']]
    )

    # 3. 稳健缩放 - 标准化特征，抵抗离群值影响
    robust_scaler = RobustScaler()
    df[['view_s', 'like_s', 'comment_s', 'duration_s']] = robust_scaler.fit_transform(
        df[['view_q', 'like_q', 'comment_q', 'duration_q']]
    )

    # 返回变换后的特征矩阵
    features = ['view_s', 'like_s', 'comment_s', 'duration_s']
    X = df[features].values

    return df, X, features

def find_optimal_k(X, k_range=(2, 10)):
    """
    寻找最佳聚类数K

    Args:
        X: 特征矩阵
        k_range: K的范围，默认(2, 10)

    Returns:
        最佳K值
    """
    print(f"寻找最佳聚类数K (范围: {k_range[0]}-{k_range[1]})...")

    silhouette_scores = []
    ch_scores = []
    k_values = range(k_range[0], k_range[1] + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # 计算轮廓系数
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)

        # 计算Calinski-Harabasz指数
        ch_score = calinski_harabasz_score(X, labels)
        ch_scores.append(ch_score)

        print(f"  K = {k}, 轮廓系数 = {sil_score:.3f}, CH指数 = {ch_score:.1f}")

    # 选择轮廓系数最高的K
    best_k_sil = k_values[np.argmax(silhouette_scores)]
    best_score_sil = max(silhouette_scores)

    # 可选：绘制K值评估图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, silhouette_scores, 'o-', color='blue')
    plt.axvline(x=best_k_sil, color='red', linestyle='--')
    plt.xlabel('聚类数 (K)')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数评估')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, ch_scores, 'o-', color='green')
    plt.xlabel('聚类数 (K)')
    plt.ylabel('Calinski-Harabasz指数')
    plt.title('CH指数评估')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('k_evaluation.png')
    print(f"已保存K值评估图: k_evaluation.png")

    print(f"最佳K = {best_k_sil}, 轮廓系数 = {best_score_sil:.3f}")
    return best_k_sil

def perform_clustering(df, X, k):
    """
    执行K-means聚类

    Args:
        df: 数据DataFrame
        X: 特征矩阵
        k: 聚类数

    Returns:
        带有聚类标签的DataFrame
    """
    print(f"执行K-means聚类 (K = {k})...")

    # 执行聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X)

    # 计算每个簇的样本数
    cluster_counts = df['cluster_id'].value_counts().sort_index()
    print("各簇样本数:")
    for cluster_id, count in cluster_counts.items():
        print(f"  簇 {cluster_id}: {count} 样本 ({count/len(df)*100:.1f}%)")

    return df

def analyze_clusters(df):
    """
    分析聚类结果并分配业务标签

    Args:
        df: 带有聚类标签的DataFrame

    Returns:
        带有业务标签的DataFrame和聚类统计信息
    """
    print("分析聚类结果...")

    # 全局中位数
    view_median = df['view_count'].median()
    eng_median = df['eng_rate'].median()
    print(f"全局中位数 - 观看量: {view_median:.1f}, 互动率: {eng_median:.6f}")

    # 计算每个簇的统计信息
    stats = df.groupby('cluster_id').agg({
        'view_count': ['mean', 'median'],
        'like_count': ['mean', 'median'],
        'comment_count': ['mean', 'median'],
        'duration_minutes': ['mean', 'median'],
        'like_rate': ['mean', 'median'],
        'comment_rate': ['mean', 'median'],
        'eng_rate': ['mean', 'median']
    })

    # 扁平化多级列名
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]

    # 分配业务标签
    stats['biz_label'] = '未分类'

    for idx, row in stats.iterrows():
        if row['view_count_mean'] >= view_median and row['eng_rate_mean'] >= eng_median:
            stats.at[idx, 'biz_label'] = '明星爆款'
        elif row['view_count_mean'] >= view_median and row['eng_rate_mean'] < eng_median:
            stats.at[idx, 'biz_label'] = '过眼云烟'
        elif row['view_count_mean'] < view_median and row['eng_rate_mean'] >= eng_median:
            stats.at[idx, 'biz_label'] = '潜力小众'
        else:
            stats.at[idx, 'biz_label'] = '沉寂视频'

    # 将业务标签添加到原始数据
    df = df.merge(stats[['biz_label']], left_on='cluster_id', right_index=True)

    # 打印业务标签分布
    label_counts = df['biz_label'].value_counts()
    print("业务标签分布:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} 样本 ({count/len(df)*100:.1f}%)")

    # 打印各簇的统计信息
    print("各簇统计信息:")
    stats_display = stats[['view_count_mean', 'eng_rate_mean', 'duration_minutes_mean', 'biz_label']].copy()
    stats_display = stats_display.round(2)
    print(stats_display)

    return df, stats

def visualize_results(df, X, features):
    """
    可视化聚类结果

    Args:
        df: 带有聚类和业务标签的DataFrame
        X: 特征矩阵
        features: 特征名称列表
    """
    print("生成可视化结果...")

    # 1. UMAP降维可视化
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings = umap_model.fit_transform(X)

    # 创建可视化
    plt.figure(figsize=(12, 10))

    # 2. 业务标签散点图
    plt.subplot(2, 2, 1)
    sns.scatterplot(
        x=embeddings[:, 0], 
        y=embeddings[:, 1],
        hue=df['biz_label'], 
        palette='tab10', 
        s=30,
        alpha=0.7
    )
    plt.title('UMAP投影 - 业务标签')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(title='业务标签', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. 互动率与观看量散点图
    plt.subplot(2, 2, 2)
    sns.scatterplot(
        x='view_log', 
        y='eng_rate',
        hue='biz_label',
        data=df,
        palette='tab10',
        s=30,
        alpha=0.7
    )
    plt.axvline(x=np.log1p(df['view_count'].median()), color='gray', linestyle='--')
    plt.axhline(y=df['eng_rate'].median(), color='gray', linestyle='--')
    plt.title('观看量 vs 互动率')
    plt.xlabel('观看量 (对数)')
    plt.ylabel('互动率')
    plt.legend(title='业务标签', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. 视频时长分布
    plt.subplot(2, 2, 3)
    sns.boxplot(x='biz_label', y='duration_minutes', data=df, palette='tab10')
    plt.title('各业务标签的视频时长分布')
    plt.xlabel('业务标签')
    plt.ylabel('视频时长 (分钟)')
    plt.xticks(rotation=45)

    # 5. 互动指标比较
    plt.subplot(2, 2, 4)

    # 准备数据
    engagement_data = df.groupby('biz_label')[['like_rate', 'comment_rate']].mean().reset_index()
    engagement_data_melted = pd.melt(
        engagement_data, 
        id_vars='biz_label',
        value_vars=['like_rate', 'comment_rate'],
        var_name='互动类型',
        value_name='互动率'
    )

    # 绘制分组柱状图
    sns.barplot(x='biz_label', y='互动率', hue='互动类型', data=engagement_data_melted, palette='Set2')
    plt.title('各业务标签的互动率比较')
    plt.xlabel('业务标签')
    plt.ylabel('平均互动率')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('cluster_analysis_results.png')
    print(f"已保存聚类分析结果图: cluster_analysis_results.png")

    # 可选：保存高质量PDF版本
    plt.savefig('cluster_analysis_results.pdf', format='pdf')

    plt.show()

def save_results(df, stats, output_dir='.'):
    """
    保存分析结果

    Args:
        df: 带有聚类和业务标签的DataFrame
        stats: 聚类统计信息
        output_dir: 输出目录
    """
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存带标签的数据
    output_file = os.path.join(output_dir, f'video_clusters_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    print(f"已保存带标签的数据: {output_file}")

    # 保存聚类统计信息
    stats_file = os.path.join(output_dir, f'cluster_stats_{timestamp}.csv')
    stats.to_csv(stats_file)
    print(f"已保存聚类统计信息: {stats_file}")

def main():
    """
    主函数
    """
    print("=" * 50)
    print("视频时长与互动指标聚类分析")
    print("=" * 50)

    # 1. 加载并清洗数据
    file_path = r"D:\下载\sports_2024-2025.csv"
    df = load_and_clean_data(file_path)

    # 2. 特征变换
    df, X, features = transform_features(df)

    # 3. 寻找最佳K
    best_k = find_optimal_k(X, k_range=(2, 8))

    # 4. 执行聚类
    df = perform_clustering(df, X, best_k)

    # 5. 分析聚类结果
    df, stats = analyze_clusters(df)

    # 6. 可视化结果
    visualize_results(df, X, features)

    # 7. 保存结果
    save_results(df, stats)

    print("def find_optimal_k(X, k_range=(4, 8)):  # 从4开始，确保至少有4个簇")
    """
    寻找最佳聚类数K，最少4个簇以对应4种业务类型
    """
def analyze_clusters(df):
    """
    分析聚类结果并分配业务标签 - 修改为4类业务逻辑
    """
    print("分析聚类结果...")

    # 计算全局阈值 - 使用更合理的分位数
    view_threshold = df['view_count'].quantile(0.6)  # 60%分位数作为高曝光阈值
    like_threshold = df['like_rate'].quantile(0.6)   # 60%分位数作为高点赞率阈值  
    comment_threshold = df['comment_rate'].quantile(0.6)  # 60%分位数作为高评论率阈值
    
    print(f"阈值设定 - 观看量: {view_threshold:.1f}, 点赞率: {like_threshold:.6f}, 评论率: {comment_threshold:.6f}")

    # 计算每个簇的统计信息
    stats = df.groupby('cluster_id').agg({
        'view_count': ['mean', 'median'],
        'like_count': ['mean', 'median'], 
        'comment_count': ['mean', 'median'],
        'duration_minutes': ['mean', 'median'],
        'like_rate': ['mean', 'median'],
        'comment_rate': ['mean', 'median'],
        'eng_rate': ['mean', 'median']
    })

    # 扁平化多级列名
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]

    # 修改业务标签分配逻辑 - 按照4种类型
    stats['biz_label'] = '未分类'

    for idx, row in stats.iterrows():
        high_view = row['view_count_mean'] >= view_threshold
        high_like = row['like_rate_mean'] >= like_threshold
        high_comment = row['comment_rate_mean'] >= comment_threshold
        
        if high_view and high_like and high_comment:
            stats.at[idx, 'biz_label'] = '明星全能'
        elif high_view and high_like and not high_comment:
            stats.at[idx, 'biz_label'] = '安静点赞'
        elif high_view and not high_like and high_comment:
            stats.at[idx, 'biz_label'] = '讨论热议'
        elif high_view and not high_like and not high_comment:
            stats.at[idx, 'biz_label'] = '过眼云烟'
        else:
            # 低曝光的视频统一归为一类
            stats.at[idx, 'biz_label'] = '小众内容'
def main():
    """
    主函数 - 修改为直接使用4个簇
    """
    print("=" * 50)
    print("视频时长与互动指标聚类分析")
    print("=" * 50)

    # 1. 加载并清洗数据
    file_path = r"D:\下载\sports_2024-2025.csv"
    df = load_and_clean_data(file_path)

    # 2. 特征变换
    df, X, features = transform_features(df)

    # 3. 直接使用K=4，对应4种业务类型
    best_k = 4
    print(f"使用固定K值: {best_k} (对应4种业务类型)")
    
    # 也可以选择性地评估K值
    # best_k = find_optimal_k(X, k_range=(4, 6))

    # 4. 执行聚类
    df = perform_clustering(df, X, best_k)

    # 计算基础互动率指标 - 添加异常值处理
    df['like_rate'] = np.where(df['view_count'] > 0, df['like_count'] / df['view_count'], 0)
    df['comment_rate'] = np.where(df['view_count'] > 0, df['comment_count'] / df['view_count'], 0)
    df['eng_rate'] = df['like_rate'] + df['comment_rate']

    # 处理极端异常值
    df = df[df['like_rate'] <= 1.0]  # 点赞率不应超过100%
    df = df[df['comment_rate'] <= 1.0]  # 评论率不应超过100%
    print("分析完成!")

if __name__ == "__main__":
    main()
