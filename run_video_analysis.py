# 运行视频聚类分析

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 视频数据文件路径
file_path = "D:\下载\sports_2024-2025.csv"

def load_data(file_path):
    """加载并预处理数据"""
    print("1. 加载数据...")
    try:
        # 使用正斜杠避免Windows路径转义问题
        file_path = file_path.replace("\\", "/")
        df = pd.read_csv(file_path)
        print(f"原始数据形状: {df.shape}")

        # 检查所需列是否存在
        required_cols = ['view_count', 'like_count', 'comment_count', 'duration_seconds']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"数据中缺少必要列: {col}")

        # 处理异常值
        df = df[(df['duration_seconds'] > 0) & (df['view_count'] > 0)]
        df = df[df['like_count'] <= df['view_count']]  # 点赞数不应超过观看数
        df = df[df['comment_count'] <= df['view_count']]  # 评论数不应超过观看数

        # 计算分钟
        df['duration_minutes'] = df['duration_seconds'] / 60

        # 计算互动指标
        df['like_rate'] = df['like_count'] / df['view_count']
        df['comment_rate'] = df['comment_count'] / df['view_count']
        df['engagement_rate'] = df['like_rate'] + df['comment_rate']

        # 处理极端值
        df['like_rate'] = np.clip(df['like_rate'], 0, 1)
        df['comment_rate'] = np.clip(df['comment_rate'], 0, 1)
        df['engagement_rate'] = np.clip(df['engagement_rate'], 0, 2)

        print(f"处理后数据形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        raise

def perform_clustering(df):
    """执行视频聚类分析"""
    print("2. 准备聚类特征...")

    # 对观看量进行对数变换处理长尾分布
    df['log_view_count'] = np.log1p(df['view_count'])

    # 选择用于聚类的特征
    features = ['log_view_count', 'like_rate', 'comment_rate']
    X = df[features].copy()

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"特征维度: {X_scaled.shape}")

    # 使用KMeans聚类，固定5个簇对应5个业务标签
    print("3. 执行KMeans聚类(K=5)...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    print(f"聚类结果:")
    cluster_counts = df['cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"  簇 {cluster}: {count} 样本 ({count/len(df)*100:.1f}%)")

    return df, X_scaled, X, features

def assign_business_labels(df):
    """为每个簇分配业务标签"""
    print("4. 分配业务标签...")

    # 计算阈值
    view_threshold = df['view_count'].quantile(0.6)
    like_threshold = df['like_rate'].quantile(0.6)
    comment_threshold = df['comment_rate'].quantile(0.6)

    print(f"阈值: 观看量={view_threshold:.1f}, 点赞率={like_threshold:.4f}, 评论率={comment_threshold:.4f}")

    # 计算每个簇的统计信息
    cluster_stats = df.groupby('cluster').agg({
        'view_count': 'mean',
        'like_rate': 'mean',
        'comment_rate': 'mean'
    })

    # 计算高低属性
    cluster_stats['high_view'] = cluster_stats['view_count'] >= view_threshold
    cluster_stats['high_like'] = cluster_stats['like_rate'] >= like_threshold
    cluster_stats['high_comment'] = cluster_stats['comment_rate'] >= comment_threshold

    # 业务标签映射
    business_labels = {}

    for cluster_id, row in cluster_stats.iterrows():
        if row['high_view']:
            if row['high_like'] and row['high_comment']:
                label = '明星全能'  # 高曝光、高点赞率、高评论率
            elif row['high_like'] and not row['high_comment']:
                label = '安静点赞'  # 高曝光、高点赞率、低评论率
            elif not row['high_like'] and row['high_comment']:
                label = '讨论热议'  # 高曝光、低点赞率、高评论率
            else:
                label = '过眼云烟'  # 高曝光、低点赞率、低评论率
        else:
            label = '沉寂孤立'  # 低曝光、低点赞率、低评论率

        business_labels[cluster_id] = label

    # 将业务标签映射到数据框
    df['business_type'] = df['cluster'].map(business_labels)

    # 输出每个业务类型的数量
    type_counts = df['business_type'].value_counts()
    print("业务标签分布:")
    for btype, count in type_counts.items():
        print(f"  {btype}: {count} 样本 ({count/len(df)*100:.1f}%)")

    # 输出每个簇的统计信息
    cluster_stats['business_type'] = cluster_stats.index.map(business_labels)
    print("簇统计信息:")
    print(cluster_stats[['view_count', 'like_rate', 'comment_rate', 'business_type']].round(4))

    return df, cluster_stats

def visualize_results(df, X_scaled, X, features):
    """生成可视化结果"""
    print("5. 生成可视化结果...")

    # 设置更好的颜色方案和绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    color_palette = sns.color_palette("viridis", n_colors=len(df['business_type'].unique()))

    # 1. 使用PCA进行降维可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    plt.figure(figsize=(20, 15))

    plt.subplot(2, 2, 1)
    # 创建一个颜色映射字典
    business_types = df['business_type'].unique()
    business_type_to_index = {btype: i for i, btype in enumerate(business_types)}
    colors = [business_type_to_index[btype] for btype in df['business_type']]
    
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=colors, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='业务类型')
    plt.title('PCA降维: 业务类型分布', fontsize=14)
    plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    
    # 手动创建图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/len(business_types)), 
                        markersize=10) for i in range(len(business_types))]
    plt.legend(handles=handles, labels=business_types, title="业务类型", loc="upper right")

    # 2. 观看量与点赞率的关系，按业务类型着色
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='log_view_count', y='like_rate', hue='business_type', 
                    palette=color_palette, s=50, alpha=0.7)
    plt.axvline(x=np.log1p(df['view_count'].quantile(0.6)), color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=df['like_rate'].quantile(0.6), color='red', linestyle='--', alpha=0.7)
    plt.title('观看量 vs 点赞率', fontsize=14)
    plt.xlabel('观看量 (对数)', fontsize=12)
    plt.ylabel('点赞率', fontsize=12)
    plt.legend(title="业务类型")

    # 3. 点赞率与评论率的关系，按业务类型着色
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='like_rate', y='comment_rate', hue='business_type', 
                    palette=color_palette, s=50, alpha=0.7)
    plt.axvline(x=df['like_rate'].quantile(0.6), color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=df['comment_rate'].quantile(0.6), color='red', linestyle='--', alpha=0.7)
    plt.title('点赞率 vs 评论率', fontsize=14)
    plt.xlabel('点赞率', fontsize=12)
    plt.ylabel('评论率', fontsize=12)
    plt.legend(title="业务类型")

    # 4. 业务类型的特征雷达图
    plt.subplot(2, 2, 4)

    # 准备数据
    radar_data = df.groupby('business_type')[['view_count', 'like_rate', 'comment_rate']].mean()

    # 标准化数据用于雷达图
    scaler = MinMaxScaler()
    radar_data_scaled = pd.DataFrame(
        scaler.fit_transform(radar_data), 
        index=radar_data.index, 
        columns=radar_data.columns
    )

    # 绘制雷达图
    categories = ['观看量', '点赞率', '评论率']
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形

    # 初始化雷达图
    ax = plt.subplot(2, 2, 4, polar=True)

    # 绘制每个业务类型的雷达图
    for i, (idx, row) in enumerate(radar_data_scaled.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, linewidth=2, label=idx, color=color_palette[i])
        ax.fill(angles, values, alpha=0.1, color=color_palette[i])

    # 设置雷达图属性
    plt.xticks(angles[:-1], categories)
    ax.set_title('业务类型特征雷达图', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig('video_clustering_results.png', dpi=300, bbox_inches='tight')
    print("已保存可视化结果到: video_clustering_results.png")

    # 5. 额外的业务类型箱线图比较
    plt.figure(figsize=(15, 12))

    # 观看量箱线图
    plt.subplot(3, 1, 1)
    sns.boxplot(x='business_type', y='view_count', data=df, palette=color_palette)
    plt.title('各业务类型的观看量分布', fontsize=14)
    plt.xlabel('业务类型', fontsize=12)
    plt.ylabel('观看量', fontsize=12)
    plt.yscale('log')  # 使用对数尺度以便更好地可视化

    # 点赞率箱线图
    plt.subplot(3, 1, 2)
    sns.boxplot(x='business_type', y='like_rate', data=df, palette=color_palette)
    plt.title('各业务类型的点赞率分布', fontsize=14)
    plt.xlabel('业务类型', fontsize=12)
    plt.ylabel('点赞率', fontsize=12)

    # 评论率箱线图
    plt.subplot(3, 1, 3)
    sns.boxplot(x='business_type', y='comment_rate', data=df, palette=color_palette)
    plt.title('各业务类型的评论率分布', fontsize=14)
    plt.xlabel('业务类型', fontsize=12)
    plt.ylabel('评论率', fontsize=12)

    plt.tight_layout()
    plt.savefig('video_type_distributions.png', dpi=300, bbox_inches='tight')
    print("已保存分布图到: video_type_distributions.png")

    # 展示图表
    plt.show()

def save_results(df, cluster_stats):
    """保存聚类结果"""
    print("6. 保存分析结果...")
    df.to_csv('clustered_videos.csv', index=False)
    cluster_stats.to_csv('cluster_statistics.csv')
    print(f"已保存聚类结果到: clustered_videos.csv")
    print(f"已保存簇统计信息到: cluster_statistics.csv")

def main():
    """主函数"""
    print("="*50)
    print("视频数据五类业务标签聚类分析")
    print("="*50)

    try:
        # 1. 加载数据
        df = load_data(file_path)

        # 2. 执行聚类
        df, X_scaled, X, features = perform_clustering(df)

        # 3. 分配业务标签
        df, cluster_stats = assign_business_labels(df)

        # 4. 可视化结果
        visualize_results(df, X_scaled, X, features)

        # 5. 保存结果
        save_results(df, cluster_stats)

        print("="*50)
        print("分析完成!")
        print("="*50)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
