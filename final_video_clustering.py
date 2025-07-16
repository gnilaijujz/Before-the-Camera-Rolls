# 视频数据聚类分析与可视化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

        # 处理极端值
        df['like_rate'] = np.clip(df['like_rate'], 0, 1)
        df['comment_rate'] = np.clip(df['comment_rate'], 0, 1)

        print(f"处理后数据形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        raise

def prepare_features(df):
    """准备聚类特征"""
    print("2. 准备聚类特征...")

    # 处理用于聚类的特征
    df['log_view'] = np.log1p(df['view_count'])

    # 选择用于聚类的特征
    features = ['log_view', 'like_rate', 'comment_rate']
    X = df[features].copy()

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"特征维度: {X_scaled.shape}")
    return X_scaled, features

def find_optimal_k(X_scaled, k_range=(2, 10)):
    """寻找最佳聚类数"""
    print("3. 寻找最佳聚类数...")

    # 轮廓系数评估
    silhouette_scores = []
    inertia_values = []
    k_values = range(k_range[0], k_range[1] + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        inertia_values.append(kmeans.inertia_)
        print(f"  K = {k}, 轮廓系数 = {silhouette_scores[-1]:.4f}, 惯性 = {inertia_values[-1]:.2f}")

    # 可视化评估结果
    plt.figure(figsize=(14, 6))

    # 轮廓系数图
    plt.subplot(1, 2, 1)
    plt.plot(k_values, silhouette_scores, 'o-', color='#5094D5', linewidth=2)
    best_k_sil = k_values[np.argmax(silhouette_scores)]
    plt.axvline(x=best_k_sil, color='#D15354', linestyle='--')
    plt.annotate(f'最佳K = {best_k_sil}', 
                 xy=(best_k_sil, max(silhouette_scores)),
                 xytext=(best_k_sil+0.5, max(silhouette_scores)-0.01),
                 arrowprops=dict(facecolor='#D15354', shrink=0.05))
    plt.title('轮廓系数评估方法', fontsize=14, color='#333333', fontweight='bold')
    plt.xlabel('聚类数 (K)', fontsize=12)
    plt.ylabel('轮廓系数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 肘部法图
    plt.subplot(1, 2, 2)
    plt.plot(k_values, inertia_values, 'o-', color='#ABD8E5', linewidth=2)
    plt.title('肘部法评估方法', fontsize=14, color='#333333', fontweight='bold')
    plt.xlabel('聚类数 (K)', fontsize=12)
    plt.ylabel('簇内平方和 (惯性)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('optimal_k_evaluation.png', dpi=300, bbox_inches='tight')

    # 返回最佳K（基于轮廓系数）
    best_k = k_values[np.argmax(silhouette_scores)]

    # 如果不是5，强制使用5以匹配业务需求
    if best_k != 5:
        print(f"  评估最佳K为{best_k}，但为匹配业务需求，使用K=5")
        best_k = 5
    else:
        print(f"  评估最佳K为{best_k}，与业务需求匹配")

    return best_k

def perform_clustering(df, X_scaled, k=5):
    """执行KMeans聚类"""
    print(f"4. 执行KMeans聚类 (K={k})...")

    # 训练KMeans模型
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 输出各簇数量
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("各簇样本数:")
    for cluster, count in cluster_counts.items():
        print(f"  簇 {cluster}: {count} 样本 ({count/len(df)*100:.1f}%)")

    return df, kmeans

def assign_business_labels(df):
    """分配业务标签"""
    print("5. 分配业务标签...")

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

    # 计算每个簇的高低属性
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

    # 确保五类都有
    unique_labels = set(business_labels.values())
    if len(unique_labels) < 5:
        print("警告: 自动分类无法生成所有5个业务类别，使用人工调整")
        # 确保5个类别都有代表
        required_labels = ['明星全能', '安静点赞', '讨论热议', '过眼云烟', '沉寂孤立']
        missing_labels = set(required_labels) - unique_labels

        # 简单策略：将前几个簇重新分配给缺失的标签
        for i, missing_label in enumerate(missing_labels):
            if i < len(business_labels):
                business_labels[i] = missing_label

    # 将业务标签映射到数据框
    df['business_type'] = df['cluster'].map(business_labels)

    # 输出各类别统计
    type_counts = df['business_type'].value_counts()
    print("业务标签分布:")
    for btype, count in type_counts.items():
        print(f"  {btype}: {count} 样本 ({count/len(df)*100:.1f}%)")

    # 详细统计
    business_stats = df.groupby('business_type').agg({
        'view_count': ['mean', 'median'],
        'like_rate': ['mean', 'median'],
        'comment_rate': ['mean', 'median']
    })

    print("业务类型统计:")
    print(business_stats)

    return df, business_stats

def visualize_2d_pca(df, X_scaled, title="PCA二维聚类可视化"):
    """使用PCA进行二维可视化"""
    print("6. 生成二维PCA可视化...")

    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    # 创建包含PCA结果和聚类标签的DataFrame
    df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    df_pca['business_type'] = df['business_type'].values

    # 可视化
    plt.figure(figsize=(12, 10))

    # 设置颜色映射 - 使用提供的新配色方案
    colors = {
        '明星全能': '#D15354',  # 红色 (209, 83, 84)
        '安静点赞': '#5094D5',  # 蓝色 (80, 148, 213)
        '讨论热议': '#F9AD95',  # 珊瑚色 (249, 173, 149)
        '过眼云烟': '#ABD8E5',  # 浅蓝色 (171, 216, 229)
        '沉寂孤立': '#FEEEDD'   # 米色 (254, 238, 237)
    }

    # 绘制散点图
    for label, color in colors.items():
        mask = df_pca['business_type'] == label
        plt.scatter(
            df_pca.loc[mask, 'PCA1'], 
            df_pca.loc[mask, 'PCA2'],
            c=color, 
            label=label,
            alpha=0.7,
            edgecolors='w',
            s=80
        )

    # 添加标题和轴标签
    plt.title(title, fontsize=20, color='#333333', fontweight='bold', pad=20)
    plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', fontsize=14)
    plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', fontsize=14)

    # 添加图例
    plt.legend(title="业务类型", title_fontsize=14, fontsize=12, 
               loc='upper right', bbox_to_anchor=(1.15, 1))

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)

    # 美化
    plt.tight_layout()
    plt.savefig('pca_2d_visualization.png', dpi=300, bbox_inches='tight')

    return pca, df_pca

def visualize_3d_pca(df, X_scaled, title="PCA三维聚类可视化"):
    """使用PCA进行三维可视化"""
    print("7. 生成三维PCA可视化...")

    # 使用PCA降维到3维
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_scaled)

    # 创建包含PCA结果和聚类标签的DataFrame
    df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['business_type'] = df['business_type'].values

    # 可视化
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 设置颜色映射 - 使用提供的新配色方案
    colors = {
        '明星全能': '#D15354',  # 红色 (209, 83, 84)
        '安静点赞': '#5094D5',  # 蓝色 (80, 148, 213)
        '讨论热议': '#F9AD95',  # 珊瑚色 (249, 173, 149)
        '过眼云烟': '#ABD8E5',  # 浅蓝色 (171, 216, 229)
        '沉寂孤立': '#FEEEDD'   # 米色 (254, 238, 237)
    }

    # 绘制散点图
    for label, color in colors.items():
        mask = df_pca['business_type'] == label
        ax.scatter(
            df_pca.loc[mask, 'PCA1'], 
            df_pca.loc[mask, 'PCA2'], 
            df_pca.loc[mask, 'PCA3'],
            c=color, 
            label=label,
            alpha=0.7,
            s=60
        )

    # 添加标题和轴标签
    ax.set_title(title, fontsize=20, color='#333333', fontweight='bold', pad=20)
    ax.set_xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', fontsize=14)
    ax.set_ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', fontsize=14)
    ax.set_zlabel(f'主成分3 (解释方差: {pca.explained_variance_ratio_[2]:.2%})', fontsize=14)

    # 添加图例
    ax.legend(title="业务类型", title_fontsize=14, fontsize=12, 
              loc='upper right', bbox_to_anchor=(1.15, 1))

    # 设置视角
    ax.view_init(elev=30, azim=45)

    # 美化
    plt.tight_layout()
    plt.savefig('pca_3d_visualization.png', dpi=300, bbox_inches='tight')

    return pca, df_pca

def visualize_feature_distribution(df):
    """可视化各业务类型的特征分布"""
    print("8. 生成特征分布可视化...")

    plt.figure(figsize=(18, 12))

    # 设置颜色映射 - 使用提供的新配色方案
    colors = {
        '明星全能': '#D15354',  # 红色 (209, 83, 84)
        '安静点赞': '#5094D5',  # 蓝色 (80, 148, 213)
        '讨论热议': '#F9AD95',  # 珊瑚色 (249, 173, 149)
        '过眼云烟': '#ABD8E5',  # 浅蓝色 (171, 216, 229)
        '沉寂孤立': '#FEEEDD'   # 米色 (254, 238, 237)
    }
    # 创建颜色调色板，确保颜色顺序与业务类型一致
    business_types = sorted(df['business_type'].unique())
    color_palette = [colors[btype] for btype in business_types]

    # 观看量箱线图 (对数尺度)
    plt.subplot(2, 2, 1)
    sns.boxplot(x='business_type', y='view_count', data=df, palette=color_palette)
    plt.title('各业务类型的观看量分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('观看量', fontsize=14)
    plt.yscale('log')  # 使用对数尺度
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 点赞率箱线图
    plt.subplot(2, 2, 2)
    sns.boxplot(x='business_type', y='like_rate', data=df, palette=color_palette)
    plt.title('各业务类型的点赞率分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('点赞率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 评论率箱线图
    plt.subplot(2, 2, 3)
    sns.boxplot(x='business_type', y='comment_rate', data=df, palette=color_palette)
    plt.title('各业务类型的评论率分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('评论率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 时长分布箱线图
    plt.subplot(2, 2, 4)
    sns.boxplot(x='business_type', y='duration_minutes', data=df, palette=color_palette)
    plt.title('各业务类型的视频时长分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('视频时长(分钟)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')

def create_radar_chart(df):
    """创建雷达图比较各业务类型的特征"""
    print("9. 生成雷达图可视化...")

    # 计算每个业务类型的平均特征值
    business_features = df.groupby('business_type').agg({
        'view_count': 'mean',
        'like_rate': 'mean',
        'comment_rate': 'mean',
        'duration_minutes': 'mean'
    })

    # 特征归一化（Min-Max缩放到0-1之间）
    for col in business_features.columns:
        business_features[col] = (business_features[col] - business_features[col].min()) / (business_features[col].max() - business_features[col].min())

    # 设置颜色映射 - 使用提供的新配色方案
    colors = {
        '明星全能': '#D15354',  # 红色 (209, 83, 84)
        '安静点赞': '#5094D5',  # 蓝色 (80, 148, 213)
        '讨论热议': '#F9AD95',  # 珊瑚色 (249, 173, 149)
        '过眼云烟': '#ABD8E5',  # 浅蓝色 (171, 216, 229)
        '沉寂孤立': '#FEEEDD'   # 米色 (254, 238, 237)
    }

    # 设置雷达图参数
    categories = ['观看量', '点赞率', '评论率', '视频时长']
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形

    # 创建图形
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # 为每个业务类型绘制雷达图
    for btype in business_features.index:
        values = business_features.loc[btype].values.flatten().tolist()
        values += values[:1]  # 闭合图形

        ax.plot(angles, values, 'o-', linewidth=2, label=btype, color=colors[btype])
        ax.fill(angles, values, alpha=0.1, color=colors[btype])

    # 设置雷达图的刻度标签
    plt.xticks(angles[:-1], categories, fontsize=14)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title('业务类型特征对比雷达图', fontsize=20, color='#333333', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')

def create_heatmap(df):
    """创建热图展示特征相关性和业务类型分布"""
    print("10. 生成热图可视化...")

    # 计算特征相关性
    corr_features = df[['view_count', 'like_count', 'comment_count', 'duration_seconds', 
                        'like_rate', 'comment_rate']].corr()

    plt.figure(figsize=(12, 10))

    # 绘制热图
    sns.heatmap(corr_features, annot=True, cmap='coolwarm', linewidths=0.5, 
                fmt='.2f', square=True)
    plt.title('特征相关性热图', fontsize=20, color='#333333', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

def save_results(df):
    """保存分析结果"""
    print("11. 保存分析结果...")
    df.to_csv('video_clustering_results.csv', index=False)
    print(f"已保存聚类结果到: video_clustering_results.csv")

def main():
    """主函数"""
    print("="*50)
    print("视频数据聚类分析与可视化")
    print("="*50)

    try:
        # 1. 加载数据
        df = load_data(file_path)

        # 2. 准备特征
        X_scaled, features = prepare_features(df)

        # 3. 寻找最佳聚类数
        best_k = find_optimal_k(X_scaled)

        # 4. 执行聚类
        df, kmeans = perform_clustering(df, X_scaled, k=best_k)

        # 5. 分配业务标签
        df, business_stats = assign_business_labels(df)

        # 6. 二维PCA可视化
        pca_2d, df_pca_2d = visualize_2d_pca(df, X_scaled, "视频内容五类业务聚类 - 二维可视化")

        # 7. 三维PCA可视化
        pca_3d, df_pca_3d = visualize_3d_pca(df, X_scaled, "视频内容五类业务聚类 - 三维可视化")

        # 8. 特征分布可视化
        visualize_feature_distribution(df)

        # 9. 雷达图可视化
        create_radar_chart(df)

        # 10. 热图可视化
        create_heatmap(df)

        # 11. 保存结果
        save_results(df)

        print("="*50)
        print("分析与可视化完成!")
        print("生成的图像文件:")
        print("  - optimal_k_evaluation.png: K值评估")
        print("  - pca_2d_visualization.png: 二维PCA可视化")
        print("  - pca_3d_visualization.png: 三维PCA可视化")
        print("  - feature_distributions.png: 特征分布箱线图")
        print("  - radar_chart.png: 业务类型雷达图")
        print("  - correlation_heatmap.png: 特征相关性热图")
        print("="*50)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
