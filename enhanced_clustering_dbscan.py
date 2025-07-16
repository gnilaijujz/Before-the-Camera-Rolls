# 视频数据聚类分析与可视化 - 增强版 (无需HDBSCAN)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义高质量配色方案
COLORS = {
    '明星全能': '#D15354',  # 红色 (209, 83, 84)
    '安静点赞': '#5094D5',  # 蓝色 (80, 148, 213)
    '讨论热议': '#F9AD95',  # 珊瑚色 (249, 173, 149)
    '过眼云烟': '#ABD8E5',  # 浅蓝色 (171, 216, 229)
    '沉寂孤立': '#FEEEDD'   # 米色 (254, 238, 237)
}

# 视频数据文件路径
file_path = "D:\\下载\\sports_2024-2025.csv"

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
        df['engagement_rate'] = df['like_rate'] + df['comment_rate']  # 综合互动率

        # 计算面向中间状态的新特征
        df['like_comment_ratio'] = np.where(df['comment_rate'] > 0, 
                                          df['like_rate'] / df['comment_rate'], 
                                          df['like_rate'] * 100)  # 点赞评论比

        # 处理极端值
        df['like_rate'] = np.clip(df['like_rate'], 0, 1)
        df['comment_rate'] = np.clip(df['comment_rate'], 0, 1)
        df['engagement_rate'] = np.clip(df['engagement_rate'], 0, 2)

        print(f"处理后数据形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        raise

def prepare_features(df):
    """准备聚类特征，包含额外的区分指标"""
    print("2. 准备聚类特征...")

    # 对数变换长尾分布特征
    df['log_view'] = np.log1p(df['view_count'])

    # 选择用于聚类的特征 - 加入额外特征增强区分度
    features = [
        'log_view',      # 观看量（对数）
        'like_rate',     # 点赞率
        'comment_rate',  # 评论率
        'engagement_rate',  # 综合互动率
        'like_comment_ratio'  # 点赞评论比
    ]

    X = df[features].copy()

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"特征维度: {X_scaled.shape}, 特征列表: {features}")
    return X_scaled, features, X

def evaluate_clustering_algorithms(X_scaled, k_range=(3, 8)):
    """全面评估多种聚类算法和参数"""
    print("3. 多算法聚类评估...")

    results = []
    k_values = range(k_range[0], k_range[1] + 1)

    # 1. 评估KMeans在不同K值下的表现
    print("\n1) KMeans评估:")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # 计算多种评估指标
        sil_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)

        results.append({
            'algorithm': 'KMeans',
            'params': k,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score,
            'n_clusters': len(set(labels))
        })

        print(f"  K = {k}, 轮廓系数 = {sil_score:.4f}, 戴维斯-波尔丁指数 = {db_score:.4f}, CH指数 = {ch_score:.2f}")

    # 2. 评估DBSCAN在不同参数下的表现
    print("\n2) DBSCAN评估:")
    for eps in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for min_samples in [5, 10, 15]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

            # 排除噪声点
            if -1 in labels:
                valid_indices = labels != -1
                if sum(valid_indices) > 1 and len(set(labels[valid_indices])) > 1:
                    try:
                        sil_score = silhouette_score(X_scaled[valid_indices], labels[valid_indices])
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        results.append({
                            'algorithm': 'DBSCAN',
                            'params': f"eps={eps}, min_samples={min_samples}",
                            'silhouette': sil_score,
                            'davies_bouldin': np.nan,  # DBSCAN不适用此指标
                            'calinski_harabasz': np.nan,  # DBSCAN不适用此指标
                            'n_clusters': n_clusters,
                            'noise_points': sum(labels == -1)
                        })

                        print(f"  eps={eps}, min_samples={min_samples}, 簇数={n_clusters}, 噪声点={sum(labels == -1)}, 轮廓系数={sil_score:.4f}")
                    except:
                        continue

    # 3. 评估更多DBSCAN参数 (尝试更多参数组合)
    print("\n3) 额外DBSCAN评估:")
    for eps in [0.2, 0.25, 0.35, 0.45, 0.55, 0.65]:
        for min_samples in [3, 8, 12, 18]:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)

                # 排除噪声点
                if -1 in labels:
                    valid_indices = labels != -1
                    if sum(valid_indices) > 1 and len(set(labels[valid_indices])) > 1:
                        sil_score = silhouette_score(X_scaled[valid_indices], labels[valid_indices])
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        results.append({
                            'algorithm': 'DBSCAN_Extended',
                            'params': f"eps={eps}, min_samples={min_samples}",
                            'silhouette': sil_score,
                            'davies_bouldin': np.nan,  # DBSCAN不适用此指标
                            'calinski_harabasz': np.nan,  # DBSCAN不适用此指标
                            'n_clusters': n_clusters,
                            'noise_points': sum(labels == -1)
                        })

                        print(f"  eps={eps}, min_samples={min_samples}, 簇数={n_clusters}, 噪声点={sum(labels == -1)}, 轮廓系数={sil_score:.4f}")
            except:
                continue

    # 4. 评估层次聚类
    print("\n4) 层次聚类评估:")
    for k in k_values:
        for linkage in ['ward', 'complete', 'average']:
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = agg.fit_predict(X_scaled)

                sil_score = silhouette_score(X_scaled, labels)
                db_score = davies_bouldin_score(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)

                results.append({
                    'algorithm': 'AgglomerativeClustering',
                    'params': f"n_clusters={k}, linkage={linkage}",
                    'silhouette': sil_score,
                    'davies_bouldin': db_score,
                    'calinski_harabasz': ch_score,
                    'n_clusters': k
                })

                print(f"  n_clusters={k}, linkage={linkage}, 轮廓系数={sil_score:.4f}, 戴维斯-波尔丁指数={db_score:.4f}, CH指数={ch_score:.2f}")
            except:
                continue

    # 结果排序和可视化
    results_df = pd.DataFrame(results)
    best_silhouette = results_df.sort_values('silhouette', ascending=False).iloc[0]

    # 可视化KMeans的指标变化
    kmeans_results = results_df[results_df['algorithm'] == 'KMeans']

    plt.figure(figsize=(15, 8))

    # 轮廓系数图
    plt.subplot(1, 3, 1)
    plt.plot(kmeans_results['params'], kmeans_results['silhouette'], 'o-', color='#5094D5', linewidth=2)
    best_k_sil = kmeans_results.loc[kmeans_results['silhouette'].idxmax(), 'params']
    plt.axvline(x=best_k_sil, color='#D15354', linestyle='--')
    plt.annotate(f'最佳K = {best_k_sil}', 
                 xy=(best_k_sil, kmeans_results['silhouette'].max()),
                 xytext=(best_k_sil+0.5, kmeans_results['silhouette'].max()-0.01),
                 arrowprops=dict(facecolor='#D15354', shrink=0.05))
    plt.title('轮廓系数评估 (越高越好)', fontsize=14, color='#333333', fontweight='bold')
    plt.xlabel('聚类数 (K)', fontsize=12)
    plt.ylabel('轮廓系数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 戴维斯-波尔丁指数图
    plt.subplot(1, 3, 2)
    plt.plot(kmeans_results['params'], kmeans_results['davies_bouldin'], 'o-', color='#F9AD95', linewidth=2)
    best_k_db = kmeans_results.loc[kmeans_results['davies_bouldin'].idxmin(), 'params']
    plt.axvline(x=best_k_db, color='#D15354', linestyle='--')
    plt.annotate(f'最佳K = {best_k_db}', 
                 xy=(best_k_db, kmeans_results['davies_bouldin'].min()),
                 xytext=(best_k_db+0.5, kmeans_results['davies_bouldin'].min()+0.1),
                 arrowprops=dict(facecolor='#D15354', shrink=0.05))
    plt.title('戴维斯-波尔丁指数 (越低越好)', fontsize=14, color='#333333', fontweight='bold')
    plt.xlabel('聚类数 (K)', fontsize=12)
    plt.ylabel('戴维斯-波尔丁指数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 卡林斯基-哈拉巴斯指数图
    plt.subplot(1, 3, 3)
    plt.plot(kmeans_results['params'], kmeans_results['calinski_harabasz'], 'o-', color='#ABD8E5', linewidth=2)
    best_k_ch = kmeans_results.loc[kmeans_results['calinski_harabasz'].idxmax(), 'params']
    plt.axvline(x=best_k_ch, color='#D15354', linestyle='--')
    plt.annotate(f'最佳K = {best_k_ch}', 
                 xy=(best_k_ch, kmeans_results['calinski_harabasz'].max()),
                 xytext=(best_k_ch+0.5, kmeans_results['calinski_harabasz'].max()-50),
                 arrowprops=dict(facecolor='#D15354', shrink=0.05))
    plt.title('卡林斯基-哈拉巴斯指数 (越高越好)', fontsize=14, color='#333333', fontweight='bold')
    plt.xlabel('聚类数 (K)', fontsize=12)
    plt.ylabel('卡林斯基-哈拉巴斯指数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('clustering_evaluation.png', dpi=300, bbox_inches='tight')

    # 综合推荐最佳聚类设置
    print("\n评估结果汇总:")
    print(f"整体最佳算法与参数: {best_silhouette['algorithm']}, 参数: {best_silhouette['params']}, 轮廓系数: {best_silhouette['silhouette']:.4f}")

    # 根据指标综合推荐KMeans的最佳K值
    # 一般来说轮廓系数是最直观的指标，但我们也考虑其他指标
    all_best_k = [best_k_sil, best_k_db, best_k_ch]
    recommended_k = max(set(all_best_k), key=all_best_k.count)  # 取众数

    if recommended_k != 5:
        print(f"评估推荐的最佳K值为{recommended_k}，但为匹配业务需求5类标签，使用K=5")
        recommended_k = 5
    else:
        print(f"评估推荐的最佳K值为{recommended_k}，与业务需求匹配")

    # 推荐最佳算法
    if best_silhouette['algorithm'] == 'KMeans':
        return 'kmeans', recommended_k, results_df
    elif best_silhouette['algorithm'] == 'DBSCAN' or best_silhouette['algorithm'] == 'DBSCAN_Extended':
        eps, min_samples = best_silhouette['params'].replace('eps=', '').replace('min_samples=', '').replace(' ', '').split(',')
        return 'dbscan', (float(eps), int(min_samples)), results_df
    else:
        n_clusters, linkage = best_silhouette['params'].replace('n_clusters=', '').replace('linkage=', '').replace(' ', '').split(',')
        return 'agglomerative', (int(n_clusters), linkage), results_df

def perform_clustering(df, X_scaled, algorithm='kmeans', params=5):
    """执行指定的聚类算法"""
    print(f"4. 执行{algorithm}聚类...")

    if algorithm == 'kmeans':
        k = params
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['cluster'] = model.fit_predict(X_scaled)
        print(f"使用KMeans, 簇数 = {k}")

    elif algorithm == 'dbscan':
        eps, min_samples = params
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

        # 处理噪声点
        if -1 in labels:
            noise_points = np.where(labels == -1)[0]
            print(f"发现 {len(noise_points)} 个噪声点 ({len(noise_points)/len(labels)*100:.1f}%), 将它们分配到最近的簇")

            # 计算非噪声点的簇中心
            non_noise_indices = np.where(labels != -1)[0]
            unique_labels = np.unique(labels[non_noise_indices])
            centroids = {}

            for label in unique_labels:
                mask = labels == label
                centroids[label] = X_scaled[mask].mean(axis=0)

            # 将噪声点分配到最近的簇
            for idx in noise_points:
                point = X_scaled[idx].reshape(1, -1)
                distances = {label: np.linalg.norm(point - centroid) for label, centroid in centroids.items()}
                closest_label = min(distances, key=distances.get)
                labels[idx] = closest_label

        df['cluster'] = labels
        model = {'eps': eps, 'min_samples': min_samples}
        print(f"使用DBSCAN, eps = {eps}, min_samples = {min_samples}")

    elif algorithm == 'agglomerative':
        n_clusters, linkage = params
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        df['cluster'] = model.fit_predict(X_scaled)
        print(f"使用层次聚类, n_clusters = {n_clusters}, linkage = {linkage}")

    # 输出各簇数量
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("各簇样本数:")
    for cluster, count in cluster_counts.items():
        print(f"  簇 {cluster}: {count} 样本 ({count/len(df)*100:.1f}%)")

    return df, model

def assign_business_labels(df, X):
    """基于业务规则和聚类结果分配业务标签"""
    print("5. 分配业务标签...")

    # 计算阈值 - 使用分位数确保相对性
    view_threshold = df['view_count'].quantile(0.6)
    like_threshold = df['like_rate'].quantile(0.6)
    comment_threshold = df['comment_rate'].quantile(0.6)
    engagement_threshold = df['engagement_rate'].quantile(0.6)

    print(f"相对阈值: 观看量={view_threshold:.1f}, 点赞率={like_threshold:.4f}, 评论率={comment_threshold:.4f}, 互动率={engagement_threshold:.4f}")

    # 计算每个簇的统计信息
    cluster_stats = df.groupby('cluster').agg({
        'view_count': 'mean',
        'like_rate': 'mean',
        'comment_rate': 'mean',
        'engagement_rate': 'mean',
        'like_comment_ratio': 'mean'
    })

    # 计算每个簇的高低属性
    cluster_stats['high_view'] = cluster_stats['view_count'] >= view_threshold
    cluster_stats['high_like'] = cluster_stats['like_rate'] >= like_threshold
    cluster_stats['high_comment'] = cluster_stats['comment_rate'] >= comment_threshold
    cluster_stats['high_engagement'] = cluster_stats['engagement_rate'] >= engagement_threshold

    # 计算点赞vs评论的主导型
    # 高点赞评论比表示点赞更主导，低点赞评论比表示评论更主导
    cluster_stats['like_dominant'] = cluster_stats['like_comment_ratio'] >= cluster_stats['like_comment_ratio'].median()

    # 改进的业务标签映射逻辑
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
                # 改进：进一步区分"过眼云烟"
                if row['engagement_rate'] <= cluster_stats['engagement_rate'].quantile(0.3):
                    label = '过眼云烟'  # 高曝光、极低互动
                else:
                    # 根据点赞评论比进一步细分
                    if row['like_dominant']:
                        label = '安静点赞'  # 高曝光、中点赞率、低评论率
                    else:
                        label = '讨论热议'  # 高曝光、低点赞率、中评论率
        else:
            label = '沉寂孤立'  # 低曝光、低点赞率、低评论率

        business_labels[cluster_id] = label

    # 确保五类都有
    unique_labels = set(business_labels.values())
    required_labels = ['明星全能', '安静点赞', '讨论热议', '过眼云烟', '沉寂孤立']
    missing_labels = set(required_labels) - unique_labels

    if missing_labels:
        print(f"警告: 自动分类未生成所有5个业务类别，缺少: {missing_labels}")
        print("执行基于特征中心的业务类型分配...")

        # 构建理想的特征原型
        ideal_profiles = {
            '明星全能': [1.0, 1.0, 1.0, 1.0],  # 高观看、高点赞、高评论、高互动
            '安静点赞': [1.0, 1.0, 0.0, 0.7],  # 高观看、高点赞、低评论、中互动
            '讨论热议': [1.0, 0.0, 1.0, 0.7],  # 高观看、低点赞、高评论、中互动
            '过眼云烟': [1.0, 0.0, 0.0, 0.0],  # 高观看、低点赞、低评论、低互动
            '沉寂孤立': [0.0, 0.0, 0.0, 0.0]   # 低观看、低点赞、低评论、低互动
        }

        # 正规化簇统计数据用于计算距离
        stats_for_distance = cluster_stats[['view_count', 'like_rate', 'comment_rate', 'engagement_rate']].copy()
        for col in stats_for_distance.columns:
            if stats_for_distance[col].max() > 0:
                stats_for_distance[col] = stats_for_distance[col] / stats_for_distance[col].max()

        # 计算每个簇到每个理想原型的距离
        cluster_to_profile_distance = {}
        for cluster_id, cluster_vector in stats_for_distance.iterrows():
            cluster_to_profile_distance[cluster_id] = {}
            for profile_name, profile_vector in ideal_profiles.items():
                distance = np.linalg.norm(np.array(cluster_vector) - np.array(profile_vector))
                cluster_to_profile_distance[cluster_id][profile_name] = distance

        # 基于最小距离分配业务标签，确保每个所需标签都有对应簇
        assigned_profiles = set()
        final_business_labels = {}

        # 第一轮：处理缺失的标签，寻找最近的簇
        for profile in missing_labels:
            best_cluster = None
            min_distance = float('inf')

            for cluster_id, distances in cluster_to_profile_distance.items():
                if distances[profile] < min_distance and cluster_id not in final_business_labels.values():
                    min_distance = distances[profile]
                    best_cluster = cluster_id

            if best_cluster is not None:
                final_business_labels[best_cluster] = profile
                assigned_profiles.add(profile)

        # 第二轮：处理剩余的簇，使用原始标签或寻找最近的未分配标签
        for cluster_id, original_label in business_labels.items():
            if cluster_id not in final_business_labels:
                # 如果原始标签未被使用，继续使用
                if original_label not in assigned_profiles:
                    final_business_labels[cluster_id] = original_label
                    assigned_profiles.add(original_label)
                else:
                    # 否则寻找最近的未分配标签
                    min_distance = float('inf')
                    best_label = None

                    for profile in required_labels:
                        if profile not in assigned_profiles:
                            if cluster_to_profile_distance[cluster_id][profile] < min_distance:
                                min_distance = cluster_to_profile_distance[cluster_id][profile]
                                best_label = profile

                    # 如果找到了未分配的标签
                    if best_label:
                        final_business_labels[cluster_id] = best_label
                        assigned_profiles.add(best_label)
                    else:
                        # 否则使用最近的标签，即使它已被分配
                        best_label = min(cluster_to_profile_distance[cluster_id], 
                                        key=cluster_to_profile_distance[cluster_id].get)
                        final_business_labels[cluster_id] = best_label

        # 使用最终的业务标签映射
        business_labels = final_business_labels

    # 将业务标签映射到数据框
    df['business_type'] = df['cluster'].map(business_labels)

    # 输出各类别统计
    type_counts = df['business_type'].value_counts()
    print("\n业务标签分布:")
    for btype, count in type_counts.items():
        print(f"  {btype}: {count} 样本 ({count/len(df)*100:.1f}%)")

    # 详细统计
    business_stats = df.groupby('business_type').agg({
        'view_count': ['mean', 'median'],
        'like_rate': ['mean', 'median'],
        'comment_rate': ['mean', 'median'],
        'engagement_rate': ['mean', 'median']
    })

    print("\n业务类型统计:")
    print(business_stats.round(4))

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
    df_pca['cluster'] = df['cluster'].values

    # 可视化
    plt.figure(figsize=(14, 12))

    # 绘制散点图
    for label, color in COLORS.items():
        mask = df_pca['business_type'] == label
        # 调整点大小反映该类别的样本量
        sample_size = sum(mask)
        size_factor = 30 + (sample_size / len(df)) * 100

        plt.scatter(
            df_pca.loc[mask, 'PCA1'], 
            df_pca.loc[mask, 'PCA2'],
            c=color, 
            label=f"{label} ({sample_size}个样本)",
            alpha=0.7,
            edgecolors='w',
            s=size_factor
        )

    # 添加标题和轴标签
    plt.title(title, fontsize=20, color='#333333', fontweight='bold', pad=20)
    plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', fontsize=14)
    plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', fontsize=14)

    # 标记簇的质心
    for business_type in df['business_type'].unique():
        mask = df_pca['business_type'] == business_type
        centroid_x = df_pca.loc[mask, 'PCA1'].mean()
        centroid_y = df_pca.loc[mask, 'PCA2'].mean()
        plt.scatter(centroid_x, centroid_y, marker='*', color='black', s=300, alpha=0.8)
        plt.annotate(business_type, (centroid_x, centroid_y), fontsize=12, fontweight='bold',
                    xytext=(10, 10), textcoords='offset points')

    # 添加图例
    plt.legend(title="业务类型", title_fontsize=14, fontsize=12, 
               loc='upper right', bbox_to_anchor=(1.15, 1))

    # 添加网格线和背景美化
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

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

    # 绘制散点图
    for label, color in COLORS.items():
        mask = df_pca['business_type'] == label
        sample_size = sum(mask)
        size_factor = 20 + (sample_size / len(df)) * 80

        ax.scatter(
            df_pca.loc[mask, 'PCA1'], 
            df_pca.loc[mask, 'PCA2'], 
            df_pca.loc[mask, 'PCA3'],
            c=color, 
            label=f"{label} ({sample_size}个样本)",
            alpha=0.7,
            s=size_factor
        )

        # 添加质心
        centroid = df_pca.loc[mask, ['PCA1', 'PCA2', 'PCA3']].mean()
        ax.scatter(centroid[0], centroid[1], centroid[2], 
                  color='black', marker='*', s=200, alpha=0.8)
        ax.text(centroid[0], centroid[1], centroid[2], label, 
               color='black', fontsize=10, fontweight='bold')

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

def visualize_tsne(df, X_scaled, title="t-SNE聚类可视化"):
    """使用t-SNE进行降维可视化"""
    print("8. 生成t-SNE可视化...")

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X_scaled)

    # 创建包含t-SNE结果和聚类标签的DataFrame
    df_tsne = pd.DataFrame(data=tsne_result, columns=['t-SNE1', 't-SNE2'])
    df_tsne['business_type'] = df['business_type'].values

    # 可视化
    plt.figure(figsize=(14, 12))

    # 绘制散点图
    for label, color in COLORS.items():
        mask = df_tsne['business_type'] == label
        sample_size = sum(mask)
        size_factor = 30 + (sample_size / len(df)) * 100

        plt.scatter(
            df_tsne.loc[mask, 't-SNE1'], 
            df_tsne.loc[mask, 't-SNE2'],
            c=color, 
            label=f"{label} ({sample_size}个样本)",
            alpha=0.7,
            edgecolors='w',
            s=size_factor
        )

    # 添加标题和轴标签
    plt.title(title, fontsize=20, color='#333333', fontweight='bold', pad=20)
    plt.xlabel('t-SNE维度1', fontsize=14)
    plt.ylabel('t-SNE维度2', fontsize=14)

    # 标记簇的质心
    for business_type in df['business_type'].unique():
        mask = df_tsne['business_type'] == business_type
        centroid_x = df_tsne.loc[mask, 't-SNE1'].mean()
        centroid_y = df_tsne.loc[mask, 't-SNE2'].mean()
        plt.scatter(centroid_x, centroid_y, marker='*', color='black', s=300, alpha=0.8)
        plt.annotate(business_type, (centroid_x, centroid_y), fontsize=12, fontweight='bold',
                    xytext=(10, 10), textcoords='offset points')

    # 添加图例
    plt.legend(title="业务类型", title_fontsize=14, fontsize=12, 
               loc='upper right', bbox_to_anchor=(1.15, 1))

    # 添加网格线和背景美化
    plt.grid(True, linestyle='--', alpha=0.3)

    # 美化
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')

    return df_tsne

def visualize_feature_distributions(df):
    """可视化各业务类型的特征分布"""
    print("9. 生成特征分布可视化...")

    # 设置颜色调色板
    business_types = sorted(df['business_type'].unique())
    color_palette = [COLORS[btype] for btype in business_types]

    # 1. 箱线图组
    plt.figure(figsize=(18, 15))

    # 观看量箱线图 (对数尺度)
    plt.subplot(2, 2, 1)
    sns.boxplot(x='business_type', y='view_count', data=df, palette=color_palette, linewidth=1)
    plt.title('各业务类型的观看量分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('观看量', fontsize=14)
    plt.yscale('log')  # 使用对数尺度
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 点赞率箱线图
    plt.subplot(2, 2, 2)
    sns.boxplot(x='business_type', y='like_rate', data=df, palette=color_palette, linewidth=1)
    plt.title('各业务类型的点赞率分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('点赞率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 评论率箱线图
    plt.subplot(2, 2, 3)
    sns.boxplot(x='business_type', y='comment_rate', data=df, palette=color_palette, linewidth=1)
    plt.title('各业务类型的评论率分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('评论率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 互动率箱线图
    plt.subplot(2, 2, 4)
    sns.boxplot(x='business_type', y='engagement_rate', data=df, palette=color_palette, linewidth=1)
    plt.title('各业务类型的互动率分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('互动率 (点赞率+评论率)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')

    # 2. 小提琴图 - 更好地展示分布形状
    plt.figure(figsize=(20, 15))

    # 观看量小提琴图
    plt.subplot(2, 2, 1)
    sns.violinplot(x='business_type', y='log_view', data=df, palette=color_palette, inner='quartile')
    plt.title('各业务类型的观看量分布 (对数)', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('观看量 (对数)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 点赞率小提琴图
    plt.subplot(2, 2, 2)
    sns.violinplot(x='business_type', y='like_rate', data=df, palette=color_palette, inner='quartile')
    plt.title('各业务类型的点赞率分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('点赞率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 评论率小提琴图
    plt.subplot(2, 2, 3)
    sns.violinplot(x='business_type', y='comment_rate', data=df, palette=color_palette, inner='quartile')
    plt.title('各业务类型的评论率分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('评论率', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    # 点赞评论比小提琴图
    plt.subplot(2, 2, 4)
    df_for_plot = df.copy()
    df_for_plot['like_comment_ratio_capped'] = np.clip(df_for_plot['like_comment_ratio'], 0, 20)  # 上限截断极端值
    sns.violinplot(x='business_type', y='like_comment_ratio_capped', data=df_for_plot, palette=color_palette, inner='quartile')
    plt.title('各业务类型的点赞评论比分布', fontsize=16, color='#333333', fontweight='bold')
    plt.xlabel('业务类型', fontsize=14)
    plt.ylabel('点赞评论比 (上限20)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig('feature_distributions_violin.png', dpi=300, bbox_inches='tight')

def create_radar_chart(df):
    """创建雷达图比较各业务类型的特征"""
    print("10. 生成雷达图可视化...")

    # 计算每个业务类型的平均特征值
    business_features = df.groupby('business_type').agg({
        'view_count': 'mean',
        'like_rate': 'mean',
        'comment_rate': 'mean',
        'engagement_rate': 'mean',
        'duration_minutes': 'mean'
    })

    # 特征归一化（Min-Max缩放到0-1之间）
    for col in business_features.columns:
        business_features[col] = (business_features[col] - business_features[col].min()) / (business_features[col].max() - business_features[col].min())

    # 设置雷达图参数
    categories = ['观看量', '点赞率', '评论率', '互动率', '视频时长']
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形

    # 创建图形
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)

    # 为每个业务类型绘制雷达图
    for i, btype in enumerate(business_features.index):
        values = business_features.loc[btype].values.flatten().tolist()
        values += values[:1]  # 闭合图形

        ax.plot(angles, values, 'o-', linewidth=2, label=btype, color=COLORS[btype])
        ax.fill(angles, values, alpha=0.1, color=COLORS[btype])

    # 设置雷达图的刻度标签
    plt.xticks(angles[:-1], categories, fontsize=14)

    # 设置y轴刻度
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title('业务类型特征对比雷达图', fontsize=20, color='#333333', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')

def create_heatmap(df):
    """创建热图展示特征相关性和业务类型分布"""
    print("11. 生成热图可视化...")

    # 特征相关性热图
    plt.figure(figsize=(12, 10))

    # 选择用于分析的特征
    corr_features = df[['view_count', 'like_count', 'comment_count', 'duration_seconds', 
                        'like_rate', 'comment_rate', 'engagement_rate', 'like_comment_ratio']].corr()

    # 绘制热图
    mask = np.triu(np.ones_like(corr_features, dtype=bool))  # 上三角形掩码
    sns.heatmap(corr_features, annot=True, cmap='coolwarm', linewidths=0.5, 
                fmt='.2f', square=True, mask=mask, vmin=-1, vmax=1)
    plt.title('特征相关性热图', fontsize=20, color='#333333', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

    # 类别特征分布热图
    plt.figure(figsize=(14, 10))

    # 计算每个业务类型在各特征上的平均值（Z分数标准化）
    feature_cols = ['view_count', 'like_rate', 'comment_rate', 'engagement_rate', 
                   'duration_minutes', 'like_comment_ratio']

    # 准备热图数据
    heatmap_data = pd.DataFrame()
    for col in feature_cols:
        # 对每列进行Z分数标准化
        mean_val = df[col].mean()
        std_val = df[col].std()
        for btype in df['business_type'].unique():
            heatmap_data.loc[btype, col] = (df[df['business_type']==btype][col].mean() - mean_val) / std_val

    # 绘制热图
    sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', linewidths=0.5, fmt='.2f', center=0)
    plt.title('业务类型特征分布热图 (Z分数标准化)', fontsize=20, color='#333333', fontweight='bold', pad=20)
    plt.xlabel('特征', fontsize=14)
    plt.ylabel('业务类型', fontsize=14)

    plt.tight_layout()
    plt.savefig('business_type_heatmap.png', dpi=300, bbox_inches='tight')

def save_results(df):
    """保存分析结果"""
    print("12. 保存分析结果...")
    df.to_csv('video_clustering_results.csv', index=False)
    print(f"已保存聚类结果到: video_clustering_results.csv")

    # 保存业务类型统计摘要
    business_summary = df.groupby('business_type').agg({
        'view_count': ['mean', 'median', 'min', 'max'],
        'like_rate': ['mean', 'median', 'min', 'max'],
        'comment_rate': ['mean', 'median', 'min', 'max'],
        'engagement_rate': ['mean', 'median', 'min', 'max'],
        'duration_minutes': ['mean', 'median', 'min', 'max'],
        'cluster': 'count'
    })

    business_summary.to_csv('business_type_statistics.csv')
    print(f"已保存业务类型统计到: business_type_statistics.csv")

def main():
    """主函数"""
    print("="*50)
    print("视频数据聚类分析与可视化 - 增强版")
    print("="*50)

    try:
        # 1. 加载数据
        df = load_data(file_path)

        # 2. 准备特征
        X_scaled, features, X = prepare_features(df)

        # 3. 评估多种聚类算法
        algorithm, params, results_df = evaluate_clustering_algorithms(X_scaled)

        # 4. 执行最佳聚类算法
        df, model = perform_clustering(df, X_scaled, algorithm=algorithm, params=params)

        # 5. 分配业务标签
        df, business_stats = assign_business_labels(df, X)

        # 6. 二维PCA可视化
        pca_2d, df_pca_2d = visualize_2d_pca(df, X_scaled, "视频内容五类业务聚类 - 二维PCA可视化")

        # 7. 三维PCA可视化
        pca_3d, df_pca_3d = visualize_3d_pca(df, X_scaled, "视频内容五类业务聚类 - 三维PCA可视化")

        # 8. t-SNE可视化 (更好地保留局部结构)
        df_tsne = visualize_tsne(df, X_scaled, "视频内容五类业务聚类 - t-SNE可视化")

        # 9. 特征分布可视化
        visualize_feature_distributions(df)

        # 10. 雷达图可视化
        create_radar_chart(df)

        # 11. 热图可视化
        create_heatmap(df)

        # 12. 保存结果
        save_results(df)

        print("="*50)
        print("分析与可视化完成!")
        print("生成的图像文件:")
        print("  - clustering_evaluation.png: 聚类算法评估")
        print("  - pca_2d_visualization.png: 二维PCA可视化")
        print("  - pca_3d_visualization.png: 三维PCA可视化")
        print("  - tsne_visualization.png: t-SNE降维可视化")
        print("  - feature_distributions.png: 特征箱线图")
        print("  - feature_distributions_violin.png: 特征小提琴图")
        print("  - radar_chart.png: 业务类型雷达图")
        print("  - correlation_heatmap.png: 特征相关性热图")
        print("  - business_type_heatmap.png: 业务类型特征热图")
        print("="*50)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
