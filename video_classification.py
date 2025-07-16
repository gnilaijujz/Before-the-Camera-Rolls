# 运行视频聚类分析 - 改进版

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

        # 处理极端值
        df['like_rate'] = np.clip(df['like_rate'], 0, 1)
        df['comment_rate'] = np.clip(df['comment_rate'], 0, 1)

        print(f"处理后数据形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        raise

def direct_classification(df):
    """
    直接基于阈值进行分类而非无监督聚类
    使用百分位数来定义"高"与"低"
    """
    print("2. 基于百分位阈值进行分类...")

    # 定义高低阈值（使用中位数以上为"高"）
    high_view_threshold = df['view_count'].quantile(0.5)  # 50%分位
    high_like_threshold = df['like_rate'].quantile(0.5)   # 50%分位
    high_comment_threshold = df['comment_rate'].quantile(0.5)  # 50%分位

    print(f"阈值设定:")
    print(f"  高观看量阈值 (50%分位): {high_view_threshold:.1f}")
    print(f"  高点赞率阈值 (50%分位): {high_like_threshold:.4f}")
    print(f"  高评论率阈值 (50%分位): {high_comment_threshold:.4f}")

    # 创建高低标志
    df['high_view'] = df['view_count'] >= high_view_threshold
    df['high_like'] = df['like_rate'] >= high_like_threshold
    df['high_comment'] = df['comment_rate'] >= high_comment_threshold

    # 直接分类
    conditions = [
        # 明星全能: 高曝光、高点赞率、高评论率
        (df['high_view'] & df['high_like'] & df['high_comment']),

        # 安静点赞: 高曝光、高点赞率、低评论率
        (df['high_view'] & df['high_like'] & ~df['high_comment']),

        # 讨论热议: 高曝光、低点赞率、高评论率
        (df['high_view'] & ~df['high_like'] & df['high_comment']),

        # 过眼云烟: 高曝光、低点赞率、低评论率
        (df['high_view'] & ~df['high_like'] & ~df['high_comment']),

        # 沉寂孤立: 低曝光（不管点赞率和评论率）
        (~df['high_view'])
    ]

    choices = ['明星全能', '安静点赞', '讨论热议', '过眼云烟', '沉寂孤立']

    # 应用分类
    df['business_type'] = np.select(conditions, choices, default='未分类')

    # 统计各类别数量
    type_counts = df['business_type'].value_counts()
    print("业务标签分布:")
    for btype, count in type_counts.items():
        print(f"  {btype}: {count} 样本 ({count/len(df)*100:.1f}%)")

    # 计算各类别统计信息
    type_stats = df.groupby('business_type').agg({
        'view_count': ['mean', 'median'],
        'like_rate': ['mean', 'median'],
        'comment_rate': ['mean', 'median'],
        'duration_minutes': ['mean', 'median']
    })

    print("各类别统计:")
    print(type_stats)

    return df, type_stats

def enhanced_classification(df):
    """
    使用KMeans聚类辅助，但最终仍按业务规则分类
    解决类别分布不均的问题
    """
    print("3. 执行增强分类...")

    # 特征准备和标准化
    df['log_view'] = np.log1p(df['view_count'])

    features = ['log_view', 'like_rate', 'comment_rate']
    X = df[features].copy()

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用KMeans帮助发现自然聚类
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 计算每个簇的中心
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )

    # 反向转换观看量
    cluster_centers['view_count'] = np.expm1(cluster_centers['log_view'])

    print("聚类中心:")
    print(cluster_centers[['view_count', 'like_rate', 'comment_rate']].round(4))

    # 计算每个簇在各维度上的排名
    ranks = cluster_centers[['view_count', 'like_rate', 'comment_rate']].rank(ascending=False)

    # 基于排名分配业务标签
    business_labels = {}

    for cluster_id in range(5):
        view_rank = ranks.loc[cluster_id, 'view_count']
        like_rank = ranks.loc[cluster_id, 'like_rate']
        comment_rank = ranks.loc[cluster_id, 'comment_rate']

        # 明星全能: 观看量、点赞率、评论率都高
        if view_rank <= 3 and like_rank <= 3 and comment_rank <= 3:
            if view_rank == 1 and (like_rank <= 2 or comment_rank <= 2):
                label = '明星全能'
            else:
                # 次高的观看量和好的互动
                label = '明星全能'

        # 安静点赞: 观看量高、点赞率高、评论率低
        elif view_rank <= 3 and like_rank <= 2 and comment_rank > 3:
            label = '安静点赞'

        # 讨论热议: 观看量高、点赞率低、评论率高
        elif view_rank <= 3 and like_rank > 3 and comment_rank <= 2:
            label = '讨论热议'

        # 过眼云烟: 观看量高、点赞率和评论率都低
        elif view_rank <= 3 and like_rank > 3 and comment_rank > 3:
            label = '过眼云烟'

        # 沉寂孤立: 观看量低
        else:
            label = '沉寂孤立'

        business_labels[cluster_id] = label

    # 确保有5类
    unique_labels = set(business_labels.values())
    if len(unique_labels) < 5:
        print("警告: 自动分类无法生成所有5个业务类别，使用百分位分类法")
        # 回退到百分位分类法
        df, _ = direct_classification(df)
        return df, None

    # 分配业务标签
    df['business_type'] = df['cluster'].map(business_labels)

    # 统计各类别数量
    type_counts = df['business_type'].value_counts()
    print("业务标签分布:")
    for btype, count in type_counts.items():
        print(f"  {btype}: {count} 样本 ({count/len(df)*100:.1f}%)")

    return df, business_labels

def optimal_classification(df):
    """结合两种分类方法进行最优分类"""
    print("2. 尝试多种分类方法...")

    # 先尝试增强分类
    df_enhanced, business_labels = enhanced_classification(df)

    # 如果增强分类失败，使用直接分类
    if business_labels is None:
        return df_enhanced, None

    # 否则使用增强分类结果
    # 计算各类别统计信息
    type_stats = df_enhanced.groupby('business_type').agg({
        'view_count': ['mean', 'median'],
        'like_rate': ['mean', 'median'],
        'comment_rate': ['mean', 'median'],
        'duration_minutes': ['mean', 'median']
    })

    print("各类别统计:")
    print(type_stats)

    return df_enhanced, type_stats

def save_results(df, type_stats):
    """保存分析结果"""
    print("3. 保存分析结果...")
    df.to_csv('video_classification_results.csv', index=False)
    if type_stats is not None:
        type_stats.to_csv('video_type_statistics.csv')
    print(f"已保存分类结果到: video_classification_results.csv")

def main():
    """主函数"""
    print("="*50)
    print("视频数据五类业务标签分析")
    print("="*50)

    try:
        # 1. 加载数据
        df = load_data(file_path)

        # 2. 最优分类
        df, type_stats = optimal_classification(df)

        # 3. 保存结果
        save_results(df, type_stats)

        print("="*50)
        print("分析完成!")
        print("="*50)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
