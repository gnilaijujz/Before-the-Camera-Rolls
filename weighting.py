import pandas as pd
import numpy as np

def assign_period_weights(period_map, decay_rate=0.5):
    """按指数衰减分配时间段权重（复用原逻辑）"""
    # 按时间段的起始时间排序（近期在前，权重更高）
    sorted_periods = sorted(period_map.items(), key=lambda x: x[1], reverse=True)
    # 指数衰减计算权重（第i个时段的权重 = decay_rate^i）
    weights = {period: (decay_rate) ** i for i, (period, _) in enumerate(sorted_periods)}
    # 归一化权重（确保总和为1）
    total_weight = sum(weights.values())
    return {k: v / total_weight for k, v in weights.items()}

def add_heat_column(df, time_col='published_at', heat_col='normalized_popularity',
                    decay_rate=0.5, min_samples=5, outlier_threshold=3.0, scale_factor=100000):
    #归一化热度都太小了，加权以后全是0.00，所以用scale_factor扩大一下倍数
    """
    基于已有的period_label计算加权热度，新增heat列
    
    参数:
    - df: 已包含period_label列的DataFrame（由splitter.py生成）
    - time_col: 时间列名（用于提取时间段起始时间）
    - heat_col: 原始归一化热度列名（需计算加权的列）
    - decay_rate: 指数衰减系数（近期时段权重更高）
    - min_samples: 过滤样本量不足的分段（标记为NaN）
    - outlier_threshold: 原始热度异常值过滤阈值（Z-score）
    
    返回:
    - 新增weight（分段权重）、heat（加权后热度）列的DataFrame
    """
    df_clean = df.copy()
    
    # 1. 基础检查（确保period_label已存在）
    if 'period_label' not in df_clean.columns:
        raise ValueError("数据中缺少period_label列，请先调用split_time_periods生成")
    required_cols = [time_col, heat_col, 'period_label']
    if not set(required_cols).issubset(df_clean.columns):
        missing = [col for col in required_cols if col not in df_clean.columns]
        raise ValueError(f"数据缺少必要列：{missing}")
    
    # 2. 数据清洗（异常值和缺失值处理）
    df_clean[time_col] = pd.to_datetime(df_clean[time_col])  # 确保时间格式正确
    df_clean = df_clean.dropna(subset=[time_col, heat_col, 'period_label']).copy()  # 过滤缺失值
    
    # 异常值处理（Z-score方法，避免极端值影响）
    mean_val = df_clean[heat_col].mean()
    std_val = df_clean[heat_col].std()
    if std_val > 0:
        z_scores = (df_clean[heat_col] - mean_val) / std_val
        df_clean = df_clean[z_scores.abs() <= outlier_threshold].copy()
    
    # 3. 计算每个分段的样本量（用于过滤小样本）
    period_sample_counts = df_clean['period_label'].value_counts().to_dict()
    df_clean['period_sample_count'] = df_clean['period_label'].map(period_sample_counts)
    
    # 4. 生成时间段映射表（period_label → 起始时间）
    # 提取每个分段的起始时间（取该分段内最早的时间）
    period_start_map = df_clean.groupby('period_label')[time_col].min().to_dict()
    
    # 5. 计算分段权重（基于指数衰减）
    period_weights = assign_period_weights(period_start_map, decay_rate=decay_rate)
    
    # 6. 为每条数据添加权重和最终的heat列（放大了）
    df_clean['weight'] = df_clean['period_label'].map(period_weights) # 映射分段权重
    
    # 计算加权热度：仅保留样本量足够的分段，否则标记为NaN
    df_clean['heat'] = np.where(
        df_clean['period_sample_count'] >= min_samples,
        (df_clean[heat_col] * scale_factor) * df_clean['weight'],  # 先放大再乘权重
        np.nan
    )
    
    return df_clean