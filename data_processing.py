# data_processing/loader.py
import pandas as pd
from datetime import datetime

def load_and_clean_data(file_path):
    """加载并清洗原始数据"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"加载数据失败：{str(e)}")

    # 时间格式转换
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    # 处理缺失值（根据字段重要性填充/删除）
    df = df.dropna(subset=['video_id', 'published_at', 'popularity_normalized'])
    # 补充基础特征（如发布小时、星期几）
    df['publish_hour'] = df['published_at'].dt.hour
    df['publish_weekday'] = df['published_at'].dt.dayofweek
    return df

# data_processing/splitter.py（复用之前的时间段划分逻辑）
from datetime import timedelta

def split_time_periods(df, time_col='published_at', period_type='day', interval=7):
    """
    通用时间段划分函数
    :param df: 数据集（需包含时间列）
    :param time_col: 时间列名（需为datetime类型）
    :param period_type: 划分类型，可选 'day'（按天）、'week'（按周）、'custom'（自定义间隔天数）
    :param interval: 间隔天数（仅period_type='custom'时生效）
    :return: 新增period_label列的DataFrame、时间段映射字典
    """
    if time_col not in df.columns:
        raise ValueError(f"数据中不存在时间列：{time_col}（请检查列名是否正确）")
    
        
    # 转换为datetime类型（防御性处理）
    df[time_col] = pd.to_datetime(df[time_col])

    # 按类型划分
    if period_type == 'day':
        # 按自然日划分（每1天一个时段）
        df['period_start'] = df[time_col].dt.floor('D')
        df['period_label'] = df['period_start'].dt.strftime('%Y-%m-%d')
    
    elif period_type == 'week':
        # 按自然周划分（每周一为起始，每7天一个时段）
        df['period_start'] = df[time_col].dt.to_period('W').dt.start_time
        df['period_label'] = df['period_start'].dt.strftime('%Y-%m-%d（周%U）')
    
    elif period_type == 'custom':
        # 自定义间隔天数划分（如每3天、5天一个时段）
        min_time = df[time_col].min()
        # 生成时段边界
        period_bounds = []
        current = min_time
        while current < df[time_col].max():
            period_bounds.append(current)
            current += timedelta(days=interval)
        # 给每条数据分配时段
        df['period_label'] = pd.cut(
            df[time_col], 
            bins=period_bounds + [df[time_col].max() + timedelta(days=1)],
            labels=[f'{start.strftime("%Y-%m-%d")}~{end.strftime("%Y-%m-%d")}' 
                    for start, end in zip(period_bounds[:-1], period_bounds[1:])]
        )
    
    else:
       
        raise ValueError("period_type支持'day'/'week'/'custom'")
    # ========== 关键修改：强制过滤所有无效标签 ==========
    # 1. 记录原始数据量
    original_count = len(df)
    
    # # 2. 过滤 period_label 中的 NaN 和非字符串类型
    # valid_labels_mask = pd.notna(df['period_label']) & df['period_label'].apply(lambda x: isinstance(x, str))
    # df = df[valid_labels_mask].copy()
    
    # # 3. 记录过滤后的数据量和被过滤的标签
    # filtered_count = original_count - len(df)
    # if filtered_count > 0:
    #     print(f"警告：过滤了 {filtered_count} 条无效时间段标签（可能包含 NaN 或非字符串类型）")
    #     invalid_labels = df[~valid_labels_mask]['period_label'].unique()
    #     print(f"被过滤的无效标签示例：{invalid_labels[:5]}")
    
    # 4. 生成时间段映射（确保只包含有效字符串标签）
    period_map = df[['period_label', time_col]].drop_duplicates()\
                  .sort_values(time_col).set_index('period_label')[time_col].to_dict()
    
    return df, period_map