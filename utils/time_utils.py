# utils/time_utils.py
import pandas as pd

def extract_time_features(df, time_col="published_at"):
    """
    从时间列中提取特征：星期几、小时、是否周末、是否节假日（简化版）
    :param df: 包含时间列的DataFrame
    :param time_col: 时间列名（默认"published_at"）
    :return: 新增时间特征的DataFrame
    """
    # 复制数据避免修改原DataFrame
    df_copy = df.copy()
    
    # 确保时间列是datetime类型
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors="coerce")
    
    # 提取星期几（0=周一，6=周日）
    df_copy["publish_weekday"] = df_copy[time_col].dt.weekday
    
    # 提取小时（0-23）
    df_copy["publish_hour"] = df_copy[time_col].dt.hour
    
    # 是否周末（周六/周日：1=是，0=否）
    df_copy["is_weekend"] = df_copy["publish_weekday"].apply(lambda x: 1 if x in [5, 6] else 0)
    
    # 简化版：是否节假日（这里先默认全为0，实际使用时可补充节假日列表）
    df_copy["is_holiday"] = 0
    
    return df_copy