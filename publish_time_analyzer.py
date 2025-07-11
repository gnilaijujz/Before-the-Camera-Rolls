import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# 导入外部的weighting模块（使用新的生成heat列的版本）
from weighting import add_heat_column  # 注意：这里导入的是新的add_heat_column函数

def extract_time_features(df, time_col="published_at"):
    """提取时间特征（包含星期和小时）"""
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors="coerce")
    df_copy["hour"] = df_copy[time_col].dt.hour  # 小时（0-23）
    df_copy["weekday"] = df_copy[time_col].dt.weekday  # 星期几（0=周一，6=周日）
    df_copy["weekday_name"] = df_copy["weekday"].map({
        0: "周一", 1: "周二", 2: "周三", 3: "周四", 
        4: "周五", 5: "周六", 6: "周日"
    })
    df_copy["is_weekend"] = df_copy["weekday"].apply(lambda x: 1 if x in [5, 6] else 0)
    df_copy["is_holiday"] = 0  # 简化处理节假日
    return df_copy

def analyze_publish_time(df):
    """分析最佳发布时间（完全使用heat列作为指标）"""
    # 1. 基础数据检查（确保包含必要列：已生成period_label和热度）
    required_cols = ["published_at", "heat", "period_label"]
    if not set(required_cols).issubset(df.columns):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"数据缺少必要列（需先通过splitter生成period_label）：{missing}")
    # 使用传入的df直接分析（已包含heat列）
    df_weighted = df.copy()
    
    # 3. 提取时间特征（与heat列结合）
    df_weighted = extract_time_features(df_weighted, time_col="published_at")
    
    # 4. 过滤无效数据（确保heat列和时间特征有效）
    valid_df = df_weighted.dropna(
        subset=["hour", "weekday", "weekday_name", "heat"]
    ).reset_index(drop=True)
    
    if len(valid_df) == 0:
        return {"error": "无有效数据用于分析（heat列或时间特征缺失）"}
    
    print(f"过滤后有效样本量：{len(valid_df)}")

    # 5. 核心分析：按星期+小时分组，计算平均heat值
    daily_hour_heat = valid_df.groupby(["weekday_name", "hour"])["heat"].agg(
        mean_heat="mean",  # 平均heat值（替代weighted_heat）
        count="count"      # 样本量
    ).reset_index()
    
    # 过滤小样本分组
    daily_hour_heat = daily_hour_heat[daily_hour_heat["count"] >= 5]

    # 6. 生成每天的最佳发布时间（基于heat列排序）
    weekday_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    daily_best_hours = {}
    
    for weekday in weekday_order:
        day_data = daily_hour_heat[daily_hour_heat["weekday_name"] == weekday]
        if len(day_data) == 0:
            daily_best_hours[weekday] = {"推荐时间(小时)": [], "提示": "数据不足"}
            continue
        
        # 按heat值降序取前3小时
        best_hours = day_data.sort_values("mean_heat", ascending=False).head(3)
        daily_best_hours[weekday] = {
            "推荐时间(小时)": best_hours["hour"].tolist(),
            "平均heat值": [round(h, 3) for h in best_hours["mean_heat"].tolist()],  # 显示heat值
            "样本量": best_hours["count"].tolist()
        }

    # 7. 补充分析：聚类（基于heat列）
    features = ["hour", "weekday", "is_weekend", "is_holiday"]
    X = valid_df[features]
    y_heat = valid_df["heat"]  # 目标变量为heat

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    valid_df["time_cluster"] = kmeans.fit_predict(X[["hour", "weekday"]])
    cluster_heat = valid_df.groupby("time_cluster")["heat"].mean().reset_index()
    cluster_heat = cluster_heat.sort_values("heat", ascending=False)
    cluster_desc = {
        i: f"聚类 {i}: 平均小时={valid_df[valid_df['time_cluster']==i]['hour'].mean():.1f}, "
           f"平均heat值={cluster_heat[cluster_heat['time_cluster']==i]['heat'].values[0]:.3f}"
        for i in range(5)
    }

    # 8. 补充分析：回归（基于heat列）
    X_train, X_test, y_train, y_test = train_test_split(X, y_heat, test_size=0.3, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    reg_mse = mean_squared_error(y_test, rf_reg.predict(X_test))
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": rf_reg.feature_importances_
    }).sort_values("importance", ascending=False).to_dict(orient="records")

    # 9. 整理结果（突出heat列的使用）
    return {
        "核心推荐：周一至周日最佳发布时间（基于heat值）": daily_best_hours,
        "heat值计算参数": {
            "时间衰减率": 0.5,
            "说明": "基于period_label的指数衰减加权，近期数据权重更高"
        },
        "补充分析：聚类结果（基于heat值）": {
            "按heat值排序的聚类": cluster_heat.to_dict(orient="records"),
            "聚类描述": cluster_desc
        },
        "补充分析：特征重要性（影响heat值的因素）": feature_importance,
        "补充分析：回归均方误差（heat值预测）": round(reg_mse, 3),
        "有效样本总量": len(valid_df)
    }