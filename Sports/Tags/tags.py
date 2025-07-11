import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 创建输出目录
output_dir = "sports_video_analysis"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
df = pd.read_csv('sports_2024-2025_with_popularity.csv')

# 1. 视频时长分类
def classify_duration(duration):
    if duration <= 240:  # 4分钟=240秒
        return 'short'
    elif 240 < duration <= 1200:  # 20分钟=1200秒
        return 'medium'
    else:
        return 'long'

df['duration_type'] = df['duration_seconds'].apply(classify_duration)

# 2. 时间衰减权重计算
df['published_at'] = pd.to_datetime(df['published_at'])
latest_date = df['published_at'].max()
df['days_since_publish'] = (latest_date - df['published_at']).dt.days

# 指数衰减函数 (半衰期=90天)
half_life = 90
decay_rate = np.log(2) / half_life
df['time_weight'] = np.exp(-decay_rate * df['days_since_publish'])

# 时间调整后的流行度
df['adj_popularity'] = df['popularity_normalized'] / df['time_weight']

# 3. 标签数字化处理
def clean_tags(tag_str):
    if pd.isna(tag_str):
        return ""
    # 移除特殊字符和引号
    return re.sub(r'[^a-zA-Z0-9,\s]', '', tag_str.lower())

df['cleaned_tags'] = df['tags'].apply(clean_tags)

# TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tag_matrix = vectorizer.fit_transform(df['cleaned_tags'])

# 降维处理
svd = TruncatedSVD(n_components=50)
tag_features = svd.fit_transform(tag_matrix)
tag_feature_names = [f'tag_svd_{i}' for i in range(50)]
tag_df = pd.DataFrame(tag_features, columns=tag_feature_names)
df = pd.concat([df.reset_index(drop=True), tag_df], axis=1)

# 保存处理后的数据
processed_csv_path = os.path.join(output_dir, "processed_sports_videos.csv")
df.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ 已保存处理后的数据到: {processed_csv_path}")

# 4. 分组建模与评估
results = {}
models = {}
feature_importances = {}

# 可视化设置
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

for dtype in ['short', 'medium', 'long']:
    print(f"\n===== Modeling {dtype} videos =====")
    subset = df[df['duration_type'] == dtype]
    
    # 特征选择
    features = tag_feature_names + ['time_weight']
    X = subset[features]
    y = subset['adj_popularity']
    
    # 检查样本量
    if len(subset) < 10:
        print(f"⚠️ 警告: {dtype}类型视频只有{len(subset)}个样本，跳过建模")
        continue
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 建模
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 预测与评估
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 交叉验证
    if len(X_train) >= 5:  # 确保有足够样本进行交叉验证
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=min(5, len(X_train)), scoring='r2')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
    else:
        cv_mean = np.nan
        cv_std = np.nan
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"交叉验证R²: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # 保存结果
    results[dtype] = {
        'rmse': rmse,
        'r2': r2,
        'cv_r2': cv_mean,
        'cv_std': cv_std,
        'num_samples': len(subset)
    }
    models[dtype] = model

    # 特征重要性 - 修复错误
    # 创建特征名称列表 (50个标签特征 + 时间权重)
    feature_names = tag_feature_names + ['time_weight']
    
    # 确保长度匹配
    if len(model.feature_importances_) == len(feature_names):
        feat_importances = pd.Series(model.feature_importances_, index=feature_names)
        
        # 可视化标签特征重要性 (排除时间权重)
        tag_importances = feat_importances[tag_feature_names]
        top_tag_features = tag_importances.nlargest(10)
        
        # 创建并保存特征重要性图
        plt.figure(figsize=(10, 6))
        ax = top_tag_features.sort_values().plot.barh(color='steelblue')
        plt.title(f"{dtype.capitalize()}视频 - 前10重要标签特征", fontsize=14)
        plt.xlabel("特征重要性", fontsize=12)
        plt.ylabel("标签特征", fontsize=12)
        plt.tight_layout()
        
        # 添加数值标签
        for i, v in enumerate(top_tag_features.sort_values()):
            ax.text(v + 0.001, i, f"{v:.4f}", color='black', va='center')
        
        # 保存图像
        importance_img_path = os.path.join(output_dir, f"{dtype}_tag_importance.png")
        plt.savefig(importance_img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存特征重要性图像到: {importance_img_path}")
        
        # 保存特征重要性数据
        feature_importances[dtype] = top_tag_features
        importance_csv_path = os.path.join(output_dir, f"{dtype}_tag_importance.csv")
        top_tag_features.to_csv(importance_csv_path, header=['importance'], encoding='utf-8-sig')
        print(f"✅ 已保存特征重要性数据到: {importance_csv_path}")
        
        # 创建预测与实际值对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.title(f"{dtype.capitalize()}视频 - 预测值 vs 实际值", fontsize=14)
        plt.xlabel("实际流行度", fontsize=12)
        plt.ylabel("预测流行度", fontsize=12)
        
        # 添加R²文本
        plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes, fontsize=12)
        
        # 保存图像
        scatter_img_path = os.path.join(output_dir, f"{dtype}_prediction_scatter.png")
        plt.savefig(scatter_img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存预测对比图像到: {scatter_img_path}")
    else:
        print(f"⚠️ 特征重要性不匹配: 预期{len(feature_names)}个特征, 实际{len(model.feature_importances_)}")

# 5. 标签推荐系统
class TagRecommender:
    def __init__(self, vectorizer, svd, df):
        self.vectorizer = vectorizer
        self.svd = svd
        self.df = df
        
        # 创建标签-流行度映射
        self.tag_popularity = {}
        all_tags = []
        for tags in df['cleaned_tags']:
            if pd.notna(tags):
                all_tags.extend(tags.split(','))
        
        unique_tags = set(filter(None, all_tags))
        
        for tag in unique_tags:
            mask = df['cleaned_tags'].str.contains(tag, regex=False, na=False)
            if mask.sum() > 0:
                self.tag_popularity[tag.strip()] = df.loc[mask, 'adj_popularity'].mean()
    
    def recommend_tags(self, user_input, duration_type, top_n=10):
        # 文本预处理
        cleaned_input = clean_tags(user_input)
        
        # 向量化用户输入
        input_vec = self.vectorizer.transform([cleaned_input])
        input_svd = self.svd.transform(input_vec)
        
        # 计算与所有标签的相似度
        similarities = []
        for tag, popularity in self.tag_popularity.items():
            tag_vec = self.vectorizer.transform([tag])
            tag_svd = self.svd.transform(tag_vec)
            try:
                sim = 1 - cosine(input_svd.flatten(), tag_svd.flatten())
                similarities.append((tag, sim, popularity))
            except:
                continue
        
        # 创建结果DataFrame
        rec_df = pd.DataFrame(similarities, 
                             columns=['tag', 'similarity', 'avg_popularity'])
        
        # 过滤该视频类型的标签
        duration_tags = set()
        for tags in self.df[self.df['duration_type'] == duration_type]['cleaned_tags']:
            if pd.notna(tags):
                duration_tags.update(tags.split(','))
        
        rec_df = rec_df[rec_df['tag'].isin(duration_tags)]
        
        # 综合评分 (相似度60% + 流行度40%)
        if not rec_df.empty:
            rec_df['score'] = (0.6 * rec_df['similarity'] + 
                              0.4 * rec_df['avg_popularity'] / rec_df['avg_popularity'].max())
            return rec_df.nlargest(top_n, 'score')
        else:
            return pd.DataFrame(columns=['tag', 'score'])

# 初始化推荐器
recommender = TagRecommender(vectorizer, svd, df)

# 示例使用
user_query = "I want to create an exciting basketball highlight video"
duration_type = "short"  # 短视频类型

recommendations = recommender.recommend_tags(user_query, duration_type)

# 保存推荐结果
if not recommendations.empty:
    rec_csv_path = os.path.join(output_dir, "tag_recommendations.csv")
    recommendations.to_csv(rec_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存标签推荐结果到: {rec_csv_path}")
    
    # 可视化推荐结果
    plt.figure(figsize=(10, 6))
    ax = recommendations.sort_values('score').plot.barh(x='tag', y='score', color='forestgreen')
    plt.title("热门标签推荐", fontsize=14)
    plt.xlabel("推荐得分", fontsize=12)
    plt.ylabel("标签", fontsize=12)
    
    # 添加数值标签
    for i, v in enumerate(recommendations.sort_values('score')['score']):
        ax.text(v + 0.01, i, f"{v:.4f}", color='black', va='center')
    
    # 保存图像
    rec_img_path = os.path.join(output_dir, "tag_recommendations.png")
    plt.savefig(rec_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存标签推荐图像到: {rec_img_path}")
else:
    print("\n未找到匹配的标签推荐")

# 创建并保存模型性能报告
report_path = os.path.join(output_dir, "model_performance_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("体育视频流行度预测模型性能评估报告\n")
    f.write("="*60 + "\n\n")
    f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"总视频数量: {len(df)}\n")
    f.write(f"短视频数量: {len(df[df['duration_type']=='short'])}\n")
    f.write(f"中视频数量: {len(df[df['duration_type']=='medium'])}\n")
    f.write(f"长视频数量: {len(df[df['duration_type']=='long'])}\n\n")
    
    f.write("模型性能评估:\n")
    f.write("-"*50 + "\n")
    for dtype, metrics in results.items():
        f.write(f"{dtype.capitalize()}视频:\n")
        f.write(f"  样本量: {metrics['num_samples']}\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  R²: {metrics['r2']:.4f}\n")
        f.write(f"  交叉验证R²: {metrics['cv_r2']:.4f} ± {metrics['cv_std']:.4f}\n\n")
    
    f.write("\n关键洞察:\n")
    f.write("-"*50 + "\n")
    f.write("1. 视频时长分类策略:\n")
    f.write("   - 短视频(<4分钟): 适合快节奏内容(精彩瞬间/花絮)\n")
    f.write("   - 中视频(4-20分钟): 教程/赛事集锦\n")
    f.write("   - 长视频(>20分钟): 深度分析/纪录片\n")
    f.write("2. 时间衰减模型:\n")
    f.write("   - 采用指数衰减函数 w(t) = e^(-λt), λ=ln(2)/90\n")
    f.write("   - 有效消除发布时间对流行度的影响\n")
    f.write("3. 标签推荐系统:\n")
    f.write("   - 综合评分 = 0.6 × 相似度 + 0.4 × 标准化流行度\n")
    f.write("   - 平衡语义相关性和历史表现\n\n")
    
    if not recommendations.empty:
        f.write("\n标签推荐示例:\n")
        f.write("-"*50 + "\n")
        f.write(f"查询内容: '{user_query}'\n")
        f.write(f"视频类型: {duration_type}\n")
        f.write("推荐结果:\n")
        for i, row in recommendations.iterrows():
            f.write(f"  {i+1}. {row['tag']} (得分: {row['score']:.4f})\n")

print(f"✅ 已保存模型性能报告到: {report_path}")

# 创建并保存数据分析报告
analysis_path = os.path.join(output_dir, "data_analysis_report.txt")
with open(analysis_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("体育视频数据分析报告\n")
    f.write("="*60 + "\n\n")
    
    # 视频时长分布
    duration_counts = df['duration_type'].value_counts()
    f.write("视频时长分布:\n")
    f.write("-"*50 + "\n")
    for dtype, count in duration_counts.items():
        f.write(f"  {dtype.capitalize()}视频: {count} ({count/len(df)*100:.1f}%)\n")
    
    # 流行度分析
    f.write("\n视频流行度分析:\n")
    f.write("-"*50 + "\n")
    f.write(f"  原始平均流行度: {df['popularity_normalized'].mean():.4f}\n")
    f.write(f"  时间调整后平均流行度: {df['adj_popularity'].mean():.4f}\n")
    
    # 标签分析
    all_tags = []
    for tags in df['cleaned_tags']:
        if pd.notna(tags):
            all_tags.extend(tags.split(','))
    
    tag_counts = pd.Series(all_tags).value_counts().head(20)
    f.write("\n最常见标签(前20):\n")
    f.write("-"*50 + "\n")
    for tag, count in tag_counts.items():
        f.write(f"  {tag}: {count}次\n")

print(f"✅ 已保存数据分析报告到: {analysis_path}")

# 创建整体可视化
plt.figure(figsize=(12, 8))

# 视频时长分布
plt.subplot(2, 2, 1)
duration_counts = df['duration_type'].value_counts()
plt.pie(duration_counts, labels=duration_counts.index, autopct='%1.1f%%', 
        colors=['#66c2a5', '#fc8d62', '#8da0cb'])
plt.title("视频时长分布")

# 流行度分布
plt.subplot(2, 2, 2)
sns.histplot(df['adj_popularity'], bins=30, kde=True, color='#66c2a5')
plt.title("时间调整后流行度分布")
plt.xlabel("流行度")

# 时长与流行度关系
plt.subplot(2, 2, 3)
sns.boxplot(x='duration_type', y='adj_popularity', data=df, 
           palette=['#66c2a5', '#fc8d62', '#8da0cb'])
plt.title("不同时长视频的流行度")
plt.xlabel("视频类型")
plt.ylabel("流行度")

# 发布时间与流行度关系
plt.subplot(2, 2, 4)
sns.scatterplot(x='days_since_publish', y='adj_popularity', data=df, 
               alpha=0.6, color='#66c2a5')
plt.title("发布时间与流行度关系")
plt.xlabel("发布天数")
plt.ylabel("流行度")
plt.gca().invert_xaxis()

plt.tight_layout()
overview_img_path = os.path.join(output_dir, "data_overview.png")
plt.savefig(overview_img_path, dpi=300)
plt.close()
print(f"✅ 已保存数据概览图像到: {overview_img_path}")

print("\n" + "="*50)
print("分析完成! 所有结果已保存至目录:", output_dir)
print("="*50)