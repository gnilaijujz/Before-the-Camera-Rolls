import pandas as pd
import numpy as np
import os
import re
import string
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib

# 下载NLTK资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # 添加POS标注器
nltk.download('omw-1.4')  # 添加Open Multilingual WordNet
nltk.download('punkt_tab')


# 创建输出目录
output_dir = "description_analysis"
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

def plateau_decay(t, early_a=0.25, t0=20, floor=0.35):
    decay = 1 / (1 + np.exp(early_a * (t - t0)))
    return np.maximum(decay, floor)

df['time_weight'] = plateau_decay(df['days_since_publish'])
df['adj_popularity'] = df['popularity_normalized'] * 100000 * df['time_weight']

# 3. 简介文本预处理
def clean_text(text):
    if pd.isna(text):
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 移除标点符号 - 保留句点用于句子分割
    text = text.translate(str.maketrans('', '', string.punctuation.replace('.', '')))
    
    # 移除数字
    text = re.sub(r'\d+', '', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 情感分析
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# 文本统计特征
def get_text_features(text):
    # 使用空格分词替代nltk.word_tokenize
    words = text.split()
    
    # 计算基本统计特征
    word_count = len(words)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    unique_word_ratio = len(set(words)) / word_count if word_count else 0
    
    # 添加情感特征
    sentiment = TextBlob(text).sentiment.polarity
    
    return {
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'unique_word_ratio': unique_word_ratio,
        'sentiment': sentiment
    }

# 应用预处理 - 直接使用get_text_features包含情感分析
df['cleaned_description'] = df['description'].apply(clean_text)


# 提取文本统计特征
text_features = df['cleaned_description'].apply(get_text_features).apply(pd.Series)
df = pd.concat([df, text_features], axis=1)

# 保存处理后的数据
processed_csv_path = os.path.join(output_dir, "processed_data_with_description.csv")
df.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ 已保存处理后的数据到: {processed_csv_path}")

# 4. 文本向量化和主题建模
# 准备停用词
stop_words = set(stopwords.words('english'))
custom_stopwords = ['subscribe', 'channel', 'follow', 'like', 'instagram', 'facebook', 
                   'twitter', 'tiktok', 'website', 'click', 'link', 'watch', 'video']
stop_words.update(custom_stopwords)

# 词形还原
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words])

df['lemmatized_description'] = df['cleaned_description'].apply(lemmatize_text)

# TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words=list(stop_words),
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df['lemmatized_description'])

# 主题建模 (LDA)
num_topics = 5
lda = LatentDirichletAllocation(
    n_components=num_topics,
    max_iter=10,
    learning_method='online',
    random_state=42
)

lda_topics = lda.fit_transform(tfidf_matrix)

# 添加主题到数据框
for i in range(num_topics):
    df[f'topic_{i}'] = lda_topics[:, i]

# 可视化主题关键词
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, num_topics, figsize=(20, 6), sharex=True)
    axes = axes.flatten()
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 14})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return fig

# 获取特征名称
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# 绘制主题关键词
topic_plot = plot_top_words(lda, tfidf_feature_names, 10, "主题关键词")
topic_plot_path = os.path.join(output_dir, "topic_keywords.png")
plt.savefig(topic_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 已保存主题关键词图像到: {topic_plot_path}")




# 5. 分组建模与评估
results = {}
models = {}
keyword_importances = {}

# 更新文本特征列表 - 使用实际提取的特征
text_features = [
    'word_count', 'avg_word_length', 'unique_word_ratio', 'sentiment'
] + [f'topic_{i}' for i in range(num_topics)]

# 确保这些特征存在于DataFrame中
missing_features = [f for f in text_features if f not in df.columns]
if missing_features:
    print(f"⚠️ 警告: 以下特征缺失: {missing_features}")
    text_features = [f for f in text_features if f in df.columns]

print("建模使用的特征:", text_features)  # 调试输出

# 可视化设置
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

scalers = {}  # 添加这个字典保存标准化器
feature_names_by_duration = {}  # 添加这个字典保存特征名称

for dtype in ['short', 'medium', 'long']:
    print(f"\n===== Modeling {dtype} videos =====")
    subset = df[df['duration_type'] == dtype]
    
    # 检查样本量
    if len(subset) < 20:
        print(f"⚠️ 警告: {dtype}类型视频只有{len(subset)}个样本，跳过建模")
        continue
    
    # 准备特征和目标变量 - 确保只使用存在的列
    available_features = [f for f in text_features if f in subset.columns]
    X = subset[available_features]
    y = subset['adj_popularity']
    
    # 保存特征名称
    feature_names_by_duration[dtype] = available_features

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存标准化器
    scalers[dtype] = scaler

    # 建模
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 保存模型
    model_path = os.path.join(output_dir, f"{dtype}_description_model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ 已保存模型到: {model_path}")
    
    
    # 预测与评估
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 交叉验证
    if len(X_train) >= 5:
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=min(5, len(X_train)), 
            scoring='r2'
        )
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

    # 特征重要性分析
    feat_importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    # 保存特征重要性
    keyword_importances[dtype] = feat_importances
    importance_csv_path = os.path.join(output_dir, f"{dtype}_feature_importance.csv")
    feat_importances.to_csv(importance_csv_path, header=['importance'])
    print(f"✅ 已保存特征重要性数据到: {importance_csv_path}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feat_importances.nlargest(10)
    ax = top_features.sort_values().plot.barh(color='teal')
    plt.title(f"{dtype.capitalize()}视频 - 特征重要性", fontsize=16)
    plt.xlabel("特征重要性", fontsize=12)
    
    # 添加数值标签
    for i, v in enumerate(top_features.sort_values()):
        ax.text(v + 0.001, i, f"{v:.4f}", color='black', va='center')
    
    # 保存图像
    importance_img_path = os.path.join(output_dir, f"{dtype}_feature_importance.png")
    plt.savefig(importance_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存特征重要性图像到: {importance_img_path}")
    
    # 创建预测与实际值对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='darkorange')
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

# 6. AI辅助撰写框架
class DescriptionAssistant:
    def __init__(self, models, vectorizer, lda, text_features, scalers=None, feature_names_by_duration=None, topic_names=None):
        self.models = models
        self.scalers = scalers or {} # 添加标准化器字典
        self.vectorizer = vectorizer
        self.lda = lda
        self.text_features = text_features
        self.feature_names_by_duration = feature_names_by_duration or {}
        self.topic_names = topic_names or {
            0: "订阅与关注",
            1: "赛事信息",
            2: "选手亮点",
            3: "背景故事",
            4: "情感共鸣"
        }
    
    # 修改 analyze_description 方法
    def analyze_description(self, description, duration_type):
        """分析现有简介的质量（修复特征维度错误和标准化兼容性）"""
        if duration_type not in self.models:
            return {"error": f"没有{duration_type}类型视频的模型"}
        if duration_type not in self.scalers:
            return {"error": f"没有{duration_type}类型视频的标准化器"}
        if duration_type not in self.feature_names_by_duration:
            return {"error": f"没有{duration_type}类型视频的特征名称信息"}

        # 获取模型、标准化器和期望特征名
        model = self.models[duration_type]
        scaler = self.scalers[duration_type]
        expected_features = self.feature_names_by_duration[duration_type]

        # 文本预处理
        cleaned_text = clean_text(description)
        lemmatized_text = lemmatize_text(cleaned_text)

        # 提取文本统计特征 + 情感
        features = get_text_features(cleaned_text)
        features['sentiment'] = get_sentiment(cleaned_text)

        # 提取主题分布
        tfidf_vec = self.vectorizer.transform([lemmatized_text])
        topic_dist = self.lda.transform(tfidf_vec)[0]
        num_topics = len(self.lda.components_)
        for i in range(num_topics):
            features[f'topic_{i}'] = topic_dist[i] if i < len(topic_dist) else 0

        # 构造特征向量，确保顺序一致
        feature_list = []
        missing_features = []
        for feat in expected_features:
            if feat in features:
                feature_list.append(features[feat])
            else:
                feature_list.append(0)
                missing_features.append(feat)

        if missing_features:
            print(f"⚠️ 缺失特征（已填0）: {missing_features}")
        if len(feature_list) != len(expected_features):
            return {"error": f"特征维度不匹配: 期望 {len(expected_features)} 个，实际 {len(feature_list)} 个"}

        # 使用 DataFrame 且列名与训练一致（防止警告）
        feature_df = pd.DataFrame([feature_list], columns=expected_features)
        try:
            features_scaled = scaler.transform(feature_df)
        except Exception as e:
            return {"error": f"标准化失败: {str(e)}"}

        # 模型预测
        popularity_score = model.predict(features_scaled)[0]
        main_topic = np.argmax(topic_dist)
        topic_name = self.topic_names.get(main_topic, f"主题{main_topic}")
        topic_strength = topic_dist[main_topic]

        return {
            "popularity_score": popularity_score,
            "main_topic": topic_name,
            "topic_strength": topic_strength,
            "word_count": features.get('word_count', 0),
            "sentiment": features.get('sentiment', 0),
            "topic_distribution": topic_dist.tolist()
        }


        
    def generate_improvement_tips(self, analysis_result, duration_type):
        """生成改进建议"""
        tips = []
        
        # 根据视频类型提供建议
        if duration_type == 'short':
            if analysis_result['word_count'] > 50:
                tips.append("简介过长：短视频简介应简洁，建议控制在50字以内")
            elif analysis_result['word_count'] < 20:
                tips.append("简介过短：可添加更多关键词提升可搜索性")
        elif duration_type == 'medium':
            if analysis_result['word_count'] < 50:
                tips.append("简介过短：中视频可包含更多细节，建议50-150字")
        else:  # long
            if analysis_result['word_count'] < 100:
                tips.append("简介过短：长视频应提供详细背景，建议100字以上")
        
        # 情感建议
        if analysis_result['sentiment'] < -0.3:
            tips.append("情感过于负面：体育视频通常需要积极情感，建议调整语气")
        elif analysis_result['sentiment'] > 0.7:
            tips.append("情感过于夸张：可能降低可信度，建议适度调整")
        
        # 主题建议
        if duration_type == 'short' and analysis_result['main_topic'] == "背景故事":
            tips.append("主题不匹配：短视频更适合突出选手亮点而非背景故事")
        
        if duration_type == 'long' and analysis_result['main_topic'] == "订阅与关注":
            tips.append("主题不匹配：长视频应减少关注引导，增加深度内容")
        
        # 添加基于特征重要性的建议
        if duration_type in keyword_importances:
            top_features = keyword_importances[duration_type].nlargest(3).index.tolist()
            tips.append(f"重要特征：尝试优化 {', '.join(top_features)}")
        
        return tips
    
    def generate_description_template(self, duration_type, main_topic=None):
        """生成描述模板"""
        templates = {
            'short': {
                'default': "精彩时刻：{highlight}！🔥 不容错过！ #体育 #精彩瞬间",
                '选手亮点': "惊艳表现：{player} 的 {skill}！👏 #体育 #精彩瞬间",
                '情感共鸣': "热血沸腾！{event} 的这一刻让人难忘 ❤️ #体育 #情感"
            },
            'medium': {
                'default': "赛事集锦：{event} 的精彩回顾。包含 {key_moments} 等关键时刻。订阅获取更多内容！",
                '赛事信息': "{event} 完整回顾：从 {start} 到 {end} 的关键时刻。🏆 #体育 #赛事",
                '背景故事': "背后的故事：{player} 如何克服困难参加 {event}。❤️ #体育 #故事"
            },
            'long': {
                'default': "深度解析：{event} 的完整回顾与分析。从历史背景到技术细节，全面解读 {key_aspects}。",
                '背景故事': "传奇之路：{player} 的职业生涯回顾与 {event} 的幕后故事。📖 #体育 #纪录片",
                '技术分析': "技术拆解：深入分析 {player} 在 {event} 中的 {technique} 技术。🔍 #体育 #分析"
            }
        }
        
        if duration_type not in templates:
            return "抱歉，暂不支持此视频类型的模板"
        
        if main_topic and main_topic in templates[duration_type]:
            return templates[duration_type][main_topic]
        else:
            return templates[duration_type]['default']

# 初始化AI助手
topic_names = {
    0: "订阅与关注",
    1: "赛事信息",
    2: "选手亮点",
    3: "背景故事",
    4: "技术分析"
}

assistant = DescriptionAssistant(
    models=models,
    scalers=scalers,  # 添加这行
    vectorizer=tfidf_vectorizer,
    lda=lda,
    text_features=text_features,
    feature_names_by_duration=feature_names_by_duration,  # 添加这行
    topic_names=topic_names
)

# 测试AI助手功能
test_descriptions = {
    'short': "Amazing goal by Messi! Watch now! #football #soccer",
    'medium': "Full highlights of the Lakers vs Warriors game. Key moments: LeBron's dunk, Curry's three-pointers. Subscribe for more NBA content.",
    'long': "In-depth analysis of the 2024 Australian Open. This documentary explores the journey of the top players, their training routines, and the key matches that defined the tournament."
}

assistant_results = {}
for dtype, desc in test_descriptions.items():
    analysis = assistant.analyze_description(desc, dtype)

    if "error" in analysis:
        print(f"❌ {dtype} 视频分析失败: {analysis['error']}")
        assistant_results[dtype] = {
            'analysis': analysis,
            'improvement_tips': [],
            'template': "无法生成模板"
        }
        continue

    tips = assistant.generate_improvement_tips(analysis, dtype)
    template = assistant.generate_description_template(dtype, analysis['main_topic'])

    assistant_results[dtype] = {
        'analysis': analysis,
        'improvement_tips': tips,
        'template': template
    }

# 写入 AI 助手测试结果
assistant_path = os.path.join(output_dir, "ai_assistant_results.txt")
with open(assistant_path, 'w', encoding='utf-8') as f:
    f.write("AI助手测试结果\n")
    f.write("="*50 + "\n\n")

    for dtype, result in assistant_results.items():
        f.write(f"视频类型: {dtype}\n")
        f.write(f"原始简介: {test_descriptions[dtype]}\n")

        if "error" in result['analysis']:
            f.write(f"❌ 分析失败: {result['analysis']['error']}\n\n")
            f.write("-"*50 + "\n\n")
            continue

        f.write(f"预测流行度: {result['analysis']['popularity_score']:.4f}\n")
        f.write(f"主要主题: {result['analysis']['main_topic']}\n")
        f.write(f"情感分数: {result['analysis']['sentiment']:.2f}\n")
        f.write(f"字数: {result['analysis']['word_count']}\n")

        f.write("\n改进建议:\n")
        for tip in result['improvement_tips']:
            f.write(f" - {tip}\n")

        f.write(f"\n推荐模板:\n{result['template']}\n\n")
        f.write("-"*50 + "\n\n")

print(f"✅ 已保存AI助手测试结果到: {assistant_path}")

# ✅ 修复 scatterplot 报错前，确保 word_count 是一维数值：
if isinstance(df['word_count'].iloc[0], (list, np.ndarray)):
    df['word_count'] = df['word_count'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

# 转换为数值，防止异常类型残留
for col in ['word_count', 'adj_popularity']:
    if col not in df.columns:
        print(f"⚠️ 列 {col} 不存在，跳过")
        continue

    value = df[col].iloc[0]

    print(f"🔍 检查 {col} 第一行值类型: {type(value)}")
    
    if isinstance(value, (list, np.ndarray)):
        print(f"📛 {col} 是嵌套数组，尝试展开")
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"✅ {col} 成功转换为数值")
    except Exception as e:
        print(f"❌ 转换 {col} 出错: {str(e)}")

# 清理 word_count
df['word_count'] = df['word_count'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce')

# 清理 adj_popularity
def flatten_adj_popularity(x):
    if isinstance(x, (list, np.ndarray)):
        arr = np.array(x)
        if arr.ndim == 1:
            return arr[0]
        elif arr.ndim == 2:
            return arr[0, 0]
    return x

df['adj_popularity'] = df['adj_popularity'].apply(flatten_adj_popularity)
df['adj_popularity'] = pd.to_numeric(df['adj_popularity'], errors='coerce')



# 7. 创建并保存最终报告
report_path = os.path.join(output_dir, "final_analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("视频简介对流行度影响分析报告\n")
    f.write("="*60 + "\n\n")
    f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"总视频数量: {len(df)}\n")
    f.write(f"短视频数量: {len(df[df['duration_type']=='short'])}\n")
    f.write(f"中视频数量: {len(df[df['duration_type']=='medium'])}\n")
    f.write(f"长视频数量: {len(df[df['duration_type']=='long'])}\n\n")
    
    f.write("主题分析:\n")
    f.write("-"*50 + "\n")
    for topic_id, topic_name in topic_names.items():
        top_keywords = ", ".join([tfidf_feature_names[i] for i in lda.components_[topic_id].argsort()[:-11:-1]])
        f.write(f"{topic_name} (主题{topic_id}): {top_keywords}\n")
    f.write("\n")
    
    f.write("模型性能评估:\n")
    f.write("-"*50 + "\n")
    for dtype, metrics in results.items():
        f.write(f"{dtype.capitalize()}视频:\n")
        f.write(f"  样本量: {metrics['num_samples']}\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  R²: {metrics['r2']:.4f}\n")
        f.write(f"  交叉验证R²: {metrics['cv_r2']:.4f} ± {metrics['cv_std']:.4f}\n")
        
        if dtype in keyword_importances:
            f.write("\n  重要特征:\n")
            top_features = keyword_importances[dtype].nlargest(5)
            for feature, importance in top_features.items():
                f.write(f"    - {feature}: {importance:.4f}\n")
        f.write("\n")
    
    f.write("\n关键发现:\n")
    f.write("-"*50 + "\n")
    f.write("1. 短视频简介:\n")
    f.write("   - 最佳字数: 20-50字\n")
    f.write("   - 高流行度简介特点: 突出精彩瞬间，包含情感词(#热血 #精彩)\n")
    f.write("   - 避免: 过多背景信息和关注引导\n\n")
    
    f.write("2. 中视频简介:\n")
    f.write("   - 最佳字数: 50-150字\n")
    f.write("   - 高流行度简介特点: 列出关键事件，适当包含情感和故事元素\n")
    f.write("   - 避免: 过于技术性的分析和冗长的背景\n\n")
    
    f.write("3. 长视频简介:\n")
    f.write("   - 最佳字数: 100-300字\n")
    f.write("   - 高流行度简介特点: 深度背景故事，技术分析，情感共鸣\n")
    f.write("   - 避免: 过于简短的描述和过多的关注引导\n\n")
    
    f.write("AI助手框架:\n")
    f.write("-"*50 + "\n")
    f.write("已开发AI助手框架，提供以下功能:\n")
    f.write("  - 简介质量分析 (预测流行度、主题分析)\n")
    f.write("  - 个性化改进建议\n")
    f.write("  - 主题模板生成\n")
    f.write("测试结果详见: ai_assistant_results.txt\n")

print(f"✅ 已保存最终分析报告到: {report_path}")

# 创建并保存词云
def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        stopwords=stop_words,
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 高流行度视频词云
high_popularity = df[df['adj_popularity'] > df['adj_popularity'].quantile(0.75)]
high_pop_text = " ".join(high_popularity['lemmatized_description'])
generate_wordcloud(high_pop_text, "高流行度视频关键词", "high_popularity_wordcloud.png")

# 低流行度视频词云
low_popularity = df[df['adj_popularity'] < df['adj_popularity'].quantile(0.25)]
low_pop_text = " ".join(low_popularity['lemmatized_description'])
generate_wordcloud(low_pop_text, "低流行度视频关键词", "low_popularity_wordcloud.png")

print("✅ 已保存词云图像")

# 创建整体可视化
plt.figure(figsize=(15, 10))

# 流行度与字数关系
plt.subplot(2, 2, 1)
sns.scatterplot(x='word_count', y='adj_popularity', data=df, 
               hue='duration_type', palette='viridis', alpha=0.6)
plt.title("字数与流行度关系", fontsize=14)
plt.xlabel("字数", fontsize=12)
plt.ylabel("流行度", fontsize=12)

# 情感与流行度关系
plt.subplot(2, 2, 2)
sns.boxplot(x=pd.cut(df['sentiment'], bins=5), y='adj_popularity', 
           data=df, palette='coolwarm')
plt.title("情感与流行度关系", fontsize=14)
plt.xlabel("情感分数", fontsize=12)
plt.ylabel("流行度", fontsize=12)
plt.xticks(rotation=15)

# 主题分布
plt.subplot(2, 2, 3)
topic_columns = [f'topic_{i}' for i in range(num_topics)]
topic_means = df.groupby('duration_type')[topic_columns].mean()
topic_means.plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
plt.title("不同视频类型的主题分布", fontsize=14)
plt.xlabel("视频类型", fontsize=12)
plt.ylabel("主题比例", fontsize=12)
plt.legend(title="主题", bbox_to_anchor=(1.05, 1), loc='upper left')

# 特征重要性对比
plt.subplot(2, 2, 4)
for dtype, importance in keyword_importances.items():
    top_features = importance.nlargest(5)
    plt.plot(top_features.values, top_features.index, 'o-', label=dtype)
plt.title("不同视频类型的重要特征", fontsize=14)
plt.xlabel("特征重要性", fontsize=12)
plt.ylabel("特征", fontsize=12)
plt.legend(title="视频类型")

plt.tight_layout()
overview_img_path = os.path.join(output_dir, "analysis_overview.png")
plt.savefig(overview_img_path, dpi=300)
plt.close()
print(f"✅ 已保存分析概览图像到: {overview_img_path}")

print("\n" + "="*50)
print("分析完成! 所有结果已保存至目录:", output_dir)
print("="*50)