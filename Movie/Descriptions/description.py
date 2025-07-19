import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
from datetime import datetime
import matplotlib.cm as cm

# 全局设置
sns.set_palette("Set2")
plt.rcParams["font.family"] = "Arial"

# NLTK 下载
nltk.download('punkt')
nltk.download('stopwords')

# ========== 路径配置 ==========
DATA_PATH = "../movie_2024-2025.csv"
OUTPUT_DIR = "description_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 数据加载与预处理 ==========
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["description"], inplace=True)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

df["tokens"] = df["description"].apply(clean_text)
df["description_length"] = df["description"].apply(len)
df["word_count"] = df["tokens"].apply(len)
df["sentence_count"] = df["description"].apply(lambda x: len(sent_tokenize(x)))

# ========== 描述长度分布 ==========
def plot_hist(data, column, title, xlabel, filename):
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png")
    plt.close()

plot_hist(df, "word_count", "Description Word Count Distribution", "Number of Words", "description_wordcount_distribution")
plot_hist(df, "sentence_count", "Description Sentence Count Distribution", "Number of Sentences", "description_sentencecount_distribution")

# ========== 词频统计 ==========
all_words = [word for tokens in df["tokens"] for word in tokens]
word_freq = pd.Series(all_words).value_counts().head(30)

plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x=word_freq.values, y=word_freq.index)
plt.title("Top 30 Frequent Words in Descriptions")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_top_words.png")
plt.close()

# ========== WordCloud ==========
wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(" ".join(all_words))
plt.figure(figsize=(16, 8), dpi=300)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Descriptions")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_wordcloud.png")
plt.close()

# ========== 情感分析 ==========
df["sentiment_polarity"] = df["description"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["sentiment_subjectivity"] = df["description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

plot_hist(df, "sentiment_polarity", "Sentiment Polarity Distribution", "Polarity (-1 to +1)", "description_sentiment_polarity")
plot_hist(df, "sentiment_subjectivity", "Sentiment Subjectivity Distribution", "Subjectivity (0 to 1)", "description_sentiment_subjectivity")

# ========== TF-IDF ==========
tfidf = TfidfVectorizer(max_features=50, stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["description"])
keywords = tfidf.get_feature_names_out()
scores = tfidf_matrix.sum(axis=0).A1
keyword_scores = pd.Series(scores, index=keywords).sort_values(ascending=False)

plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x=keyword_scores.values, y=keyword_scores.index, palette="coolwarm")
plt.title("Top TF-IDF Keywords")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_tfidf_keywords.png")
plt.close()

# ========== LDA 主题建模 ==========
cv = CountVectorizer(stop_words="english", max_features=1000)
cv_matrix = cv.fit_transform(df["description"])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(cv_matrix)

def get_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(", ".join(top_words))
    return topics

topics = get_topics(lda, cv.get_feature_names_out())
with open(f"{OUTPUT_DIR}/lda_topics.txt", "w") as f:
    for i, topic in enumerate(topics):
        f.write(f"Topic {i+1}: {topic}\n")

# ========== 时间趋势分析 ==========
df["published_at"] = pd.to_datetime(df["published_at"])
df["date"] = df["published_at"].dt.date
daily_desc = df.groupby("date")["word_count"].mean()

plt.figure(figsize=(12, 6), dpi=300)
daily_desc.plot()
plt.title("Average Description Word Count Over Time")
plt.xlabel("Date")
plt.ylabel("Average Word Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_length_over_time.png")
plt.close()

# ========== 视频时长分组 ==========
def categorize_video(duration):
    if duration < 240:
        return "short"
    elif duration <= 1200:
        return "medium"
    else:
        return "long"

df["video_length_type"] = df["duration_seconds"].apply(categorize_video)

# 各类型箱线图对比
metrics = [("word_count", "Word Count"), ("sentiment_polarity", "Sentiment Polarity"), ("sentiment_subjectivity", "Sentiment Subjectivity")]
for col, label in metrics:
    plt.figure(figsize=(12, 6), dpi=300)
    sns.boxplot(data=df, x="video_length_type", y=col)
    plt.title(f"{label} by Video Length")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/by_length_{col}.png")
    plt.close()

# 不同视频类型的词云 + Top30
for vtype in ["short", "medium", "long"]:
    subset = df[df["video_length_type"] == vtype]
    tokens = [word for tokens in subset["tokens"] for word in tokens]
    freq = pd.Series(tokens).value_counts().head(30)

    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(x=freq.values, y=freq.index)
    plt.title(f"Top 30 Words in {vtype.capitalize()} Videos")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_words_{vtype}.png")
    plt.close()

    wc = WordCloud(width=1600, height=800, background_color="white").generate(" ".join(tokens))
    plt.figure(figsize=(16, 8), dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {vtype.capitalize()} Videos")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/wordcloud_{vtype}.png")
    plt.close()

# ========== 聚类分析 ==========
vectorizer = TfidfVectorizer(max_features=100)
desc_vecs = vectorizer.fit_transform(df["description"])
kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(desc_vecs)

# PCA 可视化
pca = PCA(n_components=2)
reduced = pca.fit_transform(desc_vecs.toarray())
colors = cm.tab10(df["cluster"].astype(int) / df["cluster"].nunique())

plt.figure(figsize=(10, 8), dpi=300)
plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=30, alpha=0.6)
plt.title("PCA of TF-IDF Vectors (Colored by Cluster)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_pca_scatter.png")
plt.close()

# 聚类特征平均值柱状图
cluster_metrics = df.groupby("cluster")[["word_count", "sentiment_polarity", "sentiment_subjectivity"]].mean()

plt.figure(figsize=(12, 6), dpi=300)
cluster_metrics.plot(kind="bar", colormap="Set3")
plt.title("Average Metrics per Cluster")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_barplot_metrics.png")
plt.close()

# 聚类数量分布饼图
plt.figure(figsize=(8, 8), dpi=300)
df["cluster"].value_counts().sort_index().plot.pie(
    autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel")
)
plt.title("Cluster Size Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_pie_chart.png")
plt.close()

# 聚类特征热力图
plt.figure(figsize=(10, 6), dpi=300)
sns.heatmap(cluster_metrics, annot=True, cmap="YlGnBu")
plt.title("Cluster Feature Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_feature_heatmap.png")
plt.close()

print("✅ All visualizations complete! Saved to:", OUTPUT_DIR)