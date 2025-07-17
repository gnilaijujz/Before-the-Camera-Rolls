import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime

# Ensure required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Paths
DATA_PATH = "../sports_2024-2025.csv"
OUTPUT_DIR = "description_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Drop missing descriptions
df = df.dropna(subset=["description"])
stop_words = set(stopwords.words('english'))

# Clean description
def clean_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

df["tokens"] = df["description"].apply(clean_text)
df["description_length"] = df["description"].apply(len)
df["word_count"] = df["tokens"].apply(len)
df["sentence_count"] = df["description"].apply(lambda x: len(sent_tokenize(x)))

# 1. 描述长度分布
plt.figure(figsize=(10, 6))
sns.histplot(df["word_count"], bins=30, kde=True)
plt.title("Distribution of Word Counts in Descriptions")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_wordcount_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df["sentence_count"], bins=30, kde=True)
plt.title("Distribution of Sentence Counts in Descriptions")
plt.xlabel("Number of Sentences")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_sentencecount_distribution.png")
plt.close()

# 2. 词频统计
all_words = [word for tokens in df["tokens"] for word in tokens]
word_freq = pd.Series(all_words).value_counts().head(30)
plt.figure(figsize=(12, 6))
sns.barplot(x=word_freq.values, y=word_freq.index, palette="viridis")
plt.title("Top 30 Most Frequent Words in Descriptions")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_top_words.png")
plt.close()

# 3. WordCloud
wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(" ".join(all_words))
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Description Texts")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_wordcloud.png")
plt.close()

# 4. 情感分析
df["sentiment_polarity"] = df["description"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["sentiment_subjectivity"] = df["description"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

plt.figure(figsize=(10, 6))
sns.histplot(df["sentiment_polarity"], kde=True, bins=30)
plt.title("Sentiment Polarity Distribution")
plt.xlabel("Polarity (-1 = negative, +1 = positive)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_sentiment_polarity.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df["sentiment_subjectivity"], kde=True, bins=30)
plt.title("Sentiment Subjectivity Distribution")
plt.xlabel("Subjectivity (0 = objective, 1 = subjective)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_sentiment_subjectivity.png")
plt.close()

# 5. TF-IDF 关键词提取
tfidf = TfidfVectorizer(max_features=50, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["description"])
keywords = tfidf.get_feature_names_out()
scores = tfidf_matrix.sum(axis=0).A1
keyword_scores = pd.Series(scores, index=keywords).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=keyword_scores.values, y=keyword_scores.index, palette="coolwarm")
plt.title("Top TF-IDF Keywords from Descriptions")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_tfidf_keywords.png")
plt.close()

# 6. LDA 主题建模
cv = CountVectorizer(stop_words='english', max_features=1000)
cv_matrix = cv.fit_transform(df["description"])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(cv_matrix)

def get_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(", ".join(topic_words))
    return topics

topics = get_topics(lda, cv.get_feature_names_out())

# Save topics to file
with open(f"{OUTPUT_DIR}/lda_topics.txt", "w") as f:
    for i, topic in enumerate(topics):
        f.write(f"Topic {i+1}: {topic}\n")

# 7. 描述长度随时间变化
df["published_at"] = pd.to_datetime(df["published_at"])
df["date"] = df["published_at"].dt.date
daily_desc = df.groupby("date")["word_count"].mean()

plt.figure(figsize=(12, 6))
daily_desc.plot()
plt.title("Average Description Word Count Over Time")
plt.xlabel("Date")
plt.ylabel("Avg. Word Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/description_length_over_time.png")
plt.close()

# 8. 分组：短/中/长视频分类
def categorize_video(duration):
    if duration < 240:
        return 'short'
    elif duration <= 1200:
        return 'medium'
    else:
        return 'long'

df['video_length_type'] = df['duration_seconds'].apply(categorize_video)

# 分析字段：['word_count', 'sentiment_polarity', 'sentiment_subjectivity']

for feature, title, filename in [
    ('word_count', 'Description Word Count', 'wordcount'),
    ('sentiment_polarity', 'Sentiment Polarity', 'polarity'),
    ('sentiment_subjectivity', 'Sentiment Subjectivity', 'subjectivity')
]:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='video_length_type', y=feature, palette='Set2')
    plt.title(f'{title} by Video Length Category')
    plt.xlabel("Video Length Type")
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/by_length_{filename}.png")
    plt.close()

# 分组词云 + top词频
for length_type in ['short', 'medium', 'long']:
    subset = df[df['video_length_type'] == length_type]
    all_tokens = [word for tokens in subset['tokens'] for word in tokens]
    word_freq = pd.Series(all_tokens).value_counts().head(30)

    # 条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette='coolwarm')
    plt.title(f"Top 30 Words in {length_type.capitalize()} Videos")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_words_{length_type}.png")
    plt.close()

    # WordCloud
    wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(" ".join(all_tokens))
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud of {length_type.capitalize()} Video Descriptions")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/wordcloud_{length_type}.png")
    plt.close()

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.cm as cm

# 如果你还没有 cluster 信息，我们可以先做一个聚类
if 'cluster' not in df.columns:
    vectorizer = TfidfVectorizer(max_features=100)
    desc_vecs = vectorizer.fit_transform(df['description'])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(desc_vecs)

# 1. 降维 + 聚类可视化（PCA）
pca = PCA(n_components=2)
reduced = pca.fit_transform(desc_vecs.toarray())

plt.figure(figsize=(10, 8))
colors = cm.tab10(df['cluster'].astype(int) / df['cluster'].nunique())
plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6, s=30)
plt.title("PCA Reduction of Description TF-IDF with Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_pca_scatter.png")
plt.close()

# 2. 多指标按 cluster 平均值柱状图
cluster_metrics = df.groupby("cluster")[['word_count', 'sentiment_polarity', 'sentiment_subjectivity']].mean()

plt.figure(figsize=(12, 6))
cluster_metrics.plot(kind='bar', colormap='Set3')
plt.title("Average Metrics by Cluster")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_barplot_metrics.png")
plt.close()

# 3. 各聚类样本比例饼图
cluster_counts = df['cluster'].value_counts().sort_index()
plt.figure(figsize=(8, 8))
cluster_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title("Cluster Size Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_pie_chart.png")
plt.close()

# 4. 聚类特征热力图
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_metrics, annot=True, cmap='YlGnBu')
plt.title("Feature Heatmap by Cluster")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_feature_heatmap.png")
plt.close()


print("✅ Description analysis complete. All visualizations saved to:", OUTPUT_DIR)
