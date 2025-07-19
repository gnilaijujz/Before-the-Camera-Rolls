import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re
from collections import Counter, defaultdict
import networkx as nx
from nltk.corpus import stopwords

# File path
DATA_PATH = "../game_2024-2025.csv"
OUTPUT_DIR = "tag_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Preprocess tags
def clean_and_split_tags(tag_str):
    if pd.isna(tag_str):
        return []
    return [t.strip().lower() for t in re.split(r',|;', tag_str) if t.strip()]

df['tag_list'] = df['tags'].apply(clean_and_split_tags)

# 1. 标签数量分布
df['tag_count'] = df['tag_list'].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df['tag_count'], bins=20, kde=True)
plt.title("Distribution of Tag Counts per Video")
plt.xlabel("Number of Tags")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_count_distribution.png")
plt.close()

# 2. Top-N 标签频率
all_tags = [tag for tags in df['tag_list'] for tag in tags]
tag_freq = Counter(all_tags)
top_n = 30
top_tags = tag_freq.most_common(top_n)

tags_df = pd.DataFrame(top_tags, columns=['tag', 'frequency'])
plt.figure(figsize=(12, 6))
sns.barplot(data=tags_df, x='frequency', y='tag', palette='magma')
plt.title(f"Top {top_n} Most Frequent Tags")
plt.xlabel("Frequency")
plt.ylabel("Tag")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_{top_n}_tags.png")
plt.close()

# 3. 词云图
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(tag_freq)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Tags")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_wordcloud.png")
plt.close()

# 4. 标签频率与视频表现（views, likes, comments, engagement）
tag_metrics = defaultdict(lambda: {'count': 0, 'views': 0, 'likes': 0, 'comments': 0, 'engagement': 0})

for _, row in df.iterrows():
    for tag in row['tag_list']:
        tag_metrics[tag]['count'] += 1
        tag_metrics[tag]['views'] += row['view_count']
        tag_metrics[tag]['likes'] += row['like_count']
        tag_metrics[tag]['comments'] += row['comment_count']
        tag_metrics[tag]['engagement'] += row['engagement_rate']

tag_stats = pd.DataFrame([
    [tag,
     tag_metrics[tag]['count'],
     tag_metrics[tag]['views'] / tag_metrics[tag]['count'],
     tag_metrics[tag]['likes'] / tag_metrics[tag]['count'],
     tag_metrics[tag]['comments'] / tag_metrics[tag]['count'],
     tag_metrics[tag]['engagement'] / tag_metrics[tag]['count']]
    for tag in tag_freq if tag_metrics[tag]['count'] >= 3  # Only keep tags appearing 3+ times
], columns=['tag', 'count', 'avg_views', 'avg_likes', 'avg_comments', 'avg_engagement'])

# 选择几个指标可视化
melted = tag_stats.sort_values("avg_engagement", ascending=False).head(20).melt(id_vars="tag", value_vars=["avg_views", "avg_likes", "avg_comments", "avg_engagement"])
plt.figure(figsize=(14, 7))
sns.barplot(data=melted, x="value", y="tag", hue="variable")
plt.title("Top Tags by Engagement (and other metrics)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_performance_metrics.png")
plt.close()

# 5. 标签共现网络图
cooccurrence = defaultdict(Counter)
for tags in df['tag_list']:
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            t1, t2 = sorted([tags[i], tags[j]])
            cooccurrence[t1][t2] += 1

# 构建图
G = nx.Graph()
for t1 in cooccurrence:
    for t2, freq in cooccurrence[t1].items():
        if freq >= 3:  # Only show strong co-occurrence
            G.add_edge(t1, t2, weight=freq)

plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G, k=0.5)
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]
nx.draw_networkx_nodes(G, pos, node_size=400, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=[w * 0.2 for w in weights], alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Tag Co-occurrence Network (≥3 co-tags)")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_cooccurrence_network.png")
plt.close()






from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import umap
import numpy as np

# 1. 加载预训练词向量（建议 fastText 或 GoogleNews word2vec）
# 示例：使用 gensim 中的 GoogleNews 向量（需提前下载并放在本地）
# https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
print("Loading word2vec model...")
word_vectors = KeyedVectors.load_word2vec_format("../../GoogleNews-vectors-negative300.bin.gz", binary=True, limit=200000)

# 2. 将标签转为向量（仅保留词库中存在的标签）
unique_tags = list(tag_freq.keys())
tag_vecs = []
valid_tags = []

for tag in unique_tags:
    if tag in word_vectors:
        tag_vecs.append(word_vectors[tag])
        valid_tags.append(tag)

tag_vecs = np.array(tag_vecs)

# 3. KMeans 聚类
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(tag_vecs)

# 4. 聚类标签 dataframe
cluster_df = pd.DataFrame({
    'tag': valid_tags,
    'cluster': clusters
})

# 5. 每个聚类前10个标签
cluster_top_tags = cluster_df.groupby("cluster")['tag'].apply(lambda x: ', '.join(x[:10])).reset_index()
cluster_top_tags.columns = ['cluster', 'top_tags']
print("\n=== Cluster Summary ===")
print(cluster_top_tags.to_string(index=False))

# 保存聚类结果
cluster_df.to_csv(f"{OUTPUT_DIR}/tag_clusters.csv", index=False)

# 6. 可视化：UMAP 降维 + 聚类展示
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding = reducer.fit_transform(tag_vecs)

plt.figure(figsize=(12, 9))
palette = sns.color_palette("tab10", n_colors=n_clusters)
for i in range(n_clusters):
    idxs = cluster_df['cluster'] == i
    plt.scatter(embedding[idxs, 0], embedding[idxs, 1], label=f"Cluster {i}", alpha=0.7, s=50, c=[palette[i]])

# 添加标签文本（在所有点上方绘制标签）
for i, tag in enumerate(valid_tags):
    plt.text(embedding[i, 0], embedding[i, 1], tag, fontsize=6, alpha=0.5)

plt.title("Tag Clusters in Semantic Space (UMAP Projection)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_clusters_umap_labeled.png")  # 建议另存一份，避免覆盖
plt.close()



# ====== 标签聚类表现分析图集 ======
from sklearn.preprocessing import MinMaxScaler

# 合并聚类结果和原始标签表现
cluster_performance = cluster_df.merge(tag_stats, on="tag", how="inner")

# 1. 聚类样本数量饼图
cluster_counts = cluster_performance['cluster'].value_counts().sort_index()
plt.figure(figsize=(7, 7))
cluster_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title("Tag Cluster Size Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_cluster_pie_chart.png")
plt.close()

# 2. 聚类平均指标柱状图
agg_metrics = cluster_performance.groupby("cluster")[['avg_views', 'avg_likes', 'avg_comments', 'avg_engagement']].mean()

plt.figure(figsize=(12, 6))
agg_metrics.plot(kind='bar', colormap='Set2')
plt.title("Average Performance per Tag Cluster")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_cluster_barplot_metrics.png")
plt.close()

# 3. 聚类指标热力图（归一化）
scaled = MinMaxScaler().fit_transform(agg_metrics)
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(scaled, index=agg_metrics.index, columns=agg_metrics.columns),
            annot=True, cmap='YlGnBu')
plt.title("Normalized Performance Heatmap by Tag Cluster")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_cluster_feature_heatmap.png")
plt.close()

# 4. 再次生成 UMAP 降维散点图（聚类结果对比）
plt.figure(figsize=(12, 9))
for i in range(n_clusters):
    idxs = cluster_df['cluster'] == i
    plt.scatter(embedding[idxs, 0], embedding[idxs, 1], label=f"Cluster {i}", alpha=0.7, s=50, c=[palette[i]])

plt.title("Tag Cluster Distribution in UMAP Semantic Space")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tag_cluster_umap_scatter.png")
plt.close()

print("✅ Tag cluster performance visualizations saved.")


print("✅ Tag analysis complete. All visualizations saved to:", OUTPUT_DIR)