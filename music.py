#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
music_titleanalysis.py – end‑to‑end analysis of YouTube MUSIC video titles
Ported from titleanalysis.py (sports) but with auto column detection
and music‑specific output folder.

Author : Ari
Updated: 2025‑07‑18
"""

# ==================================================
# 0. Imports & global visual style
# ==================================================
import os, re, string, joblib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------- visual defaults ----------------------------------------------
sns.set_theme(context="talk", style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight"
})

# ==================================================
# 1. NLTK resources (quiet download)
# ==================================================
for corp in ["punkt", "stopwords", "wordnet",
             "averaged_perceptron_tagger", "omw-1.4"]:
    nltk.download(corp, quiet=True)

# ==================================================
# 2. Paths
# ==================================================
DATA_PATH  = "music_2024-2025.csv"          # <-- your uploaded file
OUTPUT_DIR = "music_title_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 3. Load & column auto‑resolve
# ==================================================
df_raw = pd.read_csv(DATA_PATH)

def resolve_col(df, preferred, fallbacks, required=True, cast=None):
    """
    Find the first column present in df among [preferred]+fallbacks.
    Optionally cast via callable. Raise KeyError if required and none found.
    """
    candidates = [preferred] + list(fallbacks)
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Required column not found. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None

COL_TITLE = resolve_col(df_raw, "title",
                        ["video_title", "name", "song_title", "track"])
COL_PUB   = resolve_col(df_raw, "published_at",
                        ["publish_date", "published_date", "date", "upload_date"])
COL_DUR   = resolve_col(df_raw, "duration_seconds",
                        ["duration", "length_seconds", "length", "video_length"])
COL_POP   = resolve_col(df_raw, "popularity_normalized",
                        ["views_normalized", "views", "view_count",
                         "like_view_score", "engagement_score"])

# Copy to unified column names we’ll use downstream
df = df_raw.rename(columns={
    COL_TITLE: "title",
    COL_PUB:   "published_at",
    COL_DUR:   "duration_seconds",
    COL_POP:   "popularity_normalized"
}).copy()

# --------------------------------------------------
# If popularity is raw views, scale to ~0‑1
# --------------------------------------------------
if df["popularity_normalized"].max() > 2:  # crude heuristic
    df["popularity_normalized"] = (
        df["popularity_normalized"] / df["popularity_normalized"].max()
    )

# ==================================================
# 4. Basic per‑video engineering
# ==================================================
def classify_duration(sec: float) -> str:
    if sec <= 240:  return "short"
    if sec <= 1200: return "medium"
    return "long"

df["duration_type"] = df["duration_seconds"].apply(classify_duration)

# publication‑time decay weighting
df["published_at"]  = pd.to_datetime(df["published_at"], errors="coerce")
latest_date         = df["published_at"].max()
df["days_since_publish"] = (latest_date - df["published_at"]).dt.days

def plateau_decay(t, a=0.25, t0=20, floor=0.35):
    return np.maximum(1 / (1 + np.exp(a * (t - t0))), floor)

df["time_weight"]    = plateau_decay(df["days_since_publish"])
df["adj_popularity"] = df["popularity_normalized"] * 1e5 * df["time_weight"]

# ==================================================
# 5. Title text preprocessing
# ==================================================
STOP_WORDS = set(stopwords.words("english")).union({
    # Generic promo words
    "subscribe","channel","follow","like","watch","video",
    "instagram","facebook","twitter","tiktok","website","click","link",
    # Music‑specific fluff words you may want to down‑weight:
    "official","music","lyrics","lyric","audio","visualizer","feat","ft","remix",
    "live","mv","hd","new","out","now"
})
lemmatizer = WordNetLemmatizer()

def clean_text(tx: str) -> str:
    if pd.isna(tx): return ""
    tx = tx.lower()
    tx = re.sub(r"https?://\S+|www\.\S+", "", tx)
    tx = re.sub(r"<.*?>", "", tx)
    tx = tx.translate(str.maketrans('', '', string.punctuation.replace('.', '')))
    tx = re.sub(r"\d+", "", tx)
    return re.sub(r"\s+", " ", tx).strip()

def lemmatise(tx: str) -> str:
    return " ".join(lemmatizer.lemmatize(w) for w in word_tokenize(tx))

def text_stats(tx: str):
    words = tx.split(); wc = len(words)
    return {
        "word_count": wc,
        "avg_word_length": np.mean([len(w) for w in words]) if wc else 0,
        "unique_word_ratio": len(set(words))/wc if wc else 0,
        "sentiment": TextBlob(tx).sentiment.polarity
    }

df["cleaned_title"]    = df["title"].apply(clean_text)
df["lemmatised_title"] = df["cleaned_title"].apply(lemmatise)
df = pd.concat([df, df["cleaned_title"].apply(text_stats).apply(pd.Series)], axis=1)

# write processed CSV early so you can inspect
df.to_csv(os.path.join(OUTPUT_DIR, "processed_music_titles.csv"),
          index=False, encoding="utf-8-sig")

# ==================================================
# 6. TF‑IDF + LDA topics
# ==================================================
NUM_TOPICS = 5  # adjust if you want finer granularity
tfidf_vec = TfidfVectorizer(max_features=1000,
                            stop_words=list(STOP_WORDS),
                            ngram_range=(1,2))
tfidf_mat = tfidf_vec.fit_transform(df["lemmatised_title"])

lda = LatentDirichletAllocation(n_components=NUM_TOPICS,
                                max_iter=10,
                                learning_method="online",
                                random_state=42)
lda_topics = lda.fit_transform(tfidf_mat)
for i in range(NUM_TOPICS):
    df[f"topic_{i}"] = lda_topics[:, i]

# Topic keyword plot (gradient)
def plot_topics(model, feat_names, n=10):
    cmaps = plt.cm.get_cmap("viridis", n)
    fig, axes = plt.subplots(1, NUM_TOPICS, figsize=(20,6), sharex=True)
    for idx, topic in enumerate(model.components_):
        order   = topic.argsort()[:-n-1:-1]
        words   = [feat_names[i] for i in order]
        weights = topic[order]
        colours = cmaps(np.linspace(0.2, 0.9, n))
        axes[idx].barh(words, weights, color=colours)
        axes[idx].set_title(f"Topic {idx+1}")
        axes[idx].invert_yaxis()
    fig.suptitle("Top words per LDA topic (Music)")
    fig.tight_layout()
    return fig

plot_topics(lda, tfidf_vec.get_feature_names_out())\
    .savefig(os.path.join(OUTPUT_DIR, "music_topic_keywords.png"))

# ==================================================
# 7. Random‑Forest per duration
# ==================================================
TEXT_VARS = ["word_count","avg_word_length","unique_word_ratio","sentiment"] \
            + [f"topic_{i}" for i in range(NUM_TOPICS)]
results = {}

for dur in ["short","medium","long"]:
    sub = df[df["duration_type"]==dur]
    if len(sub) < 20:
        print(f"⚠️  Only {len(sub)} {dur} music samples – model skipped.")
        continue

    X,y = sub[TEXT_VARS], sub["adj_popularity"]
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=.2,random_state=42)

    scaler      = StandardScaler()
    X_tr, X_te  = scaler.fit_transform(X_tr), scaler.transform(X_te)
    rf = RandomForestRegressor(n_estimators=200,max_depth=10,
                               min_samples_split=5,random_state=42)
    rf.fit(X_tr,y_tr)
    joblib.dump(rf, os.path.join(OUTPUT_DIR, f"{dur}_music_title_model.joblib"))

    y_pred = rf.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(y_te,y_pred))
    r2     = r2_score(y_te,y_pred)
    cv     = cross_val_score(rf, scaler.transform(X), y, cv=5, scoring="r2")
    results[dur] = dict(rmse=rmse, r2=r2, cv_r2=cv.mean(), cv_std=cv.std(), n=len(sub))

    # Feature importance (gradient)
    imp   = pd.Series(rf.feature_importances_, index=TEXT_VARS).sort_values()
    top10 = imp.tail(10)
    cols  = plt.cm.Blues(np.linspace(0.3, 1, 10))
    plt.figure(figsize=(10,6))
    plt.barh(top10.index, top10.values, color=cols)
    plt.title(f"{dur.capitalize()} Music – top features")
    plt.xlabel("Importance")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dur}_music_feature_importance.png"))
    plt.close()

    # Prediction scatter (gradient)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(y_te, y_pred, c=y_te, cmap="viridis", alpha=.7)
    plt.colorbar(sc, label="Actual popularity")
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    plt.plot(lims, lims, "k--")
    plt.title(f"{dur.capitalize()} Music – predicted vs actual")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dur}_music_prediction_scatter.png"))
    plt.close()

# ==================================================
# 8. Word‑clouds (gradient, filtered tokens)
# ==================================================
token_ok  = re.compile(r"^[a-zA-Z]{3,}$").match  # ≥3 ASCII letters
def cloud(raw_text, title, fname):
    txt  = " ".join(w for w in raw_text.split() if token_ok(w))
    wc   = WordCloud(width=900, height=450, background_color="white",
                     colormap="viridis", stopwords=STOP_WORDS,
                     max_words=120).generate(txt)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    safe_title = title.encode("ascii", "ignore").decode()
    plt.title(safe_title)
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

hi = " ".join(df[df["adj_popularity"]>df["adj_popularity"].quantile(.75)]
              ["lemmatised_title"])
lo = " ".join(df[df["adj_popularity"]<df["adj_popularity"].quantile(.25)]
              ["lemmatised_title"])
cloud(hi, "High popularity music title keywords", "high_pop_music_title_wc.png")
cloud(lo, "Low popularity music title keywords",  "low_pop_music_title_wc.png")

# ==================================================
# 9. Overview plots (gradient)
# ==================================================
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.scatterplot(data=df, x="word_count", y="adj_popularity",
                hue="word_count", palette="viridis", alpha=.7, legend=False)
plt.title("Word count vs popularity (Music)")
plt.xlabel("Word count"); plt.ylabel("Adjusted popularity")

plt.subplot(2,2,2)
sent_bins = pd.cut(df["sentiment"], bins=5)
palette   = sns.color_palette("coolwarm", n_colors=sent_bins.cat.categories.size)
sns.boxplot(x=sent_bins, y="adj_popularity", data=df, palette=palette)
plt.title("Sentiment vs popularity (Music)")
plt.xlabel("Sentiment bin"); plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "music_overview_plots.png"))
plt.close()

# ==================================================
# 10. Text report
# ==================================================
with open(os.path.join(OUTPUT_DIR, "music_title_analysis_report.txt"),
          "w", encoding="utf-8") as f:
    f.write("="*60+"\nMusic Title Impact on Popularity Report\n"+"="*60+"\n\n")
    f.write(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"Total music videos: {len(df)}\n")
    for d in ["short","medium","long"]:
        f.write(f"{d.capitalize()} music videos: {len(df[df['duration_type']==d])}\n")
    f.write("\nModel performance:\n")
    for d,m in results.items():
        f.write(f"  {d.capitalize()} – R² {m['r2']:.3f} (±{m['cv_std']:.3f}), "
                f"RMSE {m['rmse']:.2f}, n={m['n']}\n")
    f.write("\nSee PNGs & CSV for details.\n")

print(f"✅ Music artefacts saved to '{OUTPUT_DIR}'")
