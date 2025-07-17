#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
titleanalysis.py – end‑to‑end analysis of YouTube sports titles
author : Ari (adapted from description.py)
date   : 2025‑07‑17
"""

# --------------------------------------------------
# 0. Imports & setup
# --------------------------------------------------
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

# download NLTK resources once
for corp in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "omw-1.4"]:
    nltk.download(corp, quiet=True)

# --------------------------------------------------
# 1. Paths & I/O
# --------------------------------------------------
DATA_PATH   = "sports_2024-2025_with_popularity.csv"
OUTPUT_DIR  = "title_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 2. Basic video‑level engineering (same as description.py)
# --------------------------------------------------
def classify_duration(sec: float) -> str:
    if sec <= 240:
        return "short"
    elif sec <= 1200:
        return "medium"
    return "long"

df["duration_type"] = df["duration_seconds"].apply(classify_duration)

# publication‑time decay weighting
df["published_at"] = pd.to_datetime(df["published_at"])
latest_date = df["published_at"].max()
df["days_since_publish"] = (latest_date - df["published_at"]).dt.days

def plateau_decay(t, early_a=0.25, t0=20, floor=0.35):
    return np.maximum(1 / (1 + np.exp(early_a * (t - t0))), floor)

df["time_weight"]     = plateau_decay(df["days_since_publish"])
df["adj_popularity"]  = df["popularity_normalized"] * 1e5 * df["time_weight"]

# --------------------------------------------------
# 3. TITLE text preprocessing
# --------------------------------------------------
STOP_WORDS = set(stopwords.words("english")).union({
    "subscribe","channel","follow","like","instagram","facebook",
    "twitter","tiktok","website","click","link","watch","video"
})
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace('.', '')))
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def lemmatize_text(text: str) -> str:
    return " ".join(lemmatizer.lemmatize(w) for w in word_tokenize(text))

def get_text_features(text: str) -> dict:
    words = text.split()
    wc   = len(words)
    return {
        "word_count"        : wc,
        "avg_word_length"   : np.mean([len(w) for w in words]) if wc else 0,
        "unique_word_ratio" : len(set(words)) / wc if wc else 0,
        "sentiment"         : TextBlob(text).sentiment.polarity
    }

df["cleaned_title"]        = df["title"].apply(clean_text)
df["lemmatized_title"]     = df["cleaned_title"].apply(lemmatize_text)
text_feats_df              = df["cleaned_title"].apply(get_text_features).apply(pd.Series)
df                         = pd.concat([df, text_feats_df], axis=1)

# save intermediate tidy data
df.to_csv(os.path.join(OUTPUT_DIR, "processed_data_with_titles.csv"),
          index=False, encoding="utf-8-sig")

# --------------------------------------------------
# 4. TF‑IDF & LDA topic modelling
# --------------------------------------------------
tfidf_vec = TfidfVectorizer(max_features=1000,
                            stop_words=list(STOP_WORDS),
                            ngram_range=(1, 2))
tfidf_mat = tfidf_vec.fit_transform(df["lemmatized_title"])

NUM_TOPICS = 5
lda = LatentDirichletAllocation(n_components=NUM_TOPICS,
                                max_iter=10,
                                learning_method="online",
                                random_state=42)
lda_topics = lda.fit_transform(tfidf_mat)

for i in range(NUM_TOPICS):
    df[f"topic_{i}"] = lda_topics[:, i]

# --------------------------------------------------
# 5. Visualise topic keywords (bar‑panel)
# --------------------------------------------------
def plot_top_words(model, feat_names, n=10, title="Topic keywords"):
    fig, axes = plt.subplots(1, NUM_TOPICS, figsize=(20, 6), sharex=True)
    for idx, topic in enumerate(model.components_):
        top_idx   = topic.argsort()[:-n-1:-1]
        features  = [feat_names[i] for i in top_idx]
        weights   = topic[top_idx]
        ax        = axes[idx]
        ax.barh(features, weights)
        ax.set_title(f"Topic {idx+1}")
        ax.invert_yaxis()
        for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    plt.suptitle(title)
    fig.tight_layout()
    return fig

topic_fig = plot_top_words(lda, tfidf_vec.get_feature_names_out())
topic_fig.savefig(os.path.join(OUTPUT_DIR, "topic_keywords.png"), dpi=300)

# --------------------------------------------------
# 6. Train per‑duration Random‑Forest regressors
# --------------------------------------------------
TEXT_FEATURES = ["word_count","avg_word_length","unique_word_ratio","sentiment"] \
                + [f"topic_{i}" for i in range(NUM_TOPICS)]

results, models, scalers, featnames_by_dur = {}, {}, {}, {}

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12

for dur in ["short","medium","long"]:
    subset = df[df["duration_type"] == dur]
    if len(subset) < 20:
        print(f"⚠️  Only {len(subset)} {dur} samples – skipping model.")
        continue

    X      = subset[TEXT_FEATURES]
    y      = subset["adj_popularity"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler          = StandardScaler()
    X_tr_sc, X_te_sc = scaler.fit_transform(X_tr), scaler.transform(X_te)
    model           = RandomForestRegressor(n_estimators=200, max_depth=10,
                                            min_samples_split=5, random_state=42)
    model.fit(X_tr_sc, y_tr)

    # save artefacts
    joblib.dump(model,  os.path.join(OUTPUT_DIR, f"{dur}_title_model.joblib"))
    scalers[dur]     = scaler
    models[dur]      = model
    featnames_by_dur[dur] = TEXT_FEATURES

    y_pred           = model.predict(X_te_sc)
    rmse             = np.sqrt(mean_squared_error(y_te, y_pred))
    r2               = r2_score(y_te, y_pred)
    cv_scores        = cross_val_score(model, scaler.transform(X), y,
                                       cv=5, scoring="r2")
    results[dur]     = dict(rmse=rmse, r2=r2,
                            cv_r2=cv_scores.mean(), cv_std=cv_scores.std(),
                            num_samples=len(subset))

    # --- feature importance plot
    feat_imp = pd.Series(model.feature_importances_, index=TEXT_FEATURES).sort_values()
    plt.figure(figsize=(10, 6))
    feat_imp.tail(10).plot.barh()
    plt.title(f"{dur.capitalize()} videos – top features")
    plt.xlabel("Importance")
    for i,v in enumerate(feat_imp.tail(10)):
        plt.text(v+0.001, i, f"{v:.3f}", va="center")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dur}_feature_importance.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # --- prediction scatter
    plt.figure(figsize=(8,6))
    plt.scatter(y_te, y_pred, alpha=.6)
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--')
    plt.title(f"{dur.capitalize()} – predicted vs actual")
    plt.xlabel("Actual popularity"); plt.ylabel("Predicted")
    plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dur}_prediction_scatter.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# 7. Word‑clouds for high / low popularity titles
# --------------------------------------------------
def generate_wordcloud(text, title, fname):
    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=STOP_WORDS, max_words=100).generate(text)
    plt.figure(figsize=(12,6)); plt.imshow(wc, interpolation="bilinear")
    plt.axis("off"); plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close()

hi_pop = " ".join(df[df["adj_popularity"] > df["adj_popularity"].quantile(.75)]["lemmatized_title"])
lo_pop = " ".join(df[df["adj_popularity"] < df["adj_popularity"].quantile(.25)]["lemmatized_title"])
generate_wordcloud(hi_pop, "High‑popularity title keywords", "high_pop_title_wc.png")
generate_wordcloud(lo_pop, "Low‑popularity title keywords",  "low_pop_title_wc.png")

# --------------------------------------------------
# 8. Combined overview plot
# --------------------------------------------------
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.scatterplot(data=df, x="word_count", y="adj_popularity",
                hue="duration_type", palette="viridis", alpha=.6)
plt.title("Word count vs popularity"); plt.xlabel("Word count"); plt.ylabel("Adjusted popularity")

plt.subplot(2,2,2)
sent_bins = pd.cut(df["sentiment"], bins=5)
sns.boxplot(x=sent_bins, y="adj_popularity", data=df, palette="coolwarm")
plt.title("Sentiment vs popularity"); plt.xlabel("Sentiment bin"); plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "overview_plots.png"), dpi=300)
plt.close()

# --------------------------------------------------
# 9. Text report
# --------------------------------------------------
report = os.path.join(OUTPUT_DIR, "final_title_analysis_report.txt")
with open(report, "w", encoding="utf-8") as f:
    f.write("="*60+"\n")
    f.write("Title Impact on Popularity Report\n")
    f.write("="*60+"\n\n")
    f.write(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"Total number of videos: {len(df)}\n")
    for dur in ["short","medium","long"]:
        f.write(f"{dur.capitalize()} videos: {len(df[df['duration_type']==dur])}\n")
    f.write("\nModel performance:\n")
    for dur, m in results.items():
        f.write(f"  {dur.capitalize()} – R² {m['r2']:.3f} (±{m['cv_std']:.3f} CV), "
                f"RMSE {m['rmse']:.2f}, n={m['num_samples']}\n")
    f.write("\n(See images and CSV output for further details.)\n")

print(f"✅ All artefacts saved to '{OUTPUT_DIR}'")
