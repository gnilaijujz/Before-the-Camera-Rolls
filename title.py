# --------------------------------------------------
# 0. Imports & global visual style
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

# ---------- visual defaults ----------------------------------------------
sns.set_theme(context="talk", style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight"
})

# --------------------------------------------------
# 1. NLTK resources
# --------------------------------------------------
for corp in ["punkt", "stopwords", "wordnet",
             "averaged_perceptron_tagger", "omw-1.4"]:
    nltk.download(corp, quiet=True)

# --------------------------------------------------
# 2. Paths & I/O
# --------------------------------------------------
DATA_PATH  = "sports_2024-2025_with_popularity.csv"
OUTPUT_DIR = "title_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 3. Basic engineering
# --------------------------------------------------
def classify_duration(sec: float) -> str:
    if sec <= 240:  return "short"
    if sec <= 1200: return "medium"
    return "long"

df["duration_type"] = df["duration_seconds"].apply(classify_duration)

df["published_at"]  = pd.to_datetime(df["published_at"])
latest_date         = df["published_at"].max()
df["days_since_publish"] = (latest_date - df["published_at"]).dt.days

def plateau_decay(t, a=0.25, t0=20, floor=0.35):
    return np.maximum(1 / (1 + np.exp(a * (t - t0))), floor)

df["time_weight"]    = plateau_decay(df["days_since_publish"])
df["adj_popularity"] = df["popularity_normalized"] * 1e5 * df["time_weight"]

# --------------------------------------------------
# 4. Title preprocessing
# --------------------------------------------------
STOP_WORDS = set(stopwords.words("english")).union({
    "subscribe","channel","follow","like",
    "instagram","facebook","twitter","tiktok",
    "website","click","link","watch","video"
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
    words, wc = tx.split(), 0
    wc = len(words)
    return {
        "word_count": wc,
        "avg_word_length": np.mean([len(w) for w in words]) if wc else 0,
        "unique_word_ratio": len(set(words))/wc if wc else 0,
        "sentiment": TextBlob(tx).sentiment.polarity
    }

df["cleaned_title"]    = df["title"].apply(clean_text)
df["lemmatised_title"] = df["cleaned_title"].apply(lemmatise)
df = pd.concat([df, df["cleaned_title"].apply(text_stats).apply(pd.Series)], axis=1)
df.to_csv(os.path.join(OUTPUT_DIR, "processed_data_with_titles.csv"),
          index=False, encoding="utf-8-sig")

# --------------------------------------------------
# 5. TF‑IDF + LDA topics
# --------------------------------------------------
tfidf_vec = TfidfVectorizer(max_features=1000,
                            stop_words=list(STOP_WORDS),
                            ngram_range=(1,2))
tfidf_mat = tfidf_vec.fit_transform(df["lemmatised_title"])

NUM_TOPICS = 5
lda = LatentDirichletAllocation(n_components=NUM_TOPICS,
                                max_iter=10,
                                learning_method="online",
                                random_state=42)
lda_topics = lda.fit_transform(tfidf_mat)
for i in range(NUM_TOPICS):
    df[f"topic_{i}"] = lda_topics[:, i]

# ---------- topic keyword plot (gradient bars) ---------------------------
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
    fig.suptitle("Top words per LDA topic")
    fig.tight_layout()
    return fig
plot_topics(lda, tfidf_vec.get_feature_names_out())\
    .savefig(os.path.join(OUTPUT_DIR, "topic_keywords.png"))

# --------------------------------------------------
# 6. Random‑Forest per duration
# --------------------------------------------------
TEXT_VARS = ["word_count","avg_word_length","unique_word_ratio","sentiment"] \
            + [f"topic_{i}" for i in range(NUM_TOPICS)]
results = {}

for dur in ["short","medium","long"]:
    sub = df[df["duration_type"]==dur]
    if len(sub) < 20:
        print(f"⚠️  Only {len(sub)} {dur} samples – model skipped.")
        continue

    X,y = sub[TEXT_VARS], sub["adj_popularity"]
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=.2,random_state=42)

    scaler      = StandardScaler()
    X_tr, X_te  = scaler.fit_transform(X_tr), scaler.transform(X_te)
    rf = RandomForestRegressor(n_estimators=200,max_depth=10,
                               min_samples_split=5,random_state=42)
    rf.fit(X_tr,y_tr)
    joblib.dump(rf, os.path.join(OUTPUT_DIR, f"{dur}_title_model.joblib"))

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
    plt.title(f"{dur.capitalize()} – top features")
    plt.xlabel("Importance")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dur}_feature_importance.png"))
    plt.close()

    # Prediction scatter (gradient)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(y_te, y_pred, c=y_te, cmap="viridis", alpha=.7)
    plt.colorbar(sc, label="Actual popularity")
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    plt.plot(lims, lims, "k--")
    plt.title(f"{dur.capitalize()} – predicted vs actual")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dur}_prediction_scatter.png"))
    plt.close()

# --------------------------------------------------
# 7. Word‑clouds (clean ASCII titles + gradient colormap)
# --------------------------------------------------
token_ok  = re.compile(r"^[a-zA-Z]{3,}$").match
def cloud(raw_text, title, fname):
    txt  = " ".join(w for w in raw_text.split() if token_ok(w))
    wc   = WordCloud(width=900, height=450, background_color="white",
                     colormap="viridis",  # gradient!
                     stopwords=STOP_WORDS, max_words=120).generate(txt)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    # force ASCII on the title to eradicate stray symbols
    safe_title = title.encode("ascii", "ignore").decode()
    plt.title(safe_title)
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

hi = " ".join(df[df["adj_popularity"]>df["adj_popularity"].quantile(.75)]
              ["lemmatised_title"])
lo = " ".join(df[df["adj_popularity"]<df["adj_popularity"].quantile(.25)]
              ["lemmatised_title"])
cloud(hi, "High-popularity title keywords", "high_pop_title_wc.png")
cloud(lo, "Low-popularity title keywords",  "low_pop_title_wc.png")

# --------------------------------------------------
# 8. Overview scatter & box (gradient palettes)
# --------------------------------------------------
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.scatterplot(data=df, x="word_count", y="adj_popularity",
                hue="word_count", palette="viridis", alpha=.7, legend=False)
plt.title("Word count vs popularity")
plt.xlabel("Word count"); plt.ylabel("Adjusted popularity")

plt.subplot(2,2,2)
sent_bins = pd.cut(df["sentiment"], bins=5)
palette   = sns.color_palette("coolwarm", n_colors=sent_bins.cat.categories.size)
sns.boxplot(x=sent_bins, y="adj_popularity", data=df, palette=palette)
plt.title("Sentiment vs popularity")
plt.xlabel("Sentiment bin"); plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "overview_plots.png"))
plt.close()

# --------------------------------------------------
# 9. Text report
# --------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "final_title_analysis_report.txt"),
          "w", encoding="utf-8") as f:
    f.write("="*60+"\nTitle Impact on Popularity Report\n"+"="*60+"\n\n")
    f.write(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"Total videos: {len(df)}\n")
    for d in ["short","medium","long"]:
        f.write(f"{d.capitalize()} videos: {len(df[df['duration_type']==d])}\n")
    f.write("\nModel performance:\n")
    for d,m in results.items():
        f.write(f"  {d.capitalize()} – R² {m['r2']:.3f} (±{m['cv_std']:.3f}), "
                f"RMSE {m['rmse']:.2f}, n={m['n']}\n")
    f.write("\nSee PNGs & CSV for details.\n")

print(f"✅ All artefacts saved to '{OUTPUT_DIR}'")
