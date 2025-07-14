import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ——— CONFIG ———
# → point this at the real location of your CSV on your Mac:
file_path = "/Users/arimartono/Documents/NUS_SOC/Before-the-Camera-Rolls/sports_2024-2025.csv"

# ——— LOAD & PREP ———
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV not found at {file_path}")

df = pd.read_csv(file_path)

# compute title length
df['title_length'] = df['title'].astype(str).str.len()

# categorize into Short / Medium / Long
def title_cat(n):
    if n <= 20:   return "Short"
    if n <= 60:   return "Medium"
    return "Long"

df['title_category'] = df['title_length'].apply(title_cat)

# ——— AGGREGATE ———
grouped = (
    df
    .groupby('title_category')
    [['view_count','like_count','comment_count','engagement_rate']]
    .mean()
    .reindex(['Short','Medium','Long'])   # ensure order
)

# ——— PLOT ———
plt.figure(figsize=(8,6))
sns.heatmap(
    grouped,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    cbar_kws={'label': 'Average Value'}
)
plt.title("Average Engagement by Title Length Category")
plt.xlabel("Engagement Metric")
plt.ylabel("Title Length Category")
plt.tight_layout()
plt.show()
