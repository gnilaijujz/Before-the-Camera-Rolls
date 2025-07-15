import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.neural_network import MLPRegressor
import shap

# Create output directory
output_dir = "sports_video_analysis_tags_only_refined" # Changed output directory again
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv('sports_2024-2025_with_popularity.csv')

# Video duration classification
def classify_duration(duration):
    if duration <= 240:
        return 'short'
    elif 240 < duration <= 1200:
        return 'medium'
    else:
        return 'long'

df['duration_type'] = df['duration_seconds'].apply(classify_duration)

# Time weight calculation (plateau decay) - Still calculated for 'adj_popularity' but not used as a feature
df['published_at'] = pd.to_datetime(df['published_at'])
latest_date = df['published_at'].max()
df['days_since_publish'] = (latest_date - df['published_at']).dt.days

def plateau_decay(t, early_a=0.25, t0=20, floor=0.35):
    decay = 1 / (1 + np.exp(early_a * (t - t0)))
    return np.maximum(decay, floor)

df['time_weight'] = plateau_decay(df['days_since_publish'])
df['adj_popularity'] = df['popularity_normalized']* 100000 * df['time_weight']

# Tag cleaning and vectorizations
def clean_tags(tag_str):
    if pd.isna(tag_str):
        return ""
    return re.sub(r'[^a-zA-Z0-9,\s]', '', tag_str.lower())

df['cleaned_tags'] = df['tags'].apply(clean_tags)

# --- ADJUSTMENT 1: Increase max_features for TF-IDF ---
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') # Increased from 500 to 1000
tag_matrix = vectorizer.fit_transform(df['cleaned_tags'])

# Tag clustering (KMeans)
# --- ADJUSTMENT 2: Increase n_clusters for KMeans ---
kmeans_n_clusters = 50 # Increased from 20 to 50
kmeans = KMeans(n_clusters=kmeans_n_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(tag_matrix)

def get_cluster_distribution(text):
    vec = vectorizer.transform([text])
    label = kmeans.predict(vec)[0]
    dist = np.zeros(kmeans.n_clusters)
    dist[label] = 1
    return dist

cluster_features = np.vstack(df['cleaned_tags'].apply(get_cluster_distribution))
cluster_feature_names = [f'cluster_{i}' for i in range(kmeans_n_clusters)]
cluster_df = pd.DataFrame(cluster_features, columns=cluster_feature_names)
df = pd.concat([df.reset_index(drop=True), cluster_df], axis=1)

processed_csv_path = os.path.join(output_dir, "processed_sports_videos.csv")
df.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Processed data saved to: {processed_csv_path}")

# Model training with Hyperparameter Tuning (using only tag features)
results = {}
models = {}
feature_importances = {}
best_params_storage = {}

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# --- ADJUSTMENT 3: Refine parameter grids for GridSearchCV ---
# Broader ranges and potentially new parameters for better exploration
'''
param_grid_xgb_short = {
    'n_estimators': [150, 250, 350], # Expanded range
    'learning_rate': [0.03, 0.05, 0.07],
    'max_depth': [4, 5, 6] # Slightly adjusted max_depth
}
'''

param_grid_lgbm_medium = {
    'n_estimators': [80, 120, 160], # Expanded range
    'learning_rate': [0.03, 0.05, 0.07],
    'num_leaves': [20, 31, 50], # Expanded num_leaves
    'min_child_samples': [10, 20, 30] # Added min_child_samples for regularization
}

param_grid_xgb_long = {
    'n_estimators': [100, 150, 200], # Expanded range
    'learning_rate': [0.03, 0.05, 0.07],
    'max_depth': [2, 3, 4],
    'reg_lambda': [0.8, 1.0, 1.2, 1.5] # Expanded reg_lambda
}

model_initializers = {
    #'short': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, objective='reg:squarederror'), # Specify objective
    'medium': LGBMRegressor(random_state=42, objective='regression'), # Specify objective
    'long': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, objective='reg:squarederror') # Specify objective
}

param_grids = {
    #'short': param_grid_xgb_short,
    'medium': param_grid_lgbm_medium,
    'long': param_grid_xgb_long
}


for dtype in ['short', 'medium', 'long']:
    print(f"\n===== Modeling {dtype} videos =====")
    subset = df[df['duration_type'] == dtype]
    features = cluster_feature_names
    X = subset[features]
    y = subset['adj_popularity']

    if len(subset) < 20:
        print(f"⚠️ Not enough samples for {dtype} videos ({len(subset)}), skipping modeling and tuning.")
        continue

    if len(y.unique()) < 2:
        print(f"⚠️ Not enough unique target values for {dtype} videos, skipping modeling.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ 替代 short 部分为 MLP + SHAP
    if dtype == 'short':
        print("🚀 Training MLPRegressor for short videos...")
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                           solver='adam', max_iter=500, early_stopping=True,
                           random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(mlp, X_train, y_train, cv=5, scoring='r2')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"✅ MLP RMSE: {rmse:.4f}")
        print(f"✅ MLP R²: {r2:.4f}")
        print(f"✅ MLP CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

        results[dtype] = {
            'rmse': rmse,
            'r2': r2,
            'cv_r2': cv_mean,
            'cv_std': cv_std,
            'num_samples': len(subset),
            'best_params': {'hidden_layer_sizes': (128, 64), 'activation': 'relu'}
        }
        models[dtype] = mlp

        # ✅ SHAP 可解释性分析
        print("🔍 Generating SHAP values for short videos...")
        explainer = shap.KernelExplainer(mlp.predict, shap.kmeans(X_train, 10))
        shap_values = explainer.shap_values(X_test[:100], nsamples=100)

        shap_output_path = os.path.join(output_dir, f"{dtype}_shap_summary.png")
        plt.figure()
        shap.summary_plot(shap_values, X_test[:100], feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig(shap_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved SHAP summary plot to: {shap_output_path}")

        # ✅ 散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='darkorange')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.title(f"{dtype.capitalize()} Videos - Predicted vs. Actual (MLP + Tags)")
        plt.xlabel("Actual Popularity")
        plt.ylabel("Predicted Popularity")
        plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes)
        scatter_img_path = os.path.join(output_dir, f"{dtype}_prediction_scatter_mlp_tags.png")
        plt.savefig(scatter_img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved prediction scatter plot to: {scatter_img_path}")

        continue  # skip rest

    # ✅ medium 和 long 用原有 GridSearchCV 模型
    model_base = model_initializers[dtype]
    param_grid = param_grids[dtype]
    grid_search = GridSearchCV(estimator=model_base, param_grid=param_grid,
                               scoring='r2', cv=min(5, len(X_train)), n_jobs=-1, verbose=1)
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ Error during GridSearchCV for {dtype} videos: {e}")
        results[dtype] = {'rmse': np.nan, 'r2': np.nan, 'cv_r2': np.nan, 'cv_std': np.nan, 'num_samples': len(subset), 'best_params': "N/A"}
        continue

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), scoring='r2', n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    results[dtype] = {
        'rmse': rmse,
        'r2': r2,
        'cv_r2': cv_mean,
        'cv_std': cv_std,
        'num_samples': len(subset),
        'best_params': best_params
    }
    models[dtype] = model

    # Feature importance plot
    feature_names = features
    if hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(feature_names):
        feat_importances = pd.Series(model.feature_importances_, index=feature_names)
        
        top_features = feat_importances.nlargest(10)

        plt.figure(figsize=(10, 6))
        ax = top_features.sort_values().plot.barh(color='royalblue')
        plt.title(f"{dtype.capitalize()} Videos - Top 10 Tag Cluster Feature Importances (Refined Model)")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        for i, v in enumerate(top_features.sort_values()):
            ax.text(v + 0.001, i, f"{v:.4f}", color='black', va='center')
        importance_img_path = os.path.join(output_dir, f"{dtype}_tag_cluster_importance_model_refined.png")
        plt.savefig(importance_img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved feature importance plot to: {importance_img_path}")

        feature_importances[dtype] = top_features
        importance_csv_path = os.path.join(output_dir, f"{dtype}_tag_cluster_importance_model_refined.csv")
        top_features.to_csv(importance_csv_path, header=['importance'], encoding='utf-8-sig')
        print(f"✅ Saved feature importances to CSV: {importance_csv_path}")
    else:
        print("⚠️ Feature count and importances do not match or model has no feature_importances_.")

    # Prediction scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.title(f"{dtype.capitalize()} Videos - Predicted vs. Actual Popularity (Tags Only, Refined Model)")
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes)
    scatter_img_path = os.path.join(output_dir, f"{dtype}_prediction_scatter_tags_only_refined.png")
    plt.savefig(scatter_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved prediction scatter plot to: {scatter_img_path}")


# Tag Recommendation System (based on cluster centers) - This part remains robust
class TagRecommender:
    def __init__(self, vectorizer, kmeans, df):
        self.vectorizer = vectorizer
        self.kmeans = kmeans
        self.df = df
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
        cleaned_input = clean_tags(user_input)
        input_vec = self.vectorizer.transform([cleaned_input])
        
        if input_vec.shape[0] == 0 or input_vec.sum() == 0:
            print("Warning: User input did not result in a valid vector. Cannot provide recommendations.")
            return pd.DataFrame(columns=['tag', 'score'])

        input_cluster = self.kmeans.predict(input_vec)[0]
        input_center = self.kmeans.cluster_centers_[input_cluster]

        similarities = []
        for tag, popularity in self.tag_popularity.items():
            tag_vec = self.vectorizer.transform([tag])
            if tag_vec.shape[0] == 0 or tag_vec.sum() == 0:
                continue

            tag_vec_flat = tag_vec.toarray().flatten()

            try:
                max_len = max(len(input_center), len(tag_vec_flat))
                padded_input_center = np.pad(input_center, (0, max_len - len(input_center)), 'constant')
                padded_tag_vec = np.pad(tag_vec_flat, (0, max_len - len(tag_vec_flat)), 'constant')

                sim = 1 - cosine(padded_input_center, padded_tag_vec)
                similarities.append((tag, sim, popularity))
            except Exception as e:
                continue

        rec_df = pd.DataFrame(similarities, columns=['tag', 'similarity', 'avg_popularity'])

        duration_tags = set()
        for tags in self.df[self.df['duration_type'] == duration_type]['cleaned_tags']:
            if pd.notna(tags):
                duration_tags.update(tags.split(','))
        rec_df = rec_df[rec_df['tag'].isin(duration_tags)]

        if not rec_df.empty:
            rec_df['similarity_normalized'] = rec_df['similarity'] / rec_df['similarity'].max()
            rec_df['popularity_normalized_rec'] = rec_df['avg_popularity'] / rec_df['avg_popularity'].max()

            rec_df['score'] = (0.6 * rec_df['similarity_normalized'] +
                               0.4 * rec_df['popularity_normalized_rec'])
            return rec_df.nlargest(top_n, 'score')
        else:
            return pd.DataFrame(columns=['tag', 'score'])

# Example recommendation
recommender = TagRecommender(vectorizer, kmeans, df)
user_query = "I want to create an exciting basketball highlight video"
duration_type = "short"
recommendations = recommender.recommend_tags(user_query, duration_type)

if not recommendations.empty:
    rec_csv_path = os.path.join(output_dir, "tag_recommendations.csv")
    recommendations.to_csv(rec_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved tag recommendations to: {rec_csv_path}")

    # Visualize recommendations
    plt.figure(figsize=(10, 6))
    ax = recommendations.sort_values('score').plot.barh(x='tag', y='score', color='forestgreen')
    plt.title("Top Tag Recommendations", fontsize=14)
    plt.xlabel("Recommendation Score", fontsize=12)
    plt.ylabel("Tag", fontsize=12)

    for i, v in enumerate(recommendations.sort_values('score')['score']):
        ax.text(v + 0.01, i, f"{v:.4f}", color='black', va='center')

    rec_img_path = os.path.join(output_dir, "tag_recommendations.png")
    plt.savefig(rec_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved tag recommendation plot to: {rec_img_path}")
else:
    print("\nNo matching tag recommendations found.")

# Create and save model performance report
report_path = os.path.join(output_dir, "model_performance_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("Sports Video Popularity Prediction Model Performance Report (Tags Only Features, Refined)\n")
    f.write("="*60 + "\n\n")
    f.write(f"Report Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Videos: {len(df)}\n")
    f.write(f"Short Videos: {len(df[df['duration_type']=='short'])}\n")
    f.write(f"Medium Videos: {len(df[df['duration_type']=='medium'])}\n")
    f.write(f"Long Videos: {len(df[df['duration_type']=='long'])}\n\n")

    f.write("Model Performance Evaluation (Tuned Models with Tags Only):\n")
    f.write("-"*50 + "\n")
    for dtype, metrics in results.items():
        f.write(f"{dtype.capitalize()} Videos:\n")
        f.write(f"  Number of Samples: {metrics['num_samples']}\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"  R²: {metrics['r2']:.4f}\n")
        f.write(f"  Cross-validation R² (on training set with best model): {metrics['cv_r2']:.4f} ± {metrics['cv_std']:.4f}\n")
        f.write(f"  Best Hyperparameters: {metrics['best_params']}\n\n")

    f.write("\nKey Insights:\n")
    f.write("-"*50 + "\n")
    f.write("1. Video Duration Classification Strategy:\n")
    f.write("   - Short videos (<4 minutes): Suitable for fast-paced content (highlights/snippets)\n")
    f.write("   - Medium videos (4-20 minutes): Tutorials/match compilations\n")
    f.write("   - Long videos (>20 minutes): In-depth analysis/documentaries\n")
    f.write("2. Time Decay Model:\n")
    f.write("   - Plateau decay function applied to adjust popularity based on age.\n")
    f.write("   - Note: 'time_weight' is used for 'adj_popularity' calculation but *not* as an independent variable for prediction.\n")
    f.write("3. Tag Recommendation System (KMeans Clustering):\n")
    f.write("   - Integrated score = 0.6 × Similarity + 0.4 × Normalized Popularity\n")
    f.write("   - Balances semantic relevance from cluster centers and historical performance.\n")
    f.write("4. Hyperparameter Tuning:\n")
    f.write("   - GridSearchCV was used to find optimal parameters for XGBoost and LightGBM models.\n")
    f.write("   - Models are trained exclusively on tag-based features. Refined search space and increased TF-IDF features/KMeans clusters were applied.\n\n")

    if not recommendations.empty:
        f.write("\nExample Tag Recommendations:\n")
        f.write("-"*50 + "\n")
        f.write(f"Query: '{user_query}'\n")
        f.write(f"Video Type: {duration_type}\n")
        f.write("Recommendations:\n")
        for i, row in recommendations.iterrows():
            f.write(f"  {i+1}. {row['tag']} (Score: {row['score']:.4f})\n")

print(f"✅ Saved model performance report to: {report_path}")

# Create and save data analysis report
analysis_path = os.path.join(output_dir, "data_analysis_report.txt")
with open(analysis_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("Sports Video Data Analysis Report\n")
    f.write("="*60 + "\n\n")

    # Video duration distribution
    duration_counts = df['duration_type'].value_counts()
    f.write("Video Duration Distribution:\n")
    f.write("-"*50 + "\n")
    for dtype, count in duration_counts.items():
        f.write(f"  {dtype.capitalize()} Videos: {count} ({count/len(df)*100:.1f}%)\n")

    # Popularity analysis
    f.write("\nVideo Popularity Analysis:\n")
    f.write("-"*50 + "\n")
    f.write(f"  Original Average Popularity: {df['popularity_normalized'].mean():.4f}\n")
    f.write(f"  Time-Adjusted Average Popularity: {df['adj_popularity'].mean():.4f}\n")

    # Tag analysis
    all_tags = []
    for tags in df['cleaned_tags']:
        if pd.notna(tags):
            all_tags.extend(tags.split(','))

    tag_counts = pd.Series(all_tags).value_counts().head(20)
    f.write("\nMost Common Tags (Top 20):\n")
    f.write("-"*50 + "\n")
    for tag, count in tag_counts.items():
        f.write(f"  {tag}: {count} times\n")

print(f"✅ Saved data analysis report to: {analysis_path}")

# Create overall visualization
plt.figure(figsize=(12, 8))

# Video duration distribution
plt.subplot(2, 2, 1)
duration_counts = df['duration_type'].value_counts()
plt.pie(duration_counts, labels=duration_counts.index, autopct='%1.1f%%',
        colors=['#66c2a5', '#fc8d62', '#8da0cb'])
plt.title("Video Duration Distribution")

# Popularity distribution
plt.subplot(2, 2, 2)
sns.histplot(df['adj_popularity'], bins=30, kde=True, color='#66c2a5')
plt.title("Time-Adjusted Popularity Distribution")
plt.xlabel("Popularity")

# Duration vs. Popularity
plt.subplot(2, 2, 3)
sns.boxplot(x='duration_type', y='adj_popularity', data=df,
           palette=['#66c2a5', '#fc8d62', '#8da0cb'])
plt.title("Popularity by Video Duration Type")
plt.xlabel("Video Type")
plt.ylabel("Popularity")

# Publish time vs. Popularity
plt.subplot(2, 2, 4)
sns.scatterplot(x='days_since_publish', y='adj_popularity', data=df,
               alpha=0.6, color='#66c2a5')
plt.title("Popularity vs. Days Since Publish")
plt.xlabel("Days Since Publish")
plt.ylabel("Popularity")
plt.gca().invert_xaxis()

plt.tight_layout()
overview_img_path = os.path.join(output_dir, "data_overview.png")
plt.savefig(overview_img_path, dpi=300)
plt.close()
print(f"✅ Saved data overview image to: {overview_img_path}")

print("\n" + "="*50)
print("Analysis Complete! All results saved to directory:", output_dir)
print("="*50)
