# Re-import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the re-uploaded dataset
file_path = "/Users/arimartono/Documents/NUS_SOC/Before-the-Camera-Rolls/sports_2024-2025.csv"
df = pd.read_csv(file_path)

# Compute title length
df['title_length'] = df['title'].apply(lambda x: len(str(x)))

# Define engagement metrics
metrics = ['view_count', 'like_count', 'comment_count', 'engagement_rate']

# Create scatter plots
plt.figure(figsize=(16, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(data=df, x='title_length', y=metric)
    plt.title(f'Title Length vs {metric.replace("_", " ").title()}')
    plt.xlabel("Title Length (Characters)")
    plt.ylabel(metric.replace("_", " ").title())

plt.tight_layout()
plt.show()
