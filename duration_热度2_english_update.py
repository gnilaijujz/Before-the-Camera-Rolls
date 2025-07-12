import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']  # English fonts
plt.rcParams['axes.unicode_minus'] = False    # Properly display minus sign
plt.rcParams['savefig.dpi'] = 300  # High resolution save
plt.rcParams['figure.dpi'] = 100   # Display resolution

# 1. Read and preprocess data
df = pd.read_csv(r"D:\下载\sports_2024-2025_with_popularity.csv")
df['duration_min'] = df['duration_seconds'] / 60

print(f"Total videos in dataset: {len(df)}")
print(f"Popularity range: {df['popularity_normalized'].min():.4f} - {df['popularity_normalized'].max():.4f}")
print(f"Duration range: {df['duration_min'].min():.1f} - {df['duration_min'].max():.1f} minutes")

# 2. Duration grouping
bins = [0, 3, 5, 10, 20, 30, np.inf]
labels = [
    '0-3 min', '3-5 min', '5-10 min',
    '10-20 min', '20-30 min', '30+ min'
]

# Keep Interval object to access left and right attributes
df['dur_group_interval'] = pd.cut(df['duration_min'], bins=bins, right=False)
# Save string labels for display
df['dur_group'] = pd.cut(df['duration_min'], bins=bins, labels=labels, right=False)

# Print count and percentage for each duration group
dur_group_counts = df['dur_group'].value_counts().sort_index()
dur_group_percent = dur_group_counts / len(df) * 100
print("\nVideo count and percentage by duration group:")
for group, count in dur_group_counts.items():
    percent = dur_group_percent[group]
    print(f"{group}: {count} videos ({percent:.1f}%)")

# 3. Divide each group into 6 sub-intervals, labeled 0-5
def assign_subbin(x):
    grp_interval = x['dur_group_interval']
    if pd.isna(grp_interval):
        return np.nan

    low, high = grp_interval.left, grp_interval.right
    # For the last open interval (30+, inf), set inf to max+0.001
    if np.isinf(high):
        high = df.loc[df['dur_group'] == x['dur_group'], 'duration_min'].max() + 1e-3

    span = high - low
    # Calculate segment number within the group
    idx = int(np.floor((x['duration_min'] - low) / span * 6))
    return min(max(idx, 0), 5)

df['sub_bin'] = df.apply(assign_subbin, axis=1)

# Calculate actual duration range for each sub-interval for terminal output
def get_subbin_ranges(dur_group_interval, n_bins=6):
    """Calculate actual duration range for each sub-interval in a duration group"""
    low, high = dur_group_interval.left, dur_group_interval.right
    if np.isinf(high):
        high = df.loc[df['dur_group_interval'] == dur_group_interval, 'duration_min'].max() + 1e-3

    ranges = []
    span = high - low
    for i in range(n_bins):
        bin_low = low + i * span / n_bins
        bin_high = low + (i + 1) * span / n_bins
        ranges.append((bin_low, bin_high))
    return ranges

# Output sub-interval ranges to terminal
print("\nEach duration group is divided into 6 equal sub-intervals, deeper color indicates higher popularity")
print("\nActual duration ranges for sub-intervals:")
for dur_group in df['dur_group'].unique():
    # Get corresponding interval object
    interval = df.loc[df['dur_group'] == dur_group, 'dur_group_interval'].iloc[0]

    ranges = get_subbin_ranges(interval)
    print(f"\n{dur_group}:")
    for j, (low, high) in enumerate(ranges):
        # Handle infinity
        high_str = f"{high:.1f}" if high < 1000 else "max"
        print(f"  Segment {j+1}: {low:.1f}-{high_str} minutes")

# 4. Calculate average popularity for each 'group × segment'
pivot = (
    df
    .groupby(['dur_group', 'sub_bin'])['popularity_normalized']
    .mean()
    .unstack(fill_value=0)
)

# Calculate video count for each cell for terminal output
count_pivot = (
    df
    .groupby(['dur_group', 'sub_bin']).size()
    .unstack(fill_value=0)
)

# Output popularity values and video counts to terminal
print("\nPopularity values and video counts by duration group and sub-interval:")
for i, dur_group in enumerate(pivot.index):
    print(f"\n{dur_group}:")
    for j in range(6):
        heat_value = pivot.iloc[i, j]
        count = count_pivot.iloc[i, j]
        print(f"  Segment {j+1}: Popularity {heat_value:.4f}, {count} videos")

# 5. Create custom color map - attractive yellow-red gradient
colors = ["#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
custom_cmap = LinearSegmentedColormap.from_list("custom_heat", colors)

# 6. Draw simplified heatmap
plt.figure(figsize=(14, 10))

# Draw main heatmap
ax = sns.heatmap(
    pivot,
    cmap=custom_cmap,
    annot=True,
    fmt=".3f",
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': 'Average Popularity', 'shrink': 0.8}
)

# 7. Add appropriate title
title_text = "Popularity Distribution in Duration Sub-intervals"
plt.title(title_text, fontsize=16, fontweight='bold', pad=15)

# Remove x-axis labels, keep y-axis labels
ax.set_xticklabels([])
ax.set_yticklabels(pivot.index, rotation=0, fontsize=12)

# Remove axis labels
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig('duration_heat_subintervals_clean.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Create additional bar chart - comparing overall popularity across duration groups
plt.figure(figsize=(12, 6))

# Calculate average popularity for each duration group
duration_heat_mean = df.groupby('dur_group')['popularity_normalized'].mean().reset_index()
duration_heat_mean = duration_heat_mean.sort_values('popularity_normalized', ascending=False)

# Set gradient colors for bar chart
bars = plt.bar(
    duration_heat_mean['dur_group'], 
    duration_heat_mean['popularity_normalized'], 
    color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(duration_heat_mean)))
)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, height + 0.0005,
        f'{height:.4f}', ha='center', va='bottom', 
        fontsize=11, fontweight='bold'
    )

plt.title('Average Popularity by Duration Group', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Duration Group', fontsize=12, labelpad=10)
plt.ylabel('Average Popularity', fontsize=12, labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('duration_heat_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show()
