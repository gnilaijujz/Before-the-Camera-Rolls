import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取并预处理
df = pd.read_csv(r"D:\下载\sports_2024-2025_with_popularity.csv")
df['duration_min'] = df['duration_seconds'] / 60

# 2. 业务时长分组
bins = [0, 3, 5, 10, 20, 30, np.inf]
labels = [
    '0-3 min', '3-5 min', '5-10 min',
    '10-20 min', '20-30 min', '30+ min'
]
df['dur_group'] = pd.cut(df['duration_min'], bins=bins, labels=labels, right=False)

# 3. 对每个组再等分成 6 段，标号 0～5
def assign_subbin(x):
    grp = x['dur_group']
    if pd.isna(grp):
        return np.nan
    low, high = grp.left, grp.right
    # 对于最后一个开区间 (30+, inf) 把 inf 临时设为  max+0.001
    if np.isinf(high):
        high = df.loc[df['dur_group']==grp, 'duration_min'].max() + 1e-3
    span = high - low
    # 计算该视频在组内所处的段号
    idx = int(np.floor((x['duration_min'] - low) / span * 6))
    return min(max(idx, 0), 5)

df['sub_bin'] = df.apply(assign_subbin, axis=1)

# 4. 计算「组 × 段」的平均热度
pivot = (
    df
    .groupby(['dur_group', 'sub_bin'])['popularity_normalized']
    .mean()
    .unstack(fill_value=0)
)

# 5. 绘制热力图
plt.figure(figsize=(8, 5))
sns.set_style("white")

ax = sns.heatmap(
    pivot,
    cmap="YlOrRd",           # 浅黄→深红，深色表示平均热度高
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '平均热度'}
)

# 6. 美化：横轴标为 1–6 段，图例中说明每段对应组内时长
ax.set_title("各时长组 6 等分子区间的平均热度", fontsize=14, pad=10)
ax.set_xlabel("子区间序号(内时长等分1–6)", fontsize=12)
ax.set_ylabel("视频时长分组", fontsize=12)
ax.set_xticklabels([f"{i+1}" for i in pivot.columns], rotation=0)
ax.set_yticklabels(pivot.index, rotation=0)
plt.tight_layout()
plt.show()