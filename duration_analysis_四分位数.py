import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\下载\sports_2024-2025_with_popularity.csv")
df['duration_min'] = df['duration_seconds'] / 60

# 1. 计算 0%、25%、50%、75%、100% 分位点
q = df['duration_min'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values

# 2. 构造 “下限–上限” 的标签
labels = [f"{q[i]:.2f}–{q[i+1]:.2f}" for i in range(len(q)-1)]

# 3. 用 pd.cut 做分箱
df['group_quantile_custom'] = pd.cut(
    df['duration_min'],
    bins=q,
    labels=labels,
    include_lowest=True,  # 第一档包含下界
    right=True           # 默认右开区间 (a, b]
)

# 4. 统计
counts_custom = df['group_quantile_custom'].value_counts().sort_index()
print(counts_custom)


