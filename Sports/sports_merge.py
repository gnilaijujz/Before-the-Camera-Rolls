import pandas as pd

# === 1. 读取四个季度的文件 ===
files = ['sports_q1.csv', 'sports_q2.csv', 'sports_q3.csv', 'sports_q4.csv','sports_q1_2025.csv','sports_q2_2025.csv']
dfs = [pd.read_csv(file) for file in files]

# === 2. 合并所有数据 ===
merged_df = pd.concat(dfs, ignore_index=True)

# === 3. 按 video_id 去重（保留第一个出现的） ===
merged_df = merged_df.drop_duplicates(subset='video_id', keep='first')

# === 4. 保存为新文件 ===
merged_df.to_csv('sports_2024-2025.csv', index=False, encoding='utf-8-sig')

print(f"✅ 合并完成，共 {len(merged_df)} 条视频，已保存为 sports_2024-2025.csv")
