import pandas as pd

# 加载数据
df = pd.read_csv('sports_2024-2025.csv', encoding='UTF-8-SIG')

# 定义需要归一化的列
columns_to_normalize = ['view_count', 'like_count', 'comment_count']

# 进行最小-最大归一化
for column in columns_to_normalize:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# 定义权重
weights = {
    'view_count': 0.50,
    'like_count': 0.25,
    'comment_count': 0.25
}

# 计算热度值
df['popularity'] = (df['view_count'] * weights['view_count'] +
              df['like_count'] * weights['like_count'] +
              df['comment_count'] * weights['comment_count'] )

# 对 popularity 列进行最小-最大归一化
df['popularity_normalized'] = (df['popularity'] - df['popularity'].min()) / (df['popularity'].max() - df['popularity'].min())

# 将结果保存为新的 CSV 文件
new_file_path = 'sports_2024-2025_with_popularity.csv'
df.to_csv(new_file_path, index=False, encoding='UTF-8-SIG')