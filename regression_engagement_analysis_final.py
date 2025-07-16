
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.stats import pearsonr
from scipy import stats

def analyze_engagement_metrics(file_path=r"D:\下载\sports_2024-2025.csv.xlsx"):
    """
    Analyze relationships between video engagement metrics (like_count, view_count, comment_count)

    Parameters:
        file_path: Excel file path containing video data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist!")
        return

    print(f"Processing file: {file_path}")

    # Read data
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check if required columns exist
    required_columns = ['like_count', 'view_count', 'comment_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: File is missing required columns: {', '.join(missing_columns)}")
        return

    # Data cleaning
    print(f"Original row count: {len(df)}")

    # Handle missing values in engagement metrics
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where all engagement metrics are missing
    df = df.dropna(subset=required_columns, how='all')
    print(f"Row count after cleaning: {len(df)}")

    # Clip extreme values to avoid outliers skewing the analysis (keep only 99.5% of data)
    for col in required_columns:
        threshold = df[col].quantile(0.995)
        df[col+'_filtered'] = df[col].clip(upper=threshold)

    # 使用提供的七色系方案 - 选择编号1,2,3的颜色
    color_palette = {
        'deep_sea_blue': (8/255, 51/255, 110/255),    # 1: 深海蓝，沉稳冷静
        'indigo_blue': (16/255, 92/255, 164/255),    # 2: 靛蓝，科技感强
        'blue_gray': (56/255, 136/255, 192/255)      # 3: 蓝灰，理性柔和
    }

    # 用于保存关系强度分析
    relationship_strength = []

    # Set up figure for correlation analysis
    plt.figure(figsize=(22, 16))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Views vs Likes Scatter Plot - 使用深海蓝
    plt.subplot(2, 2, 1)

    # 添加散点图
    sns.scatterplot(x='view_count_filtered', y='like_count_filtered', data=df, alpha=0.7,
                   color=color_palette['deep_sea_blue'], s=70)

    # 添加回归线及其95%置信区间
    sns.regplot(x='view_count_filtered', y='like_count_filtered', data=df, 
                scatter=False, ci=95, line_kws={"color": color_palette['deep_sea_blue'], 
                                            "lw": 2, "linestyle": "-"})

    plt.title('Views vs Likes Relationship', fontsize=18, fontweight='bold', color=color_palette['deep_sea_blue'], pad=15)
    plt.xlabel('View Count', fontsize=14, fontweight='bold')
    plt.ylabel('Like Count', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Calculate and display correlation coefficient and p-value
    corr_views_likes, p_value_vl = pearsonr(df['view_count_filtered'].fillna(0), df['like_count_filtered'].fillna(0))

    # 计算线性回归的斜率和截距
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['view_count_filtered'].fillna(0), 
                                                                 df['like_count_filtered'].fillna(0))

    # 添加相关系数、p值和回归方程
    plt.annotate(f'''Correlation: {corr_views_likes:.2f}
y = {slope:.4f}x + {intercept:.2f}''',
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=14, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8,
                      ec=color_palette['deep_sea_blue'], lw=2))


    # 保存关系强度分析
    strength_vl = "Strong" if abs(corr_views_likes) > 0.7 else "Moderate" if abs(corr_views_likes) > 0.3 else "Weak"
    significance_vl = "Significant" if p_value_vl < 0.05 else "Not significant"
    relationship_strength.append(f"Views-Likes: {strength_vl} correlation ({corr_views_likes:.2f}), {significance_vl} (p={p_value_vl:.4f})")

    # 2. Views vs Comments Scatter Plot - 使用靛蓝
    plt.subplot(2, 2, 2)

    # 添加散点图
    sns.scatterplot(x='view_count_filtered', y='comment_count_filtered', data=df, alpha=0.7,
                   color=color_palette['indigo_blue'], s=70)

    # 添加回归线及其95%置信区间
    sns.regplot(x='view_count_filtered', y='comment_count_filtered', data=df, 
                scatter=False, ci=95, line_kws={"color": color_palette['indigo_blue'], 
                                            "lw": 2, "linestyle": "-"})

    plt.title('Views vs Comments Relationship', fontsize=18, fontweight='bold', color=color_palette['indigo_blue'], pad=15)
    plt.xlabel('View Count', fontsize=14, fontweight='bold')
    plt.ylabel('Comment Count', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Calculate and display correlation coefficient and p-value
    corr_views_comments, p_value_vc = pearsonr(df['view_count_filtered'].fillna(0), df['comment_count_filtered'].fillna(0))

    # 计算线性回归的斜率和截距
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['view_count_filtered'].fillna(0), 
                                                                 df['comment_count_filtered'].fillna(0))

    # 添加相关系数、p值和回归方程
    plt.annotate(f'''Correlation: {corr_views_comments:.2f}
y = {slope:.4f}x + {intercept:.2f}''',
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=14, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8,
                      ec=color_palette['indigo_blue'], lw=2))


    # 保存关系强度分析
    strength_vc = "Strong" if abs(corr_views_comments) > 0.7 else "Moderate" if abs(corr_views_comments) > 0.3 else "Weak"
    significance_vc = "Significant" if p_value_vc < 0.05 else "Not significant"
    relationship_strength.append(f"Views-Comments: {strength_vc} correlation ({corr_views_comments:.2f}), {significance_vc} (p={p_value_vc:.4f})")

    # 3. Likes vs Comments Scatter Plot - 使用蓝灰
    plt.subplot(2, 2, 3)

    # 添加散点图
    sns.scatterplot(x='like_count_filtered', y='comment_count_filtered', data=df, alpha=0.7,
                   color=color_palette['blue_gray'], s=70)

    # 添加回归线及其95%置信区间
    sns.regplot(x='like_count_filtered', y='comment_count_filtered', data=df, 
                scatter=False, ci=95, line_kws={"color": color_palette['blue_gray'], 
                                            "lw": 2, "linestyle": "-"})

    plt.title('Likes vs Comments Relationship', fontsize=18, fontweight='bold', color=color_palette['blue_gray'], pad=15)
    plt.xlabel('Like Count', fontsize=14, fontweight='bold')
    plt.ylabel('Comment Count', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Calculate and display correlation coefficient and p-value
    corr_likes_comments, p_value_lc = pearsonr(df['like_count_filtered'].fillna(0), df['comment_count_filtered'].fillna(0))

    # 计算线性回归的斜率和截距
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['like_count_filtered'].fillna(0), 
                                                                 df['comment_count_filtered'].fillna(0))

    # 添加相关系数、p值和回归方程
    plt.annotate(f'''Correlation: {corr_likes_comments:.2f}
y = {slope:.4f}x + {intercept:.2f}''',
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=14, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8,
                      ec=color_palette['blue_gray'], lw=2))


    # 保存关系强度分析
    strength_lc = "Strong" if abs(corr_likes_comments) > 0.7 else "Moderate" if abs(corr_likes_comments) > 0.3 else "Weak"
    significance_lc = "Significant" if p_value_lc < 0.05 else "Not significant"
    relationship_strength.append(f"Likes-Comments: {strength_lc} correlation ({corr_likes_comments:.2f}), {significance_lc} (p={p_value_lc:.4f})")

    # 4. Correlation Heatmap - 使用从深到浅的蓝色渐变
    plt.subplot(2, 2, 4)

    # Rename columns for better display in heatmap
    correlation_matrix = df[['view_count_filtered', 'like_count_filtered', 'comment_count_filtered']].corr()
    correlation_matrix.index = ['Views', 'Likes', 'Comments']
    correlation_matrix.columns = ['Views', 'Likes', 'Comments']

    # 使用七色系中的深海蓝、靛蓝、蓝灰创建自定义色谱
    custom_cmap = sns.color_palette([color_palette['deep_sea_blue'],
                                     color_palette['indigo_blue'],
                                     color_palette['blue_gray']], as_cmap=True)

    sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, vmin=0, vmax=1,
                linewidths=.5, cbar_kws={"shrink": .8},
                annot_kws={"size": 16, "weight": "bold", "color": "white"})
    plt.title('Engagement Metrics Correlation Heatmap', fontsize=18, fontweight='bold', color=color_palette['indigo_blue'], pad=15)

    # Add overall title with more space
    plt.suptitle('Analysis of Video Engagement Metrics Relationships', fontsize=24, fontweight='bold', y=0.96, color=color_palette['deep_sea_blue'])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save chart
    output_path = "video_engagement_analysis_with_regression.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Engagement metrics relationship chart saved to: {output_path}")

    # 输出关系强度分析
    print("\n=== Relationship Strength Analysis ===")
    for analysis in relationship_strength:
        print(analysis)

    # Analyze the relationship between duration and engagement metrics
    if 'duration_seconds' in df.columns:
        # Convert to minutes and create duration categories
        df['duration_min'] = df['duration_seconds'] / 60

        def classify_duration(minutes):
            if pd.isna(minutes):
                return None
            if minutes < 4:
                return 'Short (0-4 min)'
            elif minutes < 20:
                return 'Medium (4-20 min)'
            else:
                return 'Long (20+ min)'

        df['duration_category'] = df['duration_min'].apply(classify_duration)

        # Set up figure for duration-engagement analysis
        plt.figure(figsize=(20, 12))

        # 使用提供的三种蓝色
        bar_colors = [
            color_palette['deep_sea_blue'],  # 深海蓝
            color_palette['indigo_blue'],    # 靛蓝
            color_palette['blue_gray']       # 蓝灰
        ]

        # 1. Average engagement metrics by duration category (normalized)
        metrics = ['view_count', 'like_count', 'comment_count']
        metrics_display = ['Views', 'Likes', 'Comments']
        agg_data = df.groupby('duration_category')[metrics].mean().reset_index()

        # Handle potential null values
        agg_data = agg_data.fillna(0)

        # Create bar chart comparing average engagement metrics across duration categories
        plt.subplot(1, 2, 1)

        # Create three bars for each category
        bar_width = 0.25
        index = np.arange(len(agg_data))

        # Normalize data to make metrics comparable
        for i, metric in enumerate(metrics):
            # Normalize to make all metrics visible on same scale
            max_val = agg_data[metric].max()
            if max_val > 0:  # Avoid division by zero
                normalized = agg_data[metric] / max_val

                plt.bar(index + i*bar_width - bar_width,
                        normalized,
                        bar_width,
                        alpha=0.9,
                        color=bar_colors[i],
                        label=f'{metrics_display[i]} (Normalized)')

                # Add original values as labels
                for j, v in enumerate(agg_data[metric]):
                    plt.text(j + i*bar_width - bar_width,
                            normalized[j] + 0.02,
                            f'{int(v)}',
                            ha='center', va='bottom',
                            fontweight='bold', fontsize=10,
                            color='black')

        plt.xticks(index, agg_data['duration_category'], fontsize=12, fontweight='bold')
        plt.title('Average Engagement Metrics by Duration Category (Normalized)', fontsize=18, fontweight='bold', color=color_palette['deep_sea_blue'], pad=15)
        plt.ylabel('Normalized Value', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 2. Engagement efficiency metrics by duration category
        plt.subplot(1, 2, 2)

        # Calculate efficiency metrics
        df['likes_per_view'] = df['like_count'] / df['view_count'].replace(0, np.nan)
        df['comments_per_view'] = df['comment_count'] / df['view_count'].replace(0, np.nan)
        df['comments_per_like'] = df['comment_count'] / df['like_count'].replace(0, np.nan)

        # Group by duration category and calculate mean efficiency metrics
        efficiency_metrics = ['likes_per_view', 'comments_per_view', 'comments_per_like']
        efficiency_display = ['Likes per View', 'Comments per View', 'Comments per Like']
        eff_data = df.groupby('duration_category')[efficiency_metrics].mean().reset_index()
        eff_data = eff_data.fillna(0)

        # Create bar chart
        index = np.arange(len(eff_data))
        bar_width = 0.25

        for i, metric in enumerate(efficiency_metrics):
            plt.bar(index + i*bar_width - bar_width,
                    eff_data[metric],
                    bar_width,
                    alpha=0.9,
                    color=bar_colors[i],
                    label=efficiency_display[i])

            # Add value labels
            for j, v in enumerate(eff_data[metric]):
                plt.text(j + i*bar_width - bar_width,
                        v + 0.001,
                        f'{v:.4f}',
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=10,
                        color='black')

        plt.xticks(index, eff_data['duration_category'], fontsize=12, fontweight='bold')
        plt.title('Engagement Efficiency Metrics by Duration Category', fontsize=18, fontweight='bold', color=color_palette['indigo_blue'], pad=15)
        plt.ylabel('Efficiency Ratio', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Add overall title with more space
        plt.suptitle('Video Duration and Engagement Metrics Analysis', fontsize=24, fontweight='bold', y=0.96, color=color_palette['deep_sea_blue'])

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])

        # Save chart
        output_path2 = "video_duration_engagement_analysis.png"
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Duration-engagement relationship chart saved to: {output_path2}")

    # Display basic statistics for engagement metrics
    engagement_columns = ['like_count', 'comment_count', 'view_count']
    print("\nVideo Engagement Metrics Summary Statistics:")
    print(df[engagement_columns].describe())

    # Print correlation matrix
    print("\nPearson Correlation Coefficients between Engagement Metrics:")
    print(correlation_matrix)

    # Suggest further analysis
    print("\nPotential Further Analysis Directions:")
    print("1. Engagement Ratio Analysis: Like/view, comment/view ratios to reveal audience participation levels")
    print("2. Temporal Trend Analysis: Study how engagement metrics evolve over time")
    print("3. Topic Analysis: Compare engagement across different video topics/categories")
    print("4. Outlier Analysis: Identify and study exceptionally successful or unsuccessful videos")
    print("5. Engagement Prediction Model: Build models to predict engagement based on video characteristics")

    return df

# If this script is run directly
if __name__ == "__main__":
    analyze_engagement_metrics()
