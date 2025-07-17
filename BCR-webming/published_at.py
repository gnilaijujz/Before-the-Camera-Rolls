import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免Qt线程问题
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import signal
import sys
import traceback
import io
from matplotlib.colors import LinearSegmentedColormap 

warnings.filterwarnings('ignore')

# 信号处理：优雅退出
def signal_handler(signal, frame) -> None:
    print("\n程序正在优雅退出...")
    plt.close('all')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 马卡龙色系配置（核心美化方案）
MACARON_COLORS = {
    'pink': "#FBB4C1",      # 马卡龙粉
    'blue': "#96D7BA",      # 马卡龙蓝
    'purple': "#AFB8DA",    # 马卡龙紫
    'yellow': "#FCE89A",    # 马卡龙黄
    'green': "#6FC29D",     # 马卡龙绿
    'peach': "#FACFBA",     # 马卡龙桃
    'mint': "#AAECF5",      # 马卡龙薄荷
    'lavender': "#E7D1F1",  # 马卡龙薰衣草
}

def create_macaron_cmap(base_color, num_colors=256):
    rgb = matplotlib.colors.hex2color(base_color)
    cmap_vals = []
    for i in range(num_colors):
        factor = i / (num_colors - 1)
        new_rgb = (rgb[0] + (1 - rgb[0]) * factor * 0.9,  
                   rgb[1] + (1 - rgb[1]) * factor * 0.9,  
                   rgb[2] + (1 - rgb[2]) * factor * 0.9)  
        cmap_vals.append(new_rgb)
    return LinearSegmentedColormap.from_list('MacaronHeat', cmap_vals)

# 为热力图创建马卡龙色阶（可根据需求调整基础色）
MACARON_HEAT_CMAP = create_macaron_cmap(MACARON_COLORS['pink'])  
MACARON_COOL_CMAP = create_macaron_cmap(MACARON_COLORS['blue'])  
MACARON_DIVERGING_CMAP = create_macaron_cmap(MACARON_COLORS['purple'])  
MACARON_CMAPS = {
    'main': 'Pastel1',
    'heat': MACARON_HEAT_CMAP,     
    'cool': MACARON_COOL_CMAP,     
    'diverging': MACARON_DIVERGING_CMAP  # 修复语法错误（移除多余换行）
}

# 字体配置
def setup_font() -> bool:
    """设置中文字体，失败则使用英文"""
    try:
        plt.rcParams.update({
            'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'font.size': 10,
            'grid.color': '#F0F0F0',
            'grid.linestyle': '--',
            'axes.edgecolor': '#DDDDDD',
            'axes.linewidth': 1.0
        })
        plt.style.use('seaborn-v0_8-whitegrid')
        # 测试字体渲染
        with plt.figure(figsize=(1, 1)):
            plt.text(0.5, 0.5, '测试', fontsize=12)
        return True
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        return False

chinese_font_available = setup_font()
sns.set_style("whitegrid")


class YouTubeTimeAnalyzer:
    """YouTube运动视频发布时间分析系统"""
    
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.df: pd.DataFrame = None
        self.chinese_font = chinese_font_available
        print("🎯 YouTube发布时间深度分析系统")
        print("=" * 60)
        
    def load_and_preprocess_data(self) -> None:
        """加载并预处理数据"""
        print("📊 加载数据...")
        # 尝试多种编码读取
        for encoding in ['iso-8859-1', 'utf-8', 'gbk']:
            try:
                self.df = pd.read_csv(self.file_path, encoding=encoding)
                print(f"✅ 使用 {encoding} 编码成功读取文件")
                break
            except UnicodeDecodeError:
                continue
        
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")
        print(f"published_at列缺失值: {self.df['published_at'].isnull().sum()}")
        
        # 时间格式处理
        self.df['published_at'] = pd.to_datetime(self.df['published_at'])
        if self.df['published_at'].dt.tz is not None:
            self.df['published_at'] = self.df['published_at'].dt.tz_convert(None)  # 移除时区
        
        # 提取时间维度
        self.df['year'] = self.df['published_at'].dt.year
        self.df['month'] = self.df['published_at'].dt.month
        self.df['day'] = self.df['published_at'].dt.day
        self.df['weekday'] = self.df['published_at'].dt.dayofweek
        self.df['hour'] = self.df['published_at'].dt.hour
        self.df['minute'] = self.df['published_at'].dt.minute
        self.df['week_of_year'] = self.df['published_at'].dt.isocalendar().week
        self.df['day_of_year'] = self.df['published_at'].dt.dayofyear
        self.df['quarter'] = self.df['published_at'].dt.quarter
        
        # 时间标签映射（强制使用英文，满足可视化英文需求）
        self.df['weekday_name'] = self.df['weekday'].map({
            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 
            4: 'Fri', 5: 'Sat', 6: 'Sun'
        })
        self.df['month_name'] = self.df['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        print("✅ 数据预处理完成")
        
    def controlled_time_analysis(self) -> None:
        """控制变量分析（消除累积效应）"""
        print("\n🔬 步骤1: 控制变量分析 - 消除时间累积偏差")
        print("-" * 50)
        
        current_date = datetime.now()
        self.df['days_since_published'] = (current_date - self.df['published_at']).dt.days
        self.df['days_since_published'] = self.df['days_since_published'].replace(0, 1)  # 避免除零
        
        # 标准化指标
        self.df['daily_view_rate'] = self.df['view_count'] / self.df['days_since_published']
        self.df['daily_like_rate'] = self.df['like_count'] / self.df['days_since_published']
        self.df['daily_comment_rate'] = self.df['comment_count'] / self.df['days_since_published']
        
        # 对数修正
        self.df['log_days'] = np.log(self.df['days_since_published'])
        self.df['view_per_log_time'] = self.df['view_count'] / self.df['log_days']
        
        # 同期相对表现
        self.df['publish_period'] = self.df['published_at'].dt.strftime('%Y-W%W')
        period_avg = self.df.groupby('publish_period').agg({
            'view_count': 'mean', 'like_count': 'mean', 
            'comment_count': 'mean', 'engagement_rate': 'mean'
        }).add_suffix('_period_avg')
        self.df = self.df.merge(period_avg, on='publish_period', how='left')
        
        self.df['relative_view_performance'] = self.df['view_count'] / (self.df['view_count_period_avg'] + 1e-6)
        self.df['relative_engagement_performance'] = self.df['engagement_rate'] / (self.df['engagement_rate_period_avg'] + 1e-6)
        
        print("✅ 控制变量计算完成")
        
    def basic_statistics(self) -> None:
        """基础统计分析"""
        print("\n📈 步骤2: 基础时间统计")
        print("-" * 30)
        
        print(f"数据时间范围: {self.df['published_at'].min()} 至 {self.df['published_at'].max()}")
        print(f"总时间跨度: {(self.df['published_at'].max() - self.df['published_at'].min()).days} 天")
        print(f"平均每日发布量: {len(self.df) / (self.df['published_at'].max() - self.df['published_at'].min()).days:.2f}")
        
        print("\n发布时间分布:")
        print("按年份:", self.df['year'].value_counts().sort_index().to_dict())
        print("按月份:", self.df['month'].value_counts().sort_index().to_dict()) 
        print("按星期:", self.df['weekday_name'].value_counts().to_dict())
        
        # 最活跃时间段
        peak_hour = self.df['hour'].value_counts().idxmax()
        peak_weekday = self.df['weekday_name'].value_counts().idxmax()
        peak_month = self.df['month_name'].value_counts().idxmax()
        
        print(f"\n📊 发布活跃度最高:")
        print(f"  小时: {peak_hour}点")
        print(f"  星期: {peak_weekday}")
        print(f"  月份: {peak_month}")
    
    # 以下为可视化方法（移至类内部，作为类方法）
    def time_series_visualization(self) -> None:
        """Time series visualization"""
        print(f"\n📊 Step 3: Time Series Visualization")
        print("-" * 30)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Daily publication volume
        daily_counts = self.df.groupby(self.df['published_at'].dt.date).size()
        axes[0, 0].plot(daily_counts.index, daily_counts.values, 
                       alpha=0.7, linewidth=1.5, color=MACARON_COLORS['blue'])
        axes[0, 0].fill_between(daily_counts.index, 0, daily_counts.values, 
                              color=MACARON_COLORS['blue'], alpha=0.2)
        axes[0, 0].set_title('Daily Publication Volume Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Moving average
        rolling_mean = daily_counts.rolling(window=7).mean()
        axes[0, 1].plot(daily_counts.index, daily_counts.values, 
                       alpha=0.3, label='Raw Data', color=MACARON_COLORS['blue'])
        axes[0, 1].plot(rolling_mean.index, rolling_mean.values, 
                       color=MACARON_COLORS['pink'], linewidth=2, 
                       label='7-day Moving Average')
        axes[0, 1].fill_between(rolling_mean.index, 0, rolling_mean.values, 
                              color=MACARON_COLORS['pink'], alpha=0.2)
        axes[0, 1].set_title('Publication Trend (7-day Moving Average)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly publication volume
        try:
            monthly_counts = self.df.groupby(self.df['published_at'].dt.to_period('M')).size()
            x_dates = monthly_counts.index.to_timestamp()
            
            bars = axes[1, 0].bar(x_dates, monthly_counts.values, width=20, alpha=0.8, 
                                color=MACARON_COLORS['green'], edgecolor='white')
            axes[1, 0].set_title('Monthly Publication Volume')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, monthly_counts.values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{value}', ha='center', va='bottom', fontsize=9)
        except Exception as e:
            print(f"Monthly chart error: {e}")
            axes[1, 0].text(0.5, 0.5, 'Monthly data display failed', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Weekly publication volume
        weekly_counts = self.df.groupby('week_of_year').size()
        axes[1, 1].plot(weekly_counts.index, weekly_counts.values, 
                       marker='o', linewidth=2, markersize=4, 
                       color=MACARON_COLORS['purple'], markerfacecolor='white')
        axes[1, 1].set_title('Weekly Publication Volume')
        axes[1, 1].set_xlabel('Week of Year')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_series_analysis.png', dpi=300)
        plt.close()

    def cyclical_pattern_analysis(self) -> None:
        """Cyclical pattern analysis"""
        print(f"\n🔄 Step 4: Cyclical Pattern Analysis")
        print("-" * 30)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Weekday distribution
        weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_counts = self.df['weekday_name'].value_counts().reindex(weekday_order)
        
        bars1 = axes[0, 0].bar(weekday_counts.index, weekday_counts.values, 
                              alpha=0.8, color=MACARON_COLORS['peach'],
                              edgecolor='white', linewidth=1)
        axes[0, 0].set_title('Weekday Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars1, weekday_counts.values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{value}', ha='center', va='bottom', fontsize=9)
        
        # 2. Hour distribution
        hour_counts = self.df['hour'].value_counts().sort_index()
        bars2 = axes[0, 1].bar(hour_counts.index, hour_counts.values, 
                              alpha=0.8, color=MACARON_COLORS['blue'],
                              edgecolor='white', linewidth=1)
        axes[0, 1].set_title('Hour Distribution')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Month distribution
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_counts = self.df['month_name'].value_counts().reindex(month_order)
        
        bars3 = axes[1, 0].bar(month_counts.index, month_counts.values, 
                              alpha=0.8, color=MACARON_COLORS['green'],
                              edgecolor='white', linewidth=1)
        axes[1, 0].set_title('Month Distribution')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Quarter distribution
        quarter_counts = self.df['quarter'].value_counts().sort_index()
        axes[1, 1].bar(['Q1', 'Q2', 'Q3', 'Q4'], quarter_counts.values, 
                      alpha=0.8, color=[MACARON_COLORS['pink'], MACARON_COLORS['blue'], 
                                       MACARON_COLORS['green'], MACARON_COLORS['yellow']],
                      edgecolor='white', linewidth=1)
        axes[1, 1].set_title('Quarter Distribution')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('cyclical_pattern_analysis.png', dpi=300)
        plt.close()

    def performance_heatmap_analysis(self) -> None:
        """Performance heatmap analysis"""
        print(f"\n🌡️ Step 5: Performance Heatmap Analysis")
        print("-" * 30)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Weekday-Hour Engagement Rate Heatmap
        weekday_hour_engagement = self.df.groupby(['weekday', 'hour'])['engagement_rate'].mean().unstack(fill_value=0)
        sns.heatmap(weekday_hour_engagement, annot=True, fmt='.3f', 
                   cmap=MACARON_CMAPS['heat'], ax=axes[0, 0], 
                   cbar_kws={'label': 'Engagement Rate'},
                   linewidths=0.5, linecolor='white',
                   annot_kws={'size': 8})  # 调整字体大小避免重叠
        axes[0, 0].set_title('Weekday-Hour Engagement Rate Heatmap')
        axes[0, 0].set_ylabel('Weekday (0=Mon)')
        axes[0, 0].set_xlabel('Hour')
        
        # 2. Month-Weekday Engagement Rate Heatmap
        month_weekday_engagement = self.df.groupby(['month', 'weekday'])['engagement_rate'].mean().unstack(fill_value=0)
        sns.heatmap(month_weekday_engagement, annot=True, fmt='.3f', 
                   cmap=MACARON_CMAPS['heat'], ax=axes[0, 1], 
                   cbar_kws={'label': 'Engagement Rate'},
                   linewidths=0.5, linecolor='white',
                   annot_kws={'size': 8})  # 调整字体大小避免重叠
        axes[0, 1].set_title('Month-Weekday Engagement Rate Heatmap')
        axes[0, 1].set_ylabel('Month')
        axes[0, 1].set_xlabel('Weekday (0=Mon)')
        
        # 3. Controlled Weekday-Hour Heatmap
        weekday_hour_controlled = self.df.groupby(['weekday', 'hour'])['relative_engagement_performance'].mean().unstack(fill_value=0)
        sns.heatmap(weekday_hour_controlled, annot=True, fmt='.2f', 
                   cmap=MACARON_CMAPS['diverging'], ax=axes[1, 0], 
                   cbar_kws={'label': 'Relative Performance'},
                   linewidths=0.5, linecolor='white',
                   annot_kws={'size': 8})  # 调整字体大小避免重叠
        axes[1, 0].set_title('Controlled Weekday-Hour Performance')
        axes[1, 0].set_ylabel('Weekday (0=Mon)')
        axes[1, 0].set_xlabel('Hour')
        
        # 4. Month-Hour Daily View Rate Heatmap
        month_hour_views = self.df.groupby(['month', 'hour'])['daily_view_rate'].mean().unstack(fill_value=0)
        sns.heatmap(month_hour_views, annot=True, fmt='.1f',  
                   cmap=MACARON_CMAPS['cool'], ax=axes[1, 1], 
                   cbar_kws={'label': 'Daily View Rate'},
                   linewidths=0.5, linecolor='white',
                   annot_kws={'size': 7})  # 更小编号避免重叠
        axes[1, 1].set_title('Month-Hour Daily View Rate Heatmap')
        axes[1, 1].set_ylabel('Month')
        axes[1, 1].set_xlabel('Hour')
        
        plt.tight_layout()
        plt.savefig('performance_heatmap_analysis.png', dpi=300)
        plt.close()

    def clustering_analysis(self) -> None:
        """Clustering analysis"""
        print(f"\n🎯 Step 8: Clustering Analysis")
        print("-" * 30)
        
        try:
            # Feature preparation and clustering
            time_features = ['weekday', 'hour', 'month', 'day_of_year']
            X = self.df[time_features].values
            X_scaled = StandardScaler().fit_transform(X)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            self.df['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Visualize clustering results
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            colors = [MACARON_COLORS['pink'], MACARON_COLORS['blue'], 
                     MACARON_COLORS['green'], MACARON_COLORS['yellow']]
            
            # 1. Weekday-Hour clustering distribution
            for i in range(4):
                cluster_data = self.df[self.df['cluster'] == i]
                axes[0, 0].scatter(cluster_data['weekday'], cluster_data['hour'], 
                                 c=colors[i], label=f'Cluster {i}', alpha=0.6, s=30, edgecolor='white')
            axes[0, 0].set_title('Weekday-Hour Clustering Distribution')
            axes[0, 0].set_xlabel('Weekday')
            axes[0, 0].set_ylabel('Hour')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Month-Hour clustering distribution
            for i in range(4):
                cluster_data = self.df[self.df['cluster'] == i]
                axes[0, 1].scatter(cluster_data['month'], cluster_data['hour'], 
                                 c=colors[i], label=f'Cluster {i}', alpha=0.6, s=30, edgecolor='white')
            axes[0, 1].set_title('Month-Hour Clustering Distribution')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Hour')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Cluster performance comparison
            cluster_performance = self.df.groupby('cluster')[['engagement_rate', 'daily_view_rate', 
                                                           'daily_like_rate', 'relative_engagement_performance']].mean()
            cluster_performance.plot(kind='bar', ax=axes[1, 0], width=0.8,
                                    color=[MACARON_COLORS['peach'], MACARON_COLORS['mint'],
                                          MACARON_COLORS['lavender'], MACARON_COLORS['yellow']],
                                    edgecolor='white')
            axes[1, 0].set_title('Cluster Performance Metrics Comparison')
            axes[1, 0].set_ylabel('Average Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Cluster size distribution
            cluster_sizes = self.df['cluster'].value_counts().sort_index()
            axes[1, 1].pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in range(4)], 
                          autopct='%1.1f%%', startangle=90, colors=colors,
                          wedgeprops={'edgecolor': 'white', 'linewidth': 1})
            axes[1, 1].set_title('Cluster Size Distribution')
            
            plt.tight_layout()
            plt.savefig('clustering_analysis.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Clustering analysis error: {e}")

    def early_performance_analysis(self, days_cutoff=30) -> None:
        """Early performance analysis"""
        print(f"\n⚡ Step 9: Early Performance Analysis (Within {days_cutoff} Days After Publication)")
        print("-" * 50)
        
        old_enough_videos = self.df[self.df['days_since_published'] >= days_cutoff].copy()
        if len(old_enough_videos) == 0:
            print(f"⚠️ Not enough historical data (need videos published at least {days_cutoff} days ago)")
            return
            
        # Calculate early performance metrics
        old_enough_videos['estimated_early_engagement'] = (
            old_enough_videos['engagement_rate'] * days_cutoff / old_enough_videos['days_since_published']
        )
        old_enough_videos['estimated_early_views'] = (
            old_enough_videos['view_count'] * days_cutoff / old_enough_videos['days_since_published']
        )
        
        # Visualize early performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Early engagement rate by hour
        early_engagement_by_hour = old_enough_videos.groupby('hour')['estimated_early_engagement'].mean()
        bars1 = axes[0, 0].bar(early_engagement_by_hour.index, early_engagement_by_hour.values, 
                              alpha=0.8, color=MACARON_COLORS['peach'], edgecolor='white')
        axes[0, 0].set_title(f'Early Engagement Rate by Hour ({days_cutoff} Days)')
        axes[0, 0].set_xlabel('Publication Hour')
        axes[0, 0].set_ylabel('Estimated Early Engagement Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        best_hour_early = early_engagement_by_hour.idxmax()
        axes[0, 0].axvline(x=best_hour_early, color=MACARON_COLORS['pink'], linestyle='--', 
                          alpha=0.7, label=f'Best: {best_hour_early}h')
        axes[0, 0].legend()
        
        # 2. Early engagement rate by weekday
        early_engagement_by_weekday = old_enough_videos.groupby('weekday_name')['estimated_early_engagement'].mean()
        bars2 = axes[0, 1].bar(early_engagement_by_weekday.index, early_engagement_by_weekday.values, 
                              alpha=0.8, color=MACARON_COLORS['blue'], edgecolor='white')
        axes[0, 1].set_title(f'Early Engagement Rate by Weekday ({days_cutoff} Days)')
        axes[0, 1].set_xlabel('Publication Weekday')
        axes[0, 1].set_ylabel('Estimated Early Engagement Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Early views by hour
        early_views_by_hour = old_enough_videos.groupby('hour')['estimated_early_views'].mean()
        axes[1, 0].plot(early_views_by_hour.index, early_views_by_hour.values, 
                       marker='o', linewidth=2, markersize=6, color=MACARON_COLORS['green'],
                       markerfacecolor='white', markeredgecolor=MACARON_COLORS['green'])
        axes[1, 0].set_title(f'Early Views by Hour ({days_cutoff} Days)')
        axes[1, 0].set_xlabel('Publication Hour')
        axes[1, 0].set_ylabel('Estimated Early Views')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Long-term vs early performance comparison
        current_engagement_by_hour = self.df.groupby('hour')['engagement_rate'].mean()
        x = range(24)
        width = 0.35
        
        axes[1, 1].bar([i - width/2 for i in x], 
                      [current_engagement_by_hour.get(i, 0) for i in x],
                      width, label='Current Average', 
                      alpha=0.8, color=MACARON_COLORS['blue'], edgecolor='white')
        axes[1, 1].bar([i + width/2 for i in x], 
                      [early_engagement_by_hour.get(i, 0) for i in x],
                      width, label='Early Performance', 
                      alpha=0.8, color=MACARON_COLORS['pink'], edgecolor='white')
        
        axes[1, 1].set_title('Long-Term vs Early Performance Comparison')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Engagement Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('early_performance_analysis.png', dpi=300)
        plt.close()

    # 补充缺失的方法（run_complete_analysis中调用的）
    def controlled_comparison_analysis(self) -> None:
        """控制变量对比分析"""
        print(f"\n⚖️ Step 6: Controlled Variable Comparison Analysis")
        print("-" * 40)
        
        metrics = ['engagement_rate', 'daily_like_rate', 'relative_engagement_performance']
        metric_names = ['Raw Engagement', 'Daily Like Rate', 'Relative Performance']
        
        results = {}
        for metric, name in zip(metrics, metric_names):
            hour_analysis = self.df.groupby('hour')[metric].mean()
            weekday_analysis = self.df.groupby('weekday_name')[metric].mean()
            results[name] = {
                'best_hour': hour_analysis.idxmax(),
                'best_hour_value': hour_analysis.max(),
                'best_weekday': weekday_analysis.idxmax()
            }
        
        print("🔍 Best times by different metrics:")
        for metric_name, result in results.items():
            print(f"  {metric_name}:")
            print(f"    Best hour: {result['best_hour']}h ({result['best_hour_value']:.4f})")
            print(f"    Best weekday: {result['best_weekday']}")
        
        # 一致性分析
        hour_recommendations = [result['best_hour'] for result in results.values()]
        weekday_recommendations = [result['best_weekday'] for result in results.values()]
        
        hour_consensus = len(set(hour_recommendations)) == 1
        weekday_consensus = len(set(weekday_recommendations)) == 1
        
        print(f"\n🎯 Consistency:")
        if hour_consensus:
            print(f"✅ Consistent hour recommendation: {hour_recommendations[0]}h")
        else:
            print(f"⚠️ Inconsistent hour recommendations: {hour_recommendations}")
            
        if weekday_consensus:
            print(f"✅ Consistent weekday recommendation: {weekday_recommendations[0]}")
        else:
            print(f"⚠️ Inconsistent weekday recommendations: {weekday_recommendations}")

    def statistical_testing(self) -> None:
        """统计显著性检验"""
        print(f"\n📊 Step 7: Statistical Significance Testing")
        print("-" * 30)
        
        # 1. 卡方检验
        weekday_counts = self.df['weekday'].value_counts().sort_index()
        hour_counts = self.df['hour'].value_counts().sort_index()
        
        chi2_weekday, p_weekday = stats.chisquare(weekday_counts.values)
        chi2_hour, p_hour = stats.chisquare(hour_counts.values)
        
        print(f"📈 Chi-square test results:")
        print(f"  Weekday distribution: χ² = {chi2_weekday:.4f}, p = {p_weekday:.4f}")
        print(f"    {'✅ Significantly uneven' if p_weekday < 0.05 else '➖ Nearly uniform'}")
        print(f"  Hour distribution: χ² = {chi2_hour:.4f}, p = {p_hour:.4f}")
        print(f"    {'✅ Significantly uneven' if p_hour < 0.05 else '➖ Nearly uniform'}")
        
        # 2. 方差分析
        try:
            hour_groups = [group['relative_engagement_performance'].values 
                          for name, group in self.df.groupby('hour') if len(group) >= 5]
            if len(hour_groups) >= 3:
                f_stat, p_value = stats.f_oneway(*hour_groups)
                print(f"\n🧮 ANOVA results:")
                print(f"  Hour effect: F = {f_stat:.4f}, p = {p_value:.4f}")
                print(f"    {'✅ Significant differences' if p_value < 0.05 else '➖ No significant differences'}")
        except Exception as e:
            print(f"ANOVA error: {e}")
            
        # 3. 相关性分析
        time_features = ['weekday', 'hour', 'month', 'day_of_year']
        performance_features = ['engagement_rate', 'daily_view_rate', 'relative_engagement_performance']
        
        print(f"\n🔗 Correlation between time and performance:")
        for perf_feature in performance_features:
            print(f"  {perf_feature}:")
            for time_feature in time_features:
                correlation = self.df[time_feature].corr(self.df[perf_feature])
                significance = "significant" if abs(correlation) > 0.1 else "weak"
                print(f"    Correlation with {time_feature}: {correlation:.4f} ({significance})")

    def comprehensive_recommendation(self) -> None:
        """综合推荐分析（英文输出）"""
        print(f"\n🎯 Step 10: Comprehensive Recommendation Analysis")
        print("=" * 50)
        
        # 收集推荐结果
        recommendations = {
            'Raw Data': {
                'hour': self.df.groupby('hour')['engagement_rate'].mean().idxmax(),
                'weekday': self.df.groupby('weekday_name')['engagement_rate'].mean().idxmax()
            },
            'Controlled': {
                'hour': self.df.groupby('hour')['relative_engagement_performance'].mean().idxmax(),
                'weekday': self.df.groupby('weekday_name')['relative_engagement_performance'].mean().idxmax()
            },
            'Daily Performance': {
                'hour': self.df.groupby('hour')['daily_like_rate'].mean().idxmax(),
                'weekday': self.df.groupby('weekday_name')['daily_like_rate'].mean().idxmax()
            },
            'Heatmap Combination': {
                'hour': self.df.groupby(['weekday', 'hour'])['engagement_rate'].mean().unstack().stack().idxmax()[1],
                'weekday': self.df[self.df['weekday'] == self.df.groupby(['weekday', 'hour'])['engagement_rate'].mean().unstack().stack().idxmax()[0]]['weekday_name'].iloc[0]
            }
        }
        
        # 打印推荐结果
        print("📊 Recommendations from different methods:")
        for method, result in recommendations.items():
            print(f"  {method}:")
            print(f"    Best hour: {result['hour']}h")
            print(f"    Best weekday: {result['weekday']}")
        
        # 综合推荐
        hour_counter = Counter([r['hour'] for r in recommendations.values()])
        weekday_counter = Counter([r['weekday'] for r in recommendations.values()])
        most_recommended_hour = hour_counter.most_common(1)[0]
        most_recommended_weekday = weekday_counter.most_common(1)[0]
        
        print(f"\n🏆 Final Recommendation:")
        print(f"  Recommended hour: {most_recommended_hour[0]}h (endorsed by {most_recommended_hour[1]}/{len(recommendations)} methods)")
        print(f"  Recommended weekday: {most_recommended_weekday[0]} (endorsed by {most_recommended_weekday[1]}/{len(recommendations)} methods)")
        
        # 置信度评估
        hour_confidence = most_recommended_hour[1] / len(recommendations)
        weekday_confidence = most_recommended_weekday[1] / len(recommendations)
        
        print(f"\n📊 Recommendation Confidence:")
        print(f"  Hour confidence: {hour_confidence:.2%}")
        print(f"  Weekday confidence: {weekday_confidence:.2%}")
        
        # 生成行动计划
        self.generate_action_plan(most_recommended_hour[0], most_recommended_weekday[0], 
                                hour_confidence, weekday_confidence)
        
    def generate_action_plan(self, best_hour, best_weekday, hour_confidence, weekday_confidence) -> None:
        """生成行动计划（英文输出）"""
        print(f"\n📋 Action Plan")
        print("-" * 30)
        
        print(f"🎯 Core Strategy: Primary publication time - {best_weekday} {best_hour}h")
        
        if hour_confidence >= 0.75 and weekday_confidence >= 0.75:
            print(f"\n✅ High Confidence Strategy - Implement Immediately:")
            print(f"  1. Schedule key content for {best_weekday} {best_hour}h")
            print(f"  2. Test this time slot for 4 consecutive weeks")
            print(f"  3. Monitor changes in key metrics")
        elif hour_confidence >= 0.5 or weekday_confidence >= 0.5:
            print(f"\n🟡 Moderate Confidence Strategy - Test Cautiously:")
            print(f"  1. Select 3-5 videos to publish at {best_weekday} {best_hour}h")
            print(f"  2. Conduct A/B testing against current strategy")
            print(f"  3. Evaluate results after 2 weeks of data collection")
        else:
            print(f"\n⚠️ Low Confidence Strategy - Small-Scale Experiment:")
            print(f"  1. Test with 1-2 videos initially")
            print(f"  2. Simultaneously test other highly recommended times")
            print(f"  3. Establish longer-term data collection")
        
        print(f"\n⚠️ Notes:")
        print(f"  1. Time is not the only determining factor")
        print(f"  2. Content quality, titles, and thumbnails are equally important")
        print(f"  3. Audience characteristics may vary by content type")
        print(f"  4. External factors (holidays, major events) will affect performance")
        
        print(f"\n📊 Recommended Monitoring Metrics:")
        print(f"  1. View growth within 24 hours of publication")
        print(f"  2. Changes in like and comment rates")
        print(f"  3. Click-through rate (CTR) variations")
        print(f"  4. Watch time and completion rate")
        print(f"  5. Subscription conversion rate")
        
 
    def generate_final_report(self) -> None:
        """生成最终分析报告（英文输出）"""
        report_str = io.StringIO()

        def print_to_report(*args, **kwargs):
            print(*args, **kwargs, file=report_str)

        print_to_report(f"\n" + "🎯" * 20)
        print_to_report("   YouTube Sports Video Publication Time Analysis Report")
        print_to_report("🎯" * 20)
        
        # 数据概况
        print_to_report(f"\n📊 Data Overview:")
        print_to_report(f"  Number of videos analyzed: {len(self.df):,}")
        print_to_report(f"  Time span: {self.df['published_at'].min().strftime('%Y-%m-%d')} to {self.df['published_at'].max().strftime('%Y-%m-%d')}")
        print_to_report(f"  Total duration: {(self.df['published_at'].max() - self.df['published_at'].min()).days} days")
        
        # 关键发现
        print_to_report(f"\n🔍 Key Findings:")
        peak_hour = self.df['hour'].value_counts().idxmax()
        peak_weekday = self.df['weekday_name'].value_counts().idxmax()
        peak_month = self.df['month_name'].value_counts().idxmax()
        best_perf_hour = self.df.groupby('hour')['engagement_rate'].mean().idxmax()
        best_perf_weekday = self.df.groupby('weekday_name')['engagement_rate'].mean().idxmax()
        best_perf_rate = self.df.groupby('hour')['engagement_rate'].mean().max()
        
        print_to_report(f"  📈 Most active publication time: {peak_weekday} {peak_hour}h, {peak_month}")
        print_to_report(f"  🏆 Best performance time: {best_perf_weekday} {best_perf_hour}h (Engagement rate: {best_perf_rate:.4f})")
        print_to_report(f"  📅 Weekday publication ratio: {len(self.df[self.df['weekday'] < 5])/len(self.df):.1%}")
        print_to_report(f"  🌙 Night publication ratio: {len(self.df[self.df['hour'].isin([0,1,2,3,4,5,22,23])])/len(self.df):.1%}")
        
        # 控制变量后结果
        controlled_best_hour = self.df.groupby('hour')['relative_engagement_performance'].mean().idxmax()
        controlled_best_weekday = self.df.groupby('weekday_name')['relative_engagement_performance'].mean().idxmax()
        print_to_report(f"  🔬 Best time after controlled variables: {controlled_best_weekday} {controlled_best_hour}h")
        
        # 一致性检查
        raw_vs_controlled_hour = "consistent" if best_perf_hour == controlled_best_hour else "inconsistent"
        raw_vs_controlled_weekday = "consistent" if best_perf_weekday == controlled_best_weekday else "inconsistent"
        print_to_report(f"  ⚖️ Raw vs controlled analysis: Hour {raw_vs_controlled_hour}, Weekday {raw_vs_controlled_weekday}")
        
        # 统计显著性
        hour_counts = self.df['hour'].value_counts().sort_index()
        chi2_hour, p_hour = stats.chisquare(hour_counts.values)
        significance = "significant" if p_hour < 0.05 else "not significant"
        print_to_report(f"  📊 Hour distribution significance: {significance} (p={p_hour:.4f})")
        
        # 核心建议
        print_to_report(f"\n💡 Core Recommendations:")
        print_to_report(f"  1. Focus on publishing at {controlled_best_weekday} {controlled_best_hour}h")
        print_to_report(f"  2. Avoid peak publication times, find blue ocean time slots")
        print_to_report(f"  3. Establish A/B testing to verify optimal times")
        print_to_report(f"  4. Monitor long-term trend changes")
        
        # 预期效果
        avg_engagement = self.df['engagement_rate'].mean()
        best_engagement = self.df.groupby('hour')['engagement_rate'].mean().max()
        potential_improvement = (best_engagement / avg_engagement - 1) * 100
        print_to_report(f"\n📈 Expected Impact:")
        print_to_report(f"  Current average engagement rate: {avg_engagement:.4f}")
        print_to_report(f"  Best time engagement rate: {best_engagement:.4f}")
        print_to_report(f"  Potential improvement: {potential_improvement:.1f}%")
        
        print_to_report(f"\n🎉 Analysis complete! Recommend saving this report for publication strategy reference.")

        # 保存报告
        with open('youtube_time_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_str.getvalue())
        print("Report saved as youtube_time_analysis_report.txt")

        
    def run_complete_analysis(self) -> None:
        """运行完整分析流程"""
        try:
            self.load_and_preprocess_data()
            self.controlled_time_analysis()
            self.basic_statistics()
            self.time_series_visualization()
            self.cyclical_pattern_analysis()
            self.performance_heatmap_analysis()
            self.controlled_comparison_analysis()
            self.statistical_testing()
            self.clustering_analysis()
            self.early_performance_analysis()
            self.comprehensive_recommendation()
            self.generate_final_report()
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            traceback.print_exc()


def main():
    """主程序入口"""
    file_path = './sports_videos.csv'  # 修改为你的CSV文件实际路径（建议放在同目录下）
    analyzer = YouTubeTimeAnalyzer(file_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()