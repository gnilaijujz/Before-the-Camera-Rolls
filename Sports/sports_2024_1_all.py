import random
import time
import isodate
from googleapiclient.discovery import build
import pandas as pd

# === 1. 设置参数 ===
API_KEYS = [
    'AIzaSyC4FzQ3jVTBKanstgyKdRFj4qp0CDOrIaQ',
    'AIzaSyDMVEJwXu4Rs3amiDIk44QyppzWO_NF0u4',
    'AIzaSyC33-8fcs9okEn8qJIbM3K_oQ4GsJhG75o'
]

# 👇 替换为每人负责的季度时间段
PUBLISHED_AFTER = '2024-01-01T00:00:00Z'
PUBLISHED_BEFORE = '2024-03-31T23:59:59Z'

SEARCH_QUERY = 'sports'  # 加回关键词，扩大搜索结果
MAX_RESULTS_TOTAL = 1000  # 每季度目标数量
RESULTS_PER_PAGE = 50  # 每页最大视频数
MAX_PAGES = 50  # 最大翻页数

# === 2. 获取 YouTube 客户端 ===
def get_youtube_client():
    key = random.choice(API_KEYS)
    return build('youtube', 'v3', developerKey=key)

# === 3. 搜索视频ID ===
def get_video_ids():
    video_ids = []
    next_page_token = None
    page_count = 0

    while len(video_ids) < MAX_RESULTS_TOTAL and page_count < MAX_PAGES:
        try:
            youtube = get_youtube_client()
            response = youtube.search().list(
                q=SEARCH_QUERY,
                type='video',
                part='id',
                maxResults=RESULTS_PER_PAGE,
                pageToken=next_page_token,
                publishedAfter=PUBLISHED_AFTER,
                publishedBefore=PUBLISHED_BEFORE
            ).execute()

            page_ids = [item['id']['videoId'] for item in response['items'] if item['id']['kind'] == 'youtube#video']
            video_ids.extend(page_ids)

            print(f"📄 第 {page_count + 1} 页，获取 {len(page_ids)} 条，总计 {len(video_ids)} 条")

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

            page_count += 1
            time.sleep(1)
        except Exception as e:
            print(f"Search error: {e}")
            time.sleep(2)

    return video_ids[:MAX_RESULTS_TOTAL]

# === 4. 获取视频详细数据 ===
def get_video_details(video_ids):
    video_data = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        youtube = get_youtube_client()
        try:
            response = youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(batch)
            ).execute()

            for item in response['items']:
                vid = item['id']
                snippet = item['snippet']
                stats = item.get('statistics', {})

                title = snippet.get('title', '')
                description = snippet.get('description', '')
                published_at = snippet.get('publishedAt', '')
                tags = snippet.get('tags', [])
                duration = isodate.parse_duration(item['contentDetails']['duration']).total_seconds()
                url = f'https://www.youtube.com/watch?v={vid}'

                # 统计数据
                view_count = int(stats.get('viewCount', 0))
                like_count = int(stats.get('likeCount', 0))
                #dislike_count = int(stats.get('dislikeCount', 0))  # 已不再公开，结果为0
                comment_count = int(stats.get('commentCount', 0))
                #share_count = 0  # 无 API 接口获取，默认设为 0

                # 观众留存率与互动率 (占位符模拟值)
                #retention_rate = round(random.uniform(0.2, 0.9), 2)  # 模拟：0.2~0.9
                engagement_rate = round((like_count + comment_count) / max(view_count, 1), 4)

                video_data.append({
                    'video_id': vid,
                    'url': url,
                    'duration_seconds': int(duration),
                    'published_at': published_at,
                    'tags': ', '.join(tags),
                    'title': title,
                    'description': description,
                    'view_count': view_count,
                    'like_count': like_count,
                    #'dislike_count': dislike_count,
                    'comment_count': comment_count,
                    #'share_count': share_count,
                    #'retention_rate': retention_rate,
                    'engagement_rate': engagement_rate
                })
        except Exception as e:
            print(f"Details error: {e}")
            time.sleep(2)

    return video_data

# === 5. 保存为 CSV ===
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存为：{filename}")

# === 6. 主程序 ===
if __name__ == '__main__':
    print("🚀 开始爬取...")
    ids = get_video_ids()
    print(f"🔎 获取视频数量: {len(ids)}")

    info = get_video_details(ids)
    print(f"📦 获取详情数量: {len(info)}")

    save_to_csv(info, 'sports_q1.csv')  # 👈 每人改成 q2/q3/q4 即可