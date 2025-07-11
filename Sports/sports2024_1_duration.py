import random
import time
import csv
import isodate
from googleapiclient.discovery import build
import pandas as pd

# 👇 每人替换为自己3个API KEY
API_KEYS = [
    'AIzaSyC4FzQ3jVTBKanstgyKdRFj4qp0CDOrIaQ',
    'AIzaSyDMVEJwXu4Rs3amiDIk44QyppzWO_NF0u4',
    'AIzaSyC33-8fcs9okEn8qJIbM3K_oQ4GsJhG75o'
]

# 👇 每人设置自己负责的时间段（每季度）
PUBLISHED_AFTER = '2024-01-01T00:00:00Z'   # 👈 Q1 示例
PUBLISHED_BEFORE = '2024-03-31T23:59:59Z'

# 👇 搜索关键词及分类
SEARCH_QUERY = 'sports'
VIDEO_CATEGORY_ID = '17'
MAX_PAGES = 10  # 每人最多翻多少页（每页最多50条）

# 获取 YouTube 客户端（自动换 key）
def get_youtube_client():
    key = random.choice(API_KEYS)
    print(f"✅ 当前使用 API Key: {key}")
    return build('youtube', 'v3', developerKey=key)

# 获取视频ID（按季度 + 多页）
def get_video_ids():
    youtube = get_youtube_client()
    video_ids = []
    next_page_token = None

    for page in range(MAX_PAGES):
        try:
            response = youtube.search().list(
                q=SEARCH_QUERY,
                type='video',
                videoCategoryId=VIDEO_CATEGORY_ID,
                part='id',
                maxResults=50,
                pageToken=next_page_token,
                publishedAfter=PUBLISHED_AFTER,
                publishedBefore=PUBLISHED_BEFORE
            ).execute()

            for item in response['items']:
                video_ids.append(item['id']['videoId'])

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(2)

    return video_ids

# 获取视频详细信息
def get_video_info(video_ids):
    all_info = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        youtube = get_youtube_client()
        try:
            response = youtube.videos().list(
                part='contentDetails,snippet',
                id=','.join(batch)
            ).execute()

            for item in response['items']:
                vid = item['id']
                title = item['snippet']['title']
                duration = isodate.parse_duration(item['contentDetails']['duration']).total_seconds()
                url = f'https://www.youtube.com/watch?v={vid}'

                all_info.append({
                    'video_id': vid,
                    'title': title,
                    'duration_seconds': int(duration),
                    'url': url,
                    'published_at': item['snippet']['publishedAt']
                })
        except Exception as e:
            print(f"❌ 视频详情获取失败: {e}")
            time.sleep(2)

    return all_info

# 保存为 CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存为: {filename}")

# 主程序
if __name__ == '__main__':
    print("🚀 开始爬取 ...")
    ids = get_video_ids()
    print(f"🎯 获取视频数量：{len(ids)}")

    info = get_video_info(ids)
    print(f"📦 获取完整信息数量：{len(info)}")

    # 每人保存不同文件名（如 q1.csv）
    save_to_csv(info, 'sports_q1.csv')
