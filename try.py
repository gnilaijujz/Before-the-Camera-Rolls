from googleapiclient.discovery import build
import isodate
import time
import csv

# 替换为你的API Key
API_KEY = 'AIzaSyC4FzQ3jVTBKanstgyKdRFj4qp0CDOrIaQ'

# 初始化 YouTube API 客户端
youtube = build('youtube', 'v3', developerKey=API_KEY)

# 搜索参数
SEARCH_QUERY = 'sports'
MAX_RESULTS = 50  # 每页最多50个视频
VIDEO_CATEGORY_ID = '17'  # 体育类（Sports）

def get_video_ids_from_search(query, category_id, max_pages=5):
    video_ids = []
    next_page_token = None

    for _ in range(max_pages):
        search_response = youtube.search().list(
            q=query,
            type='video',  # ✅ 添加这个，确保只返回视频
            videoCategoryId=category_id,
            part='id',
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in search_response['items']:
            if item['id']['kind'] == 'youtube#video':  # ✅ 再保险
                video_ids.append(item['id']['videoId'])

        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            break
        time.sleep(1)

    return video_ids


def get_video_durations(video_ids):
    video_data = []

    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i + 50]
        video_response = youtube.videos().list(
            part='contentDetails,snippet',
            id=','.join(batch_ids)
        ).execute()

        for item in video_response['items']:
            vid = item['id']
            title = item['snippet']['title']
            duration_iso = item['contentDetails']['duration']
            duration_sec = int(isodate.parse_duration(duration_iso).total_seconds())
            url = f"https://www.youtube.com/watch?v={vid}"
            video_data.append({
                'video_id': vid,
                'title': title,
                'duration_seconds': duration_sec,
                'url': url
            })

    return video_data

def save_to_csv(data, filename='sports_videos.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        fieldnames = ['video_id', 'title', 'duration_seconds', 'url']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for item in data:
            writer.writerow(item)

    print(f"✅ 已导出到 CSV 文件：{filename}")

if __name__ == '__main__':
    print("🔍 正在获取视频ID ...")
    video_ids = get_video_ids_from_search(SEARCH_QUERY, VIDEO_CATEGORY_ID, max_pages=5)

    print(f"✅ 找到 {len(video_ids)} 个视频，正在获取时长信息 ...")
    video_info = get_video_durations(video_ids)

    print("💾 正在保存到 CSV ...")
    save_to_csv(video_info)
