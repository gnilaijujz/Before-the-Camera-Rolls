import asyncio
import websockets
import json
import requests
from typing import Dict, List, Optional
import re
import time  # 添加时间模块导入

# 星火大模型配置
API_KEY = "Bearer LUKPbgPcUBLzhjwNYDZD:sJduuCDURqeqSWSaRxmi"
API_URL = "https://spark-api-open.xf-yun.com/v1/chat/completions"

# 用户会话管理
user_sessions: Dict[str, Dict] = {}

async def handle_connection(websocket, path):
    user_id = str(id(websocket))
    user_sessions[user_id] = {
        "messages": [],
        "websocket": websocket,
        "buffer": [],          # 响应缓冲区
        "is_sending": False,   # 是否正在发送响应
        "last_chunk": ""       # 上一个响应片段
    }
    print(f"新连接: {user_id}")

    try:
        async for message in websocket:
            user_input = message
            print(f"收到用户输入 [{user_id}]: {user_input}")
            
            # 添加用户消息到会话
            user_sessions[user_id]["messages"].append({
                "role": "user",
                "content": user_input
            })
            
            # 调用大模型并获取响应
            await call_spark_api(user_id)
            
    except Exception as e:
        print(f"连接错误 [{user_id}]: {e}")
        await send_error(user_id, "连接异常，请重试")
    finally:
        if user_id in user_sessions:
            del user_sessions[user_id]
        print(f"连接关闭 [{user_id}]")

async def call_spark_api(user_id: str) -> None:
    """调用星火API并处理流式响应"""
    messages = user_sessions[user_id]["messages"]
    
    # 清空缓冲区
    user_sessions[user_id]["buffer"] = []
    user_sessions[user_id]["last_chunk"] = ""
    user_sessions[user_id]["is_sending"] = True
    
    try:
        response = requests.post(
            API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": API_KEY
            },
            json={
                "model": "4.0Ultra",
                "user": user_id,
                "messages": messages,
                "stream": True  # 启用流式响应
            },
            stream=True
        )

        if response.status_code == 200:
            # 收集流式响应到缓冲区
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').replace('data: ', '')
                    if decoded_line != '[DONE]':
                        try:
                            data = json.loads(decoded_line)
                            content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                # 智能合并文本片段
                                merged_content = merge_text_fragments(
                                    user_sessions[user_id]["last_chunk"], 
                                    content
                                )
                                
                                # 如果成功合并，更新最后一个片段
                                if merged_content != user_sessions[user_id]["last_chunk"] + content:
                                    user_sessions[user_id]["last_chunk"] = merged_content
                                    # 如果合并后的内容包含完整句子，添加到缓冲区
                                    if contains_complete_sentence(merged_content):
                                        user_sessions[user_id]["buffer"].append(merged_content)
                                        user_sessions[user_id]["last_chunk"] = ""
                                else:
                                    # 如果未合并，直接添加到最后一个片段
                                    user_sessions[user_id]["last_chunk"] += content
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
            
            # 添加最后一个片段（如果有剩余内容）
            if user_sessions[user_id]["last_chunk"]:
                user_sessions[user_id]["buffer"].append(user_sessions[user_id]["last_chunk"])
                user_sessions[user_id]["last_chunk"] = ""
            
            # 合并缓冲区内容并一次性发送
            full_response = ' '.join(user_sessions[user_id]["buffer"])
            
            # 添加AI回复到会话历史
            user_sessions[user_id]["messages"].append({
                "role": "assistant",
                "content": full_response
            })
            
            # 发送完整响应到前端
            await user_sessions[user_id]["websocket"].send(json.dumps({
                "type": "response",
                "content": full_response,
                "timestamp": int(time.time() * 1000)  # 修复：使用time.time()
            }))
            
        else:
            error_msg = f"接口错误: {response.status_code} - {response.text}"
            print(error_msg)
            await send_error(user_id, "抱歉，暂时无法获取AI响应。")
            
    except Exception as e:
        print(f"API调用错误: {e}")
        await send_error(user_id, "抱歉，AI服务暂时不可用。")
    finally:
        user_sessions[user_id]["is_sending"] = False

def merge_text_fragments(prev: str, current: str) -> str:
    """智能合并文本片段，处理标点符号和空格"""
    prev = prev.strip()
    current = current.strip()
    
    # 如果前一个片段为空，直接返回当前片段
    if not prev:
        return current
    
    # 如果当前片段以标点符号开头，且前一个片段没有以标点符号结尾，添加空格
    if (current.startswith(('，', '。', '！', '？', ',', '.', '!', '?')) and 
        not prev.endswith(('，', '。', '！', '？', ',', '.', '!', '?'))):
        return prev + ' ' + current
    
    # 如果前一个片段以标点符号结尾，且当前片段以字母或数字开头，添加空格
    if (prev.endswith(('，', '。', '！', '？', ',', '.', '!', '?')) and 
        current and current[0].isalnum()):
        return prev + ' ' + current
    
    # 直接连接两个片段
    return prev + current

def contains_complete_sentence(text: str) -> bool:
    """检查文本是否包含完整句子（以句号、问号或感叹号结尾）"""
    return bool(re.search(r'[。！？.!?]$', text))

async def send_error(user_id: str, message: str) -> None:
    """发送错误消息到前端"""
    if user_id in user_sessions:
        try:
            await user_sessions[user_id]["websocket"].send(json.dumps({
                "type": "error",
                "content": message
            }))
        except:
            pass

# 启动WebSocket服务
start_server = websockets.serve(handle_connection, "localhost", 8765)
print("后端服务启动: ws://localhost:8765")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()