import asyncio
import websockets
import json
import aiohttp
import re
import time
from typing import Dict, List, Optional

# 星火大模型配置
API_KEY = "Bearer LUKPbgPcUBLzhjwNYDZD:sJduuCDURqeqSWSaRxmi"
API_URL = "https://spark-api-open.xf-yun.com/v1/chat/completions"

def load_analysis_report():
    """读取 YouTube 分析报告的 TXT 文件内容"""
    report_path = "youtube_time_analysis_report.txt"
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"警告：分析报告文件 {report_path} 未找到")
        return ""
    except Exception as e:
        print(f"读取报告文件出错：{e}")
        return ""

ANALYSIS_REPORT_CONTENT = load_analysis_report()

# 用户会话管理
user_sessions: Dict[str, Dict] = {}

async def handle_connection(websocket):
    user_id = str(id(websocket))
    user_sessions[user_id] = {
        "messages": [],
        "websocket": websocket,
        "formatted_buffer": "",
        "pending_sentence": "",
        "is_first_chunk": True,
        "current_list_level": 0,
        "is_in_code_block": False,
        "is_sending": False
    }
    print(f"新连接: {user_id}")

    try:
        async for message in websocket:
            user_input = message
            print(f"收到用户输入 [{user_id}]: {user_input}")
            
            user_sessions[user_id]["messages"].append({
                "role": "user",
                "content": user_input
            })
            
            await call_spark_api(user_id)
            
    except Exception as e:
        print(f"连接错误 [{user_id}]: {e}")
        await send_error(user_id, "连接异常，请重试")
    finally:
        if user_id in user_sessions:
            del user_sessions[user_id]
        print(f"连接关闭 [{user_id}]")

async def call_spark_api(user_id: str) -> None:
    await user_sessions[user_id]["websocket"].send(json.dumps({
        "type": "response",
        "content": "这是测试响应，说明后端通信正常。"
    }))
    
    """使用aiohttp的异步版本"""
    messages = user_sessions[user_id]["messages"]
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": API_KEY
                },
                # In the call_spark_api function, modify the system message:
    json={
        "model": "4.0Ultra",
        "user": user_id,
        "messages": [{
            "role": "system",
            "content": f"""Based on the following report provide suggestions in English: {ANALYSIS_REPORT_CONTENT}
    Output requirements:
    1. Use Markdown format
    2. Use ## for headings
    3. Use - or 1. for lists
    4. Wrap key data in `backticks`
    5. Highlight important suggestions with **bold**"""
        }] + messages,
        "stream": True
    },
                timeout=30
            ) as response:
                
                if response.status == 200:
                    full_response = ""
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode('utf-8').replace('data: ', '')
                            if decoded_line != '[DONE]':
                                try:
                                    data = json.loads(decoded_line)
                                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                    
                                    if content:
                                        full_response += content
                                        # 发送Markdown格式的响应
                                        await user_sessions[user_id]["websocket"].send(json.dumps({
                                            "type": "markdown",
                                            "content": full_response
                                        }))
                                        
                                except json.JSONDecodeError:
                                    continue
                    
                    # 最终处理并添加Markdown格式
                    markdown_response = format_markdown(full_response)
                    await user_sessions[user_id]["websocket"].send(json.dumps({
                        "type": "final",
                        "content": markdown_response
                    }))
                    
                    # 保存助手消息到会话历史
                    user_sessions[user_id]["messages"].append({
                        "role": "assistant",
                        "content": markdown_response
                    })
    except Exception as e:
        print(f"API调用错误 [{user_id}]: {e}")
        await send_error(user_id, "AI服务暂时不可用，请稍后再试")

def format_markdown(text: str) -> str:
    """格式化文本为Markdown"""
    # 这里可以添加自定义的Markdown格式化逻辑
    # 例如确保标题、列表等格式正确
    return text

async def process_content_chunk(user_id: str, content: str):
    """处理每个内容块并维护格式状态"""
    session = user_sessions[user_id]
    session["pending_sentence"] += content
    
    # 检测完整句子
    if contains_complete_sentence(session["pending_sentence"]):
        formatted = format_chunk(
            session["pending_sentence"],
            is_first=session["is_first_chunk"],
            list_level=session["current_list_level"],
            is_in_code=session["is_in_code_block"]
        )
        
        # 更新状态
        session["formatted_buffer"] += formatted
        session["pending_sentence"] = ""
        session["is_first_chunk"] = False
        
        # 检测列表状态
        session["current_list_level"] = detect_list_level(formatted, session["current_list_level"])
        session["is_in_code_block"] = detect_code_block(formatted, session["is_in_code_block"])
        
        # 发送部分响应
        await send_partial_response(user_id)

def format_chunk(chunk: str, is_first: bool, list_level: int = 0, is_in_code: bool = False) -> str:
    """智能格式化单个数据块"""
    # 1. 首块特殊处理
    if is_first and chunk.lstrip().startswith(('Based on', '根据')):
        chunk = '## 🎯 优化建议\n\n' + chunk.lstrip()
    
    # 2. 列表项处理（保持层级）
    if list_level > 0:
        chunk = re.sub(r'^(\d+\.|\-)', '    ' * list_level + r'\1', chunk, flags=re.MULTILINE)
    
    # 3. 关键数据标记
    chunk = re.sub(
        r'(\b\d+\.\d+\b|\b[A-Z][a-z]+(?=\s+rate\b)|(\b[0-9]{1,2}:[0-9]{2}\s[AP]M\b))', 
        r'`\1\2`', 
        chunk
    )
    
    # 4. 代码块处理
    if is_in_code:
        chunk = f"```\n{chunk}\n```"
    
    return chunk

async def finalize_response(user_id: str):
    """最终处理未完成的响应"""
    session = user_sessions[user_id]
    if session["pending_sentence"]:
        formatted = format_chunk(
            session["pending_sentence"],
            is_first=False,
            list_level=session["current_list_level"],
            is_in_code=session["is_in_code_block"]
        )
        session["formatted_buffer"] += formatted + "\n\n---\n*数据更新于：%s*" % time.strftime("%Y-%m-%d")
        
    await send_partial_response(user_id, is_final=True)
    session["messages"].append({
        "role": "assistant",
        "content": session["formatted_buffer"]
    })

async def send_partial_response(user_id: str, is_final: bool = False):
    """发送部分响应"""
    payload = {
        "type": "partial" if not is_final else "final",
        "content": user_sessions[user_id]["formatted_buffer"],
        "is_complete": is_final,
        "timestamp": int(time.time() * 1000)
    }
    await user_sessions[user_id]["websocket"].send(json.dumps(payload))

def detect_list_level(text: str, current_level: int) -> int:
    """检测列表层级变化"""
    new_level = current_level
    if re.search(r'^\s*\d+\.', text, re.MULTILINE):
        new_level += 1
    elif re.search(r'^\s*\-', text, re.MULTILINE):
        new_level = max(0, current_level - 1)
    return min(new_level, 3)  # 限制最大层级

def detect_code_block(text: str, is_currently_in_code: bool) -> bool:
    """检测代码块状态"""
    backtick_count = text.count('`')
    if backtick_count % 2 != 0:
        return not is_currently_in_code
    return is_currently_in_code

def contains_complete_sentence(text: str) -> bool:
    """增强版句子检测"""
    return bool(re.search(r'[。！？.!?][\s”’]?$', text))

async def send_error(user_id: str, message: str):
    """发送错误消息"""
    if user_id in user_sessions:
        try:
            await user_sessions[user_id]["websocket"].send(json.dumps({
                "type": "error",
                "content": message
            }))
        except Exception:
            pass

async def _run_server():
    """启动WebSocket服务"""
    server = await websockets.serve(handle_connection, "localhost", 8765)
    print("后端服务启动: ws://localhost:8765")
    try:
        await asyncio.Future()
    finally:
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(_run_server())
    except KeyboardInterrupt:
        print("服务器已手动停止。")