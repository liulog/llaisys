import time
import uuid
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import List

app = FastAPI()

# 定义 Pydantic 模型
class Message(BaseModel):
    role: str
    content: str

class Request(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False

def generate_response():
    for i in range(5):  # 假设生成 5 条数据
        print(f"Sending response {i+1}")  # 打印日志以确保数据发送
        time.sleep(1)  # 模拟延迟
        response = {
            "id": str(uuid.uuid4()),  # 唯一的响应 ID
            "created": int(time.time()),  # 响应的创建时间
            "model": "dummy_model",  # 模型名称
            "choices": [
                {
                    "message": {
                        "role": "assistant",  # 角色为 assistant
                        "content": f"Response {i+1}"
                    },
                    "finish_reason": "stop",  # 指示消息已完成
                    "index": 0
                }
            ]
        }
        yield json.dumps(response) + '\n'  # 加上换行符，确保每个 JSON 数据块独立


# 接收请求体并返回流式响应
@app.post("/infer")
async def infer(request: Request):
    print("Received request:", request)
    # 如果请求需要流式响应
    if request.stream:
        return StreamingResponse(generate_response(), media_type="application/json")
    else:
        return {"message": "Response without streaming."}
