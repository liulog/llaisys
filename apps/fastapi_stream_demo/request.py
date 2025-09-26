import requests
import json

def get_streaming_response():
    request_data = {
        "model": "dummy_model",  # 模型名称
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "What can you do?"}
        ],
        "stream": True  # 启用流式响应
    }

    # 使用 requests 发送流式请求
    with requests.post('http://127.0.0.1:8000/infer', json=request_data, stream=True) as response:
        # 检查是否成功
        if response.status_code == 200:
            print("Response received successfully")
            # 打印整个响应内容
            for line in response.iter_lines():
                print("Raw line:", line)  # 打印原始行内容
                if line:
                    try:
                        data = json.loads(line)
                        print("Received:", data)  # 打印解析后的响应数据
                    except json.JSONDecodeError:
                        print("Error decoding JSON:", line)
                        continue
        else:
            print(f"Request failed with status code {response.status_code}")

# 调用函数获取流式响应
get_streaming_response()
