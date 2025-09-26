import requests
import json

def get_streaming_response():
    request_data = {
        "model": "qwen2",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ],
        "stream": True
    }

    with requests.post('http://127.0.0.1:8000/chat', json=request_data, stream=True) as response:
        if response.status_code == 200:
            print("Response received successfully")
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data['choices'][0]['message']['content']
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print("Error decoding JSON:", line)
                        continue
                    except KeyError as e:
                        print(f"Error extracting content: Missing key {e}")
                        continue
        else:
            print(f"Request failed with status code {response.status_code}")

get_streaming_response()
