from llaisys_infer import load_hf_tokenizer, load_llaisys_model, llaisys_infer, llaisys_infer_stream

import uuid
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import List
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

# Request Body
class Request(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False

tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre Service
    global tokenizer, model
    model_path = "../../../DeepSeek-R1-Distill-Qwen-1.5B"
    if not model_path or not model_path.strip():
        raise ValueError("Model path must be provided. (DeepSeek-R1-Distill-Qwen-1.5B)")
    device_name = "nvidia"
    tokenizer = load_hf_tokenizer(model_path, device_name)
    model = load_llaisys_model(model_path, device_name)
    print("Model and tokenizer loaded successfully.")
   
    # Start Service
    yield

    # Post Service
    print("Shutting down, performing cleanup.")

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def infer(request: Request):
    if request.model != "qwen2":
        return {"error": "Model not supported."}

    if request.stream:
        conversation = [message.model_dump() for message in request.messages]
        def generate_response():
            for response_message in llaisys_infer_stream(model=model, tokenizer=tokenizer, conversation=conversation):
                response = {
                    "id": str(uuid.uuid4()),  # Unique response ID
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": response_message
                            },
                            "finish_reason": "stop",
                            "index": 0
                        }
                    ]
                }
                print("Generated response:", response)
                yield json.dumps(response) + '\n'
        return StreamingResponse(generate_response(), media_type="application/json")

    else:
        conversation = [message.model_dump() for message in request.messages]
        print("Messages dict:", conversation)
        _, response_message = llaisys_infer(model=model, tokenizer=tokenizer, conversation=conversation)

        response = {
            "id": str(uuid.uuid4()),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_message
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
        return response