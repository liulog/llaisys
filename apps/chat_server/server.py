from tools.llaisys_infer import load_hf_tokenizer, load_llaisys_model, llaisys_infer, llaisys_infer_stream

import uuid
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import List
from pydantic import BaseModel
import asyncio

class Message(BaseModel):
    role: str
    content: str

# Request Body (Not exactly the same as OpenAI, only some fields are supported)
# Reference: 
# https://huggingface.co/docs/inference-providers/tasks/chat-completion
class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_logprobs: int = None
    seed: int = None

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

# Note: non-stream return should be normal JSON format.
@app.post("/chat")
async def infer(request: ChatRequest):
    if request.model != "qwen2":
        return {"error": "Model not supported."}
    if request.stream:
        return {"error": "Streaming not supported in this endpoint. Use /v1/chat/completions instead."}
    else:
        conversation = [message.model_dump() for message in request.messages]
        _, response_message = llaisys_infer(model=model, tokenizer=tokenizer, conversation=conversation, max_new_tokens=request.max_tokens)
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

# Note: stream return should be SSE format.
# Format is described here:
# https://huggingface.co/docs/inference-providers/tasks/chat-completion
@app.post("/v1/chat/completions")
async def infer(request: ChatRequest):
    if request.model != "qwen2":
        return {"error": "Model not supported."}

    if request.stream:
        conversation = [message.model_dump() for message in request.messages]
        print("Received conversation:", conversation, flush=True)
        async def generate_response():
            for response_message in llaisys_infer_stream(model=model, tokenizer=tokenizer, conversation=conversation, max_new_tokens=request.max_tokens):
                response = {
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": response_message
                            },
                        }
                    ]
                }
                # print("Generated response:", response)
                yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            # End of stream
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate_response(), media_type="text/event-stream")

    else:
        print("Non-streaming mode is not implemented yet.")
        return {"error": "Non-streaming mode not implemented"}
