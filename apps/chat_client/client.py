import streamlit as st
from huggingface_hub import InferenceClient

st.title("LLAISYS Chat, powered by Qwen2")
# It depends on the local deployment of the inference server.
client = InferenceClient(base_url="http://127.0.0.1:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def stream_generator(stream):
    try:
        processed_response = ""
        # think_ended used for markdown display formatting
        think_ended = False
        first_chunk = True
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    # Handle content between <think> and </think>
                    if not think_ended:
                        if "</think>" in content:
                            think_part, answer_part = content.split("</think>", 1)
                            if not processed_response:
                                content = f"> {think_part}\n\n{answer_part}"
                            else:
                                content = f"{think_part}\n\n{answer_part}"
                            processed_response += answer_part.replace('\n', '')
                            think_ended = True
                        else:
                            if first_chunk:
                                first_chunk = False
                                content = f"> {content}"
                        yield content
        
                    # Handle content after </think>
                    else:
                        yield content
                        processed_response += content.replace('\n', '')
                        continue

        # record the historical message
        st.session_state.messages.append({"role": "assistant", "content": processed_response})
        st.info(processed_response)

    except Exception as e:
        st.error(f"流式处理错误: {e}")
    finally:
        try:
            stream.close()
            print("Stream closed", flush=True)
        except:
            pass

# Get user input from chat_input
if prompt := st.chat_input("Ask me anything!"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            # Request a streaming response from llaisys
            stream = client.chat.completions.create(
                model="qwen2",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
                max_tokens=1024,
            )

            # Call safe_stream_generator to process the stream response first
            response_text = st.write_stream(stream_generator(stream))
            
        except Exception as e:
            st.error(f"连接失败: {e}")
