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


if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 每次都创建新的流式连接
            stream = client.chat.completions.create(
                model="qwen2",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
                max_tokens=256,
            )
            
            # 使用完整的生成器处理流
            def safe_stream_generator():
                try:
                    full_response = ""
                    think_mode = True
                    first_chunk = True
                    
                    for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                content = delta.content
                                full_response += content
                                
                                if think_mode:
                                    if "</think>" in content:
                                        parts = content.split("</think>", 1)
                                        think_part = parts[0]
                                        main_part = parts[1] if len(parts) > 1 else ""
                                        
                                        if first_chunk:
                                            yield f"> {think_part}"
                                        else:
                                            yield think_part
                                        
                                        think_mode = False
                                        
                                        if main_part:
                                            yield f"\n\n{main_part}"
                                    else:
                                        if first_chunk:
                                            yield f"> {content}"
                                        else:
                                            lines = content.split('\n')
                                            formatted_lines = []
                                            for i, line in enumerate(lines):
                                                if i == 0:
                                                    formatted_lines.append(line)
                                                else:
                                                    formatted_lines.append(f"\n> {line}")
                                            yield ''.join(formatted_lines)
                                else:
                                    yield content
                                
                                first_chunk = False
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"流式处理错误: {e}")
                finally:
                    # 确保连接关闭
                    try:
                        stream.close()
                    except:
                        pass
            
            response_text = st.write_stream(safe_stream_generator())
            
        except Exception as e:
            st.error(f"连接失败: {e}")
            # 可选：添加重试按钮
            if st.button("重试"):
                st.rerun()