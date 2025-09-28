# 导入streamlit库用于创建web应用，random库用于生成随机数，time库用于控制时间
import streamlit as st
import random
import time

# 定义一个函数，用于生成流式响应，模拟聊天机器人的回答
def response_generator():
    # 从预设的回答中随机选择一个
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    # 将选择的回答逐词输出，每输出一个词暂停0.2秒
    for word in response.split():
        yield word + " "
        time.sleep(0.2)

# 设置web应用的标题为"Simple chat"
st.title("Simple chat")

# 初始化聊天历史记录，如果'st.session_state'中没有'messages'键，则创建它
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示之前会话中的消息历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("What is up?"):
    # 将用户的消息添加到聊天历史记录中
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 在聊天窗口中显示用户的消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 在聊天窗口中显示助手的响应
    with st.chat_message("assistant"):
        # 使用streamlit的流式数据显示功能显示助手的逐词响应
        response = st.write_stream(response_generator())
    # 将助手的响应也添加到聊天历史记录中
    st.session_state.messages.append({"role": "assistant", "content": response})