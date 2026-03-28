import uuid

import streamlit as st

from llm import stream_ai_response

st.set_page_config(page_title="소득세 챗봇", page_icon=":robot:")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title(":robot: 소득세 챗봇")
st.caption("소득세 관련 질문을 무엇이든 물어보세요.")

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input("소득세 관련 질문을 입력하세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 준비중입니다..."):
        chunks: list[str] = []

        def token_stream():
            for delta in stream_ai_response(
                user_question, session_id=st.session_state.session_id
            ):
                chunks.append(delta)
                yield delta

        with st.chat_message("assistant"):
            st.write_stream(token_stream)

        st.session_state.message_list.append(
            {"role": "assistant", "content": "".join(chunks)}
        )