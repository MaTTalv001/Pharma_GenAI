import os
import time
import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError

def main():
    st.title("生成AIプロトタイピングApp集")
    st.warning("試作品につき、品質の保証はありません", icon="🚨")

    mode = st.sidebar.radio("ユースケース", ["チャットモード", "開発モード"])

    if mode == "チャットモード":
        chat_mode()
    elif mode == "開発モード":
        development_mode()

def chat_mode():
    st.header("チャットモード")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("メッセージを入力してください")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.spinner('回答を生成中...'):
            response = generate_text(st.session_state.chat_messages)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

def development_mode():
    st.header("開発モード")
    
    if "dev_messages" not in st.session_state:
        st.session_state.dev_messages = []

    for message in st.session_state.dev_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("開発に関する質問を入力してください")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.dev_messages.append({"role": "user", "content": prompt})

        with st.spinner('回答を生成中...'):
            response = generate_text(st.session_state.dev_messages)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.dev_messages.append({"role": "assistant", "content": response})

def generate_text(messages):
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")
    model_id = "anthropic.claude-v2:1"

    # メッセージ履歴を適切な形式に変換
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"role": "human", "content": msg["content"]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": msg["content"]})

    # リクエストボディの作成
    request_body = {
        "prompt": "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in formatted_messages]) + "\n\nAssistant:",
        "max_tokens_to_sample": 500,
        "temperature": 0.7,
        "top_p": 1,
    }

    request = json.dumps(request_body)

    try:
        response = client.invoke_model(modelId=model_id, body=request)
        response_body = json.loads(response["body"].read())
        return response_body.get("completion", "No response generated.")
    except ClientError as e:
        print(f"An error occurred: {e}")
        return "エラーが発生しました。もう一度お試しください。"

# タイプライター風の演出
def stream_data(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if __name__ == "__main__":
    main()