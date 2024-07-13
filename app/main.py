import os
import time
import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError

# AWS Bedrock clientの初期化
# region = os.environ.get('AWS_DEFAULT_REGION', 'ap-northeast-1')
# print(f"Using region: {region}")
# bedrock = boto3.client('bedrock-runtime', region_name=region)

def main():
    st.title("生成AIプロトタイピングApp集")
    st.warning("試作品につき、品質の保証はありません", icon="🚨")


    # サイドバーにラジオボタンを追加
    mode = st.sidebar.radio("ユースケース", ["チャットモード", "開発モード"])

    if mode == "チャットモード":
        research_mode()
    elif mode == "開発モード":
        development_mode()

def research_mode():
    st.header("チャットモード")
    # テキスト入力を追加
    user_input = st.text_input("研究テーマを入力してください")
    
    if st.button("生成"):
        if user_input:
            response = generate_text(user_input)
            stream_response = stream_data(response)
            st.write_stream(stream_response)
        else:
            st.warning("テキストを入力してください")
            

def development_mode():
    st.header("開発モード")
    st.write("ここに開発モードの内容を記述します。")
    # 開発モードの具体的な内容をここに追加
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # アプリの再実行の際に履歴のチャットメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Say something")
    if prompt:
        # チャットメッセージコンテナにユーザーメッセージを表示
        st.chat_message("user").markdown(prompt)
        # チャット履歴にユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner('回答を生成中...'):
            bot_response = generate_text(prompt)
            # チャットメッセージコンテナにアシスタントのレスポンスを表示
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            # チャット履歴にアシスタントのレスポンスを追加
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            # st.write(f"User: {prompt}")
            # response = "System: " + generate_text(prompt)
            # stream_response = stream_data(response)
            # st.write_stream(stream_response)


def generate_text(prompt):
    # Create a Bedrock Runtime client
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")  # リージョンは必要に応じて変更してください

    # Set the model ID for Claude v2
    model_id = "anthropic.claude-v2:1"

    # Format the request payload
    request_body = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": 500,
        "temperature": 0.7,
        "top_p": 1,
    }

    # Convert the request to JSON
    request = json.dumps(request_body)

    try:
        # Invoke the model
        response = client.invoke_model(modelId=model_id, body=request)
        
        # Decode the response
        response_body = json.loads(response["body"].read())
        
        # Extract the generated text
        return response_body.get("completion", "No response generated.")
    
    except ClientError as e:
        print(f"An error occurred: {e}")
        return None

# タイプライター風の演出    
def stream_data(response):
        for word in response.split():
            yield word + " "
            time.sleep(0.05)  


if __name__ == "__main__":
    main()