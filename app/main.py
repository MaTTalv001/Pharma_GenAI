import os
import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError

# AWS Bedrock clientの初期化
# region = os.environ.get('AWS_DEFAULT_REGION', 'ap-northeast-1')
# print(f"Using region: {region}")
# bedrock = boto3.client('bedrock-runtime', region_name=region)

def main():
    st.title("研究開発アプリケーション")

    # サイドバーにラジオボタンを追加
    mode = st.sidebar.radio("ユースケース", ["研究モード", "開発モード"])

    if mode == "研究モード":
        research_mode()
    elif mode == "開発モード":
        development_mode()

def research_mode():
    st.header("研究モード")
    # テキスト入力を追加
    user_input = st.text_input("研究テーマを入力してください")
    
    if st.button("生成"):
        if user_input:
            response = generate_text(user_input)
            st.write(response)
        else:
            st.warning("テキストを入力してください")

def development_mode():
    st.header("開発モード")
    st.write("ここに開発モードの内容を記述します。")
    # 開発モードの具体的な内容をここに追加


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

if __name__ == "__main__":
    main()