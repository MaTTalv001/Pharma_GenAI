import os
import pandas as pd
import time
import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PubMedLoader

# AWS認証情報の設定（環境変数から読み込む）
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS_DEFAULT_REGION"]
os.environ['AWS_PROFILE'] = 'default'

def main():
    st.title("生成AIプロトタイピングApp集")
    st.warning("試作品につき、品質の保証はありません", icon="🚨")
    st.divider()

    mode = st.sidebar.radio("ユースケース", ["研究モード", "シンプルチャット", "PubMed検索","PubMed検索・要約" ])

    if mode == "研究モード":
        research_mode()
    elif mode == "シンプルチャット":
        simplechat_mode()
    elif mode == "PubMed検索・要約":
        literature_search_mode()
    elif mode == "PubMed検索":
        pubmed_search_mode()

def research_mode():
    st.header("研究モード")
    
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

def simplechat_mode():
    st.header("シンプルチャット")
    
    if "dev_messages" not in st.session_state:
        st.session_state.dev_messages = []

    for message in st.session_state.dev_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("質問を入力してください")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.dev_messages.append({"role": "user", "content": prompt})

        with st.spinner('回答を生成中...'):
            response = generate_text(st.session_state.dev_messages)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.dev_messages.append({"role": "assistant", "content": response})

def pubmed_search_base(mode):
    st.header("PubMed検索" if mode == "search" else "PubMed検索・要約")
    if mode == "search":
        st.markdown("- LLMを用いてクエリを最適化します")
    if mode == "summary":
        st.markdown("- LLMを用いてクエリを最適化します")
        st.markdown("- LLMを用いて各論文を要約します(処理に時間がかかるため件数上限は少なくしています)")

    llm = BedrockLLM(credentials_profile_name="default", model_id="anthropic.claude-v2:1")

    query_optimization_prompt = PromptTemplate(
        input_variables=["query"],
        template="""Pubmedで次のクエリで検索したいのですが、より確実に情報を拾えるようにクエリを拡張してほしい。そのままpubmedの検索に用いるので、あなたの一言や解説は一切不要で、生成したクエリだけを返してほしい。そのままpubmedの検索に用いるので、「最適化されたクエリ:」などの余計な文言は一切不要です。

良い例
クエリ: COVID-19の家族内伝播
最適化されたクエリ: COVID-19 AND (家族内伝播 OR household transmission)

間違いの例
クエリ: COVID-19の家族内伝播
最適化されたクエリ: 以下が生成したクエリです：COVID-19 AND household transmission

クエリ: {query}
最適化されたクエリ:"""
    )

    query_optimization_chain = LLMChain(llm=llm, prompt=query_optimization_prompt)

    user_query = st.text_input("研究したい医学トピックを入力してください:")
    max_docs = st.slider("検索する文献数", min_value=1, max_value=100 if mode == "search" else 10, value=50 if mode == "search" else 5)

    if st.button("検索" if mode == "search" else "検索・要約"):
        with st.spinner('検索中...'):
            optimized_query = query_optimization_chain.run(user_query).strip()
            optimized_query = optimized_query.split('\n')[-1].strip()
            st.write(f"最適化されたクエリ: {optimized_query}")

            loader = PubMedLoader(query=optimized_query, load_max_docs=max_docs)
            docs = loader.load()

            results = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                content = doc.page_content

                pmid = metadata.get('uid', '不明')
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != '不明' else ''

                result = {
                    "タイトル": metadata.get('Title', '不明'),
                    "出版日": metadata.get('Published', '不明'),
                    "PMID": pmid,
                    "PubMed URL": pubmed_url,
                    "概要": content[:500] + "..." if len(content) > 500 else content
                }

                if mode == "summary":
                    with st.spinner(f'文献 {i} を要約中...'):
                        summary_prompt = PromptTemplate(
                            input_variables=["text"],
                            template="以下の医学論文の要約を日本語で3文以内で作成してください：\n{text}\n要約："
                        )
                        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                        result["LLMによる要約"] = summary_chain.run(content).strip()

                results.append(result)

                if mode == "summary":
                    st.subheader(f"文献 {i}")
                    st.write(f"タイトル: {result['タイトル']}")
                    st.write(f"出版日: {result['出版日']}")
                    if pubmed_url:
                        st.markdown(f"PMID: [{pmid}]({pubmed_url})")
                    else:
                        st.write("PMID: 不明")
                    with st.expander("概要"):
                        st.write(result['概要'])
                    st.write("LLMによる要約:")
                    st.write(result['LLMによる要約'])
                    st.divider()

            # DataFrameの作成と表示
            df = pd.DataFrame(results)
            st.subheader("検索結果一覧")
            st.dataframe(df[["タイトル", "出版日", "PMID", "PubMed URL"]])

            # CSVファイルとしてダウンロード
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="検索結果をCSVでダウンロード",
                data=csv,
                file_name="pubmed_search_results.csv",
                mime="text/csv",
            )

            if mode == "search":
                # 個別の論文詳細表示
                st.subheader("論文詳細")
                for i, result in enumerate(results, 1):
                    with st.expander(f"論文 {i}: {result['タイトル']}"):
                        st.write(f"タイトル: {result['タイトル']}")
                        st.write(f"出版日: {result['出版日']}")
                        if result['PubMed URL']:
                            st.markdown(f"PMID: [{result['PMID']}]({result['PubMed URL']})")
                        else:
                            st.write(f"PMID: {result['PMID']}")
                        st.write("概要:")
                        st.write(result['概要'])

def literature_search_mode():
    pubmed_search_base("summary")

def pubmed_search_mode():
    pubmed_search_base("search")

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