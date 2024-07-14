import os
import pandas as pd
import time
import boto3
import json
import streamlit as st
from botocore.exceptions import ClientError
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_community.document_loaders import PubMedLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain_community.utilities import ArxivAPIWrapper
import arxiv

# AWS認証情報の設定（環境変数から読み込む）
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS_DEFAULT_REGION"]
os.environ['AWS_PROFILE'] = 'default'

def main():
    st.title("生成AIプロトタイピングApp集")
    
    st.divider()

    mode = st.sidebar.radio("ユースケース", ["研究モード", "シンプルチャット", "PubMed検索・要約", "wiki検索", "arxiv検索"])
    st.sidebar.warning("試作品につき、品質の保証はありません", icon="🚨")

    if mode == "研究モード":
        research_mode()
    elif mode == "シンプルチャット":
        simplechat_mode()
    elif mode == "PubMed検索・要約":
        pubmed_search_mode()
    elif mode == "wiki検索":
        wiki_search_mode()
    elif mode == "arxiv検索":
        arxiv_search_mode()

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
    st.info("通常のチャットモードです。LLMの事前学習内容を参照して回答されます。", icon=None)
    
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

def pubmed_search_mode():
    st.header("PubMed検索・要約")
    st.info("LLMでクエリを最適化し、関連記事をPubMedから検索と要約（オプション）を行います", icon=None)

    summarize = st.checkbox("LLMによる要約を行う(件数上限は少なくなります)", value=False)
    
    # if summarize:
    #     st.markdown("- LLMを用いてクエリを最適化します")
    #     st.markdown("- LLMを用いて各論文を要約します")
    # else:
    #     st.markdown("- LLMを用いてクエリを最適化します")

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
    max_docs = st.slider("検索する文献数", min_value=1, max_value=100 if not summarize else 10, value=50 if not summarize else 5)

    if st.button("検索" if not summarize else "検索・要約"):
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

                if summarize:
                    with st.spinner(f'文献 {i} を要約中...'):
                        summary_prompt = PromptTemplate(
                            input_variables=["text"],
                            template="以下の医学論文の要約を日本語で3文以内で作成してください：\n{text}\n要約："
                        )
                        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                        result["LLMによる要約"] = summary_chain.run(content).strip()

                results.append(result)

                if summarize:
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

            if not summarize:
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
def wiki_search_mode():
    
    st.title("Wikipedia検索")
    st.info("Wikipediaから関連記事を検索して回答を返します", icon=None)

    # 言語選択
    lang = st.radio("言語を選択してください:", ["日本語", "英語"])
    lang_code = "ja" if lang == "日本語" else "en"

    # BedrockLLMの設定
    llm = BedrockLLM(model_id="anthropic.claude-v2:1")

    # WikipediaRetrieverの設定
    retriever = WikipediaRetriever(lang=lang_code, top_k_results=5)

    # RetrievalQAの設定
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # 新しい質問の入力
    query = st.text_input("質問を入力してください:")

    if st.button("検索"):
        if query:
            with st.spinner('回答を生成中...'):
                result = qa({"query": query})
                answer = result['result']
                sources = result['source_documents']

                st.subheader("回答:")
                st.write(answer)
                
                st.subheader("Wikipedia記事:")
                for i, source in enumerate(sources, 1):
                    with st.expander(f"記事 {i}: {source.metadata.get('title', '不明')}"):
                        st.write(f"タイトル: {source.metadata.get('title', '不明')}")
                        st.write(f"URL: {source.metadata.get('source', '不明')}")
                        st.write("内容:")
                        st.write(source.page_content)

def arxiv_search_mode():
    st.header("arXiv検索・要約")
    st.info("LLMでクエリを最適化し、関連論文をarXivから検索と要約（オプション）を行います", icon=None)

    summarize = st.checkbox("LLMによる要約を行う(件数上限は少なくなります)", value=False)

    llm = BedrockLLM(credentials_profile_name="default", model_id="anthropic.claude-v2:1")

    query_optimization_prompt = PromptTemplate(
        input_variables=["query"],
        template="""以下の研究トピックについて、arXivでの検索に最適なクエリ(英語)を生成してください。
        より確実に情報を拾えるようにクエリを拡張してほしい。そのままarXivの検索に用いるので、あなたの一言や解説は一切不要で、生成したクエリだけを返してほしい。
        
        正しい例：
        研究トピック:LLMの比較
        最適化されたクエリ:Comparative analysis of language models

        正しい例：
        研究トピック:LLMを用いたデータ分析
        最適化されたクエリ:Data analysis using language models

        研究トピック: {query}

        最適化されたクエリ:"""
    )

    query_optimization_chain = LLMChain(llm=llm, prompt=query_optimization_prompt)

    user_query = st.text_input("研究したいトピックを入力してください:")
    max_docs = st.slider("検索する文献数", min_value=1, max_value=20 if not summarize else 5, value=5)

    if st.button("検索" if not summarize else "検索・要約"):
        with st.spinner('検索中...'):
            optimized_query = query_optimization_chain.run(user_query).strip()
            st.write(f"最適化されたクエリ: {optimized_query}")

            # arxivライブラリを直接使用して検索
            search = arxiv.Search(
                query=optimized_query,
                max_results=max_docs,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers_data = []
            for i, result in enumerate(search.results(), 1):
                paper_data = {
                    "Title": result.title,
                    "Authors": ", ".join(author.name for author in result.authors),
                    "Published": result.published.strftime("%Y-%m-%d"),
                    "URL": result.entry_id,
                    "Summary": result.summary,
                    "Journal Ref": result.journal_ref,
                    "DOI": result.doi,
                }

                st.subheader(f"論文 {i}")
                st.write(f"タイトル: {paper_data['Title']}")
                st.write(f"著者: {paper_data['Authors']}")
                st.write(f"出版日: {paper_data['Published']}")
                st.write(f"URL: {paper_data['URL']}")
                if paper_data['Journal Ref']:
                    st.write(f"ジャーナル参照: {paper_data['Journal Ref']}")
                if paper_data['DOI']:
                    st.write(f"DOI: {paper_data['DOI']}")
                
                with st.expander("要約"):
                    st.write(paper_data['Summary'])

                if summarize:
                    with st.spinner(f'論文 {i} を要約中...'):
                        summary_prompt = PromptTemplate(
                            input_variables=["text"],
                            template="以下の論文の要約を日本語で3文以内で作成してください：\n{text}\n要約："
                        )
                        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                        llm_summary = summary_chain.run(paper_data["Summary"]).strip()
                        st.write("LLMによる要約:")
                        st.write(llm_summary)
                        paper_data["LLM要約"] = llm_summary

                papers_data.append(paper_data)
                st.divider()

            # DataFrameの作成と表示
            df = pd.DataFrame(papers_data)
            st.subheader("検索結果一覧")
            st.dataframe(df[["Title", "Authors", "Published", "URL"]])

            # CSVファイルとしてダウンロード
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="検索結果をCSVでダウンロード",
                data=csv,
                file_name="arxiv_search_results.csv",
                mime="text/csv",
            )


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