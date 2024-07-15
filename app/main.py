import os
import pandas as pd
import numpy as np
import time
import boto3
import json
import requests
import feedparser
from bs4 import BeautifulSoup
from io import BytesIO
import arxiv
import wikipedia
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from botocore.exceptions import ClientError
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_community.document_loaders import PubMedLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.schema import Document


# AWS認証情報の設定
if 'aws_credentials' in st.secrets:
    os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
    os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
    os.environ['AWS_DEFAULT_REGION'] = st.secrets["aws_credentials"]["AWS_DEFAULT_REGION"]
else:
    # ローカル環境用のフォールバック
    os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key_id'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_access_key'
    os.environ['AWS_DEFAULT_REGION'] = 'your_region'

def get_llm():
    return BedrockLLM(
        model_id="anthropic.claude-v2:1",
        region_name=os.environ['AWS_DEFAULT_REGION']
    )

def main():
    st.title("生成AIプロトタイピングApp集")
    st.divider()

    mode = st.sidebar.radio("ユースケース", ["シンプルチャット", "企画相談","PubMed検索・要約", "Wikipedia検索", "arxiv検索", "ニュース取得"])
    st.sidebar.warning("試作品につき、品質の保証はありません", icon="🚨")

    if mode == "研究モード":
        research_mode()
    elif mode == "シンプルチャット":
        simplechat_mode()
    elif mode == "PubMed検索・要約":
        pubmed_search_mode()
    elif mode == "Wikipedia検索":
        wiki_search_mode()
    elif mode == "arxiv検索":
        arxiv_search_mode()
    elif mode == "企画相談":
        idea_consultation_mode()
    elif mode == "ニュース取得":
        news_feed_mode()

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
                #st.markdown(response)
                st.write_stream(stream_data(response))
            st.session_state.dev_messages.append({"role": "assistant", "content": response})

def pubmed_search_mode():
    st.header("PubMed検索・要約")
    st.info("LLMでクエリを最適化し、PubMedから検索と要約（オプション）を行います", icon=None)

    summarize = st.checkbox("LLMによる要約を行う(件数上限は少なくなります)", value=False)
    
    llm = get_llm()

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

            df = pd.DataFrame(results)
            st.subheader("検索結果一覧")
            st.dataframe(df[["タイトル", "出版日", "PMID", "PubMed URL"]])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="検索結果をCSVでダウンロード",
                data=csv,
                file_name="pubmed_search_results.csv",
                mime="text/csv",
            )

            if not summarize:
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
    st.header("Wikipedia検索")
    st.info("Wikipediaから関連記事を検索して回答を返します。候補記事をベクトル化する過程で少々時間がかかります", icon=None)

    # セッション状態の初期化
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'search_words' not in st.session_state:
        st.session_state.search_words = []

    lang = st.radio("言語を選択してください:", ["日本語", "英語"])
    lang_code = "ja" if lang == "日本語" else "en"

    llm = get_llm()
    embeddings = BedrockEmbeddings()
    retriever = WikipediaRetriever(lang=lang_code, top_k_results=3)

    # テキスト分割器の設定
    text_splitter = CharacterTextSplitter(chunk_size=2400, chunk_overlap=200)

    
    query = st.text_input("質問を入力してください:")
    
    if st.button("検索"):
        if query:
            with st.spinner('回答を生成中...'):
                # Step 1: 適切な検索キーワードの生成
                keyword_prompt = PromptTemplate(
                    input_variables=["query"],
                    template="以下の質問に対して、最も適切なWikipedia検索キーワードを3つ提案してください。カンマ区切りで出力してください。\n\n質問: {query}\n\n検索キーワード:"
                )
                keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt)
                search_keywords = keyword_chain.run(query).strip().split(',')
                st.session_state.search_words = search_keywords

                st.write(f"検索キーワード: {', '.join(search_keywords)}")

                # Step 2: 生成されたキーワードでWikipedia検索
                docs = []
                for keyword in search_keywords:
                    new_docs = retriever.get_relevant_documents(keyword.strip())
                    # 各記事の最初の部分のみを使用
                    docs.extend([Document(page_content=doc.page_content, metadata=doc.metadata) for doc in new_docs])

                # Step 3: テキストの分割とベクトルストアの作成
                chunks = text_splitter.split_documents(docs)
                # チャンク数を制限
                max_chunks = 2400
                if len(chunks) > max_chunks:
                    chunks = chunks[:max_chunks]
                    st.warning(f"記事が長いため、最初の{max_chunks}チャンクのみを使用します。")

                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore

                # Step 4: 類似度検索用のリトリーバー作成
                similar_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                # Step 5: RetrievalQAチェーンの作成と実行
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=similar_retriever)
                result = qa_chain.run(query)

                st.subheader("回答:")
                st.write(result)

                # 関連Wikipedia記事の表示を最適化
                st.subheader("関連Wikipedia記事:")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"記事 {i}: {doc.metadata.get('title', '不明')}"):
                        st.write(f"タイトル: {doc.metadata.get('title', '不明')}")
                        st.write(f"URL: {doc.metadata.get('source', '不明')}")
                        st.write("内容（最初の500文字）:")
                        st.write(doc.page_content[:500] + "...")

    # 追加の質問セクション
    if st.session_state.vectorstore is not None:
        st.subheader("追加の質問")
        additional_query = st.text_input("追加の質問を入力してください:")
        if st.button("追加の質問を送信") and additional_query:
            with st.spinner('追加の回答を生成中...'):
                similar_retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=similar_retriever)
                additional_answer = qa_chain.run(additional_query)
                st.subheader("追加の回答:")
                st.write(additional_answer)

def arxiv_search_mode():
    st.header("arXiv検索・要約")
    st.info("LLMでクエリを最適化し、関連論文をarXivから検索と要約（オプション）を行います", icon=None)

    summarize = st.checkbox("LLMによる要約を行う(件数上限は少なくなります)", value=False)

    llm = get_llm()

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
def idea_consultation_mode():
    st.header("企画相談")
    st.info("企画を考える壁打ち相手として、より深い考察と構造化された回答を提供します。", icon="💡")
    
    if "idea_messages" not in st.session_state:
        st.session_state.idea_messages = []

    for message in st.session_state.idea_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("アイデアや質問を入力してください")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.idea_messages.append({"role": "user", "content": prompt})

        with st.spinner('回答を生成中...'):
            response = generate_idea_response(st.session_state.idea_messages)
            with st.chat_message("assistant"):
                st.markdown(response)
                #st.write_stream(stream_data(response))
            st.session_state.idea_messages.append({"role": "assistant", "content": response})

def generate_idea_response(messages):
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ['AWS_DEFAULT_REGION']
    )
    model_id = "anthropic.claude-v2:1"

    # システムプロンプトの追加
    system_prompt = """# 優秀な壁打ちコンサルティング役のシステムプロンプト

あなたは卓越した壁打ちコンサルティング役です。ユーザーの相談に対して、以下の点を踏まえ、プロフェッショナルなコンサルティングを提供してください。
## 出力の体裁
- 改行やマークダウン用い、可読性を高くする。

## 基本姿勢

1. **傾聴と共感**
   - クライアントの話を聞き、感情や懸念に対して共感を示す。

2. **オープンエンドな質問**
   - 「どのようにお考えですか？」「具体的にはどういった状況でしょうか？」など、クライアントが自由に表現できる質問をする。

3. **不足している論点の深掘り**
   - 曖昧な部分や詳細が不足している箇所について、適切な質問で深掘りする。

4. **質問による自己反省の促進**
   - クライアントの自己分析を促すため、回答前に質問で返す。

5. **構造的・論理的・定量的なアドバイス**
   - 具体的で実行可能な提案を、データや事例に基づいて行う。
   - 論理的フレームワークを用いて問題を分析し、解決策を提示する。
   - 提案の難度、効果、費用などを★マークの数（1〜5）で表現する。

6. **建設的フィードバック**
   - クライアントの考えや提案に対して、具体的かつ建設的なフィードバックを提供する。

7. **プロセスの透明性**
   - コンサルティングの進行方法や期待される成果を明確に説明する。
   - 幅広い視点、深掘り、リフレーミングを適宜使い分ける。

8. **クライアント目標との整合性**
   - クライアントの目標やニーズに基づいた解決策を提案する。

9. **進捗確認と調整**
   - 定期的に進捗を確認し、必要に応じて方針を調整する。

10. **クライアントの自主性尊重**
    - 最終的な決定権はクライアントにあることを尊重し、自信を持って決定できるようサポートする。

## 高度なコンサルティング技法

1. **ソクラテス式対話法**
   - 質問を通じてクライアントの自己発見を促す。

2. **GROWモデル**
   - 目標設定(Goal)、現状認識(Reality)、選択肢検討(Options)、意志確認(Will)の4ステップでサポート。

3. **PEST分析**
   - 政治(Political)、経済(Economic)、社会(Social)、技術(Technological)の外部環境要因を分析。

4. **SWOT分析**
   - 強み(Strengths)、弱み(Weaknesses)、機会(Opportunities)、脅威(Threats)から内部・外部環境を分析。

5. **具体例の活用**
   - アドバイスや提案時に、関連する具体的な成功事例や失敗事例を提示する。

6. **業界知識の活用**
   - クライアントの業界や分野の専門知識を引き出し、それを基に議論を展開する。

7. **リスク分析と対策**
   - 提案や戦略に関連するリスクを特定し、具体的な対策を検討する。

8. **イノベーションと創造性の促進**
   - 従来の枠にとらわれない創造的な解決策を探ることを奨励する。

9. **文化的配慮**
    - クライアントの文化的背景や組織文化を考慮に入れた提案を行う。

11. **倫理的考慮**
    - 提案や戦略の倫理的側面を検討し、社会的責任を考慮する。

12. **フィードバックループの構築**
    - 提案実施後のフィードバックと継続的な改善プロセスの重要性を強調する。

13. **デザイン思考の適用**
    - ユーザー中心のアプローチを取り入れ、エンパシーマッピングやプロトタイピングなどの手法を活用する。

14. **シナリオプランニング**
    - 複数の将来シナリオを想定し、それぞれに対する戦略を検討する。

## 注意事項

- 常に建設的で、クライアントのアイデアを発展させることを目指す。
- 専門用語を使用する際は、必ず分かりやすく説明を加える。
- クライアントの理解度を確認しながら進める。
- 必要に応じて、議論の要約や次のステップの提案を行う。

このガイドラインに従い、クライアントにとって最大の価値を生み出すコンサルティングを提供してください。"""

    # メッセージ履歴を適切な形式に変換
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"role": "Human", "content": msg["content"]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "Assistant", "content": msg["content"]})

    # リクエストボディの作成
    request_body = {
        "prompt": "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in formatted_messages]) + "\n\nAssistant:",
        "max_tokens_to_sample": 1000,
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
    
def news_feed_mode():
    st.header("ニュース取得")
    st.info("複数のRSSフィードからニュース記事を取得します。オプションで自社戦略との関連性を計算できます。", icon="📰")

    company_strategy = "日本の製薬企業で低分子創薬に強みを持っている。感染症や中枢神経系の疾患に積極的に取り組んでいる。今後の成長に向け、AI活用を含むデジタルトランスレーション(DX)や海外事業展開に積極的である"

    calculate_similarity = st.checkbox("AIで自社戦略との関連性を計算する（処理時間が長くなります）", value=False)

    if st.button("ニュース取得"):
        with st.spinner('ニュースを取得中...'):
            # RSSフィードのURLリスト
            rss_urls = [
                'https://business.nikkei.com/rss/sns/nb.rdf',
                'https://business.nikkei.com/rss/sns/nb-x.rdf',
                'https://business.nikkei.com/rss/sns/nb-plus.rdf',
                'https://xtech.nikkei.com/rss/xtech-it.rdf',
                'https://xtech.nikkei.com/rss/xtech-mono.rdf',
                'https://xtech.nikkei.com/rss/xtech-hlth.rdf',
                'https://xtech.nikkei.com/rss/index.rdf',
                'https://rss.itmedia.co.jp/rss/2.0/news_bursts.xml'
            ]

            news_articles = []

            def clean_html(html_content):
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text().strip()

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            for url in rss_urls:
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    feed = feedparser.parse(BytesIO(response.content))
                    
                    if feed.entries:
                        for entry in feed.entries:
                            article = {
                                'title': entry.get('title', ''),
                                'link': entry.get('link', ''),
                                'published': entry.get('published', '') or entry.get('updated', '')
                            }
                            
                            if 'summary' in entry:
                                article['summary'] = clean_html(entry.summary)
                            elif 'description' in entry:
                                article['summary'] = clean_html(entry.description)
                            elif 'content' in entry:
                                article['summary'] = clean_html(entry.content[0].value)
                            else:
                                article['summary'] = ''
                            
                            news_articles.append(article)
                
                except Exception as e:
                    st.error(f"Error processing {url}: {str(e)}")

            if calculate_similarity:
                with st.spinner('関連性を計算中...'):
                    # 埋め込みモデルの読み込み
                    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

                    # 自社戦略の埋め込み
                    strategy_embedding = model.encode([company_strategy])[0]

                    # 各ニュース記事の類似度計算
                    for article in news_articles:
                        article_text = article['title'] + " " + article['summary']
                        article_embedding = model.encode([article_text])[0]
                        similarity = cosine_similarity([strategy_embedding], [article_embedding])[0][0]
                        article['similarity_score'] = similarity

                    # データフレームの作成と類似度でのソート
                    df = pd.DataFrame(news_articles)
                    df = df.sort_values('similarity_score', ascending=False)
                    st.subheader("取得したニュース一覧（自社戦略との関連性順）")
            else:
                # 類似度計算なしの場合
                df = pd.DataFrame(news_articles)
                st.subheader("取得したニュース一覧")

            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="ニュース一覧をCSVでダウンロード",
                data=csv,
                file_name="news_articles.csv",
                mime="text/csv",
            )

            st.subheader("ニュース詳細10件")
            for i, article in enumerate(df[0:10].to_dict('records')):
                title = f"ニュース {i+1}: {article['title']}"
                if calculate_similarity:
                    title += f" (関連性スコア: {article['similarity_score']:.4f})"
                
                with st.expander(title):
                    st.write(f"タイトル: {article['title']}")
                    st.write(f"公開日: {article['published']}")
                    st.write(f"リンク: {article['link']}")
                    st.write("要約:")
                    st.write(article['summary'])
                    if calculate_similarity:
                        st.write(f"関連性スコア: {article['similarity_score']:.4f}")

def generate_text(messages):
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ['AWS_DEFAULT_REGION']
    )
    model_id = "anthropic.claude-v2:1"

    # メッセージ履歴を適切な形式に変換
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"role": "Human", "content": msg["content"]})
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
        time.sleep(0.1)

if __name__ == "__main__":
    main()