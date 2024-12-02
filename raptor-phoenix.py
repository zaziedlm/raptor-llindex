from dotenv import load_dotenv
from pathlib import Path
import requests
import os

# .envファイルの読み込み
load_dotenv()

from phoenix.otel import register

# Add Phoenix API Key for tracing
PHOENIX_API_KEY = os.getenv("LLAMATRACE_API_KEY")
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

# configure the Phoenix tracer
tracer_provider = register(
    project_name=os.getenv("LLAMATRACE_PROJECT"),
    endpoint=os.getenv("LLAMATRACE_ENDPOINT"),
)

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
# Initialize the LlamaIndexInstrumentor
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

from llama_index.core import SimpleDirectoryReader

# Wikipediaからデータを取得して保存する関数
def fetch_and_save_wiki_data(wiki_titles_string):
    # カンマ区切りの文字列をリストに変換
    wiki_titles = [title.strip() for title in wiki_titles_string.split(",")]

    # Wikipediaからのデータ読み込み
    for title in wiki_titles:
        response = requests.get(
            "https://ja.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                # 'exintro': True,
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]

        data_path = Path("data")
        if not data_path.exists():
            Path.mkdir(data_path)

        with open(data_path / f"{title}.txt", "w", encoding="utf-8") as fp:
            fp.write(wiki_text)

def runraptor(query) :
    # # .envファイルの読み込み
    # load_dotenv()

    # 環境変数からカンマ区切りの文字列を取得
    wiki_titles_string = os.getenv("SCRAPING_WIKI_TITLES")

    # Wikipediaからデータを取得して保存
    fetch_and_save_wiki_data(wiki_titles_string)

    # データの読み込み
    documents = SimpleDirectoryReader("data").load_data()

    import chromadb
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.packs.raptor import RaptorPack

    # RaptorPackの初期化
    client = chromadb.PersistentClient(path="./raptor_db")
    collection = client.get_or_create_collection("raptor")

    vector_store = ChromaVectorStore(chroma_collection=collection)

    # OpenAIのモデルの初期化
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

    raptor_pack = RaptorPack(
        documents,
        embed_model=embed_model,  # クラスタのembeddingsに使用
        llm=llm,  # サマリの生成に使用
        vector_store=vector_store,  # ストレージの設定
        similarity_top_k=5,  # 各レイヤーごとのtop-k、または"collapsed"の場合は全体のtop-k
        mode="collapsed",  # モード設定のデフォルト値
        transformations=[
            SentenceSplitter(chunk_size=400, chunk_overlap=50)
        ],  # インデックスへの投入時にtransformationを適用
    )

    # 検索実行 for debug.
    # Collapsed Tree mode.
    nodes = raptor_pack.run("オグリキャップの主な勝ち鞍は？", mode="collapsed")

    formatted_nodes = [f"ID: {n.id_}\nScore: {n.get_score()}\n\n{n.get_content()[:200]}..." for n in nodes]
    print("total nodes: ", len(formatted_nodes))
    print()
    print("\n----\n".join(formatted_nodes))

    # Tree Traversal mode.
    nodes = raptor_pack.run("オグリキャップの主な勝ち鞍は？", mode="tree_traversal")

    formatted_nodes = [f"ID: {n.id_}\nScore: {n.get_score()}\n\n{n.get_content()[:200]}..." for n in nodes]
    print("total nodes: ", len(formatted_nodes))
    print()
    print("\n----\n".join(formatted_nodes))

    # 問い合わせ実行
    from llama_index.packs.raptor import RaptorRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine

    collapsed_retriever = RaptorRetriever([], embed_model=embed_model, llm=llm, vector_store=vector_store, mode="collapsed")
    collapsed_retriever_query_engine = RetrieverQueryEngine.from_args(collapsed_retriever, llm=llm)

    query = "オグリキャップの主な勝ち鞍を教えて。勝ち鞍とは優勝したレースのことです。"
    response = collapsed_retriever_query_engine.query(query)
    print(f"{query}:{str(response)}")

    tree_traversal_retriever = RaptorRetriever([], embed_model=embed_model, llm=llm, vector_store=vector_store, mode="tree_traversal")
    tree_traversal_retriever_query_engine = RetrieverQueryEngine.from_args(tree_traversal_retriever, llm=llm)

    response = tree_traversal_retriever_query_engine.query(query)
    print(f"{query}:{str(response)}")

if __name__ == "__main__":

    runraptor("オグリキャップの主な勝ち鞍を教えて。勝ち鞍とは優勝したレースのことです。")

