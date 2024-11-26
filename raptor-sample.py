from dotenv import load_dotenv
from pathlib import Path
import requests
import os

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

if __name__ == "__main__":

    # .envファイルの読み込み
    load_dotenv()

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

