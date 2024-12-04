import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.packs.raptor import RaptorPack


# RaptorPackの初期化
client = chromadb.PersistentClient(path="./raptor_db")
collection = client.get_or_create_collection("raptor")

vector_store = ChromaVectorStore(chroma_collection=collection)

# OpenAIのモデルの初期化
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

raptor_pack = RaptorPack(
    [],
    embed_model=embed_model,  # クラスタのembeddingsに使用
    llm=llm,  # サマリの生成に使用
    vector_store=vector_store,  # ストレージの設定
    similarity_top_k=5,  # 各レイヤーごとのtop-k、または"collapsed"の場合は全体のtop-k
    mode="collapsed",  # モード設定のデフォルト値
)

# # 検索実行 for debug.
# # Collapsed Tree mode.
# nodes = raptor_pack.run("オグリキャップの主な勝ち鞍は？", mode="collapsed")

# formatted_nodes = [f"ID: {n.id_}\nScore: {n.get_score()}\n\n{n.get_content()[:200]}..." for n in nodes]
# print("total nodes: ", len(formatted_nodes))
# print()
# print("\n----\n".join(formatted_nodes))

# # Tree Traversal mode.
# nodes = raptor_pack.run("オグリキャップの主な勝ち鞍は？", mode="tree_traversal")

# formatted_nodes = [f"ID: {n.id_}\nScore: {n.get_score()}\n\n{n.get_content()[:200]}..." for n in nodes]
# print("total nodes: ", len(formatted_nodes))
# print()
# print("\n----\n".join(formatted_nodes))

# 問い合わせ実行
from llama_index.packs.raptor import RaptorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

collapsed_retriever = RaptorRetriever([], embed_model=embed_model, llm=llm, vector_store=vector_store, mode="collapsed")
collapsed_retriever_query_engine = RetrieverQueryEngine.from_args(collapsed_retriever, llm=llm)

query = "オグリキャップの主な勝ち鞍を教えて。勝ち鞍とは優勝したレースのことです。"
response = collapsed_retriever_query_engine.query(query)
print("\ncollapsed_retriever:")
print(f"{query}:\n{str(response)}")

tree_traversal_retriever = RaptorRetriever([], embed_model=embed_model, llm=llm, vector_store=vector_store, mode="tree_traversal")
tree_traversal_retriever_query_engine = RetrieverQueryEngine.from_args(tree_traversal_retriever, llm=llm)

response = tree_traversal_retriever_query_engine.query(query)
print("\ntree_traversal_retriever:")
print(f"{query}:\n{str(response)}")