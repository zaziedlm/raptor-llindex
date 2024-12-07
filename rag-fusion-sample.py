# 参考記事： LlamaIndexでRAG Fusionを試す
# https://zenn.dev/kun432/scraps/87f2c5ac61ccbf
from dotenv import load_dotenv
import os
import asyncio

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# .envファイルの読み込み
load_dotenv()

PHOENIX_API_KEY = os.getenv('LLAMATRACE_API_KEY')
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
    project_name=os.getenv("LLAMATRACE_PROJECT"),
    endpoint=os.getenv("LLAMATRACE_ENDPOINT"),
)

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
# Initialize the LlamaIndexInstrumentor
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# OpenAIのモデルの初期化
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
embed_model = OpenAIEmbedding(model="text-embedding-3-small",
                              embed_batch_size=256)

from chromadb import PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

from llama_index.core import StorageContext


# インデックスの作成
chroma_client = PersistentClient(path="./raptor_db")
chroma_collection = chroma_client.get_or_create_collection("raptor")

# # Chroma DB から全てのデータを取得
# all_docs = chroma_collection.get()

# # データ内容を確認
# for doc in all_docs:
#     # print(f"ID: {doc['id']}, Content: {doc['document']}")
#     print(doc)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_store_index = VectorStoreIndex.from_vector_store(vector_store)

# from llama_index.core import StorageContext
# from llama_index.core import VectorStoreIndex
# storage_context = StorageContext.from_defaults(persist_dir="./raptor_db")
# vector_store_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# インデックスに登録されたドキュメント数を確認
print("Number of documents in the docstore:", len(vector_store_index.docstore.docs))

# 登録されたトークンの確認
for doc_id, doc in vector_store_index.docstore.docs.items():
    print(f"Document ID: {doc_id}, Content: {doc.get_content()}")

# Chroma DB コレクションの確認
print(f"Collection name: {chroma_collection.name}")
print(f"Number of documents in Chroma DB collection: {len(chroma_collection.get())}")

# from llama_index.core import StorageContext
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index_struct = storage_context.from_defaults(persist_dir="./raptor_db")
# vector_store_index = VectorStoreIndex(index_struct=index_struct, storage_context=storage_context)

# vector_store_index = VectorStoreIndex(
#     # documents=[],  # あるいは事前に取り込んだドキュメントリスト
#     index_struct=index_struct,
#     embedding=embed_model,
#     vector_store=vector_store
# )


from llama_index.core import PromptTemplate

query_str = "オグリキャップの主な勝ち鞍を教えて。勝ち鞍とは優勝したレースのことです。"
query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)
query_gen_prompt = PromptTemplate(query_gen_prompt_str)

def generate_queries(llm, query_str: str, num_queries: int = 4):
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries

queries = generate_queries(llm, query_str, num_queries=5)
print(queries)

from tqdm.asyncio import tqdm

async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict

# get retrievers
# from llama_index.retrievers.bm25 import BM25Retriever
# BM25Retriever については、日本語トークナイザーの問題ありで、除外としている
#
# LlamaIndexでBM25Retrieverを試す
# https://zenn.dev/kun432/scraps/41929e657e66d7

# # Retriever を作成
# retriever = vector_store_index.as_retriever()

async def get_retrievers(vector_index):
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)
     
    # bm25_retriever = BM25Retriever.from_defaults(
    #     docstore=vector_index.docstore,
    #     similarity_top_k=5,
    # )
    # results_dict = await run_queries(queries, [vector_retriever, bm25_retriever])
    results_dict = await run_queries(queries, [vector_retriever])
    return results_dict

# 非同期処理の実行
results_dict = asyncio.run(get_retrievers(vector_store_index))    
    
from typing import List
from llama_index.core.schema import NodeWithScore


def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]

final_results = fuse_results(results_dict, similarity_top_k=3)

for i, n in enumerate(final_results):
    print(f"Index: {i}\nScore: {n.score}\nText: {n.text}\n********\n")
