import chromadb

# ChromaDBのクライアントを初期化
client = chromadb.PersistentClient(path="./raptor_db")

# コレクションを取得
collection = client.get_or_create_collection("raptor")

# コレクション内のデータ件数を確認
data_count = collection.count()
print(f"データ件数: {data_count}")

# コレクション内の全データを取得
# 大量のデータがある場合は、`limit` を指定して分割取得すると効率的
data_count = 10
all_data = collection.get(include=["documents", "metadatas"], limit=data_count)

# チャンクデータの内容を確認
if all_data and "documents" in all_data:
    documents = all_data["documents"]
    metadatas = all_data.get("metadatas", [{}] * len(documents))  # メタデータがない場合に空辞書を補完

    for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
        print(f"\nデータ {idx + 1}:")
        print(f"チャンク内容: {doc}")
        print(f"メタデータ: {meta}")
else:
    print("コレクション内にデータはありません。")