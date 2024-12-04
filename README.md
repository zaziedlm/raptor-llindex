# Raptor-LLIndex プロジェクト

## 概要
Raptor LLIndexは、高速で効率的なインデックス作成と検索を提供するRaptor実装のサンプルです。

## ファイル一覧

### raptor-sample.py
- .envファイルを読み込み、環境変数を設定します。
- 環境変数から取得したWikipediaのタイトルに基づいてデータを取得し、dataディレクトリに保存します。
- SimpleDirectoryReaderを使用してdataディレクトリからデータを読み込みます。
- ChromaVectorStoreを初期化し、OpenAIのEmbeddingモデルとLLMモデルを設定します。
- RaptorPackを初期化し、データのチャンク分割とEmbeddingを行います。
- RaptorRetrieverを使用してクエリを実行し、結果を表示します。

### raptor-oss-trace.py
- .envファイルを読み込み、環境変数を設定します。
- 環境変数から取得したWikipediaのタイトルに基づいてデータを取得し、dataディレクトリに保存します。
- SimpleDirectoryReaderを使用してdataディレクトリからデータを読み込みます。
- ChromaVectorStoreを初期化し、OpenAIのEmbeddingモデルとLLMモデルを設定します。
- RaptorPackを初期化し、データのチャンク分割とEmbeddingを行います。
- RaptorRetrieverを使用してクエリを実行し、結果を表示します。
- Phoenixのトレーシング機能を設定し、トレースデータを収集します。

### raptor-phoenix.py
- .envファイルを読み込み、環境変数を設定します。
- 環境変数から取得したWikipediaのタイトルに基づいてデータを取得し、dataディレクトリに保存します。
- SimpleDirectoryReaderを使用してdataディレクトリからデータを読み込みます。
- ChromaVectorStoreを初期化し、OpenAIのEmbeddingモデルとLLMモデルを設定します。
- RaptorPackを初期化し、データのチャンク分割とEmbeddingを行います。
- RaptorRetrieverを使用してクエリを実行し、結果を表示します。
- Phoenixのトレーシング機能を設定し、トレースデータを収集します。

### retrieve-raptor.py
- 永続DBを使ってRaptorRetrieverとRetrieverQueryEngineを使用してクエリを実行し、結果を表示します。

### view-chromedb.py
- ChromaVectorStoreからデータを取得し、チャンク内容とメタデータを表示します。
