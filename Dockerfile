# Python 3.12 slim をベースイメージに採用
FROM python:3.12-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要な OS パッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージのコピーとインストール (キャッシュ効率化)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードとデータをコピー
COPY app/ ./app/
COPY lib/ ./lib/
COPY arxiv_graph.kuzu ./arxiv_graph.kuzu
COPY graph.html ./graph.html
COPY query.py ingest.py visualize.py ./

# 環境変数の設定 (Cloud Run の規約に対応)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV KUZU_DB_PATH=/app/arxiv_graph.kuzu

# Cloud Run がリッスンするポートを公開
EXPOSE 8080

# Cloud Run の $PORT 環境変数をバインドして uvicorn を起動
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
