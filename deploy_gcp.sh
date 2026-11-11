#!/usr/bin/env bash
set -e

# ==============================================================================
# deploy_gcp.sh - Google Cloud Run デプロイスクリプト
# ==============================================================================
# 使用法:
#   1. gcloud にログイン & プロジェクト設定:
#      gcloud auth login
#      gcloud config set project <YOUR_PROJECT_ID>
#
#   2. デプロイ実行:
#      export OPENROUTER_API_KEY="your-openrouter-key"
#      ./deploy_gcp.sh
# ==============================================================================

SERVICE_NAME="${SERVICE_NAME:-arxiv-graphrag}"
REGION="${REGION:-asia-northeast1}"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)

if [ -z "$PROJECT_ID" ]; then
    echo "❌ エラー: GCP Project が設定されていません。"
    echo "以下のコマンドでプロジェクトを設定してください:"
    echo "  gcloud config set project <YOUR_PROJECT_ID>"
    exit 1
fi

echo "=========================================================="
echo "🚀 Google Cloud Run へのデプロイを開始します"
echo "  サービス名: $SERVICE_NAME"
echo "  リージョン: $REGION"
echo "  プロジェクト: $PROJECT_ID"
echo "=========================================================="

# 必要な API の有効化 (初回のみ)
echo "📦 Cloud Build & Cloud Run API の有効化を確認中..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com --project="$PROJECT_ID"

# 環境変数の準備
ENV_VARS="KUZU_DB_PATH=/app/arxiv_graph.kuzu"
if [ -n "$OPENROUTER_API_KEY" ]; then
    ENV_VARS="$ENV_VARS,OPENROUTER_API_KEY=$OPENROUTER_API_KEY"
    echo "🔑 OPENROUTER_API_KEY を設定しました。"
else
    echo "⚠️ 警告: OPENROUTER_API_KEY がローカルシェルで未定義です。"
    echo "   API キーなしの場合、グラフ検索(No-LLM)のみ動作します。"
    echo "   後から Cloud Run コンソールで環境変数を設定することも可能です。"
fi

# Cloud Run ソースデプロイ (Cloud Build でコンテナ化され自動デプロイされる)
echo "🚀 Cloud Run へデプロイ中 (これには数分かかります)..."
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars "$ENV_VARS" \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300s \
    --concurrency 80

echo ""
echo "=========================================================="
echo "✅ デプロイが正常に完了しました！"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project="$PROJECT_ID" --region="$REGION" --format="value(status.url)")
echo "🌐 サービス公開URL: $SERVICE_URL"
echo "📄 Swagger API Docs: $SERVICE_URL/docs"
echo "🖥️ Web UI ダッシュボード: $SERVICE_URL/"
echo "=========================================================="
