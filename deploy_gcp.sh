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
#      Create Secret Manager secrets named APP_ACCESS_TOKEN and OPENROUTER_API_KEY.
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

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
RUNTIME_SERVICE_ACCOUNT="${RUNTIME_SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

echo "=========================================================="
echo "🚀 Google Cloud Run へのデプロイを開始します"
echo "  サービス名: $SERVICE_NAME"
echo "  リージョン: $REGION"
echo "  プロジェクト: $PROJECT_ID"
echo "=========================================================="

# 必要な API の有効化 (初回のみ)
echo "📦 Cloud Build, Cloud Run, Secret Manager API の有効化を確認中..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com --project="$PROJECT_ID"

if ! gcloud secrets describe APP_ACCESS_TOKEN --project="$PROJECT_ID" >/dev/null 2>&1; then
    echo "❌ Secret Manager に APP_ACCESS_TOKEN がありません。"
    echo "   README の手順で高エントロピーのトークンを登録してから再実行してください。"
    exit 1
fi

SECRET_MAPPINGS="APP_ACCESS_TOKEN=APP_ACCESS_TOKEN:latest"
if gcloud secrets describe OPENROUTER_API_KEY --project="$PROJECT_ID" >/dev/null 2>&1; then
    SECRET_MAPPINGS="$SECRET_MAPPINGS,OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest"
else
    echo "⚠️ OPENROUTER_API_KEY secret がないため、LLM回答は無効です。"
fi

echo "🔐 Cloud Run 実行サービスアカウントに必要なSecret Manager権限を付与します: $RUNTIME_SERVICE_ACCOUNT"
gcloud secrets add-iam-policy-binding APP_ACCESS_TOKEN \
    --project="$PROJECT_ID" \
    --member="serviceAccount:$RUNTIME_SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" >/dev/null
if gcloud secrets describe OPENROUTER_API_KEY --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud secrets add-iam-policy-binding OPENROUTER_API_KEY \
        --project="$PROJECT_ID" \
        --member="serviceAccount:$RUNTIME_SERVICE_ACCOUNT" \
        --role="roles/secretmanager.secretAccessor" >/dev/null
fi

# Cloud Run ソースデプロイ (Cloud Build でコンテナ化され自動デプロイされる)
echo "🚀 Cloud Run へデプロイ中 (これには数分かかります)..."
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --service-account "$RUNTIME_SERVICE_ACCOUNT" \
    --set-env-vars "KUZU_DB_PATH=/app/arxiv_graph.kuzu" \
    --set-secrets "$SECRET_MAPPINGS" \
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
