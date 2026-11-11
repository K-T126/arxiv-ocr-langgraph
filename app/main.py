"""
main.py - FastAPI アプリケーションのエントリーポイント
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import router as api_router, set_service
from app.service import GraphRAGService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("arxiv-graphrag")

templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service_instance
    db_path = os.getenv("KUZU_DB_PATH", "./arxiv_graph.kuzu")
    logger.info("Starting up FastAPI application. Loading KùzuDB from: %s", db_path)

    try:
        service_instance = GraphRAGService(db_path=db_path)
        set_service(service_instance)
        logger.info("GraphRAGService successfully initialized.")
    except Exception as e:
        logger.error("Failed to initialize GraphRAGService: %s", e)
        # 起動を阻害せず、後でエラーを返せるようにする

    yield

    logger.info("Shutting down FastAPI application.")


app = FastAPI(
    title="arXiv GraphRAG API Service",
    description="LLM学術論文の派生関係やベンチマーク評価をKùzuDBとLangGraphで多段階推論・質問回答するAPIサービス",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS を許可（外部フロントエンドや開発環境からのアクセスに対応）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API ルーターの登録
app.include_router(api_router)


# --- Web 画面 & ヘルスチェック ---


@app.get("/health", summary="ヘルスチェック (Cloud Run 用)")
async def health_check():
    """
    Cloud Run の死活監視・ヘルスチェック用エンドポイント。
    データベース接続と統計状態を検証します。
    """
    if service_instance is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "error": "Database service not initialized"},
        )

    try:
        stats = service_instance.get_stats()
        return {
            "status": "healthy",
            "models_count": stats.get("model_count", 0),
            "evals_count": stats.get("eval_relationships", 0),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)},
        )


@app.get("/graph/view", response_class=HTMLResponse, summary="全画面インタラクティブナレッジグラフ")
async def graph_view():
    """
    PyVis で生成したスタンドアロンのインタラクティブグラフ HTML をブラウザで直接閲覧します。
    """
    if service_instance is None:
        return HTMLResponse(
            "<h3>Error: Database service is not initialized</h3>", status_code=500
        )

    try:
        # すでにローカルに graph.html が存在すれば高速にそれを読み出し、なければオンデマンド生成
        local_html_path = "./graph.html"
        if os.path.exists(local_html_path):
            with open(local_html_path, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content)

        html_content = service_instance.generate_html_visualization()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(
            f"<h3>Error generating graph visualization: {str(e)}</h3>", status_code=500
        )


@app.get("/", response_class=HTMLResponse, summary="GraphRAG チャット & 可視化 Web UI")
async def index(request: Request):
    """
    ブラウザから直接質問を入力し、GraphRAG による回答とナレッジグラフを閲覧できる統合 Web 画面を提供します。
    """
    stats = {}
    if service_instance:
        try:
            stats = service_instance.get_stats()
        except Exception:
            pass

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "stats": stats,
            "has_api_key": bool(os.getenv("OPENROUTER_API_KEY")),
        },
    )
