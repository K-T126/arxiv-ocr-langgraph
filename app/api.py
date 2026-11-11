"""
api.py - FastAPI ルーター定義
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.service import GraphRAGService

router = APIRouter(prefix="/api", tags=["GraphRAG"])


# --- Pydantic スキーマ ---


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="LLMモデルやベンチマークに関する質問文",
        example="LLaMAモデルの派生関係とMMLUスコアについて教えてください",
    )
    model: Optional[str] = Field(
        None,
        description="OpenRouter で使用するモデル名 (未指定の場合はデフォルト: Qwen2.5-72B-Instruct)",
        example="qwen/qwen-2.5-72b-instruct",
    )
    no_llm: bool = Field(
        False,
        description="Trueの場合、LLMによる回答生成をスキップしグラフ検索結果のみを返します",
    )
    show_context: bool = Field(
        False,
        description="Trueの場合、サーバーログ等にコンテキストを出力します",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    entities: List[str]
    graph_context: str
    error: Optional[str] = None


class StatsResponse(BaseModel):
    model_count: int
    benchmark_count: int
    derived_relationships: int
    eval_relationships: int
    models: List[str]


class GraphDataResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# --- サービス取得 Dependency ---

_service_instance: Optional[GraphRAGService] = None


def set_service(service: GraphRAGService):
    global _service_instance
    _service_instance = service


def get_service() -> GraphRAGService:
    if _service_instance is None:
        raise HTTPException(status_code=500, detail="GraphRAGService is not initialized")
    return _service_instance


# --- エンドポイント ---


@router.post("/query", response_model=QueryResponse, summary="GraphRAG 質問回答 API")
async def query_endpoint(
    req: QueryRequest,
    service: GraphRAGService = Depends(get_service),
):
    """
    指定された質問に対して、KùzuDB のナレッジグラフ検索と LangGraph 多段階推論を用いて回答を生成します。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問文が空です。")

    try:
        res = service.query(
            question=req.question,
            model=req.model,
            no_llm=req.no_llm,
            show_context=req.show_context,
        )
        return QueryResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"クエリ処理中にエラーが発生しました: {str(e)}")


@router.get("/stats", response_model=StatsResponse, summary="ナレッジグラフ統計情報取得")
async def stats_endpoint(service: GraphRAGService = Depends(get_service)):
    """
    KùzuDB に蓄積されているモデル数、ベンチマーク数、派生関係数などのサマリー情報を返します。
    """
    try:
        stats = service.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計情報取得エラー: {str(e)}")


@router.get("/graph", response_model=GraphDataResponse, summary="グラフ生データ (ノード/エッジ) 取得")
async def graph_endpoint(service: GraphRAGService = Depends(get_service)):
    """
    フロントエンドでインタラクティブ可視化を行うためのノードとエッジの JSON データを返します。
    """
    try:
        data = service.get_graph_data()
        return GraphDataResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"グラフデータ取得エラー: {str(e)}")


@router.get("/models", response_model=List[str], summary="登録済みモデル名一覧")
async def models_endpoint(service: GraphRAGService = Depends(get_service)):
    """
    ナレッジグラフに登録されている Model 名の一覧を返します。
    """
    try:
        stats = service.get_stats()
        return stats.get("models", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"モデル一覧取得エラー: {str(e)}")
