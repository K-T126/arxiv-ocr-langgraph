#!/usr/bin/env python3
"""
query.py - 質問回答（GraphRAG 検索）スクリプト (LangGraph 版)

README.md Step 3 の仕様に従い、以下のパイプラインを LangGraph ワークフローとして実行します:
1. ExtractEntitiesNode: 質問から LLM モデル名やベンチマーク名などの重要エンティティを特定
2. RetrieveGraphNode: KùzuDB (arxiv_graph.kuzu) から派生関係やベンチマーク評価などのサブグラフを取得
3. GenerateAnswerNode: OpenRouter (LangChain ChatOpenAI) + Qwen/Qwen2.5-72B-Instruct にグラフ情報を
   コンテキストとして埋め込み、ユーザーの質問に日本語で多段階推論した回答を生成
"""

import argparse
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
import kuzu
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- LangGraph 状態定義 ---


class GraphRAGState(TypedDict, total=False):
    question: str
    entities: List[str]
    graph_context: str
    answer: str
    error: Optional[str]
    no_llm: bool
    db_path: str
    show_context: bool


# --- グラフ検索ヘルパー ---


def retrieve_graph_context_from_db(conn: kuzu.Connection, entities: Optional[List[str]] = None) -> str:
    """KùzuDB からモデルのノード、派生関係、ベンチマーク評価を取得し、テキストに整形する"""
    lines = []
    lines.append("=== KùzuDB Knowledge Graph Context ===")

    # 1. すべての Model ノードとその派生親 (DERIVED_FROM)
    model_query = """
    MATCH (m:Model)
    OPTIONAL MATCH (m)-[:DERIVED_FROM]->(parent:Model)
    RETURN m.name, m.method_type, m.summary, parent.name
    """
    try:
        res = conn.execute(model_query)
        models_info = []
        while res.has_next():
            row = res.get_next()
            m_name, method_type, summary, parent_name = row[0], row[1], row[2], row[3]
            line = f"- Model: {m_name} (Method: {method_type or 'Unknown'})"
            if parent_name:
                line += f" -> Derived From: {parent_name}"
            if summary:
                line += f" | Summary: {summary}"
            models_info.append(line)

        if models_info:
            lines.append("\n[Models and Derivations (Lineage)]:")
            lines.extend(models_info)
        else:
            lines.append("\n[Models and Derivations]: No models found in database.")
    except Exception as e:
        logger.warning("Failed to query models: %s", e)

    # 2. ベンチマーク評価 (EVALUATED_ON)
    eval_query = """
    MATCH (m:Model)-[r:EVALUATED_ON]->(b:Benchmark)
    RETURN m.name, b.name, r.score, r.metric
    """
    try:
        res = conn.execute(eval_query)
        evals_info = []
        while res.has_next():
            row = res.get_next()
            m_name, b_name, score, metric = row[0], row[1], row[2], row[3]
            evals_info.append(
                f"- {m_name} on {b_name}: Score = {score or 'N/A'} (Metric: {metric or 'N/A'})"
            )

        if evals_info:
            lines.append("\n[Benchmark Evaluations]:")
            lines.extend(evals_info)
        else:
            lines.append("\n[Benchmark Evaluations]: No evaluation records found in database.")
    except Exception as e:
        logger.warning("Failed to query evaluations: %s", e)

    return "\n".join(lines)


# --- LangGraph グラフ構築 ---


def create_query_graph(llm: Optional[ChatOpenAI], db_path: str):
    """質問回答用の LangGraph ワークフローを作成する"""

    def extract_entities_node(state: GraphRAGState) -> Dict[str, Any]:
        """Node 1: 質問からキーワード/エンティティの抽出"""
        question = state.get("question", "")
        # よくあるモデル名やベンチマーク名を正規表現で抽出
        tokens = re.findall(r"[A-Za-z0-9_\-\.]+", question)
        return {"entities": tokens}

    def retrieve_graph_node(state: GraphRAGState) -> Dict[str, Any]:
        """Node 2: KùzuDB からグラフコンテキストを取得"""
        target_db = state.get("db_path", db_path)
        if not os.path.exists(target_db):
            return {
                "graph_context": "Database not found.",
                "error": f"Database path does not exist: {target_db}",
            }

        try:
            db = kuzu.Database(target_db)
            conn = kuzu.Connection(db)
            context = retrieve_graph_context_from_db(conn, entities=state.get("entities"))
            return {"graph_context": context}
        except Exception as e:
            logger.error("Error retrieving from KùzuDB: %s", e)
            return {"graph_context": "", "error": str(e)}

    def generate_answer_node(state: GraphRAGState) -> Dict[str, Any]:
        """Node 3: グラフコンテキストをもとに回答生成"""
        context = state.get("graph_context", "")
        question = state.get("question", "")

        if state.get("show_context"):
            print(f"\n{context}\n")

        # --no-llm モードの場合
        if state.get("no_llm"):
            return {
                "answer": f"[No-LLM Mode: ナレッジグラフ検索結果]\n{context}"
            }

        if not llm:
            return {
                "answer": (
                    f"⚠️ LLM が設定されていないか、API キーが見つかりません。\n\n"
                    f"【ローカルグラフ検索結果】:\n{context}"
                )
            }

        system_prompt = (
            "あなたは学術論文のナレッジグラフを活用する高度な AI リサーチアシスタントです。\n"
            "提供される【ナレッジグラフ情報】（LLM の派生関係・系統樹、ベンチマークスコア等）に基づいて、"
            "ユーザーの質問に正確かつ論理的に日本語で回答してください。\n\n"
            "回答の指針:\n"
            "1. ナレッジグラフに記載されている派生元・派生先モデルの関係や、ベンチマークスコア（MMLU, GSM8K等）を根拠として明確に提示してください。\n"
            "2. もしナレッジグラフに直接の情報が存在しない場合は、グラフ上にデータがない旨を断った上で、あなたの持っている一般知識を補足として回答してください。\n"
            "3. 丁寧で分かりやすい構成（箇条書きなどを活用）で回答してください。"
        )

        user_prompt = (
            f"【ナレッジグラフ情報】\n"
            f"{context}\n\n"
            f"【質問】\n"
            f"{question}\n\n"
            f"上記の情報をもとに日本語で回答してください。"
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return {"answer": response.content}
        except Exception as e:
            logger.error("LLM API Call failed: %s", e)
            return {
                "answer": (
                    f"⚠️ OpenRouter API 呼び出しでエラーが発生しました: {e}\n\n"
                    f"【取得されたナレッジグラフ情報（ローカル検索結果）】:\n"
                    f"{context}"
                ),
                "error": str(e),
            }

    workflow = StateGraph(GraphRAGState)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("retrieve_graph", retrieve_graph_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("extract_entities")
    workflow.add_edge("extract_entities", "retrieve_graph")
    workflow.add_edge("retrieve_graph", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# --- メイン実行 ---


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about LLM lineage and benchmarks using GraphRAG (LangGraph Edition)"
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question to ask (if omitted, interactive mode starts)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./arxiv_graph.kuzu",
        help="Path to KùzuDB directory (default: ./arxiv_graph.kuzu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for OpenRouter (default: env OPENROUTER_MODEL or qwen/qwen-2.5-72b-instruct)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display retrieved graph context before generating answer",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation and output retrieved graph context only",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    # no-llm モードでない場合のみ API キーを必須とする
    if not args.no_llm and not api_key:
        logger.error("OPENROUTER_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    model_name = args.model or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    llm = None
    if api_key and not args.no_llm:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            temperature=0.2,
        )

    if not os.path.exists(args.db_path):
        logger.error(
            "Database path does not exist: %s. Please run ingest.py first to create and populate the graph.",
            args.db_path,
        )
        sys.exit(1)

    # LangGraph アプリケーションのコンパイル
    app = create_query_graph(llm=llm, db_path=args.db_path)

    if args.question:
        # 単発実行モード
        print(f"\n[質問]: {args.question}")
        state: GraphRAGState = {
            "question": args.question,
            "no_llm": args.no_llm,
            "show_context": args.show_context,
            "db_path": args.db_path,
        }
        res = app.invoke(state)
        print(f"\n[回答]:\n{res.get('answer', '')}\n")
    else:
        # 対話型モード
        print("=== LangGraph GraphRAG 質問対話システム (終了するには 'exit' または 'quit' と入力) ===")
        while True:
            try:
                q = input("\n質問を入力してください > ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    print("終了します。")
                    break

                print("\n検索 & 回答生成中 (LangGraph)...")
                state = {
                    "question": q,
                    "no_llm": args.no_llm,
                    "show_context": args.show_context,
                    "db_path": args.db_path,
                }
                res = app.invoke(state)
                print(f"\n[回答]:\n{res.get('answer', '')}\n")
            except (KeyboardInterrupt, EOFError):
                print("\n終了します。")
                break


if __name__ == "__main__":
    main()
