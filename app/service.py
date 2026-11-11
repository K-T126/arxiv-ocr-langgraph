"""
service.py - GraphRAG バックエンドサービス層

KùzuDB への読み取りアクセス、LangGraph 質問回答ワークフローの実行、
PyVis を使ったグラフ可視化 HTML / JSON 生成をカプセル化します。
"""

import html
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import kuzu
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pyvis.network import Network

from query import GraphRAGState, retrieve_graph_context_from_db

logger = logging.getLogger(__name__)


class GraphRAGService:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("KUZU_DB_PATH", "./arxiv_graph.kuzu")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"KùzuDB file not found at: {self.db_path}")

        logger.info("Initializing GraphRAGService with KùzuDB (read-only): %s", self.db_path)
        # read_only=True で複数スレッド・プロセスの同時読み取りに対応
        self.db = kuzu.Database(self.db_path, read_only=True)
        self._llm_cache: Dict[str, ChatOpenAI] = {}

    def get_connection(self) -> kuzu.Connection:
        """軽量な Connection オブジェクトを取得"""
        return kuzu.Connection(self.db)

    def _get_llm(self, model_name: Optional[str] = None) -> Optional[ChatOpenAI]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None

        chosen_model = model_name or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")
        if chosen_model not in self._llm_cache:
            self._llm_cache[chosen_model] = ChatOpenAI(
                model=chosen_model,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=api_key,
                temperature=0.2,
            )
        return self._llm_cache[chosen_model]

    def _build_workflow(self, llm: Optional[ChatOpenAI]):
        """質問回答用 LangGraph ワークフローを構築"""

        def extract_entities_node(state: GraphRAGState) -> Dict[str, Any]:
            question = state.get("question", "")
            tokens = re.findall(r"[A-Za-z0-9_\-\.]+", question)
            return {"entities": tokens}

        def retrieve_graph_node(state: GraphRAGState) -> Dict[str, Any]:
            try:
                conn = self.get_connection()
                context = retrieve_graph_context_from_db(conn, entities=state.get("entities"))
                return {"graph_context": context}
            except Exception as e:
                logger.error("Error retrieving from KùzuDB: %s", e)
                return {"graph_context": "", "error": str(e)}

        def generate_answer_node(state: GraphRAGState) -> Dict[str, Any]:
            context = state.get("graph_context", "")
            question = state.get("question", "")

            if state.get("no_llm"):
                return {"answer": f"[No-LLM Mode: ナレッジグラフ検索結果]\n{context}"}

            if not llm:
                return {
                    "answer": (
                        "⚠️ LLM が設定されていないか、OPENROUTER_API_KEY が環境変数に設定されていません。\n\n"
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

    def query(
        self,
        question: str,
        model: Optional[str] = None,
        no_llm: bool = False,
        show_context: bool = False,
    ) -> Dict[str, Any]:
        """GraphRAG パイプラインを実行して質問に回答"""
        llm = None if no_llm else self._get_llm(model_name=model)
        app = self._build_workflow(llm)

        state: GraphRAGState = {
            "question": question,
            "no_llm": no_llm,
            "show_context": show_context,
            "db_path": self.db_path,
        }

        result = app.invoke(state)
        return {
            "question": question,
            "answer": result.get("answer", ""),
            "entities": result.get("entities", []),
            "graph_context": result.get("graph_context", ""),
            "error": result.get("error"),
        }

    def get_stats(self) -> Dict[str, Any]:
        """データベースの統計情報サマリーを取得"""
        conn = self.get_connection()

        # モデル数
        res_m = conn.execute("MATCH (m:Model) RETURN count(m)")
        model_count = res_m.get_next()[0] if res_m.has_next() else 0

        # ベンチマーク数
        res_b = conn.execute("MATCH (b:Benchmark) RETURN count(b)")
        bench_count = res_b.get_next()[0] if res_b.has_next() else 0

        # 派生関係 (DERIVED_FROM) 数
        res_d = conn.execute("MATCH (:Model)-[r:DERIVED_FROM]->(:Model) RETURN count(r)")
        derived_count = res_d.get_next()[0] if res_d.has_next() else 0

        # 評価関係 (EVALUATED_ON) 数
        res_e = conn.execute("MATCH (:Model)-[r:EVALUATED_ON]->(:Benchmark) RETURN count(r)")
        eval_count = res_e.get_next()[0] if res_e.has_next() else 0

        # モデル名一覧
        res_names = conn.execute("MATCH (m:Model) RETURN m.name ORDER BY m.name")
        models = []
        while res_names.has_next():
            row = res_names.get_next()
            if row[0]:
                models.append(row[0])

        return {
            "model_count": model_count,
            "benchmark_count": bench_count,
            "derived_relationships": derived_count,
            "eval_relationships": eval_count,
            "models": models,
        }

    def get_graph_data(self) -> Dict[str, Any]:
        """フロントエンド描画用のノード・エッジ JSON データを生成"""
        conn = self.get_connection()
        nodes = []
        edges = []
        node_ids = set()

        # Model ノード
        res_m = conn.execute("MATCH (m:Model) RETURN m.name, m.method_type, m.summary")
        while res_m.has_next():
            row = res_m.get_next()
            name, method_type, summary = row[0], row[1], row[2]
            if name and name not in node_ids:
                node_ids.add(name)
                nodes.append(
                    {
                        "id": name,
                        "label": name,
                        "type": "Model",
                        "method_type": method_type,
                        "summary": summary,
                        "color": "#3B82F6",
                    }
                )

        # Benchmark ノード
        res_b = conn.execute("MATCH (b:Benchmark) RETURN b.name")
        while res_b.has_next():
            row = res_b.get_next()
            name = row[0]
            if name and name not in node_ids:
                node_ids.add(name)
                nodes.append(
                    {
                        "id": name,
                        "label": name,
                        "type": "Benchmark",
                        "color": "#10B981",
                    }
                )

        # DERIVED_FROM エッジ
        res_d = conn.execute(
            "MATCH (child:Model)-[r:DERIVED_FROM]->(parent:Model) RETURN child.name, parent.name"
        )
        while res_d.has_next():
            row = res_d.get_next()
            child, parent = row[0], row[1]
            if child in node_ids and parent in node_ids:
                edges.append(
                    {
                        "from": child,
                        "to": parent,
                        "label": "derived_from",
                        "type": "DERIVED_FROM",
                        "color": "#2563EB",
                    }
                )

        # EVALUATED_ON エッジ
        res_e = conn.execute(
            "MATCH (m:Model)-[r:EVALUATED_ON]->(b:Benchmark) RETURN m.name, b.name, r.score, r.metric"
        )
        while res_e.has_next():
            row = res_e.get_next()
            m_name, b_name, score, metric = row[0], row[1], row[2], row[3]
            if m_name in node_ids and b_name in node_ids:
                edges.append(
                    {
                        "from": m_name,
                        "to": b_name,
                        "label": score if score and score != "N/A" else "eval",
                        "score": score,
                        "metric": metric,
                        "type": "EVALUATED_ON",
                        "color": "#9CA3AF",
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def generate_html_visualization(self) -> str:
        """PyVis を用いてスタンドアロン HTML 文字列を生成"""
        conn = self.get_connection()

        net = Network(
            height="100%",
            width="100%",
            directed=True,
            bgcolor="#ffffff",
            font_color="#333333",
            cdn_resources="in_line",
        )

        net.set_options(
            """
            {
              "nodes": {
                "font": { "size": 14, "face": "Helvetica" }
              },
              "edges": {
                "smooth": { "type": "cubicBezier", "forceDirection": "none" },
                "font": { "size": 11, "align": "middle" }
              },
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": { "iterations": 150 }
              },
              "interaction": {
                "hover": true,
                "navigationButtons": true,
                "keyboard": true
              }
            }
            """
        )

        nodes_added = set()

        # Model ノード
        res_m = conn.execute("MATCH (m:Model) RETURN m.name, m.method_type, m.summary")
        while res_m.has_next():
            row = res_m.get_next()
            name, method_type, summary = row[0], row[1], row[2]
            if not name or name in nodes_added:
                continue
            tooltip = (
                f"<b>Model:</b> {html.escape(name)}<br>"
                f"<b>Method:</b> {html.escape(method_type or 'Unknown')}<br>"
                f"<b>Summary:</b> {html.escape(summary or 'No summary')}"
            )
            net.add_node(
                name,
                label=name,
                title=tooltip,
                color="#3B82F6",
                shape="dot",
                size=22,
            )
            nodes_added.add(name)

        # Benchmark ノード
        res_b = conn.execute("MATCH (b:Benchmark) RETURN b.name")
        while res_b.has_next():
            row = res_b.get_next()
            name = row[0]
            if not name or name in nodes_added:
                continue
            tooltip = f"<b>Benchmark:</b> {html.escape(name)}"
            net.add_node(
                name,
                label=name,
                title=tooltip,
                color="#10B981",
                shape="box",
                size=18,
            )
            nodes_added.add(name)

        # DERIVED_FROM エッジ
        res_d = conn.execute(
            "MATCH (child:Model)-[r:DERIVED_FROM]->(parent:Model) RETURN child.name, parent.name"
        )
        while res_d.has_next():
            row = res_d.get_next()
            child, parent = row[0], row[1]
            if child in nodes_added and parent in nodes_added:
                net.add_edge(
                    child,
                    parent,
                    label="derived_from",
                    title=f"{child} is derived from {parent}",
                    color="#2563EB",
                    arrows="to",
                    width=2,
                )

        # EVALUATED_ON エッジ
        res_e = conn.execute(
            "MATCH (m:Model)-[r:EVALUATED_ON]->(b:Benchmark) RETURN m.name, b.name, r.score, r.metric"
        )
        while res_e.has_next():
            row = res_e.get_next()
            m_name, b_name, score, metric = row[0], row[1], row[2], row[3]
            if m_name in nodes_added and b_name in nodes_added:
                edge_label = score if score and score != "N/A" else "eval"
                edge_tooltip = f"{m_name} on {b_name}: {score} ({metric})"
                net.add_edge(
                    m_name,
                    b_name,
                    label=edge_label,
                    title=edge_tooltip,
                    color="#9CA3AF",
                    dashes=True,
                    arrows="to",
                    width=1.5,
                )

        return net.generate_html()
