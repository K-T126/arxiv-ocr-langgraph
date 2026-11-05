#!/usr/bin/env python3
"""
visualize.py - グラフ可視化スクリプト (LangGraph 版)

README.md Step 4 の仕様に従い、以下の処理を実行します:
1. KùzuDB (arxiv_graph.kuzu) からすべての Model, Benchmark ノードおよび
   DERIVED_FROM, EVALUATED_ON リレーションを取得
2. PyVis を用いてインタラクティブなネットワーク HTML 図 (graph.html) を生成・保存
   ※ cdn_resources="in_line" により、外部 CDN やローカル JS パスに依存せず
     ブラウザで直接開いても確実に描画されます。
"""

import argparse
import html
import logging
import os
import sys

import kuzu
from pyvis.network import Network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_visualization(db_path: str, output_html: str):
    """KùzuDB からデータを読み出し、PyVis ネットワーク HTML を生成する"""
    if not os.path.exists(db_path):
        logger.error("KùzuDB path does not exist: %s", db_path)
        sys.exit(1)

    logger.info("Opening KùzuDB at: %s", db_path)
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # 有向グラフ (directed=True) を初期化 (cdn_resources="in_line" でスタンドアロン化)
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
        cdn_resources="in_line",
    )

    # 物理シミュレーションと操作オプションの設定
    net.set_options(
        """
        {
          "nodes": {
            "font": {
              "size": 14,
              "face": "Helvetica"
            }
          },
          "edges": {
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "none"
            },
            "font": {
              "size": 11,
              "align": "middle"
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "iterations": 150
            }
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

    # 1. Model ノードの取得と追加
    logger.info("Fetching Model nodes...")
    model_res = conn.execute("MATCH (m:Model) RETURN m.name, m.method_type, m.summary")
    model_count = 0
    while model_res.has_next():
        row = model_res.get_next()
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
            color="#3B82F6",  # Blue
            shape="dot",
            size=22,
        )
        nodes_added.add(name)
        model_count += 1

    # 2. Benchmark ノードの取得と追加
    logger.info("Fetching Benchmark nodes...")
    bench_res = conn.execute("MATCH (b:Benchmark) RETURN b.name")
    bench_count = 0
    while bench_res.has_next():
        row = bench_res.get_next()
        name = row[0]
        if not name or name in nodes_added:
            continue

        tooltip = f"<b>Benchmark:</b> {html.escape(name)}"
        net.add_node(
            name,
            label=name,
            title=tooltip,
            color="#10B981",  # Emerald Green
            shape="box",
            size=18,
        )
        nodes_added.add(name)
        bench_count += 1

    # 3. DERIVED_FROM リレーションの取得と追加
    logger.info("Fetching DERIVED_FROM relationships...")
    derived_res = conn.execute(
        "MATCH (child:Model)-[r:DERIVED_FROM]->(parent:Model) RETURN child.name, parent.name"
    )
    derived_count = 0
    while derived_res.has_next():
        row = derived_res.get_next()
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
            derived_count += 1

    # 4. EVALUATED_ON リレーションの取得と追加
    logger.info("Fetching EVALUATED_ON relationships...")
    eval_res = conn.execute(
        "MATCH (m:Model)-[r:EVALUATED_ON]->(b:Benchmark) RETURN m.name, b.name, r.score, r.metric"
    )
    eval_count = 0
    while eval_res.has_next():
        row = eval_res.get_next()
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
            eval_count += 1

    logger.info(
        "Graph Summary: %d Models, %d Benchmarks, %d Lineage Edges, %d Evaluation Edges",
        model_count,
        bench_count,
        derived_count,
        eval_count,
    )

    # HTML 保存
    net.write_html(output_html)
    logger.info("Successfully generated standalone visualization: %s", output_html)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize KùzuDB LLM Knowledge Graph using PyVis (LangGraph Edition)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./arxiv_graph.kuzu",
        help="Path to KùzuDB directory (default: ./arxiv_graph.kuzu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./graph.html",
        help="Path to output HTML file (default: ./graph.html)",
    )

    args = parser.parse_args()
    generate_visualization(args.db_path, args.output)


if __name__ == "__main__":
    main()
