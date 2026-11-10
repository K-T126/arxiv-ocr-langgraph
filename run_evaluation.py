#!/usr/bin/env python3
"""
run_evaluation.py
eval_dataset.json に定義された評価質問を GraphRAG パイプライン (query.py) に流し、
各評価観点・評価指標に基づいて自動テストを実行・集計するスクリプト。

利用例:
  # 1. ローカルグラフ検索（Context Recall / Precision）の無料高速テスト
  python run_evaluation.py --no-llm

  # 2. 無料枠 LLM を用いた回答生成・正確性テスト
  python run_evaluation.py --model openrouter/free
"""

import argparse
import json
import os
import time
from typing import Dict, Any, List

import kuzu
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from query import create_query_graph, GraphRAGState

def evaluate_dataset(dataset_path: str, model_name: str, no_llm: bool, db_path: str):
    load_dotenv(override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not no_llm and not api_key:
        print("Warning: OPENROUTER_API_KEY is not set. Switching to --no-llm mode.")
        no_llm = True

    llm = None
    if not no_llm and api_key:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            temperature=0.1,
        )

    app = create_query_graph(llm=llm, db_path=db_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    print(f"\n=======================================================")
    print(f"RAG Evaluation Runner: {len(dataset)} test cases")
    print(f"Mode: {'No-LLM (Graph Context Only)' if no_llm else f'LLM Answer ({model_name})'}")
    print(f"=======================================================\n")

    results = []

    for item in dataset:
        q_id = item["id"]
        category = item["category"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        contexts_expected = item.get("ground_truth_contexts", [])

        print(f"▶ [{q_id}] ({category})")
        print(f"  Question: {question}")

        state: GraphRAGState = {
            "question": question,
            "entities": [],
            "graph_context": "",
            "answer": "",
            "no_llm": no_llm,
            "db_path": db_path,
            "show_context": False,
        }

        start_time = time.time()
        final_state = app.invoke(state)
        elapsed = time.time() - start_time

        answer = final_state.get("answer", "")
        graph_context = final_state.get("graph_context", "")

        # 簡易自動スコアリング
        # Context Recall: 期待コンテキストの要素が含まれているか
        recalled_count = 0
        if contexts_expected:
            for ctx in contexts_expected:
                # 代表的なキーワードがコンテキストに含まれているか
                core_tokens = [t for t in ctx.replace("(", " ").replace(")", " ").replace(":", " ").replace("-", " ").split() if len(t) > 3]
                if any(t.lower() in graph_context.lower() for t in core_tokens):
                    recalled_count += 1
            context_recall = round(recalled_count / len(contexts_expected), 2)
        else:
            context_recall = 1.0  # 未登録質問の場合はコンテキスト不要

        # Rejection / Faithfulness check (グラフ外質問で「情報がない」等の拒否キーワードがあるか)
        is_rejected_properly = None
        if "グラフ外" in category:
            rejection_keywords = ["ない", "存在しない", "含まれていません", "ありません", "不明", "記載されていません"]
            is_rejected_properly = any(kw in answer for kw in rejection_keywords)

        result_record = {
            "id": q_id,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "system_answer": answer,
            "context_recall_score": context_recall,
            "is_rejected_properly": is_rejected_properly,
            "elapsed_seconds": round(elapsed, 2)
        }
        results.append(result_record)

        print(f"  Context Recall: {context_recall}")
        if is_rejected_properly is not None:
            print(f"  Proper Rejection (Faithfulness): {'PASS' if is_rejected_properly else 'FAIL'}")
        print(f"  Elapsed: {elapsed:.2f}s\n")

    # 結果保存
    output_path = "./eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"All evaluation runs completed. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Suite")
    parser.add_argument("--dataset", type=str, default="./eval_dataset.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--model", type=str, default="openrouter/free", help="Model name for answer generation")
    parser.add_argument("--no-llm", action="store_true", help="Evaluate retrieval context only without LLM")
    parser.add_argument("--db-path", type=str, default="./arxiv_graph.kuzu", help="Path to KùzuDB")

    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.model, args.no_llm, args.db_path)

if __name__ == "__main__":
    main()
