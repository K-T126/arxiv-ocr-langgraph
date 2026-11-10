#!/usr/bin/env python3
"""
generate_eval_dataset.py
KùzuDB（および元論文PDFデータ）から、RAG精度評価用データセット（質問・正解・根拠コンテキスト・評価観点・評価指標）
を自動生成するスクリプト。

4つの評価観点に対応:
1. 単一の質問 (Single-hop Factoid)
2. 多段階推論が必要な質問 (Multi-hop Reasoning)
3. 複数のモデル比較 (Model Comparison)
4. グラフ外・未登録の情報に関する質問 (Out-of-Graph / Unanswerable)

5つの評価指標に対応:
- Faithfulness (忠実性)
- Answer Relevancy (回答関連性)
- Context Precision (文脈適合率)
- Context Recall (文脈再現率)
- Answer Correctness (回答正確性)
"""

import json
import os
import kuzu

def build_evaluation_dataset(db_path: str = "./arxiv_graph.kuzu"):
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # 1. モデル情報の収集
    models = {}
    res = conn.execute("MATCH (m:Model) RETURN m.name, m.method_type, m.summary")
    while res.has_next():
        row = res.get_next()
        models[row[0]] = {
            "name": row[0],
            "method_type": row[1] or "Unknown",
            "summary": row[2] or ""
        }

    # 2. 派生関係の収集
    derivations = []
    res = conn.execute("MATCH (child:Model)-[:DERIVED_FROM]->(parent:Model) RETURN child.name, parent.name")
    while res.has_next():
        row = res.get_next()
        derivations.append({"child": row[0], "parent": row[1]})

    # 3. ベンチマーク評価の収集
    evaluations = {}
    res = conn.execute("MATCH (m:Model)-[r:EVALUATED_ON]->(b:Benchmark) RETURN m.name, b.name, r.score, r.metric")
    while res.has_next():
        row = res.get_next()
        m_name, b_name, score, metric = row[0], row[1], row[2], row[3]
        if m_name not in evaluations:
            evaluations[m_name] = {}
        evaluations[m_name][b_name] = {"score": score, "metric": metric}

    dataset = []

    # =========================================================================
    # 観点 1: 単一の質問 (Single-hop Factoid)
    # =========================================================================
    # Q1: 派生元モデルの特定
    if "LLaMA-I-65B" in models:
        dataset.append({
            "id": "Q1_single_base_model",
            "category": "単一の質問 (Single-hop)",
            "question": "LLaMA-I-65B のベース（派生元）となったモデルは何ですか？",
            "ground_truth": "LLaMA-I-65B のベースとなったモデルは LLaMA-65B です。Flan-PaLM プロトコルに従った指示データを用いてファインチューニングされています。",
            "ground_truth_contexts": [
                "(:Model {name: 'LLaMA-I-65B'})-[:DERIVED_FROM]->(:Model {name: 'LLaMA-65B'})",
                "LLaMA-I-65B: Instruction-tuned variant of LLaMA-65B fine-tuned on instruction data following the Flan-PaLM protocol."
            ],
            "target_metrics": ["Answer Correctness", "Context Precision", "Answer Relevancy"],
            "difficulty": "Easy"
        })

    # Q2: 特定モデルのベンチマークスコア
    if "LLaMA-7B" in evaluations and "MMLU" in evaluations["LLaMA-7B"]:
        score = evaluations["LLaMA-7B"]["MMLU"]["score"]
        dataset.append({
            "id": "Q2_single_benchmark_score",
            "category": "単一の質問 (Single-hop)",
            "question": "初代 LLaMA-7B の MMLU ベンチマークにおけるスコアはいくつですか？",
            "ground_truth": f"初代 LLaMA-7B の MMLU スコアは {score}（5-shot average accuracy）です。",
            "ground_truth_contexts": [
                f"(LLaMA-7B)-[:EVALUATED_ON {{score: '{score}', metric: '5-shot average accuracy'}}]->(MMLU)"
            ],
            "target_metrics": ["Answer Correctness", "Context Precision"],
            "difficulty": "Easy"
        })

    # Q3: 特定モデルの事前学習トークン数・手法
    if "LLaMA-7B" in models:
        dataset.append({
            "id": "Q3_single_training_tokens",
            "category": "単一の質問 (Single-hop)",
            "question": "LLaMA-7B は何トークンの公開データで事前学習されましたか？",
            "ground_truth": "LLaMA-7B は 1.0T（1兆）トークンの公開データを用いて事前学習（ゼロからの事前学習）されました。",
            "ground_truth_contexts": [
                "LLaMA-7B (Method: Foundation language model (pre-trained from scratch)): 7-billion parameter foundation language model trained on 1.0T tokens of publicly available data."
            ],
            "target_metrics": ["Answer Correctness", "Faithfulness"],
            "difficulty": "Easy"
        })

    # =========================================================================
    # 観点 2: 多段階推論が必要な質問 (Multi-hop Reasoning)
    # =========================================================================
    # Q4: Llama 2 から派生した全モデルとその学習手法の網羅
    llama2_derived = [d["child"] for d in derivations if d["parent"] == "Llama 2"]
    if llama2_derived:
        methods = {m: models[m]["method_type"] for m in llama2_derived if m in models}
        gt_parts = [f"{m} (学習手法: {methods.get(m, 'Unknown')})" for m in llama2_derived]
        dataset.append({
            "id": "Q4_multihop_lineage_and_method",
            "category": "多段階推論 (Multi-hop Reasoning)",
            "question": "Llama 2 から派生したモデルをすべて挙げ、それぞれの学習手法（method_type）を教えてください。",
            "ground_truth": (
                f"Llama 2 から派生したモデルは以下の {len(llama2_derived)} つです:\n" +
                "\n".join([f"- {part}" for part in gt_parts]) +
                "\nいずれのモデルも、対話用途に最適化するために教師あり微調整（SFT）、人間のフィードバックによる強化学習（RLHF）、および安全性微調整（Safety Fine-Tuning）を組み合わせて学習されています。"
            ),
            "ground_truth_contexts": [
                f"(:Model {{name: '{m}'}})-[:DERIVED_FROM]->(:Model {{name: 'Llama 2'}})" for m in llama2_derived
            ] + [
                f"Model: {m} | Method: {methods.get(m, '')}" for m in llama2_derived
            ],
            "target_metrics": ["Context Recall", "Answer Correctness", "Faithfulness"],
            "difficulty": "Hard"
        })

    # Q5: 派生関係をたどったスコア比較（初代65Bと指示チューニング版の性能向上）
    if "LLaMA-65B" in evaluations and "LLaMA-I-65B" in evaluations:
        score_base = evaluations["LLaMA-65B"].get("MMLU", {}).get("score", "N/A")
        score_inst = evaluations["LLaMA-I-65B"].get("MMLU", {}).get("score", "N/A")
        dataset.append({
            "id": "Q5_multihop_instruction_improvement",
            "category": "多段階推論 (Multi-hop Reasoning)",
            "question": "LLaMA-65B から指示チューニングによって派生したモデルの名前と、その派生によって MMLU スコアがどのように変化したかを教えてください。",
            "ground_truth": (
                f"LLaMA-65B から指示チューニング（Flan-PaLM プロトコル）によって派生したモデルは「LLaMA-I-65B」です。\n"
                f"MMLU スコアは、ベースモデルの LLaMA-65B が {score_base} であったのに対し、派生モデルの LLaMA-I-65B では {score_inst} に向上しています。"
            ),
            "ground_truth_contexts": [
                "(:Model {name: 'LLaMA-I-65B'})-[:DERIVED_FROM]->(:Model {name: 'LLaMA-65B'})",
                f"(LLaMA-65B)-[:EVALUATED_ON {{score: '{score_base}'}}]->(MMLU)",
                f"(LLaMA-I-65B)-[:EVALUATED_ON {{score: '{score_inst}'}}]->(MMLU)"
            ],
            "target_metrics": ["Context Recall", "Answer Correctness", "Context Precision"],
            "difficulty": "Medium"
        })

    # =========================================================================
    # 観点 3: 複数のモデル比較 (Model Comparison)
    # =========================================================================
    # Q6: 同一ベンチマークにおける複数モデルのスコア比較
    if "LLaMA-7B" in evaluations and "LLaMA-13B" in evaluations:
        b_common = ["BoolQ", "PIQA", "HellaSwag", "GSM8k", "MMLU"]
        comparisons = []
        contexts = []
        for b in b_common:
            s7 = evaluations["LLaMA-7B"].get(b, {}).get("score")
            s13 = evaluations["LLaMA-13B"].get(b, {}).get("score")
            if s7 and s13:
                comparisons.append(f"- {b}: LLaMA-7B = {s7}, LLaMA-13B = {s13} (LLaMA-13B が優位)")
                contexts.append(f"(LLaMA-7B)-[:EVALUATED_ON {{score: '{s7}'}}]->({b})")
                contexts.append(f"(LLaMA-13B)-[:EVALUATED_ON {{score: '{s13}'}}]->({b})")

        dataset.append({
            "id": "Q6_comparison_llama_7b_vs_13b",
            "category": "複数のモデル比較 (Model Comparison)",
            "question": "LLaMA-7B と LLaMA-13B の性能を BoolQ, PIQA, HellaSwag, GSM8k, MMLU のスコアで比較し、どちらが優れているかを説明してください。",
            "ground_truth": (
                "すべての対象ベンチマークにおいて LLaMA-13B の方が優れたスコアを記録しています:\n" +
                "\n".join(comparisons) +
                "\nLLaMA-13B はパラメータ数が増加したことで、知識問題・推論問題（GSM8k, MMLU など）において大幅な性能向上が見られます。"
            ),
            "ground_truth_contexts": contexts,
            "target_metrics": ["Answer Correctness", "Context Recall", "Answer Relevancy"],
            "difficulty": "Medium"
        })

    # Q7: Llama 2 事前学習版と Chat版（70B）の安全性・毒性スコア比較
    if "Llama 2 (Pretrained)" in evaluations and "Llama 2-Chat (70B)" in evaluations:
        toxi_pre = evaluations["Llama 2 (Pretrained)"].get("ToxiGen", {}).get("score", "N/A")
        toxi_chat = evaluations["Llama 2-Chat (70B)"].get("ToxiGen", {}).get("score", "N/A")
        truth_pre = evaluations["Llama 2 (Pretrained)"].get("TruthfulQA", {}).get("score", "N/A")
        truth_chat = evaluations["Llama 2-Chat (70B)"].get("TruthfulQA", {}).get("score", "N/A")
        dataset.append({
            "id": "Q7_comparison_pretraining_vs_chat_safety",
            "category": "複数のモデル比較 (Model Comparison)",
            "question": "Llama 2 (Pretrained) と Llama 2-Chat (70B) の安全性・信頼性を、ToxiGen（毒性生成率）および TruthfulQA のスコアで比較してください。",
            "ground_truth": (
                f"Llama 2 (Pretrained) から Llama 2-Chat (70B) への微調整（RLHF・安全性チューニング）により、安全性と信頼性が劇的に向上しています:\n"
                f"- ToxiGen（毒性生成率）: Llama 2 (Pretrained) の {toxi_pre}% から、Llama 2-Chat (70B) では {toxi_chat}% に大幅激減し、ほぼ毒性のない安全な生成を実現しています。\n"
                f"- TruthfulQA（真実性スコア）: {truth_pre} から {truth_chat} へと大きく改善しています。"
            ),
            "ground_truth_contexts": [
                f"(Llama 2 (Pretrained))-[:EVALUATED_ON {{score: '{toxi_pre}'}}]->(ToxiGen)",
                f"(Llama 2-Chat (70B))-[:EVALUATED_ON {{score: '{toxi_chat}'}}]->(ToxiGen)",
                f"(Llama 2 (Pretrained))-[:EVALUATED_ON {{score: '{truth_pre}'}}]->(TruthfulQA)",
                f"(Llama 2-Chat (70B))-[:EVALUATED_ON {{score: '{truth_chat}'}}]->(TruthfulQA)"
            ],
            "target_metrics": ["Answer Correctness", "Faithfulness", "Context Precision"],
            "difficulty": "Hard"
        })

    # =========================================================================
    # 観点 4: グラフ外・未登録の情報に関する質問 (Out-of-Graph / Unanswerable)
    # =========================================================================
    dataset.append({
        "id": "Q8_out_of_graph_gpt4",
        "category": "グラフ外・未登録の情報 (Out-of-Graph)",
        "question": "GPT-4 のパラメータ数と事前学習コーパスの内訳を教えてください。",
        "ground_truth": "提供されたナレッジグラフおよび対象論文（LLaMA / Llama 2）には、GPT-4 のパラメータ数や事前学習コーパスの内訳に関する情報は含まれていません。データベース上に該当する情報が存在しないため回答できません。",
        "ground_truth_contexts": [],
        "target_metrics": ["Faithfulness", "Answer Relevancy"],
        "difficulty": "Medium"
    })

    dataset.append({
        "id": "Q9_out_of_graph_claude",
        "category": "グラフ外・未登録の情報 (Out-of-Graph)",
        "question": "Claude 3.5 Sonnet のアーキテクチャとコンテキストウィンドウの長さを教えてください。",
        "ground_truth": "ナレッジグラフおよび論文データ内に Claude 3.5 Sonnet に関する記録（モデルノードや評価データ）は存在しません。グラフ外の情報のため回答できません。",
        "ground_truth_contexts": [],
        "target_metrics": ["Faithfulness", "Answer Relevancy"],
        "difficulty": "Easy"
    })

    dataset.append({
        "id": "Q10_out_of_graph_japanese_benchmark",
        "category": "グラフ外・未登録の情報 (Out-of-Graph)",
        "question": "LLaMA-7B の日本語言語理解ベンチマーク（JGLUE）における評価スコアを教えてください。",
        "ground_truth": "ナレッジグラフおよび LLaMA 論文には MMLU, GSM8k, BoolQ などの英語ベンチマーク評価は記録されていますが、日本語ベンチマーク（JGLUE）に関する評価スコアは記録されていません。",
        "ground_truth_contexts": [],
        "target_metrics": ["Faithfulness", "Answer Correctness"],
        "difficulty": "Medium"
    })

    return dataset

def main():
    dataset = build_evaluation_dataset()
    
    # JSON 出力
    json_path = "./eval_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Successfully generated {len(dataset)} evaluation questions into {json_path}")

    # Markdown レポート出力
    md_path = "./eval_dataset.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RAG 精度評価用データセット (Evaluation Dataset)\n\n")
        f.write("本データセットは、KùzuDB ナレッジグラフ（LLaMA 1 & Llama 2 論文）から自動抽出された実データに基づき生成された、RAG 評価用の質問・正解データセットです。\n\n")
        f.write("## 評価指標と評価対象\n")
        f.write("| 評価指標 | 測定内容 | 主な対象観点 |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| **Faithfulness (忠実性)** | 生成回答が取得コンテキストのみに基づき、ハルシネーション（勝手な創作）がないか | グラフ外・未登録の情報 |\n")
        f.write("| **Answer Relevancy (回答関連性)** | 生成回答がユーザーの質問の意図に対して的確・簡潔であるか | すべての質問 |\n")
        f.write("| **Context Precision (文脈適合率)** | 取得されたグラフコンテキストのうち、正解に必要な情報が上位に抽出されているか | 単一の質問、複数比較 |\n")
        f.write("| **Context Recall (文脈再現率)** | 正解を網羅するために必要なグラフ情報に抜け漏れがないか | 多段階推論、複数比較 |\n")
        f.write("| **Answer Correctness (回答正確性)** | 生成回答が Ground Truth（正解）の事実とどれだけ一致しているか | すべての質問 |\n\n")
        f.write("---\n\n")

        for item in dataset:
            f.write(f"### [{item['id']}] {item['question']}\n\n")
            f.write(f"- **分類**: `{item['category']}` (難易度: {item['difficulty']})\n")
            f.write(f"- **正解 (Ground Truth)**:\n  > {item['ground_truth'].replace(chr(10), chr(10) + '  > ')}\n")
            f.write(f"- **評価対象指標**: {', '.join([f'`{m}`' for m in item['target_metrics']])}\n")
            if item['ground_truth_contexts']:
                f.write("- **根拠コンテキスト (Ground Truth Contexts)**:\n")
                for c in item['ground_truth_contexts']:
                    f.write(f"  - `{c}`\n")
            else:
                f.write("- **根拠コンテキスト**: なし（グラフ外のため、取得コンテキストに含まれてはならない / 不足として拒否すべき）\n")
            f.write("\n---\n\n")

    print(f"Successfully generated markdown report into {md_path}")

if __name__ == "__main__":
    main()
