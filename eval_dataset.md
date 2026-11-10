# RAG 精度評価用データセット (Evaluation Dataset)

本データセットは、KùzuDB ナレッジグラフ（LLaMA 1 & Llama 2 論文）から自動抽出された実データに基づき生成された、RAG 評価用の質問・正解データセットです。

## 評価指標と評価対象
| 評価指標 | 測定内容 | 主な対象観点 |
| :--- | :--- | :--- |
| **Faithfulness (忠実性)** | 生成回答が取得コンテキストのみに基づき、ハルシネーション（勝手な創作）がないか | グラフ外・未登録の情報 |
| **Answer Relevancy (回答関連性)** | 生成回答がユーザーの質問の意図に対して的確・簡潔であるか | すべての質問 |
| **Context Precision (文脈適合率)** | 取得されたグラフコンテキストのうち、正解に必要な情報が上位に抽出されているか | 単一の質問、複数比較 |
| **Context Recall (文脈再現率)** | 正解を網羅するために必要なグラフ情報に抜け漏れがないか | 多段階推論、複数比較 |
| **Answer Correctness (回答正確性)** | 生成回答が Ground Truth（正解）の事実とどれだけ一致しているか | すべての質問 |

---

### [Q1_single_base_model] LLaMA-I-65B のベース（派生元）となったモデルは何ですか？

- **分類**: `単一の質問 (Single-hop)` (難易度: Easy)
- **正解 (Ground Truth)**:
  > LLaMA-I-65B のベースとなったモデルは LLaMA-65B です。Flan-PaLM プロトコルに従った指示データを用いてファインチューニングされています。
- **評価対象指標**: `Answer Correctness`, `Context Precision`, `Answer Relevancy`
- **根拠コンテキスト (Ground Truth Contexts)**:
  - `(:Model {name: 'LLaMA-I-65B'})-[:DERIVED_FROM]->(:Model {name: 'LLaMA-65B'})`
  - `LLaMA-I-65B: Instruction-tuned variant of LLaMA-65B fine-tuned on instruction data following the Flan-PaLM protocol.`

---

### [Q2_single_benchmark_score] 初代 LLaMA-7B の MMLU ベンチマークにおけるスコアはいくつですか？

- **分類**: `単一の質問 (Single-hop)` (難易度: Easy)
- **正解 (Ground Truth)**:
  > 初代 LLaMA-7B の MMLU スコアは 35.1（5-shot average accuracy）です。
- **評価対象指標**: `Answer Correctness`, `Context Precision`
- **根拠コンテキスト (Ground Truth Contexts)**:
  - `(LLaMA-7B)-[:EVALUATED_ON {score: '35.1', metric: '5-shot average accuracy'}]->(MMLU)`

---

### [Q3_single_training_tokens] LLaMA-7B は何トークンの公開データで事前学習されましたか？

- **分類**: `単一の質問 (Single-hop)` (難易度: Easy)
- **正解 (Ground Truth)**:
  > LLaMA-7B は 1.0T（1兆）トークンの公開データを用いて事前学習（ゼロからの事前学習）されました。
- **評価対象指標**: `Answer Correctness`, `Faithfulness`
- **根拠コンテキスト (Ground Truth Contexts)**:
  - `LLaMA-7B (Method: Foundation language model (pre-trained from scratch)): 7-billion parameter foundation language model trained on 1.0T tokens of publicly available data.`

---

### [Q4_multihop_lineage_and_method] Llama 2 から派生したモデルをすべて挙げ、それぞれの学習手法（method_type）を教えてください。

- **分類**: `多段階推論 (Multi-hop Reasoning)` (難易度: Hard)
- **正解 (Ground Truth)**:
  > Llama 2 から派生したモデルは以下の 4 つです:
  > - Llama 2-Chat (7B) (学習手法: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning)
  > - Llama 2-Chat (13B) (学習手法: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning)
  > - Llama 2-Chat (34B) (学習手法: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning)
  > - Llama 2-Chat (70B) (学習手法: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning)
  > いずれのモデルも、対話用途に最適化するために教師あり微調整（SFT）、人間のフィードバックによる強化学習（RLHF）、および安全性微調整（Safety Fine-Tuning）を組み合わせて学習されています。
- **評価対象指標**: `Context Recall`, `Answer Correctness`, `Faithfulness`
- **根拠コンテキスト (Ground Truth Contexts)**:
  - `(:Model {name: 'Llama 2-Chat (7B)'})-[:DERIVED_FROM]->(:Model {name: 'Llama 2'})`
  - `(:Model {name: 'Llama 2-Chat (13B)'})-[:DERIVED_FROM]->(:Model {name: 'Llama 2'})`
  - `(:Model {name: 'Llama 2-Chat (34B)'})-[:DERIVED_FROM]->(:Model {name: 'Llama 2'})`
  - `(:Model {name: 'Llama 2-Chat (70B)'})-[:DERIVED_FROM]->(:Model {name: 'Llama 2'})`
  - `Model: Llama 2-Chat (7B) | Method: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning`
  - `Model: Llama 2-Chat (13B) | Method: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning`
  - `Model: Llama 2-Chat (34B) | Method: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning`
  - `Model: Llama 2-Chat (70B) | Method: Supervised Fine-Tuning + RLHF + Safety Fine-Tuning`

---

### [Q5_multihop_instruction_improvement] LLaMA-65B から指示チューニングによって派生したモデルの名前と、その派生によって MMLU スコアがどのように変化したかを教えてください。

- **分類**: `多段階推論 (Multi-hop Reasoning)` (難易度: Medium)
- **正解 (Ground Truth)**:
  > LLaMA-65B から指示チューニング（Flan-PaLM プロトコル）によって派生したモデルは「LLaMA-I-65B」です。
  > MMLU スコアは、ベースモデルの LLaMA-65B が 63.4 であったのに対し、派生モデルの LLaMA-I-65B では 68.9 に向上しています。
- **評価対象指標**: `Context Recall`, `Answer Correctness`, `Context Precision`
- **根拠コンテキスト (Ground Truth Contexts)**:
  - `(:Model {name: 'LLaMA-I-65B'})-[:DERIVED_FROM]->(:Model {name: 'LLaMA-65B'})`
  - `(LLaMA-65B)-[:EVALUATED_ON {score: '63.4'}]->(MMLU)`
  - `(LLaMA-I-65B)-[:EVALUATED_ON {score: '68.9'}]->(MMLU)`

---

### [Q6_comparison_llama_7b_vs_13b] LLaMA-7B と LLaMA-13B の性能を BoolQ, PIQA, HellaSwag, GSM8k, MMLU のスコアで比較し、どちらが優れているかを説明してください。

- **分類**: `複数のモデル比較 (Model Comparison)` (難易度: Medium)
- **正解 (Ground Truth)**:
  > すべての対象ベンチマークにおいて LLaMA-13B の方が優れたスコアを記録しています:
  > - BoolQ: LLaMA-7B = 76.5, LLaMA-13B = 78.1 (LLaMA-13B が優位)
  > - PIQA: LLaMA-7B = 79.8, LLaMA-13B = 80.1 (LLaMA-13B が優位)
  > - HellaSwag: LLaMA-7B = 76.1, LLaMA-13B = 79.2 (LLaMA-13B が優位)
  > - GSM8k: LLaMA-7B = 11.0, LLaMA-13B = 17.8 (LLaMA-13B が優位)
  > - MMLU: LLaMA-7B = 35.1, LLaMA-13B = 46.9 (LLaMA-13B が優位)
  > LLaMA-13B はパラメータ数が増加したことで、知識問題・推論問題（GSM8k, MMLU など）において大幅な性能向上が見られます。
- **評価対象指標**: `Answer Correctness`, `Context Recall`, `Answer Relevancy`
- **根拠コンテキスト (Ground Truth Contexts)**:
  - `(LLaMA-7B)-[:EVALUATED_ON {score: '76.5'}]->(BoolQ)`
  - `(LLaMA-13B)-[:EVALUATED_ON {score: '78.1'}]->(BoolQ)`
  - `(LLaMA-7B)-[:EVALUATED_ON {score: '79.8'}]->(PIQA)`
  - `(LLaMA-13B)-[:EVALUATED_ON {score: '80.1'}]->(PIQA)`
  - `(LLaMA-7B)-[:EVALUATED_ON {score: '76.1'}]->(HellaSwag)`
  - `(LLaMA-13B)-[:EVALUATED_ON {score: '79.2'}]->(HellaSwag)`
  - `(LLaMA-7B)-[:EVALUATED_ON {score: '11.0'}]->(GSM8k)`
  - `(LLaMA-13B)-[:EVALUATED_ON {score: '17.8'}]->(GSM8k)`
  - `(LLaMA-7B)-[:EVALUATED_ON {score: '35.1'}]->(MMLU)`
  - `(LLaMA-13B)-[:EVALUATED_ON {score: '46.9'}]->(MMLU)`

---

### [Q7_comparison_pretraining_vs_chat_safety] Llama 2 (Pretrained) と Llama 2-Chat (70B) の安全性・信頼性を、ToxiGen（毒性生成率）および TruthfulQA のスコアで比較してください。

- **分類**: `複数のモデル比較 (Model Comparison)` (難易度: Hard)
- **正解 (Ground Truth)**:
  > Llama 2 (Pretrained) から Llama 2-Chat (70B) への微調整（RLHF・安全性チューニング）により、安全性と信頼性が劇的に向上しています:
  > - ToxiGen（毒性生成率）: Llama 2 (Pretrained) の 24.60% から、Llama 2-Chat (70B) では 0.01% に大幅激減し、ほぼ毒性のない安全な生成を実現しています。
  > - TruthfulQA（真実性スコア）: 50.18 から 64.14 へと大きく改善しています。
- **評価対象指標**: `Answer Correctness`, `Faithfulness`, `Context Precision`
- **根拠コンテキスト (Ground Truth Contexts)**:
  - `(Llama 2 (Pretrained))-[:EVALUATED_ON {score: '24.60'}]->(ToxiGen)`
  - `(Llama 2-Chat (70B))-[:EVALUATED_ON {score: '0.01'}]->(ToxiGen)`
  - `(Llama 2 (Pretrained))-[:EVALUATED_ON {score: '50.18'}]->(TruthfulQA)`
  - `(Llama 2-Chat (70B))-[:EVALUATED_ON {score: '64.14'}]->(TruthfulQA)`

---

### [Q8_out_of_graph_gpt4] GPT-4 のパラメータ数と事前学習コーパスの内訳を教えてください。

- **分類**: `グラフ外・未登録の情報 (Out-of-Graph)` (難易度: Medium)
- **正解 (Ground Truth)**:
  > 提供されたナレッジグラフおよび対象論文（LLaMA / Llama 2）には、GPT-4 のパラメータ数や事前学習コーパスの内訳に関する情報は含まれていません。データベース上に該当する情報が存在しないため回答できません。
- **評価対象指標**: `Faithfulness`, `Answer Relevancy`
- **根拠コンテキスト**: なし（グラフ外のため、取得コンテキストに含まれてはならない / 不足として拒否すべき）

---

### [Q9_out_of_graph_claude] Claude 3.5 Sonnet のアーキテクチャとコンテキストウィンドウの長さを教えてください。

- **分類**: `グラフ外・未登録の情報 (Out-of-Graph)` (難易度: Easy)
- **正解 (Ground Truth)**:
  > ナレッジグラフおよび論文データ内に Claude 3.5 Sonnet に関する記録（モデルノードや評価データ）は存在しません。グラフ外の情報のため回答できません。
- **評価対象指標**: `Faithfulness`, `Answer Relevancy`
- **根拠コンテキスト**: なし（グラフ外のため、取得コンテキストに含まれてはならない / 不足として拒否すべき）

---

### [Q10_out_of_graph_japanese_benchmark] LLaMA-7B の日本語言語理解ベンチマーク（JGLUE）における評価スコアを教えてください。

- **分類**: `グラフ外・未登録の情報 (Out-of-Graph)` (難易度: Medium)
- **正解 (Ground Truth)**:
  > ナレッジグラフおよび LLaMA 論文には MMLU, GSM8k, BoolQ などの英語ベンチマーク評価は記録されていますが、日本語ベンチマーク（JGLUE）に関する評価スコアは記録されていません。
- **評価対象指標**: `Faithfulness`, `Answer Correctness`
- **根拠コンテキスト**: なし（グラフ外のため、取得コンテキストに含まれてはならない / 不足として拒否すべき）

---

