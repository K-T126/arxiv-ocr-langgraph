#!/usr/bin/env python3
"""
ingest.py - arXiv検索・PDF自動取得・要約抽出・グラフDB登録スクリプト (LangGraph 版)

README.md Step 2 の仕様に従い、以下のパイプラインを LangGraph ワークフローとして実行します:
1. arXiv API で cs.CL かつ LLM 関連論文を検索し、PDF を data/ にダウンロード
2. [LangGraph Ingest Pipeline]
   - TextExtractNode: PyMuPDF (fitz) で論文本文テキストを抽出
   - AnalyzePaperNode: OpenRouter (LangChain ChatOpenAI) + Qwen/Qwen2.5-72B-Instruct で Pydantic 構造化抽出
   - ValidateNode: 抽出データのスキーマ妥当性検証
   - HealNode: スキーマ不備時の自己修復/再試行ループ
   - SaveKuzuNode: KùzuDB (arxiv_graph.kuzu) にノード (Model, Benchmark) とリレーションを MERGE 登録
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

import arxiv
from dotenv import load_dotenv
import kuzu
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import pymupdf
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# --- Pydantic スキーマ定義 ---


class Evaluation(BaseModel):
    benchmark: str = Field(
        ...,
        description="Name of the evaluation benchmark or dataset (e.g., 'MMLU', 'GSM8K', 'HumanEval')",
    )
    score: Optional[str] = Field(
        None,
        description="Score achieved on this benchmark (e.g., '65.4', '78.2%')",
    )
    metric: Optional[str] = Field(
        None,
        description="Metric used for evaluation (e.g., 'accuracy', 'pass@1', 'BLEU')",
    )
    conditions: Optional[str] = Field(None, description="Evaluation setup such as split, shot count, or model size")
    evidence: Optional[str] = Field(None, description="Short verbatim evidence or table reference supporting this result")


class ModelInfo(BaseModel):
    model_name: str = Field(
        ...,
        description="Name of the model or method introduced, evaluated, or fine-tuned (e.g., 'Alpaca', 'Vicuna-13B')",
    )
    base_model: Optional[str] = Field(
        None,
        description="Parent or base model this model is derived from (e.g., 'LLaMA-7B', 'Mistral-7B'). None if foundational.",
    )
    method_type: Optional[str] = Field(
        None,
        description="Type of training, adaptation, or method (e.g., 'Instruction Tuning', 'RLHF', 'DPO', 'Pre-training')",
    )
    evaluations: List[Evaluation] = Field(
        default_factory=list,
        description="Benchmark evaluations for this model",
    )
    summary: Optional[str] = Field(
        None,
        description="Short summary of the model and its main contributions",
    )
    evidence: Optional[str] = Field(None, description="Short evidence or section/table reference supporting this model and lineage")


class PaperAnalysis(BaseModel):
    paper_title: str = Field(..., description="Title of the paper")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier when known")
    source_url: Optional[str] = Field(None, description="Canonical paper URL when known")
    models: List[ModelInfo] = Field(
        default_factory=list,
        description="List of LLM models or methods analyzed or proposed in this paper",
    )


# --- LangGraph 状態定義 ---


class PaperIngestState(TypedDict, total=False):
    pdf_path: str
    paper_title: str
    arxiv_id: str
    source_url: str
    raw_text: str
    analysis: Optional[PaperAnalysis]
    error: Optional[str]
    retry_count: int
    is_valid: bool
    db_path: str
    dry_run: bool


# --- KùzuDB 初期化 ---


def init_kuzu_db(db_path: str):
    """KùzuDB データベースとスキーマ（ノード・リレーション）を初期化する"""
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Model ノード
    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Model(
            name STRING,
            method_type STRING,
            summary STRING,
            PRIMARY KEY (name)
        )
        """
    )

    # Benchmark ノード
    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Benchmark(
            name STRING,
            PRIMARY KEY (name)
        )
        """
    )

    # DERIVED_FROM リレーション: (:Model)-[:DERIVED_FROM]->(:Model)
    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(
            FROM Model TO Model
        )
        """
    )

    # EVALUATED_ON リレーション: (:Model)-[:EVALUATED_ON {score, metric}]->(:Benchmark)
    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS EVALUATED_ON(
            FROM Model TO Benchmark,
            score STRING,
            metric STRING
        )
        """
    )

    # Provenance and observation nodes preserve sources and repeated results.
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Paper(arxiv_id STRING, title STRING, source_url STRING, PRIMARY KEY (arxiv_id))")
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Evaluation(evaluation_id STRING, score STRING, metric STRING, conditions STRING, evidence STRING, PRIMARY KEY (evaluation_id))")
    conn.execute("CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(FROM Model TO Paper)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS DERIVED_FROM_REPORTED(FROM Model TO Model, paper_id STRING, evidence STRING)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS HAS_EVALUATION(FROM Model TO Evaluation)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS EVALUATION_ON(FROM Evaluation TO Benchmark)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS REPORTED_IN(FROM Evaluation TO Paper)")

    return db, conn


# --- PDF テキスト抽出 ---


def extract_text_from_pdf(pdf_path: str, max_chars: int = 250000) -> str:
    """PyMuPDF を用いて PDF から本文テキストを抽出する"""
    logger.info("Extracting text from PDF: %s", pdf_path)
    doc = pymupdf.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text.append(page.get_text())

    extracted = "\n\n".join(full_text)
    doc.close()

    if len(extracted) > max_chars:
        logger.warning(
            "Text truncated from %d to %d characters to respect token limits.",
            len(extracted),
            max_chars,
        )
        extracted = extracted[:max_chars]

    return extracted


# --- KùzuDB への保存処理 ---


def save_to_kuzu(conn: kuzu.Connection, analysis: PaperAnalysis, paper_title: Optional[str] = None, arxiv_id: Optional[str] = None, source_url: Optional[str] = None):
    """抽出したデータを KùzuDB に書き込む"""
    title = analysis.paper_title or paper_title or "Untitled"
    paper_id = analysis.arxiv_id or arxiv_id or "title:" + hashlib.sha256(title.strip().casefold().encode()).hexdigest()[:24]
    paper_url = analysis.source_url or source_url or (f"https://arxiv.org/abs/{paper_id}" if not paper_id.startswith("title:") else "")
    conn.execute("MERGE (p:Paper {arxiv_id: $id}) ON CREATE SET p.title = $title, p.source_url = $url ON MATCH SET p.title = $title, p.source_url = $url", parameters={"id": paper_id, "title": title, "url": paper_url})

    for model in analysis.models:
        model_name = model.model_name.strip()
        method_type = model.method_type or "Unknown"
        summary = model.summary or ""

        # 1. Model ノードを登録 (MERGE)
        conn.execute(
            """
            MERGE (m:Model {name: $name})
            ON CREATE SET m.method_type = $method_type, m.summary = $summary
            ON MATCH SET m.method_type = $method_type, m.summary = $summary
            """,
            parameters={
                "name": model_name,
                "method_type": method_type,
                "summary": summary,
            },
        )
        logger.info("Saved Model node: %s", model_name)
        conn.execute("MATCH (m:Model {name: $name}), (p:Paper {arxiv_id: $paper}) MERGE (m)-[:MENTIONED_IN]->(p)", parameters={"name": model_name, "paper": paper_id})

        # 2. ベースモデルが存在する場合、親ノード作成と DERIVED_FROM リレーション登録
        if model.base_model:
            base_model_name = model.base_model.strip()
            if base_model_name and base_model_name.lower() != model_name.lower():
                conn.execute(
                    """
                    MERGE (b:Model {name: $name})
                    ON CREATE SET b.method_type = 'Base Model', b.summary = 'Referenced as base model'
                    """,
                    parameters={"name": base_model_name},
                )
                conn.execute(
                    """
                    MATCH (m:Model {name: $child}), (b:Model {name: $parent})
                    MERGE (m)-[:DERIVED_FROM]->(b)
                    """,
                    parameters={"child": model_name, "parent": base_model_name},
                )
                conn.execute("MATCH (m:Model {name: $child}), (b:Model {name: $parent}) MERGE (m)-[r:DERIVED_FROM_REPORTED {paper_id: $paper}]->(b) ON CREATE SET r.evidence = $evidence", parameters={"child": model_name, "parent": base_model_name, "paper": paper_id, "evidence": model.evidence or ""})
                logger.info(
                    "Created relationship: (:Model %s)-[:DERIVED_FROM]->(:Model %s)",
                    model_name,
                    base_model_name,
                )

        # 3. 評価ベンチマークの登録と EVALUATED_ON リレーション作成
        for ev in model.evaluations:
            b_name = ev.benchmark.strip()
            if not b_name:
                continue

            score = ev.score or "N/A"
            metric = ev.metric or "N/A"

            conn.execute(
                """
                MERGE (b:Benchmark {name: $name})
                """,
                parameters={"name": b_name},
            )
            evidence_key = "|".join([paper_id, model_name.casefold(), b_name.casefold(), score, metric, ev.conditions or "", ev.evidence or ""])
            evaluation_id = hashlib.sha256(evidence_key.encode()).hexdigest()
            conn.execute("MERGE (e:Evaluation {evaluation_id: $id}) ON CREATE SET e.score = $score, e.metric = $metric, e.conditions = $conditions, e.evidence = $evidence", parameters={"id": evaluation_id, "score": score, "metric": metric, "conditions": ev.conditions or "", "evidence": ev.evidence or ""})
            conn.execute("MATCH (m:Model {name: $model}), (e:Evaluation {evaluation_id: $id}), (b:Benchmark {name: $benchmark}), (p:Paper {arxiv_id: $paper}) MERGE (m)-[:HAS_EVALUATION]->(e) MERGE (e)-[:EVALUATION_ON]->(b) MERGE (e)-[:REPORTED_IN]->(p)", parameters={"model": model_name, "id": evaluation_id, "benchmark": b_name, "paper": paper_id})

            conn.execute(
                """
                MATCH (m:Model {name: $mname}), (b:Benchmark {name: $bname})
                MERGE (m)-[r:EVALUATED_ON]->(b)
                ON CREATE SET r.score = $score, r.metric = $metric
                ON MATCH SET r.score = $score, r.metric = $metric
                """,
                parameters={
                    "mname": model_name,
                    "bname": b_name,
                    "score": score,
                    "metric": metric,
                },
            )
            logger.info(
                "Created evaluation: (%s)-[:EVALUATED_ON {score: %s}]->(%s)",
                model_name,
                score,
                b_name,
            )


# --- LangGraph ノード定義 ---


def create_ingest_graph(llm: Optional[ChatOpenAI], db_path: str):
    """LangGraph による論文取り込みワークフローを構築する"""

    def extract_text_node(state: PaperIngestState) -> Dict[str, Any]:
        """Node 1: PDF からテキスト抽出"""
        pdf_path = state["pdf_path"]
        try:
            text = extract_text_from_pdf(pdf_path)
            return {"raw_text": text, "retry_count": state.get("retry_count", 0)}
        except Exception as e:
            logger.error("Text extraction failed for %s: %s", pdf_path, e)
            return {"raw_text": "", "error": str(e), "is_valid": False}

    def analyze_paper_node(state: PaperIngestState) -> Dict[str, Any]:
        """Node 2: LLM (Qwen2.5-72B via OpenRouter) で Pydantic 構造化抽出"""
        if state.get("dry_run"):
            logger.info("[Dry-Run] Skipping LLM extraction node.")
            return {"analysis": None, "is_valid": True}

        if not llm:
            return {"error": "LLM client not configured", "is_valid": False}

        raw_text = state.get("raw_text", "")
        paper_title = state.get("paper_title", "Untitled")

        system_prompt = (
            "You are an expert AI research assistant specializing in Large Language Models (LLMs).\n"
            "Analyze the given academic paper and extract structured information about models, "
            "their derivation lineage (e.g. Alpaca derived from LLaMA), and benchmark evaluation scores.\n\n"
            "Output MUST be valid JSON matching this schema:\n"
            "{\n"
            '  "paper_title": "string",\n'
            '  "arxiv_id": "string or null",\n'
            '  "source_url": "string or null",\n'
            '  "models": [\n'
            "    {\n"
            '      "model_name": "string (e.g., Alpaca-7B)",\n'
            '      "base_model": "string or null",\n'
            '      "method_type": "string or null",\n'
            '      "evaluations": [\n'
            '        {"benchmark": "string", "score": "string or null", "metric": "string or null", "conditions": "string or null", "evidence": "short quote or table reference"}\n'
            "      ],\n"
            '      "summary": "string or null"\n'
            '      "evidence": "short quote or section/table reference"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Respond ONLY with the JSON object."
        )

        user_prompt = (
            f"Paper Title: {paper_title}\n\n"
            f"--- Paper Text Content ---\n"
            f"{raw_text}\n\n"
            f"Extract all primary models/methods, their base models, and benchmark evaluations."
        )

        try:
            # LangChain structured output 呼び出し（フォールバック付き）
            structured_llm = llm.with_structured_output(PaperAnalysis)
            analysis = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return {"analysis": analysis, "error": None}
        except Exception as e:
            logger.warning("Structured output invoke failed: %s. Trying JSON mode parse...", e)
            # フォールバック: 通常メッセージ呼び出しから JSON 正規表現抽出
            try:
                res = llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                content = res.content or ""
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                json_str = match.group(1).strip() if match else content.strip()
                data = json.loads(json_str)
                analysis = PaperAnalysis(**data)
                return {"analysis": analysis, "error": None}
            except Exception as e2:
                logger.error("JSON fallback parsing failed: %s", e2)
                return {"analysis": None, "error": str(e2)}

    def validate_analysis_node(state: PaperIngestState) -> Dict[str, Any]:
        """Node 3: 抽出データの検証"""
        if state.get("dry_run"):
            return {"is_valid": True}

        analysis = state.get("analysis")
        if not analysis or not isinstance(analysis, PaperAnalysis):
            logger.warning("Validation failed: analysis is empty or invalid type.")
            return {"is_valid": False}

        if not analysis.models:
            logger.warning("Validation note: No models were extracted for '%s'.", state.get("paper_title"))
            # モデルなしでも空の有効な解析として認めるか、リトライするか
            return {"is_valid": True}

        logger.info(
            "Validation passed: extracted %d model(s) for '%s'",
            len(analysis.models),
            analysis.paper_title,
        )
        return {"is_valid": True}

    def heal_node(state: PaperIngestState) -> Dict[str, Any]:
        """Node 4: 自己修復ノード（プロンプト補正 & リトライカウント増加）"""
        current_retries = state.get("retry_count", 0) + 1
        logger.info("[Self-Healing] Retry %d/%d for '%s'", current_retries, MAX_RETRIES, state.get("paper_title"))
        return {"retry_count": current_retries}

    def save_kuzu_node(state: PaperIngestState) -> Dict[str, Any]:
        """Node 5: KùzuDB への永続化"""
        if state.get("dry_run"):
            logger.info("[Dry-Run] Skipped database write.")
            return {}

        analysis = state.get("analysis")
        if not analysis or not analysis.models:
            logger.info("No models to save for '%s'.", state.get("paper_title"))
            return {}

        try:
            target_db = state.get("db_path", db_path)
            db, conn = init_kuzu_db(target_db)
            save_to_kuzu(conn, analysis, paper_title=state.get("paper_title"), arxiv_id=state.get("arxiv_id"), source_url=state.get("source_url"))
            logger.info("Saved analysis for '%s' to KùzuDB at %s", state.get("paper_title"), target_db)
        except Exception as e:
            logger.error("Failed to save into KùzuDB: %s", e)
        return {}

    # グラフ構築
    workflow = StateGraph(PaperIngestState)
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("analyze_paper", analyze_paper_node)
    workflow.add_node("validate_analysis", validate_analysis_node)
    workflow.add_node("heal", heal_node)
    workflow.add_node("save_kuzu", save_kuzu_node)

    workflow.set_entry_point("extract_text")
    workflow.add_edge("extract_text", "analyze_paper")
    workflow.add_edge("analyze_paper", "validate_analysis")

    def route_after_validation(state: PaperIngestState):
        if state.get("is_valid"):
            return "save_kuzu"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "heal"
        logger.warning("Max retries reached. Proceeding to save_kuzu with partial data.")
        return "save_kuzu"

    workflow.add_conditional_edges("validate_analysis", route_after_validation)
    workflow.add_edge("heal", "analyze_paper")
    workflow.add_edge("save_kuzu", END)

    return workflow.compile()


# --- arXiv 検索 & PDF ダウンロード ---


def _save_pdf_from_url(url: str, target_path: str):
    headers = {"User-Agent": "arxiv-ocr-langgraph/1.0 (mailto:researcher@example.com)"}
    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_arxiv_papers(
    query: str = 'cat:cs.CL AND (LLM OR "Large Language Model")',
    max_results: int = 5,
    download_dir: str = "./data",
    arxiv_id: Optional[str] = None,
) -> List[tuple]:
    os.makedirs(download_dir, exist_ok=True)
    if max_results <= 0 and not arxiv_id:
        logger.info("max_results is 0 and no arxiv_id provided. Skipping download.")
        return []

    client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=3)

    if arxiv_id:
        clean_id = arxiv_id.strip()
        if "arxiv.org" in clean_id:
            clean_id = clean_id.split("/")[-1].replace(".pdf", "")
        logger.info("Fetching specific arXiv paper by ID: %s", clean_id)
        search = arxiv.Search(id_list=[clean_id])
    else:
        logger.info("Searching arXiv: query='%s', max_results=%d", query, max_results)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

    results = []
    for paper in client.results(search):
        paper_id = paper.get_short_id()
        safe_filename = f"{paper_id}.pdf"
        target_path = os.path.join(download_dir, safe_filename)

        if os.path.exists(target_path):
            logger.info("PDF already exists locally: %s", target_path)
        else:
            logger.info("Downloading PDF: %s -> %s", paper.title, target_path)
            try:
                if hasattr(paper, "download_pdf"):
                    paper.download_pdf(dirpath=download_dir, filename=safe_filename)
                else:
                    _save_pdf_from_url(paper.pdf_url, target_path)
            except Exception as e:
                logger.error("Failed to download PDF for %s: %s", paper.title, e)
                continue

        results.append((paper, target_path))

    return results


# --- メイン実行 ---


def main():
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into KùzuDB Knowledge Graph using LangGraph"
    )
    parser.add_argument(
        "--query",
        type=str,
        default='cat:cs.CL AND (LLM OR "Large Language Model")',
        help='arXiv search query (default: cat:cs.CL AND (LLM OR "Large Language Model"))',
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Number of papers to fetch (default: 5)",
    )
    parser.add_argument(
        "--arxiv-id",
        type=str,
        default=None,
        help="Specific arXiv paper ID or URL to process (e.g., 2302.13971)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to an existing local PDF file to process directly",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./arxiv_graph.kuzu",
        help="Path to KùzuDB directory (default: ./arxiv_graph.kuzu)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save downloaded PDFs (default: ./data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for OpenRouter (default: env OPENROUTER_MODEL or qwen/qwen-2.5-72b-instruct)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download PDF and extract text only, without calling OpenRouter LLM",
    )
    parser.add_argument(
        "--seed-sample-data",
        action="store_true",
        help="Seed sample LLM lineage and evaluation data into KùzuDB for testing (LLaMA, Alpaca, Vicuna)",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    # dry-run や seed-sample-data でない場合のみ API キーを必須とする親切設計
    if not args.dry_run and not args.seed_sample_data and not api_key:
        logger.error("OPENROUTER_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    model_name = args.model or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    llm = None
    if api_key and not args.dry_run:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            temperature=0.1,
        )

    # KùzuDB 初期化
    logger.info("Initializing KùzuDB at: %s", args.db_path)
    db, conn = init_kuzu_db(args.db_path)

    if args.seed_sample_data:
        logger.info("Seeding sample LLM lineage and evaluation data into KùzuDB...")
        sample_analysis = PaperAnalysis(
            paper_title="Sample LLM Family Benchmarks",
            models=[
                ModelInfo(
                    model_name="LLaMA-7B",
                    base_model=None,
                    method_type="Pre-training",
                    summary="Foundational open-source LLM released by Meta AI.",
                    evaluations=[
                        Evaluation(benchmark="MMLU", score="35.1", metric="accuracy"),
                        Evaluation(benchmark="GSM8K", score="11.0", metric="accuracy"),
                    ],
                ),
                ModelInfo(
                    model_name="Alpaca-7B",
                    base_model="LLaMA-7B",
                    method_type="Instruction Tuning",
                    summary="Stanford Alpaca, fine-tuned from LLaMA-7B using 52k self-instruct demonstrations.",
                    evaluations=[
                        Evaluation(benchmark="MMLU", score="40.5", metric="accuracy"),
                        Evaluation(benchmark="GSM8K", score="14.2", metric="accuracy"),
                    ],
                ),
                ModelInfo(
                    model_name="Vicuna-13B",
                    base_model="LLaMA-13B",
                    method_type="Instruction Tuning",
                    summary="Chat assistant model fine-tuned on ShareGPT conversations from LLaMA.",
                    evaluations=[
                        Evaluation(benchmark="MMLU", score="52.1", metric="accuracy"),
                        Evaluation(benchmark="GSM8K", score="27.6", metric="accuracy"),
                    ],
                ),
            ],
        )
        save_to_kuzu(conn, sample_analysis)
        logger.info("Sample data successfully seeded.")

    targets = []
    if args.pdf:
        if not os.path.exists(args.pdf):
            logger.error("Specified PDF file does not exist: %s", args.pdf)
            sys.exit(1)
        title = os.path.splitext(os.path.basename(args.pdf))[0]
        targets.append((title, args.pdf, None, None))
    else:
        papers = download_arxiv_papers(
            query=args.query,
            max_results=args.max_results,
            download_dir=args.data_dir,
            arxiv_id=args.arxiv_id,
        )
        for p, pdf_path in papers:
            paper_id = p.get_short_id()
            targets.append((p.title, pdf_path, paper_id, f"https://arxiv.org/abs/{paper_id}"))

    if not targets:
        logger.warning("No papers found or specified to process.")
        return

    logger.info("Found %d paper(s) to process via LangGraph pipeline.", len(targets))

    # LangGraph パイプラインをコンパイル
    app = create_ingest_graph(llm=llm, db_path=args.db_path)

    for title, pdf_path, paper_id, source_url in targets:
        logger.info("==========================================")
        logger.info("[LangGraph] Processing paper: %s", title)
        initial_state: PaperIngestState = {
            "pdf_path": pdf_path,
            "paper_title": title,
            "arxiv_id": paper_id,
            "source_url": source_url,
            "raw_text": "",
            "analysis": None,
            "error": None,
            "retry_count": 0,
            "is_valid": False,
            "db_path": args.db_path,
            "dry_run": args.dry_run,
        }

        try:
            final_state = app.invoke(initial_state)
            if args.dry_run:
                logger.info(
                    "[Dry-Run] Extracted %d characters. Skipping LLM call and DB write.",
                    len(final_state.get("raw_text", "")),
                )
            else:
                logger.info("Successfully completed LangGraph ingestion for '%s'.", title)
        except Exception as e:
            logger.error("LangGraph execution failed for %s: %s", title, e, exc_info=True)

    logger.info("All LangGraph ingestion tasks completed.")


if __name__ == "__main__":
    main()
