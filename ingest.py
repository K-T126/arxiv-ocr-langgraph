#!/usr/bin/env python3
"""
ingest.py - arXiv検索・PDF自動取得・要約抽出・グラフDB登録スクリプト (LangGraph 版)

README.md Step 2 の仕様に従い、以下のパイプラインを LangGraph ワークフローとして実行します:
1. arXiv API で cs.CL かつ LLM 関連論文を検索し、PDF を data/ にダウンロード
2. [LangGraph Ingest Pipeline]
   - TextExtractNode: PyMuPDF で全ページを取得し、ページ範囲付きチャンクへ分割
   - AnalyzeChunkNode: 各チャンクから Pydantic 構造化抽出
   - ValidateNode: チャンク単位のスキーマ妥当性検証
   - HealNode: エラー内容に応じた失敗チャンクの再試行/コンテキスト超過時の再分割
   - SaveKuzuNode: 全チャンク成功後に統合結果を KùzuDB へ保存
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
CHUNK_CHAR_LIMIT = 24000

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
    chunks: List[str]
    chunk_sources: List[str]
    text_char_count: int
    chunk_index: int
    chunk_analyses: List[PaperAnalysis]
    chunk_analysis: Optional[PaperAnalysis]
    retry_feedback: str
    analysis: Optional[PaperAnalysis]
    error: Optional[str]
    ingest_status: str
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


def extract_text_chunks_from_pdf(pdf_path: str, max_chars_per_chunk: int = CHUNK_CHAR_LIMIT) -> tuple[List[str], List[str], int]:
    """Extract every page and split at page boundaries (or within oversized pages)."""
    logger.info("Extracting text from PDF: %s", pdf_path)
    chunks: List[str] = []
    sources: List[str] = []
    current_pages: List[str] = []
    current_start = 1
    current_size = 0
    text_char_count = 0

    def flush(end_page: int):
        nonlocal current_pages, current_size, current_start
        if current_pages:
            chunks.append("\n\n".join(current_pages))
            sources.append(f"pages {current_start}-{end_page}")
            current_pages = []
            current_size = 0

    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            page_text = (page.get_text() or "").strip()
            text_char_count += len(page_text)
            if not page_text:
                continue

            # Split unusually long single pages without dropping any characters.
            pieces = [page_text[i:i + max_chars_per_chunk] for i in range(0, len(page_text), max_chars_per_chunk)]
            for piece_index, piece in enumerate(pieces):
                if current_size and current_size + len(piece) > max_chars_per_chunk:
                    flush(page_num - 1 if piece_index == 0 else page_num)
                    current_start = page_num
                if not current_pages:
                    current_start = page_num
                current_pages.append(piece)
                current_size += len(piece)
                if len(pieces) > 1:
                    flush(page_num)
                    current_start = page_num
    flush(len(doc))
    if text_char_count == 0:
        raise ValueError("PDFからテキストを抽出できませんでした。スキャン画像PDFの場合はOCRが必要です。")
    logger.info("Extracted %d characters into %d chunks", text_char_count, len(chunks))
    return chunks, sources, text_char_count


def merge_chunk_analyses(
    analyses: List[PaperAnalysis], paper_title: str, arxiv_id: Optional[str], source_url: Optional[str]
) -> PaperAnalysis:
    """Merge per-chunk findings while retaining distinct evaluation observations."""
    merged_models: Dict[str, ModelInfo] = {}
    for analysis in analyses:
        for model in analysis.models:
            key = " ".join(model.model_name.casefold().split())
            if not key:
                continue
            current = merged_models.get(key)
            if current is None:
                merged_models[key] = model.model_copy(deep=True)
                continue
            if not current.base_model and model.base_model:
                current.base_model = model.base_model
            if not current.method_type and model.method_type:
                current.method_type = model.method_type
            if model.summary and (not current.summary or len(model.summary) > len(current.summary)):
                current.summary = model.summary
            if model.evidence and model.evidence not in (current.evidence or ""):
                current.evidence = " | ".join(filter(None, [current.evidence, model.evidence]))
            seen = {(ev.benchmark.casefold(), ev.score or "", ev.metric or "", ev.conditions or "") for ev in current.evaluations}
            for evaluation in model.evaluations:
                ev_key = (evaluation.benchmark.casefold(), evaluation.score or "", evaluation.metric or "", evaluation.conditions or "")
                if ev_key not in seen:
                    current.evaluations.append(evaluation.model_copy(deep=True))
                    seen.add(ev_key)
                elif evaluation.evidence:
                    for existing in current.evaluations:
                        if (existing.benchmark.casefold(), existing.score or "", existing.metric or "", existing.conditions or "") == ev_key:
                            if evaluation.evidence not in (existing.evidence or ""):
                                existing.evidence = " | ".join(filter(None, [existing.evidence, evaluation.evidence]))
                            break
    return PaperAnalysis(
        paper_title=paper_title,
        arxiv_id=arxiv_id,
        source_url=source_url,
        models=list(merged_models.values()),
    )


def attach_chunk_source(analysis: PaperAnalysis, source: str) -> PaperAnalysis:
    """Persist the PDF page range alongside each extracted claim's evidence."""
    prefix = f"[{source}]"
    for model in analysis.models:
        model.evidence = f"{prefix} {model.evidence}" if model.evidence else prefix
        for evaluation in model.evaluations:
            evaluation.evidence = f"{prefix} {evaluation.evidence}" if evaluation.evidence else prefix
    return analysis


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
    """Build a chunked extraction workflow with bounded, actionable retries."""

    def extract_text_node(state: PaperIngestState) -> Dict[str, Any]:
        try:
            chunks, sources, char_count = extract_text_chunks_from_pdf(state["pdf_path"])
            return {"chunks": chunks, "chunk_sources": sources, "text_char_count": char_count,
                    "chunk_index": 0, "chunk_analyses": [], "retry_count": 0, "error": None}
        except Exception as exc:
            logger.exception("Text extraction failed for %s", state.get("pdf_path"))
            return {"chunks": [], "error": f"PDF text extraction failed: {exc}", "is_valid": False}

    def analyze_chunk_node(state: PaperIngestState) -> Dict[str, Any]:
        if not llm:
            return {"chunk_analysis": None, "error": "LLM client not configured", "is_valid": False}

        index = state.get("chunk_index", 0)
        chunks = state.get("chunks", [])
        if index >= len(chunks):
            return {"chunk_analysis": None, "error": f"Invalid chunk index: {index}", "is_valid": False}

        system_prompt = (
            "You extract structured facts from academic paper excerpts. Only report facts explicitly supported "
            "by the excerpt; do not infer missing scores or lineage. An excerpt may contain no relevant models, "
            "in which case return an empty models list. Preserve evaluation conditions and attach a short quote "
            "or section/table reference as evidence. Return only a JSON object with this shape: "
            '{"paper_title":"string","arxiv_id":null,"source_url":null,"models":[{"model_name":"string",'
            '"base_model":null,"method_type":null,"summary":null,"evidence":null,"evaluations":'
            '[{"benchmark":"string","score":null,"metric":null,"conditions":null,"evidence":null}]}]}'
        )
        source = state.get("chunk_sources", [])[index]
        user_prompt = (
            f"Paper title: {state.get('paper_title', 'Untitled')}\n"
            f"Excerpt location: {source}\nChunk: {index + 1}/{len(chunks)}\n\n"
            f"--- Excerpt ---\n{chunks[index]}\n\n"
            "Extract models/methods, base models, and benchmark evaluations found in this excerpt."
        )
        feedback = state.get("retry_feedback", "")
        if feedback:
            user_prompt += f"\n\nThe previous attempt failed. Fix this issue on this retry:\n{feedback}"

        try:
            result = llm.with_structured_output(PaperAnalysis).invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            if isinstance(result, dict):
                result = PaperAnalysis.model_validate(result)
            if not isinstance(result, PaperAnalysis):
                raise ValueError(f"Expected PaperAnalysis, received {type(result).__name__}")
            return {"chunk_analysis": attach_chunk_source(result, source), "error": None, "is_valid": False}
        except Exception as structured_error:
            try:
                response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                content = response.content or ""
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                data = json.loads(match.group(1).strip() if match else content.strip())
                return {"chunk_analysis": attach_chunk_source(PaperAnalysis.model_validate(data), source), "error": None, "is_valid": False}
            except Exception as fallback_error:
                detail = f"Structured output: {structured_error}; JSON fallback: {fallback_error}"
                logger.error("Chunk %d/%d extraction failed: %s", index + 1, len(chunks), detail)
                return {"chunk_analysis": None, "error": detail, "is_valid": False}

    def validate_chunk_node(state: PaperIngestState) -> Dict[str, Any]:
        result = state.get("chunk_analysis")
        if state.get("error") or not isinstance(result, PaperAnalysis):
            return {"is_valid": False, "error": state.get("error") or "LLM returned no valid PaperAnalysis"}
        invalid_models = [model for model in result.models if not model.model_name.strip()]
        invalid_evaluations = [
            (model.model_name, evaluation)
            for model in result.models
            for evaluation in model.evaluations
            if not evaluation.benchmark.strip()
        ]
        if invalid_models or invalid_evaluations:
            problems = []
            if invalid_models:
                problems.append(f"{len(invalid_models)} model record(s) have an empty model_name")
            if invalid_evaluations:
                problems.append(f"{len(invalid_evaluations)} evaluation record(s) have an empty benchmark")
            return {"is_valid": False, "error": "; ".join(problems)}
        accumulated = list(state.get("chunk_analyses", []))
        accumulated.append(result)
        next_index = state.get("chunk_index", 0) + 1
        logger.info("Validated chunk %d/%d (%d models)", next_index, len(state.get("chunks", [])), len(result.models))
        return {"is_valid": True, "chunk_analyses": accumulated, "chunk_index": next_index,
                "retry_count": 0, "retry_feedback": "", "error": None}

    def heal_node(state: PaperIngestState) -> Dict[str, Any]:
        retries = state.get("retry_count", 0) + 1
        error = state.get("error") or "The response was invalid; return schema-valid JSON."
        chunks = list(state.get("chunks", []))
        sources = list(state.get("chunk_sources", []))
        index = state.get("chunk_index", 0)
        normalized_error = error.casefold()
        context_error_markers = ("context length", "context_length", "too many tokens", "maximum tokens", "prompt is too long", "request too large")
        if (any(marker in normalized_error for marker in context_error_markers)
                and index < len(chunks) and len(chunks[index]) > 2000):
            chunk = chunks[index]
            midpoint = len(chunk) // 2
            split_at = chunk.rfind(" ", midpoint - 1000, midpoint + 1000)
            split_at = split_at if split_at > 0 else midpoint
            chunks[index:index + 1] = [chunk[:split_at], chunk[split_at:]]
            sources[index:index + 1] = [sources[index], sources[index]]
            logger.warning("Context error: split failing excerpt into smaller chunks (%d and %d chars)",
                           len(chunks[index]), len(chunks[index + 1]))
            # A smaller replacement excerpt gets its own retry budget.
            retries = 0
            error = "The previous excerpt exceeded the model context limit and was split into smaller excerpts."
        logger.info("Retry %d/%d for chunk %d of '%s'", retries, MAX_RETRIES,
                    index + 1, state.get("paper_title"))
        return {"retry_count": retries, "chunks": chunks, "chunk_sources": sources,
                "retry_feedback": error,
                "chunk_analysis": None}

    def finalize_analysis_node(state: PaperIngestState) -> Dict[str, Any]:
        analysis = merge_chunk_analyses(state.get("chunk_analyses", []), state.get("paper_title", "Untitled"),
                                        state.get("arxiv_id"), state.get("source_url"))
        status = "no_relevant_models" if not analysis.models else "ready_to_save"
        return {"analysis": analysis, "ingest_status": status, "error": None}

    def fail_node(state: PaperIngestState) -> Dict[str, Any]:
        return {"ingest_status": "failed", "error": state.get("error") or "Ingestion failed"}

    def dry_run_node(state: PaperIngestState) -> Dict[str, Any]:
        return {"ingest_status": "dry_run", "error": None}

    def save_kuzu_node(state: PaperIngestState) -> Dict[str, Any]:
        analysis = state.get("analysis")
        if not isinstance(analysis, PaperAnalysis) or not analysis.models:
            return {"ingest_status": "no_relevant_models"}
        try:
            target_db = state.get("db_path", db_path)
            database, conn = init_kuzu_db(target_db)
            save_to_kuzu(conn, analysis, paper_title=state.get("paper_title"),
                         arxiv_id=state.get("arxiv_id"), source_url=state.get("source_url"))
            logger.info("Saved analysis for '%s' to KùzuDB at %s", state.get("paper_title"), target_db)
            return {"ingest_status": "succeeded", "error": None}
        except Exception as exc:
            logger.exception("Failed to save into KùzuDB")
            return {"ingest_status": "failed", "error": f"Database save failed: {exc}"}

    workflow = StateGraph(PaperIngestState)
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("analyze_chunk", analyze_chunk_node)
    workflow.add_node("validate_chunk", validate_chunk_node)
    workflow.add_node("heal", heal_node)
    workflow.add_node("finalize_analysis", finalize_analysis_node)
    workflow.add_node("fail", fail_node)
    workflow.add_node("dry_run_done", dry_run_node)
    workflow.add_node("save_kuzu", save_kuzu_node)
    workflow.set_entry_point("extract_text")

    def route_after_extract(state: PaperIngestState):
        if state.get("error"):
            return "fail"
        return "dry_run_done" if state.get("dry_run") else "analyze_chunk"

    def route_after_validation(state: PaperIngestState):
        if state.get("is_valid"):
            return "analyze_chunk" if state.get("chunk_index", 0) < len(state.get("chunks", [])) else "finalize_analysis"
        return "heal" if state.get("retry_count", 0) < MAX_RETRIES else "fail"

    workflow.add_conditional_edges("extract_text", route_after_extract)
    workflow.add_edge("analyze_chunk", "validate_chunk")
    workflow.add_conditional_edges("validate_chunk", route_after_validation)
    workflow.add_edge("heal", "analyze_chunk")
    workflow.add_conditional_edges(
        "finalize_analysis",
        lambda state: "save_kuzu" if state.get("ingest_status") == "ready_to_save" else END,
    )
    workflow.add_edge("save_kuzu", END)
    workflow.add_edge("fail", END)
    workflow.add_edge("dry_run_done", END)
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

    if args.seed_sample_data:
        logger.info("Initializing KùzuDB at: %s", args.db_path)
        db, conn = init_kuzu_db(args.db_path)
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

    failed = 0
    for title, pdf_path, paper_id, source_url in targets:
        logger.info("==========================================")
        logger.info("[LangGraph] Processing paper: %s", title)
        initial_state: PaperIngestState = {
            "pdf_path": pdf_path,
            "paper_title": title,
            "arxiv_id": paper_id,
            "source_url": source_url,
            "analysis": None,
            "error": None,
            "retry_count": 0,
            "is_valid": False,
            "db_path": args.db_path,
            "dry_run": args.dry_run,
        }

        try:
            final_state = app.invoke(initial_state, config={"recursion_limit": 10000})
            status = final_state.get("ingest_status", "failed")
            if status == "dry_run":
                logger.info(
                    "[Dry-Run] Extracted %d characters into %d chunks. Skipping LLM call and DB write.",
                    final_state.get("text_char_count", 0),
                    len(final_state.get("chunks", [])),
                )
            elif status == "succeeded":
                logger.info("Successfully completed ingestion for '%s'.", title)
            elif status == "no_relevant_models":
                logger.info("No relevant models found in '%s'; nothing was saved.", title)
            else:
                failed += 1
                logger.error("Ingestion failed for '%s': %s", title, final_state.get("error", "unknown error"))
        except Exception as e:
            failed += 1
            logger.error("LangGraph execution failed for %s: %s", title, e, exc_info=True)

    logger.info("All LangGraph ingestion tasks completed.")
    if failed:
        logger.error("%d of %d paper(s) failed.", failed, len(targets))
        sys.exit(1)


if __name__ == "__main__":
    main()
