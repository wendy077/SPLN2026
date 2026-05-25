"""
Avaliação quantitativa dos três retrievers e dos dois módulos de QA.

Utiliza o ficheiro data/eval_queries.json com 20 queries anotadas manualmente.

Métricas dos retrievers:
  - Precision@1   — proporção de queries em que o 1º resultado é relevante
  - Hit@3         — existe pelo menos um documento relevante nos 3 primeiros?
  - MRR           — Mean Reciprocal Rank (posição do 1º resultado correto)

Métricas do QA:
  - Exact Match (EM)
  - F1 token-level

Exemplos de uso:
    python src/evaluate.py
    python src/evaluate.py --no-qa          # só retrievers, mais rápido
    python src/evaluate.py --top-k 5
    python src/evaluate.py --save-results data/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"
EVAL_QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries.json"


# ---------------------------------------------------------------------------
# EM / F1 helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def compute_exact(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def soft_match(prediction: str, expected: str) -> bool:
    """Verifica se a resposta esperada está contida na predição (ou vice-versa)."""
    pred = normalize_answer(prediction)
    exp = normalize_answer(expected)
    return exp in pred or pred in exp


# ---------------------------------------------------------------------------
# Retriever evaluation
# ---------------------------------------------------------------------------

def reciprocal_rank(results: list[dict], relevant_doc_ids: list[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result["doc_id"] in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(results: list[dict], relevant_doc_ids: list[str], k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r["doc_id"] in relevant_doc_ids)
    return hits / k

def hit_at_k(results: list[dict], relevant_doc_ids: list[str], k: int) -> float:
    top_k = results[:k]
    return float(any(r["doc_id"] in relevant_doc_ids for r in top_k))

def evaluate_retriever(retriever, queries: list[dict], top_k: int = 5) -> dict:
    rr_scores = []
    p1_scores = []
    per_query = []
    h3_scores = []

    for q in queries:
        results = retriever.search(
            query=q["question"],
            top_k=top_k,
            unique_docs=True,
            filter_type=q.get("filter_type"),
        )

        rr = reciprocal_rank(results, q["relevant_doc_ids"])
        p1 = precision_at_k(results, q["relevant_doc_ids"], k=1)
        h3 = hit_at_k(results, q["relevant_doc_ids"], k=3)
        h3_scores.append(h3)

        rr_scores.append(rr)
        p1_scores.append(p1)

        per_query.append({
            "id": q["id"],
            "question": q["question"],
            "mrr": round(rr, 4),
            "p@1": round(p1, 4),
            "top1_doc": results[0]["doc_id"] if results else None,
            "top1_title": results[0]["title"] if results else None,
            "hit@3": round(h3, 4),
        })

    return {
        "mrr": round(float(np.mean(rr_scores)), 4),
        "precision_at_1": round(float(np.mean(p1_scores)), 4),
        "num_queries": len(queries),
        "per_query": per_query,
        "hit_at_3": round(float(np.mean(h3_scores)), 4),
    }


# ---------------------------------------------------------------------------
# QA evaluation
# ---------------------------------------------------------------------------

def evaluate_qa_extractive(retriever, queries: list[dict]) -> dict:
    from qa_extractive import ExtractiveQAModel, answer_question, get_qa_model_name

    model_name = get_qa_model_name()
    print(f"  Loading extractive QA model: {model_name}")
    qa_model = ExtractiveQAModel(model_name)

    em_scores = []
    f1_scores = []
    per_query = []

    for q in queries:
        answers = answer_question(
            question=q["question"],
            retriever=retriever,
            qa_model=qa_model,
            top_k_retrieval=8,
            top_k_answers=1,
            filter_type=q.get("filter_type"),
            unique_docs=True,
        )

        prediction = answers[0]["answer"] if answers else ""
        expected = q["expected_answer"]

        em = compute_exact(prediction, expected)
        f1 = compute_f1(prediction, expected)

        em_scores.append(em)
        f1_scores.append(f1)

        per_query.append({
            "id": q["id"],
            "question": q["question"],
            "expected": expected,
            "predicted": prediction,
            "em": em,
            "f1": round(f1, 4),
        })

    return {
        "exact_match": round(100.0 * float(np.mean(em_scores)), 2),
        "f1": round(100.0 * float(np.mean(f1_scores)), 2),
        "num_queries": len(queries),
        "per_query": per_query,
    }


def evaluate_qa_abstractive(retriever, queries: list[dict]) -> dict:
    from qa_abstractive import AbstractiveQAModel, GEN_MODEL_NAME, answer_question

    print(f"  Loading abstractive QA model: {GEN_MODEL_NAME}")
    generator = AbstractiveQAModel(GEN_MODEL_NAME)

    em_scores = []
    f1_scores = []
    per_query = []

    for q in queries:
        result = answer_question(
            question=q["question"],
            retriever=retriever,
            generator=generator,
            top_k_retrieval=5,
            filter_type=q.get("filter_type"),
            unique_docs=True,
        )

        prediction = result["answer"]
        expected = q["expected_answer"]

        em = compute_exact(prediction, expected)
        f1 = compute_f1(prediction, expected)

        em_scores.append(em)
        f1_scores.append(f1)

        per_query.append({
            "id": q["id"],
            "question": q["question"],
            "expected": expected,
            "predicted": prediction,
            "method": result.get("answer_method", "generative"),
            "em": em,
            "f1": round(f1, 4),
        })

    method_counts = Counter(q["method"] for q in per_query)

    return {
        "exact_match": round(100.0 * float(np.mean(em_scores)), 2),
        "f1": round(100.0 * float(np.mean(f1_scores)), 2),
        "num_queries": len(queries),
        "method_counts": dict(method_counts),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_retriever_table(name: str, metrics: dict) -> None:
    print(f"\n  {name}")
    print(f"    MRR:          {metrics['mrr']:.4f}")
    print(f"    Precision@1:  {metrics['precision_at_1']:.4f}")
    print(f"    Hit@3:        {metrics['hit_at_3']:.4f}")

def print_qa_table(name: str, metrics: dict) -> None:
    print(f"\n  {name}")
    print(f"    Exact Match:  {metrics['exact_match']:.2f}%")
    print(f"    F1:           {metrics['f1']:.2f}%")
    if "method_counts" in metrics:
        for method, count in sorted(metrics["method_counts"].items()):
            print(f"    [{method}: {count} queries]")


def print_comparison_table(results: dict) -> None:
    print("\n" + "=" * 60)
    print("RETRIEVER COMPARISON")
    print("=" * 60)
    print(f"{'Retriever':<18} {'MRR':>8} {'P@1':>8} {'Hit@3':>8}")
    print("-" * 52)
    for name in ["BM25", "SBERT", "Hybrid"]:
        key = name.lower()
        if key not in results:
            continue
        m = results[key]
        print(
            f"{name:<18} "
            f"{m['mrr']:>8.4f} "
            f"{m['precision_at_1']:>8.4f} "
            f"{m['hit_at_3']:>8.4f}"
        )

    if "qa" in results:
        print()
        print("=" * 60)
        print("QA MODULE COMPARISON")
        print("=" * 60)
        print(f"{'Module':<22} {'Exact Match':>12} {'F1':>8}")
        print("-" * 60)
        for name in ["Extractive", "Abstractive"]:
            key = name.lower()
            if key not in results["qa"]:
                continue
            m = results["qa"][key]
            print(f"{name:<22} {m['exact_match']:>11.2f}% {m['f1']:>7.2f}%")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrievers and QA modules on annotated queries.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve per query.")
    parser.add_argument("--no-qa", action="store_true", help="Skip QA evaluation (faster).")
    parser.add_argument("--save-results", type=str, default=None, help="Save full results to a JSON file.")
    args = parser.parse_args()

    # Load queries
    if not EVAL_QUERIES_PATH.exists():
        print(f"ERROR: eval queries not found at {EVAL_QUERIES_PATH}")
        sys.exit(1)

    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation queries.")

    # Load shared data
    from retriever_bm25 import BM25Retriever, load_chunks
    from retriever_sbert import SBERTRetriever, load_model_name
    from retriever_hybrid import HybridRetriever

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    model_name = load_model_name()

    print("\nBuilding retrievers...")
    bm25 = BM25Retriever(chunks)
    sbert = SBERTRetriever(chunks, embeddings, model_name)
    hybrid = HybridRetriever(chunks, embeddings, model_name)

    # Wrap BM25 and SBERT to accept same kwargs as hybrid (filter_type, unique_docs)
    class BM25Wrapper:
        def __init__(self, r, max_results: int):
            self._r = r
            self._max_results = max_results

        def search(self, query, top_k=5, unique_docs=False, filter_type=None, **kw):
            # Search all chunks first, then apply type filtering and document deduplication.
            # This avoids losing valid documents when the type filter is applied after ranking.
            results = self._r.search(query, top_k=self._max_results)
            if filter_type:
                results = [r for r in results if r["type"] == filter_type]
            seen = set()
            out = []
            for r in results:
                if unique_docs and r["doc_id"] in seen:
                    continue
                seen.add(r["doc_id"])
                out.append(r)
                if len(out) >= top_k:
                    break
            return out

    class SBERTWrapper:
        def __init__(self, r, max_results: int):
            self._r = r
            self._max_results = max_results

        def search(self, query, top_k=5, unique_docs=False, filter_type=None, **kw):
            # Search all chunks first, then apply type filtering and document deduplication.
            # This makes the comparison with the hybrid retriever more stable.
            results = self._r.search(query, top_k=self._max_results, unique_docs=False)
            if filter_type:
                results = [r for r in results if r["type"] == filter_type]
            seen = set()
            out = []
            for r in results:
                if unique_docs and r["doc_id"] in seen:
                    continue
                seen.add(r["doc_id"])
                out.append(r)
                if len(out) >= top_k:
                    break
            return out

    all_results: dict = {}

    print("\n" + "=" * 60)
    print("Evaluating BM25...")
    bm25_metrics = evaluate_retriever(BM25Wrapper(bm25, len(chunks)), queries, top_k=args.top_k)
    all_results["bm25"] = bm25_metrics
    print_retriever_table("BM25", bm25_metrics)

    print("\nEvaluating SBERT...")
    sbert_metrics = evaluate_retriever(SBERTWrapper(sbert, len(chunks)), queries, top_k=args.top_k)
    all_results["sbert"] = sbert_metrics
    print_retriever_table("SBERT", sbert_metrics)

    print("\nEvaluating Hybrid (BM25 + SBERT, α=0.6)...")
    hybrid_metrics = evaluate_retriever(hybrid, queries, top_k=args.top_k)
    all_results["hybrid"] = hybrid_metrics
    print_retriever_table("Hybrid", hybrid_metrics)

    if not args.no_qa:
        all_results["qa"] = {}

        print("\n" + "=" * 60)
        print("Evaluating Extractive QA...")
        ext_metrics = evaluate_qa_extractive(hybrid, queries)
        all_results["qa"]["extractive"] = ext_metrics
        print_qa_table("Extractive QA", ext_metrics)

        print("\nEvaluating Abstractive QA...")
        abs_metrics = evaluate_qa_abstractive(hybrid, queries)
        all_results["qa"]["abstractive"] = abs_metrics
        print_qa_table("Abstractive QA", abs_metrics)

    print_comparison_table(all_results)

    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
