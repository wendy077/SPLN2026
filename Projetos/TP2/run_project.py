"""
Ponto de entrada principal para o TP2.

Exemplos:
    python run_project.py check
    python run_project.py demo
    python run_project.py search "Who released OK Computer?" --filter-type album
    python run_project.py ask "Who is the lead singer of Radiohead?" --mode extractive --filter-type artist
    python run_project.py ask "Explain why To Pimp a Butterfly is considered important." --mode abstractive --filter-type album
    python run_project.py evaluate
    python run_project.py evaluate --no-qa
    python run_project.py visualize-embeddings
    python run_project.py visualize-embeddings --method umap
    python run_project.py visualize-retrieval "What genre is Nirvana associated with?" --filter-type genre
    python run_project.py rebuild-index
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"
SBERT_CONFIG_PATH = PROJECT_ROOT / "models" / "sbert_config.json"
QA_FINETUNED_DIR = PROJECT_ROOT / "models" / "qa_finetuned"

# Permite importar os módulos existentes em src/ sem obrigar a instalar o projeto como pacote.
sys.path.insert(0, str(SRC_DIR))


FILTER_TYPES = ["artist", "album", "genre", "festival", "award", "event", "movement"]


DEMO_FACTUAL_QUESTIONS = [
    ("Who released OK Computer?", "album"),
    ("What genre is Nirvana associated with?", "genre"),
    ("Which artist released the album Blonde?", "album"),
    ("Which band released the album Nevermind?", "album"),
]

DEMO_ABSTRACTIVE_QUESTIONS = [
    ("Explain why To Pimp a Butterfly is considered important.", "album"),
    ("What is trip hop?", "genre"),
]

DEMO_BOTH_QUESTIONS = [
    ("Who is the lead singer of Radiohead?", "artist"),
]


class ProjectError(RuntimeError):
    pass


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def require_file(path: Path, hint: str | None = None) -> None:
    if path.exists():
        return

    message = f"Missing required file: {path}"
    if hint:
        message += f"\nHint: {hint}"
    raise ProjectError(message)


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def check_project() -> None:
    print_header("Project status")

    require_file(CORPUS_PATH, "Run the corpus construction/repair scripts or restore data/raw/corpus.jsonl.")
    require_file(CHUNKS_PATH, "Run: python src/preprocess.py")
    require_file(EMBEDDINGS_PATH, "Run: python src/build_sbert_index.py")
    require_file(SBERT_CONFIG_PATH, "Run: python src/build_sbert_index.py")

    docs = load_jsonl(CORPUS_PATH)
    chunks = load_jsonl(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Corpus documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Documents by section: {dict(Counter(doc.get('section') for doc in docs))}")
    print(f"Documents by source type: {dict(Counter(doc.get('source_type') for doc in docs))}")
    print(f"Chunks by type: {dict(Counter(chunk.get('type') for chunk in chunks))}")

    if embeddings.shape[0] != len(chunks):
        raise ProjectError(
            f"Embedding/chunk mismatch: {embeddings.shape[0]} embeddings for {len(chunks)} chunks. "
            "Run: python src/build_sbert_index.py"
        )

    if (QA_FINETUNED_DIR / "config.json").exists():
        print(f"Fine-tuned QA model: available at {QA_FINETUNED_DIR}")
    else:
        print("Fine-tuned QA model: not found. qa_extractive.py will use the default SQuAD model.")

    print("Status: OK")


def build_retriever():
    from retriever_hybrid import HybridRetriever, load_chunks, load_model_name

    require_file(CHUNKS_PATH, "Run: python src/preprocess.py")
    require_file(EMBEDDINGS_PATH, "Run: python src/build_sbert_index.py")
    require_file(SBERT_CONFIG_PATH, "Run: python src/build_sbert_index.py")

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    model_name = load_model_name()

    print(f"Loading retriever model: {model_name}")
    return HybridRetriever(chunks, embeddings, model_name)


def print_retrieval_results(results: Iterable[dict], max_chars: int = 800) -> None:
    for result in results:
        print("=" * 100)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        if "bm25_score" in result:
            print(f"BM25 score: {result['bm25_score']:.4f}")
        if "sbert_score" in result:
            print(f"SBERT score: {result['sbert_score']:.4f}")
        print(f"Title: {result['title']}")
        print(f"Type: {result['type']}")
        if result.get("artist"):
            print(f"Artist: {result['artist']}")
        print(f"Source: {result['source']}")
        print(f"URL: {result['url']}")
        print()
        print(result["text"][:max_chars])


def command_search(args: argparse.Namespace) -> None:
    retriever = build_retriever()
    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        unique_docs=args.unique_docs,
        filter_type=args.filter_type,
    )
    print_retrieval_results(results)


def command_ask(args: argparse.Namespace) -> None:
    retriever = build_retriever()

    if args.mode in {"extractive", "both"}:
        import qa_extractive

        print_header("Extractive QA")
        qa_model_name = qa_extractive.get_qa_model_name()
        print(f"Loading QA model: {qa_model_name}")
        qa_model = qa_extractive.ExtractiveQAModel(qa_model_name)
        answers = qa_extractive.answer_question(
            question=args.question,
            retriever=retriever,
            qa_model=qa_model,
            top_k_retrieval=args.top_k_retrieval,
            top_k_answers=args.top_k_answers,
            alpha=args.alpha,
            filter_type=args.filter_type,
            unique_docs=args.unique_docs,
        )
        qa_extractive.print_answers(args.question, answers)

    if args.mode in {"abstractive", "both"}:
        import qa_abstractive

        print_header("Abstractive QA")
        generator = qa_abstractive.AbstractiveQAModel(qa_abstractive.GEN_MODEL_NAME)
        result = qa_abstractive.answer_question(
            question=args.question,
            retriever=retriever,
            generator=generator,
            top_k_retrieval=args.top_k_retrieval,
            alpha=args.alpha,
            filter_type=args.filter_type,
            unique_docs=args.unique_docs,
        )
        qa_abstractive.print_result(result, show_prompt=args.show_prompt)


def command_demo(args: argparse.Namespace) -> None:
    """Runs a small end-to-end demonstration without rebuilding the data."""
    check_project()
    retriever = build_retriever()

    import qa_extractive
    import qa_abstractive

    print_header("Loading QA models")
    qa_model_name = qa_extractive.get_qa_model_name()
    print(f"Loading extractive QA model: {qa_model_name}")
    qa_model = qa_extractive.ExtractiveQAModel(qa_model_name)

    print(f"Loading abstractive QA model: {qa_abstractive.GEN_MODEL_NAME}")
    generator = qa_abstractive.AbstractiveQAModel(qa_abstractive.GEN_MODEL_NAME)

    print_header("Extractive QA — factual examples")
    for question, filter_type in DEMO_FACTUAL_QUESTIONS:
        answers = qa_extractive.answer_question(
            question=question,
            retriever=retriever,
            qa_model=qa_model,
            top_k_retrieval=args.top_k_retrieval,
            top_k_answers=3,
            alpha=args.alpha,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_extractive.print_answers(question, answers)

    print_header("Abstractive QA — explanatory examples")
    for question, filter_type in DEMO_ABSTRACTIVE_QUESTIONS:
        result = qa_abstractive.answer_question(
            question=question,
            retriever=retriever,
            generator=generator,
            top_k_retrieval=args.top_k_retrieval,
            alpha=args.alpha,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_abstractive.print_result(result, show_prompt=args.show_prompt)

    print_header("Both modes — comparison example")
    for question, filter_type in DEMO_BOTH_QUESTIONS:
        answers = qa_extractive.answer_question(
            question=question,
            retriever=retriever,
            qa_model=qa_model,
            top_k_retrieval=args.top_k_retrieval,
            top_k_answers=1,
            alpha=args.alpha,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_extractive.print_answers(question, answers)

        result = qa_abstractive.answer_question(
            question=question,
            retriever=retriever,
            generator=generator,
            top_k_retrieval=args.top_k_retrieval,
            alpha=args.alpha,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_abstractive.print_result(result, show_prompt=args.show_prompt)


def command_rebuild_index(args: argparse.Namespace) -> None:
    """Recreates chunks and SBERT embeddings from data/raw/corpus.jsonl."""
    require_file(CORPUS_PATH, "Run the corpus construction/repair scripts or restore data/raw/corpus.jsonl.")

    print_header("Rebuilding processed chunks")
    run_command([sys.executable, "src/preprocess.py"])

    print_header("Rebuilding SBERT index")
    run_command([sys.executable, "src/build_sbert_index.py"])

    print_header("Final status")
    check_project()

def command_demo_full(args: argparse.Namespace) -> None:
    """Demo completo para apresentação: QA textual + visualizações + instrução app."""
    check_project()
    retriever = build_retriever()

    import qa_extractive
    import qa_abstractive

    print_header("Loading QA models")
    qa_model_name = qa_extractive.get_qa_model_name()
    print(f"Loading extractive QA model: {qa_model_name}")
    qa_model = qa_extractive.ExtractiveQAModel(qa_model_name)

    print(f"Loading abstractive QA model: {qa_abstractive.GEN_MODEL_NAME}")
    generator = qa_abstractive.AbstractiveQAModel(qa_abstractive.GEN_MODEL_NAME)

    # --- Perguntas factuais seguras para o módulo extrativo ---
    print_header("Extractive QA — factual questions")
    for question, filter_type in DEMO_FACTUAL_QUESTIONS:
        answers = qa_extractive.answer_question(
            question=question,
            retriever=retriever,
            qa_model=qa_model,
            top_k_retrieval=5,
            top_k_answers=3,
            alpha=0.6,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_extractive.print_answers(question, answers)

    # --- Perguntas explicativas/definicionais para o módulo abstrativo ---
    print_header("Abstractive QA — explanatory questions")
    for question, filter_type in DEMO_ABSTRACTIVE_QUESTIONS:
        result = qa_abstractive.answer_question(
            question=question,
            retriever=retriever,
            generator=generator,
            top_k_retrieval=5,
            alpha=0.6,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_abstractive.print_result(result, show_prompt=False)

    # --- Comparação entre os dois modos ---
    print_header("Both modes — comparison")
    for question, filter_type in DEMO_BOTH_QUESTIONS:
        print_header(f"Extractive answer — {question}")
        answers = qa_extractive.answer_question(
            question=question,
            retriever=retriever,
            qa_model=qa_model,
            top_k_retrieval=5,
            top_k_answers=1,
            alpha=0.6,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_extractive.print_answers(question, answers)

        print_header(f"Abstractive answer — {question}")
        result = qa_abstractive.answer_question(
            question=question,
            retriever=retriever,
            generator=generator,
            top_k_retrieval=5,
            alpha=0.6,
            filter_type=filter_type,
            unique_docs=False,
        )
        qa_abstractive.print_result(result, show_prompt=False)

    # --- Visualização das métricas ---
    print_header("Visualisation — evaluation metrics")
    run_command([
        sys.executable, str(SRC_DIR / "visualize_eval_metrics.py"),
    ])
    print("→ Chart saved: data/eval_metrics.png")
    print("→ Table saved: data/eval_metrics_table.md")

    # --- Visualização dos embeddings ---
    print_header("Visualisation — SBERT embedding space")
    run_command([
        sys.executable, str(SRC_DIR / "visualize_embeddings.py"),
        "--method", "tsne",
        "--output", "data/embeddings_plot.png",
    ])
    print("→ Chart saved: data/embeddings_plot.png")

    # --- Instrução final ---
    print_header("Interactive app")
    print("To launch the Gradio interface:")
    print("  python app.py            # local: http://127.0.0.1:7860")


def command_rebuild_corpus(args: argparse.Namespace) -> None:
    """
    Rebuilds the final corpus from seeds.

    This command contacts external sites and Wikipedia, so it is slower and may be affected by rate limits.
    For normal use, prefer `rebuild-index`, because the submitted zip already includes the final corpus.
    """
    print_header("Building initial corpus")
    run_command([sys.executable, "src/build_corpus.py"])

    print_header("Auditing corpus")
    run_command([sys.executable, "src/audit_corpus.py"])

    print_header("Repairing flagged documents")
    run_command([sys.executable, "src/repair_corpus.py"])

    repaired = PROJECT_ROOT / "data" / "raw" / "corpus_repaired.jsonl"
    if repaired.exists():
        backup = PROJECT_ROOT / "data" / "raw" / "corpus_before_rebuild_repair.jsonl"
        CORPUS_PATH.replace(backup)
        repaired.replace(CORPUS_PATH)
        print(f"Replaced corpus with repaired version. Backup: {backup}")

    print("Note: album and artist Wikipedia replacement is handled by repair_corpus.py above.")

    command_rebuild_index(args)


def command_evaluate(args: argparse.Namespace) -> None:
    """Runs quantitative evaluation of retrievers and QA modules on annotated queries."""
    run_command(
        [sys.executable, "src/evaluate.py"]
        + (["--no-qa"] if args.no_qa else [])
        + ["--top-k", str(args.top_k)]
        + (["--save-results", args.save_results] if args.save_results else [])
    )


def command_visualize_embeddings(args: argparse.Namespace) -> None:
    """Projects document embeddings to 2D and saves a plot."""
    run_command([
        sys.executable, "src/visualize_embeddings.py",
        "--method", args.method,
        "--label-mode", args.label_mode,
        "--output", args.output,
    ])

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TP2 SPLN: music information retrieval and question answering pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Validate files and print corpus/index statistics.")
    check_parser.set_defaults(func=lambda args: check_project())

    search_parser = subparsers.add_parser("search", help="Run the hybrid BM25 + SBERT retriever.")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--alpha", type=float, default=0.6)
    search_parser.add_argument("--unique-docs", action="store_true")
    search_parser.add_argument("--filter-type", choices=FILTER_TYPES, default=None)
    search_parser.set_defaults(func=command_search)

    ask_parser = subparsers.add_parser("ask", help="Ask a question using extractive, abstractive, or both QA modules.")
    ask_parser.add_argument("question", type=str)
    ask_parser.add_argument("--mode", choices=["extractive", "abstractive", "both"], default="both")
    ask_parser.add_argument("--top-k-retrieval", type=int, default=5)
    ask_parser.add_argument("--top-k-answers", type=int, default=5)
    ask_parser.add_argument("--alpha", type=float, default=0.6)
    ask_parser.add_argument("--unique-docs", action="store_true")
    ask_parser.add_argument("--show-prompt", action="store_true")
    ask_parser.add_argument("--filter-type", choices=FILTER_TYPES, default=None)
    ask_parser.set_defaults(func=command_ask)

    demo_parser = subparsers.add_parser("demo", help="Run an end-to-end demo with representative questions.")
    demo_parser.add_argument("--top-k-retrieval", type=int, default=5)
    demo_parser.add_argument("--alpha", type=float, default=0.6)
    demo_parser.add_argument("--show-prompt", action="store_true")
    demo_parser.set_defaults(func=command_demo)

    demo_full_parser = subparsers.add_parser(
            "demo-full",
            help="Full presentation demo: QA pipeline + retrieval chart + embedding plot + app instructions.",
        )
    demo_full_parser.set_defaults(func=command_demo_full)

    rebuild_index_parser = subparsers.add_parser(
        "rebuild-index",
        help="Recreate processed chunks and SBERT embeddings from the final corpus.",
    )
    rebuild_index_parser.set_defaults(func=command_rebuild_index)

    rebuild_corpus_parser = subparsers.add_parser(
        "rebuild-corpus",
        help="Rebuild the corpus from seeds, repair it, and rebuild the index. Slow and network-dependent.",
    )
    rebuild_corpus_parser.set_defaults(func=command_rebuild_corpus)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate retrievers and QA modules on annotated queries (data/eval_queries.json).",
    )
    eval_parser.add_argument("--no-qa", action="store_true", help="Skip QA evaluation (faster).")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--save-results", type=str, default=None, metavar="PATH")
    eval_parser.set_defaults(func=command_evaluate)

    viz_emb_parser = subparsers.add_parser(
        "visualize-embeddings",
        help="Project document embeddings to 2D (t-SNE or UMAP) and save a plot.",
    )
    viz_emb_parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    viz_emb_parser.add_argument("--output", type=str, default="data/embeddings_plot.png")
    viz_emb_parser.add_argument("--label-mode", choices=["all", "key", "none"], default="key")
    viz_emb_parser.set_defaults(func=command_visualize_embeddings)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except ProjectError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()