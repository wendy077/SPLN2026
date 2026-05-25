"""
Interface web interativa para o sistema de QA musical (TP2 SPLN).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"
SBERT_CONFIG_PATH = PROJECT_ROOT / "models" / "sbert_config.json"
EVAL_QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries.json"

FILTER_TYPES = ["(all types)", "artist", "album", "genre", "festival", "award", "event", "movement"]
QA_MODES = ["Extractive", "Abstractive", "Both"]
LOW_CONFIDENCE_SCORE_THRESHOLD = 0.35
LOW_CONFIDENCE_MARGIN_THRESHOLD = 0.04


def load_example_questions() -> list[list[object]]:
    """Demo examples first, then additional evaluation examples."""
    demo_examples = [
        [
            "Who released Blonde?",
            "album",
            0.6,
            5,
            "Extractive",
        ],
        [
            "What genre is Tame Impala associated with?",
            "artist",
            0.6,
            5,
            "Both",
        ],
        [
            "Explain why 2014 Forest Hills Drive is considered important.",
            "album",
            0.6,
            5,
            "Abstractive",
        ],
    ]

    if not EVAL_QUERIES_PATH.exists():
        return demo_examples + [
            ["Who released OK Computer?", "album", 0.6, 5, "Extractive"],
            ["Who is the lead singer of Radiohead?", "artist", 0.6, 5, "Both"],
            ["What genre is Nirvana associated with?", "genre", 0.6, 5, "Both"],
            ["Explain why To Pimp a Butterfly is considered important.", "album", 0.6, 5, "Abstractive"],
        ]

    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    seen = {example[0] for example in demo_examples}
    extra_examples = []

    for q in queries:
        question = q["question"]
        if question in seen:
            continue

        if question.lower().startswith(("explain", "why", "what is")):
            mode = "Abstractive"
        else:
            mode = "Both"

        extra_examples.append([
            question,
            q.get("filter_type") or "(all types)",
            0.6,
            5,
            mode,
        ])

    return demo_examples + extra_examples


EXAMPLE_QUESTIONS = load_example_questions()


def md_link(title: str, url: str | None) -> str:
    """Render a markdown link when a URL is available."""
    if not url:
        return title
    safe_title = title.replace("[", "\\[").replace("]", "\\]")
    return f"[{safe_title}]({url})"


def confidence_warning(retrieved: list[dict]) -> str:
    """Warn when the top result is weak or too close to the second result."""
    if not retrieved:
        return "⚠️ **No documents retrieved.**\n\n"

    top_score = retrieved[0]["score"]
    margin = top_score - retrieved[1]["score"] if len(retrieved) > 1 else 1.0

    if top_score < LOW_CONFIDENCE_SCORE_THRESHOLD:
        return "⚠️ **Low confidence retrieval:** the top document score is low.\n\n"
    if margin < LOW_CONFIDENCE_MARGIN_THRESHOLD:
        return "⚠️ **Ambiguous retrieval:** the first two documents have very similar scores.\n\n"
    return ""


def format_retrieved_docs(retrieved: list[dict]) -> str:
    if not retrieved:
        return "No documents retrieved."

    docs_lines = [confidence_warning(retrieved)]
    for r in retrieved:
        snippet = r["text"][:300].replace("\n", " ")
        title = md_link(r["title"], r.get("url"))
        docs_lines.append(
            f"**#{r['rank']} {title}** ({r['type']})\n"
            f"Final score: `{r['score']:.4f}` | "
            f"BM25 raw: `{r.get('bm25_score', 0.0):.4f}` | "
            f"SBERT cosine: `{r.get('sbert_score', 0.0):.4f}`\n"
            f"Source: {r.get('source', 'unknown')}\n\n"
            f"> {snippet}…\n"
        )
    return "\n".join(docs_lines)


def load_retriever():
    from retriever_hybrid import HybridRetriever

    def load_chunks(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(SBERT_CONFIG_PATH, encoding="utf-8") as f:
        model_name = json.load(f)["model_name"]

    return HybridRetriever(chunks, embeddings, model_name)


def load_qa_models():
    import qa_extractive
    import qa_abstractive

    ext_model_name = qa_extractive.get_qa_model_name()
    ext_model = qa_extractive.ExtractiveQAModel(ext_model_name)

    gen_model = qa_abstractive.AbstractiveQAModel(qa_abstractive.GEN_MODEL_NAME)
    return ext_model, gen_model


print("Loading retriever and QA models — this may take a moment...")
_retriever = load_retriever()
_ext_model, _gen_model = load_qa_models()
print("Models ready.")


def answer(question: str, filter_type_str: str, alpha: float, top_k: int, mode: str) -> tuple[str, str, str]:
    """Return extractive answer, abstractive answer, and retrieved documents markdown."""
    import qa_extractive
    import qa_abstractive

    if not question.strip():
        return "Please enter a question.", "", ""

    filter_type = None if filter_type_str == "(all types)" else filter_type_str
    top_k = int(top_k)

    if mode in {"Extractive", "Both"}:
        try:
            ext_answers = qa_extractive.answer_question(
                question=question,
                retriever=_retriever,
                qa_model=_ext_model,
                top_k_retrieval=top_k,
                top_k_answers=3,
                alpha=alpha,
                filter_type=filter_type,
                unique_docs=True,
            )
            if ext_answers:
                best = ext_answers[0]
                source_title = md_link(best["title"], best.get("url"))
                ext_text = (
                    f"**{best['answer']}**\n\n"
                    f"Source: *{source_title}* ({best['type']}) — {best['source']}\n"
                    f"Method: `{best.get('answer_method', 'model')}` | "
                    f"Score: {best['final_score']:.3f}"
                )
                if len(ext_answers) > 1:
                    others = ", ".join(f"*{a['answer']}*" for a in ext_answers[1:])
                    ext_text += f"\n\nOther candidates: {others}"
            else:
                ext_text = "No answer found."
        except Exception as exc:
            ext_text = f"Error: {exc}"
    else:
        ext_text = "_Not executed. Select Extractive or Both to run this module._"

    if mode in {"Abstractive", "Both"}:
        try:
            abs_result = qa_abstractive.answer_question(
                question=question,
                retriever=_retriever,
                generator=_gen_model,
                top_k_retrieval=top_k,
                alpha=alpha,
                filter_type=filter_type,
                unique_docs=True,
            )
            abs_text = (
                f"{abs_result['answer']}\n\n"
                f"Method: `{abs_result.get('answer_method', 'generative')}`"
            )
        except Exception as exc:
            abs_text = f"Error: {exc}"
    else:
        abs_text = "_Not executed. Select Abstractive or Both to run this module._"

    retrieved = _retriever.search(
        query=question,
        top_k=top_k,
        alpha=alpha,
        unique_docs=True,
        filter_type=filter_type,
    )
    docs_text = format_retrieved_docs(retrieved)

    return ext_text, abs_text, docs_text


def build_interface() -> object:
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio is not installed. Run: pip install gradio")
        sys.exit(1)

    with gr.Blocks(title="Music QA — TP2 SPLN") as demo:
        gr.Markdown(
            "# 🎵 Music Information Retrieval & Question Answering\n"
            "**TP2 — Scripting e Processamento de Linguagem Natural**\n\n"
            "Corpus: 110 documentos sobre artistas, álbuns, géneros e eventos musicais.\n"
            "Retriever: BM25 + SBERT (híbrido). QA: extrativo (DistilBERT fine-tuned em SQuAD) + abstrativo (FLAN-T5)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_box = gr.Textbox(
                    label="Pergunta / Question",
                    placeholder="Who released OK Computer?",
                    lines=2,
                )
                with gr.Row():
                    filter_type_dd = gr.Dropdown(
                        choices=FILTER_TYPES,
                        value="(all types)",
                        label="Filter by type",
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="Semantic weight (α)",
                        info="0 = only BM25, 1 = only SBERT",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top-k documents",
                    )

                mode_radio = gr.Radio(
                    choices=QA_MODES,
                    value="Both",
                    label="QA mode",
                )

                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("### Extractive QA")
                ext_output = gr.Markdown(label="Extractive answer")
                gr.Markdown("---\n### Abstractive QA")
                abs_output = gr.Markdown(label="Abstractive answer")
                gr.Markdown("---\n### Retrieved documents")
                docs_output = gr.Markdown(label="Retrieved context")

        gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=[question_box, filter_type_dd, alpha_slider, top_k_slider, mode_radio],
            label="Example questions",
        )

        submit_btn.click(
            fn=answer,
            inputs=[question_box, filter_type_dd, alpha_slider, top_k_slider, mode_radio],
            outputs=[ext_output, abs_output, docs_output],
        )
        question_box.submit(
            fn=answer,
            inputs=[question_box, filter_type_dd, alpha_slider, top_k_slider, mode_radio],
            outputs=[ext_output, abs_output, docs_output],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Gradio interface for the TP2 QA pipeline.")
    parser.add_argument("--port", type=int, default=7860, help="Local port (default: 7860).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_interface()
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()