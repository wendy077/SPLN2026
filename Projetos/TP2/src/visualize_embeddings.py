"""
Visualização do espaço de embeddings SBERT dos documentos do corpus.

Projecta os embeddings de cada documento (média dos chunks) em 2D usando t-SNE,
coloridos por tipo (artist, album, genre, festival, award, event, movement).

Exemplos de uso:
    python src/visualize_embeddings.py
    python src/visualize_embeddings.py --method umap
    python src/visualize_embeddings.py --output embeddings_plot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "embeddings_plot.png"

KEY_LABELS = {
    "Radiohead", "Nirvana", "Kendrick Lamar", "OK Computer",
    "To Pimp a Butterfly", "Grunge", "Lollapalooza",
    "Childish Gambino", "Coachella", "Grammy Awards",
}

# Cores e labels para cada tipo de documento
TYPE_STYLE = {
    "artist":   {"color": "#534AB7", "marker": "o", "label": "Artist"},
    "album":    {"color": "#1D9E75", "marker": "s", "label": "Album"},
    "genre":    {"color": "#D85A30", "marker": "^", "label": "Genre"},
    "festival": {"color": "#BA7517", "marker": "D", "label": "Festival"},
    "award":    {"color": "#185FA5", "marker": "P", "label": "Award"},
    "event":    {"color": "#993556", "marker": "*", "label": "Event"},
    "movement": {"color": "#5F5E5A", "marker": "X", "label": "Movement"},
}


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def aggregate_doc_embeddings(chunks: list[dict], embeddings: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Calcula o embedding médio por documento (média dos seus chunks)."""
    from collections import defaultdict

    doc_chunk_indices: dict[str, list[int]] = defaultdict(list)
    doc_meta: dict[str, dict] = {}

    for i, chunk in enumerate(chunks):
        doc_id = chunk["doc_id"]
        doc_chunk_indices[doc_id].append(i)
        if doc_id not in doc_meta:
            doc_meta[doc_id] = {
                "doc_id": doc_id,
                "title": chunk["title"],
                "type": chunk["type"],
            }

    doc_ids = list(doc_chunk_indices.keys())
    doc_embeddings = np.array([
        embeddings[doc_chunk_indices[doc_id]].mean(axis=0)
        for doc_id in doc_ids
    ])

    # Renormalizar após média
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    doc_embeddings = doc_embeddings / norms

    docs = [doc_meta[doc_id] for doc_id in doc_ids]
    return docs, doc_embeddings


def reduce_dimensions(embeddings: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state, metric="cosine")
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("[WARN] umap-learn not installed. Falling back to t-SNE.")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),
            random_state=random_state,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        )
        return reducer.fit_transform(embeddings)

    raise ValueError(f"Unknown method: {method!r}. Use 'tsne' or 'umap'.")


def should_label(doc: dict, label_mode: str) -> bool:
    if label_mode == "all":
        return True
    if label_mode == "none":
        return False
    return doc["title"] in KEY_LABELS


def plot(docs: list[dict], coords_2d: np.ndarray, method: str, output: Path, label_mode: str = "key") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#F8F8F5")
    ax.set_facecolor("#F8F8F5")

    types_in_data = sorted({d["type"] for d in docs})

    for doc_type in types_in_data:
        style = TYPE_STYLE.get(doc_type, {"color": "#888", "marker": "o", "label": doc_type.capitalize()})
        mask = [d["type"] == doc_type for d in docs]
        xs = coords_2d[mask, 0]
        ys = coords_2d[mask, 1]

        ax.scatter(
            xs, ys,
            c=style["color"],
            marker=style["marker"],
            label=f"{style['label']} ({mask.count(True)})",
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # Etiquetas opcionais. Para slides, usar --label-mode key ou --label-mode none.
    for i, doc in enumerate(docs):
        if not should_label(doc, label_mode):
            continue
        title = doc["title"]
        if len(title) > 18:
            title = title[:17] + "…"
        title = title.replace("$", r"\$")  
        ax.annotate(
            title,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=5.5,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    method_label = "t-SNE" if method == "tsne" else "UMAP"
    ax.set_title(
        f"Espaço de embeddings SBERT — {method_label}\n"
        f"({len(docs)} documentos, modelo all-MiniLM-L6-v2)",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel(f"{method_label} dim 1", fontsize=10)
    ax.set_ylabel(f"{method_label} dim 2", fontsize=10)
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise SBERT document embeddings in 2D.")
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne",
                        help="Dimensionality reduction method (default: tsne).")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output image path (default: data/embeddings_plot.png).")
    parser.add_argument("--label-mode", choices=["all", "key", "none"], default="key",
                        help="Labels to draw: all, key or none (default: key).")
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        print(f"ERROR: chunks not found at {CHUNKS_PATH}\nRun: python src/preprocess.py")
        sys.exit(1)
    if not EMBEDDINGS_PATH.exists():
        print(f"ERROR: embeddings not found at {EMBEDDINGS_PATH}\nRun: python src/build_sbert_index.py")
        sys.exit(1)

    print("Loading chunks and embeddings...")
    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Aggregating embeddings per document ({len(set(c['doc_id'] for c in chunks))} documents)...")
    docs, doc_embeddings = aggregate_doc_embeddings(chunks, embeddings)

    method = args.method
    print(f"Reducing to 2D with {method.upper()} ...")
    coords_2d = reduce_dimensions(doc_embeddings, method=method)

    print("Generating plot...")
    plot(docs, coords_2d, method=method, output=Path(args.output), label_mode=args.label_mode)


if __name__ == "__main__":
    main()
