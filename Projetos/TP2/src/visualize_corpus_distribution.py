from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "corpus_distribution.png"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Plot corpus distribution by document type.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    docs = load_jsonl(CORPUS_PATH)
    counts = Counter(d.get("type", "unknown") for d in docs)
    labels = sorted(counts, key=lambda k: (-counts[k], k))
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_title("Corpus distribution by document type")
    ax.set_ylabel("Number of documents")
    ax.set_xlabel("Document type")
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output}")


if __name__ == "__main__":
    main()
