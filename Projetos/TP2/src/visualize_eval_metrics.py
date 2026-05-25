from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "eval_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval_metrics.png"
DEFAULT_TABLE = PROJECT_ROOT / "data" / "eval_metrics_table.md"


def main() -> None:
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description="Plot metrics from data/eval_results.json.")
    parser.add_argument("--input", default=str(RESULTS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Missing {input_path}. Run: "
            "python run_project.py evaluate --save-results data/eval_results.json"
        )

    results = json.loads(input_path.read_text(encoding="utf-8"))
    retrievers = [name for name in ["bm25", "sbert", "hybrid"] if name in results]

    # For the plot and table, use the most intuitive metrics for presentation.
    # P@3 was removed because most queries have only one relevant document annotated,
    # which makes Hit@3 more informative for this setup.
    metric_specs = [
        ("mrr", "MRR"),
        ("precision_at_1", "P@1"),
        ("hit_at_3", "Hit@3"),
    ]

    x = np.arange(len(retrievers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (metric_key, metric_label) in enumerate(metric_specs):
        values = [results[name][metric_key] for name in retrievers]
        offset = (i - (len(metric_specs) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=metric_label)

    ax.set_xticks(x, [name.upper() for name in retrievers])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retriever evaluation")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Markdown table with retriever metrics.
    lines = [
        "| Retriever | MRR | P@1 | Hit@3 |",
        "|---|---:|---:|---:|",
    ]

    for name in retrievers:
        r = results[name]
        lines.append(
            f"| {name.upper()} | {r['mrr']:.4f} | "
            f"{r['precision_at_1']:.4f} | "
            f"{r['hit_at_3']:.4f} |"
        )

    if "qa" in results:
        lines += [
            "",
            "| QA module | Exact Match | F1 |",
            "|---|---:|---:|",
        ]

        for name in ["extractive", "abstractive"]:
            if name in results["qa"]:
                q = results["qa"][name]
                lines.append(
                    f"| {name} | {q['exact_match']:.2f}% | {q['f1']:.2f}% |"
                )

    table_output = Path(args.table_output)
    table_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Plot saved: {output}")
    print(f"Table saved: {table_output}")


if __name__ == "__main__":
    main()