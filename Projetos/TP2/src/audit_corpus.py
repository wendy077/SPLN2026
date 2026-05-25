import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
AUDIT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus_audit.csv"


def load_docs():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    docs = load_docs()

    hashes = defaultdict(list)
    for doc in docs:
        text_hash = hashlib.md5(doc["text"].encode("utf-8")).hexdigest()
        hashes[text_hash].append(doc["doc_id"])

    duplicate_hashes = {
        h for h, ids in hashes.items()
        if len(ids) > 1
    }

    rows = []

    for doc in docs:
        text = doc["text"]
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        title = doc["title"].lower()
        text_lower = text.lower()

        title_in_text = title in text_lower

        warning_flags = []

        if len(text) < 2500:
            warning_flags.append("short_text")

        if text_hash in duplicate_hashes:
            warning_flags.append("duplicate_text")

        if not title_in_text:
            warning_flags.append("title_not_in_text")

        rows.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "type": doc["type"],
            "section": doc["section"],
            "source_type": doc["source_type"],
            "source": doc["source"],
            "url": doc["url"],
            "text_chars": len(text),
            "title_in_text": title_in_text,
            "warnings": ", ".join(warning_flags),
        })

    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_PATH, index=False)

    print("Total documents:", len(docs))
    print("By type:", Counter(doc["type"] for doc in docs))
    print("By section:", Counter(doc["section"] for doc in docs))
    print("By source type:", Counter(doc["source_type"] for doc in docs))
    print("Top sources:", Counter(doc["source"] for doc in docs).most_common(15))
    print()
    print("Documents with warnings:", (df["warnings"] != "").sum())
    print("Audit saved to:", AUDIT_PATH)

    print()
    print(df[df["warnings"] != ""][[
        "title", "source", "text_chars", "warnings", "url"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()