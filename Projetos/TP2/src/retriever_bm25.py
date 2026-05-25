import argparse
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "type": chunk["type"],
                "source": chunk["source"],
                "url": chunk["url"],
                "text": chunk["text"],
            })

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    retriever = BM25Retriever(chunks)

    results = retriever.search(args.query, top_k=args.top_k)

    for result in results:
        print("=" * 100)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Title: {result['title']}")
        print(f"Type: {result['type']}")
        print(f"Source: {result['source']}")
        print(f"URL: {result['url']}")
        print()
        print(result["text"][:800])


if __name__ == "__main__":
    main()