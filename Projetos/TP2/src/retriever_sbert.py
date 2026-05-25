import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"
CONFIG_PATH = PROJECT_ROOT / "models" / "sbert_config.json"


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model_name() -> str:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config["model_name"]


class SBERTRetriever:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, model_name: str):
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = SentenceTransformer(model_name)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                f"Mismatch: {len(self.chunks)} chunks but {len(self.embeddings)} embeddings"
            )

    def search(self, query: str, top_k: int = 5, unique_docs: bool = False) -> list[dict]:
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Como os embeddings estão normalizados, dot product = cosine similarity.
        scores = self.embeddings @ query_embedding

        candidate_indices = np.argsort(scores)[::-1]

        results = []
        seen_docs = set()

        for idx in candidate_indices:
            chunk = self.chunks[idx]

            if unique_docs and chunk["doc_id"] in seen_docs:
                continue

            seen_docs.add(chunk["doc_id"])

            results.append({
                "rank": len(results) + 1,
                "score": float(scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "type": chunk["type"],
                "source": chunk["source"],
                "url": chunk["url"],
                "text": chunk["text"],
            })

            if len(results) >= top_k:
                break

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--unique-docs", action="store_true")
    args = parser.parse_args()

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDINGS_PATH}\n"
            f"Run first: python src/build_sbert_index.py"
        )

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    model_name = load_model_name()

    retriever = SBERTRetriever(chunks, embeddings, model_name)
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        unique_docs=args.unique_docs,
    )

    for result in results:
        print("=" * 100)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Title: {result['title']}")
        print(f"Type: {result['type']}")
        print(f"Source: {result['source']}")
        print(f"URL: {result['url']}")
        print()
        print(result["text"][:900])


if __name__ == "__main__":
    main()