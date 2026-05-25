import argparse
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"
CONFIG_PATH = PROJECT_ROOT / "models" / "sbert_config.json"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model_name() -> str:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config["model_name"]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score == min_score:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


class HybridRetriever:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, model_name: str):
        self.chunks = chunks
        self.embeddings = embeddings

        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.model = SentenceTransformer(model_name)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                f"Mismatch: {len(self.chunks)} chunks but {len(self.embeddings)} embeddings"
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.6,
        unique_docs: bool = False,
        filter_type: str | None = None,
    ) -> list[dict]:
        """
        alpha controla o peso semântico:
        alpha = 0.0 -> só BM25
        alpha = 1.0 -> só SBERT
        alpha = 0.6 -> 60% SBERT, 40% BM25
        """

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        sbert_scores = self.embeddings @ query_embedding

        bm25_norm = normalize_scores(bm25_scores)
        sbert_norm = normalize_scores(sbert_scores)

        final_scores = (1 - alpha) * bm25_norm + alpha * sbert_norm

        candidate_indices = np.argsort(final_scores)[::-1]

        results = []
        seen_docs = set()

        for idx in candidate_indices:
            chunk = self.chunks[idx]

            if filter_type and chunk["type"] != filter_type:
                continue

            if unique_docs and chunk["doc_id"] in seen_docs:
                continue

            seen_docs.add(chunk["doc_id"])

            results.append({
                "rank": len(results) + 1,
                "score": float(final_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "sbert_score": float(sbert_scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "type": chunk["type"],
                "artist": chunk.get("artist"),
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
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--unique-docs", action="store_true")
    parser.add_argument(
        "--filter-type",
        type=str,
        default=None,
        choices=["artist", "album", "genre", "festival", "award", "event", "movement"],
    )

    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    model_name = load_model_name()

    retriever = HybridRetriever(chunks, embeddings, model_name)

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        unique_docs=args.unique_docs,
        filter_type=args.filter_type,
    )

    for result in results:
        print("=" * 100)
        print(f"Rank: {result['rank']}")
        print(f"Final score: {result['score']:.4f}")
        print(f"BM25 score: {result['bm25_score']:.4f}")
        print(f"SBERT score: {result['sbert_score']:.4f}")
        print(f"Title: {result['title']}")
        print(f"Type: {result['type']}")
        print(f"Source: {result['source']}")
        print(f"URL: {result['url']}")
        print()
        print(result["text"][:900])


if __name__ == "__main__":
    main()