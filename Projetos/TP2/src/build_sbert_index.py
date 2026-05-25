import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
MODELS_DIR = PROJECT_ROOT / "models"

EMBEDDINGS_PATH = MODELS_DIR / "sbert_chunk_embeddings.npy"
CONFIG_PATH = MODELS_DIR / "sbert_config.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def text_for_embedding(chunk: dict) -> str:
    # Incluir o título ajuda muito o SBERT a perceber o contexto do chunk.
    return f"{chunk['title']} ({chunk['type']}). {chunk['text']}"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(CHUNKS_PATH)
    texts = [text_for_embedding(chunk) for chunk in chunks]

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    config = {
        "model_name": MODEL_NAME,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "chunks_path": "data/processed/chunks.jsonl",
        "embeddings_path": "models/sbert_chunk_embeddings.npy",
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Saved embeddings:", EMBEDDINGS_PATH)
    print("Saved config:", CONFIG_PATH)
    print("Shape:", embeddings.shape)


if __name__ == "__main__":
    main()