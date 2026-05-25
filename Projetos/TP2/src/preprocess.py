import json
import re
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

MAX_WORDS = 170
OVERLAP_WORDS = 35
MIN_WORDS = 40


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")

    # Remove boilerplate comum da Britannica
    text = re.sub(
        r"While every effort has been made to follow citation style rules.*?if you have any questions\.",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove perguntas introdutórias típicas de páginas enciclopédicas
    text = re.sub(
        r"(What|How|Why|When|Where|Who)[^?]{10,160}\?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[,\s;:.-]+", "", text)
    return text.strip()

def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_words(text: str) -> list[str]:
    words = text.split()

    if len(words) <= MAX_WORDS:
        return [" ".join(words)] if len(words) >= MIN_WORDS else []

    chunks = []
    step = MAX_WORDS - OVERLAP_WORDS

    for start in range(0, len(words), step):
        chunk = words[start:start + MAX_WORDS]

        if len(chunk) < MIN_WORDS:
            break

        chunks.append(" ".join(chunk))

    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    all_chunks = []

    for doc in docs:
        text = clean_text(doc["text"])
        text_chunks = chunk_words(text)

        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = f"{doc['doc_id']}_chunk_{idx:04d}"

            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "type": doc["type"],
                "section": doc.get("section"),
                "source_type": doc.get("source_type"),
                "source": doc.get("source"),
                "url": doc.get("url"),
                "chunk_index": idx,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
            }

            if "artist" in doc:
                chunk["artist"] = doc["artist"]

            all_chunks.append(chunk)

    return all_chunks


def main() -> None:
    docs = load_jsonl(INPUT_PATH)
    chunks = build_chunks(docs)

    save_jsonl(OUTPUT_PATH, chunks)

    print("Documents:", len(docs))
    print("Chunks:", len(chunks))
    print("Chunks by type:", Counter(chunk["type"] for chunk in chunks))
    print("Chunks by section:", Counter(chunk["section"] for chunk in chunks))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()