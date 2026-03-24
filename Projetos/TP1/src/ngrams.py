import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def load_sources(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+", text)
    return tokens


def build_bigrams(tokens: list[str]) -> list[tuple]:
    return list(zip(tokens, tokens[1:]))


# função para construir trigramas
def build_trigrams(tokens: list[str]) -> list[tuple]:
    return list(zip(tokens, tokens[1:], tokens[2:]))


def process_source(source: dict) -> None:
    source_id = source["id"]
    input_path = CLEAN_DIR / f"{source_id}_sentences.json"

    # ficheiros de saída separados para bigramas e trigramas
    bigrams_output_path = CLEAN_DIR / f"{source_id}_bigrams.json"
    trigrams_output_path = CLEAN_DIR / f"{source_id}_trigrams.json"

    sentences = read_json(input_path)

    bigram_counter = Counter()
    trigram_counter = Counter()

    for sentence in sentences:
        tokens = tokenize(sentence)
        bigram_counter.update(build_bigrams(tokens))
        trigram_counter.update(build_trigrams(tokens))

    # guardar bigramas (compatibilidade com select_sentences.py)
    bigrams_serializable = [
        {"bigram": list(bigram), "freq": freq}
        for bigram, freq in bigram_counter.most_common()
    ]
    write_json(bigrams_output_path, bigrams_serializable)
    print(f"[OK] {source_id}: {len(bigrams_serializable)} bigramas -> {bigrams_output_path}")

    # guardar trigramas
    trigrams_serializable = [
        {"trigram": list(trigram), "freq": freq}
        for trigram, freq in trigram_counter.most_common()
    ]
    write_json(trigrams_output_path, trigrams_serializable)
    print(f"[OK] {source_id}: {len(trigrams_serializable)} trigramas -> {trigrams_output_path}")


def main():
    sources = load_sources(SOURCES_FILE)
    for source in sources:
        process_source(source)


if __name__ == "__main__":
    main()