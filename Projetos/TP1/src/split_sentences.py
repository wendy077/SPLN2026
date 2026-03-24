import re
import json
from pathlib import Path

import spacy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

MIN_WORDS = 5


def load_sources(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.split()).strip()


def is_valid_sentence(sentence: str) -> bool:
    words = sentence.split()
    lower = sentence.lower().strip()

    if len(words) < MIN_WORDS:
        return False

    # headings/títulos curtos
    if len(words) <= 8 and sentence.istitle():
        return False

    # frases demasiado curtas para scoring útil
    if len(words) < 6:
        return False

    # remover metadados/títulos conhecidos
    bad_prefixes = (
        "terrae novae 2030+ strategy roadmap",
        "josef aschbacher director general",
        "exploring the solar system",
        "about terrae novae",
        "foreword",
    )
    if any(lower.startswith(prefix) for prefix in bad_prefixes):
        return False

    # remover linhas com muitos blocos de metadados
    if "director general european space agency" in lower:
        return False

    # remove linhas muito parecidas com títulos/metadados
    bad_starts = (
        "why we go to space",
        "reaching beyond earth orbit",
        "on to mars",
        "about terrae novae",
        "foreword",
        "josef aschbacher",
        "exploring the solar system",
        "preamble",
        "introduction",
        "conclusion",
    )
    if any(lower.startswith(prefix) for prefix in bad_starts) and len(words) < 12:
        return False

    # remove frases com demasiados números/enumeração
    if re.search(r"\b\d+\.\s", sentence) and len(words) < 12:
        return False

    # remove lixo visual corrompido
    if re.search(r"(?:\b[A-Za-z]\b\s+){6,}", sentence):
        return False

    # remove frases com percentagem alta de tokens muito curtos
    short_tokens = sum(1 for w in words if len(w) <= 2)
    if len(words) > 0 and short_tokens / len(words) > 0.45:
        return False

    return True

def clean_sentence_artifacts(sentence: str) -> str:
    replacements = {
        "Why We Go to Space At NASA,": "At NASA,",
        "Reaching beyond Earth orbit While": "While",
        "On to Mars Supported": "Supported",
        "Foreword – by Josef Aschbacher, Director General At the beginning": "At the beginning",
        "2 Introduction In 2014,": "In 2014,",
        "Exploring the Solar System – by the Human spaceflight and Exploration Science Advisory Committee 1 Preamble As recent events have shown,": "As recent events have shown,",
    }

    for old, new in replacements.items():
        sentence = sentence.replace(old, new)

    return sentence.strip()

def split_sentences(text: str, nlp) -> list[str]:
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        s = normalize_sentence(sent.text)
        s = clean_sentence_artifacts(s)
        if is_valid_sentence(s):
            sentences.append(s)

    return sentences


def process_source(source: dict, nlp) -> None:
    source_id = source["id"]
    input_path = CLEAN_DIR / f"{source_id}.txt"
    output_path = CLEAN_DIR / f"{source_id}_sentences.json"

    text = read_text(input_path)
    sentences = split_sentences(text, nlp)

    write_json(output_path, sentences)
    print(f"[OK] {source_id}: {len(sentences)} frases -> {output_path}")


def main():
    sources = load_sources(SOURCES_FILE)
    nlp = spacy.load("en_core_web_lg")

    for source in sources:
        process_source(source, nlp)


if __name__ == "__main__":
    main()