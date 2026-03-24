import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def load_sources(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_basic(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("￾", "")
    text = text.replace("%", "")
    # remover caracteres unicode fora do range Latin que o LaTeX não suporta
    text = re.sub(r"[^\x00-\x7F\u00C0-\u024F\u2019\u2018\u201C\u201D\u2013\u2014]", "", text)
    # remover referências bibliográficas da Wikipedia do tipo [1], [30], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # remover marcadores da Wikipedia do tipo [update], [note 1], [note 2], etc.
    text = re.sub(r'\[update\]', '', text)
    text = re.sub(r'\[note\s*\d*\]', '', text)
    return text

def remove_common_noise(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # cabeçalhos e rodapés comuns do PDF
        if stripped == "ESA UNCLASSIFIED – Releasable to the Public":
            continue
        if re.fullmatch(r"Page \d+/\d+", stripped):
            continue

        # ruído comum
        if stripped.startswith("Figure "):
            continue
        if stripped == "Table of Contents":
            continue
        if stripped == "Contact":
            continue
        if stripped == "© European Space Agency":
            continue
        if stripped == "explorationstrategy@esa.int":
            continue

        # remover linhas de lista da Wikipedia que começam com "- "
        # (entradas de tabelas, listas de programas espaciais, etc.)
        if re.match(r'^-\s+\S', stripped):
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def remove_table_of_contents_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    toc_pattern = re.compile(r".*\.{5,}\s*\d+\s*$")

    for line in lines:
        stripped = line.strip()

        # linhas típicas do índice
        if toc_pattern.match(stripped):
            continue
        if re.fullmatch(r"\d+(\.\d+)*\s+.+", stripped) and "strategy" in stripped.lower():
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def fix_hyphenation(text: str) -> str:
    # junta palavras partidas por mudança de linha: explora-\ntion -> exploration
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text


def join_broken_lines(text: str) -> str:
    lines = text.split("\n")
    result = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            result.append("")
            continue

        if not result:
            result.append(stripped)
            continue

        prev = result[-1]

        # se a linha anterior já terminou frase/estrutura, mantém quebra
        if prev.endswith((".", "!", "?", ":", ";")):
            result.append(stripped)
            continue

        # listas com bullet
        if stripped.startswith(("•", "-", "*")):
            result.append(stripped)
            continue

        # títulos curtos ou secções
        if len(stripped.split()) <= 5 and stripped.istitle():
            result.append(stripped)
            continue

        # caso normal: junta à linha anterior
        result[-1] = prev + " " + stripped

    return "\n".join(result)

def fix_joined_headings(text: str) -> str:
    replacements = {
        "Why We Go to Space At NASA,": "Why We Go to Space\n\nAt NASA,",
        "Reaching beyond Earth orbit While": "Reaching beyond Earth orbit\n\nWhile",
        "On to Mars Supported": "On to Mars\n\nSupported",
        "About Terrae Novae The mission": "About Terrae Novae\n\nThe mission",
        "Foreword – by Josef Aschbacher, Director General At the beginning": (
            "Foreword – by Josef Aschbacher, Director General\n\nAt the beginning"
        ),
        "Exploring the Solar System – by the Human spaceflight and Exploration Science Advisory Committee (HESAC) The Human Spaceflight and Exploration Science Advisory Committee (HESAC)": (
            "Exploring the Solar System – by the Human spaceflight and Exploration Science Advisory Committee (HESAC)\n\n"
            "The Human Spaceflight and Exploration Science Advisory Committee (HESAC)"
        ),
        "1 Preamble As recent events have shown,": "1 Preamble\n\nAs recent events have shown,",
        "2 Introduction In 2014,": "2 Introduction\n\nIn 2014,",
        "6 Conclusion In essence,": "6 Conclusion\n\nIn essence,",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_corrupted_pdf_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        # remove URLs isolados
        if stripped.startswith("http://") or stripped.startswith("https://"):
            continue
        if stripped.startswith("(https://"):
            continue

        # remove referências numeradas soltas do PDF
        if re.match(r"^\d+\s+ESA/", stripped):
            continue
        if re.match(r"^\d+\s+EL3", stripped):
            continue

        # remove linhas com demasiadas letras soltas separadas por espaços
        if re.search(r"(?:\b[A-Za-z]\b\s+){8,}", stripped):
            continue

        # remove linhas com aspeto de gráfico/legenda corrompida
        alpha_chars = sum(c.isalpha() for c in stripped)
        spaces = stripped.count(" ")
        if alpha_chars > 20 and spaces > alpha_chars * 0.7:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

def collapse_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def clean_text(text: str) -> str:
    text = normalize_basic(text)
    text = remove_common_noise(text)
    text = remove_table_of_contents_lines(text)
    text = fix_hyphenation(text)
    text = join_broken_lines(text)
    text = fix_joined_headings(text)
    text = remove_corrupted_pdf_lines(text)
    text = collapse_blank_lines(text)
    return text


def process_source(source: dict) -> None:
    source_id = source["id"]
    raw_path = RAW_DIR / f"{source_id}.txt"
    clean_path = CLEAN_DIR / f"{source_id}.txt"

    text = read_text(raw_path)
    cleaned = clean_text(text)
    write_text(clean_path, cleaned)

    print(f"[OK] Limpo: {clean_path}")


def main():
    sources = load_sources(SOURCES_FILE)
    for source in sources:
        process_source(source)


if __name__ == "__main__":
    main()