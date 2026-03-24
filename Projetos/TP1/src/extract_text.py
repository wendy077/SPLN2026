import json
import sys
from pathlib import Path

import pdfplumber
import requests
import trafilatura


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_sources(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro de fontes não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("O sources.json deve conter uma lista de fontes.")

    return data


def extract_web_text(url: str) -> str:
    try:
        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Erro ao obter URL {url}: {e}") from e

    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        deduplicate=True,
    )

    if not extracted:
        raise ValueError(f"Não foi possível extrair texto útil da página: {url}")

    return extracted.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    pages_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text.strip())
                else:
                    print(
                        f"[AVISO] Página {i} sem texto extraído em {pdf_path.name}",
                        file=sys.stderr,
                    )
    except Exception as e:
        raise RuntimeError(f"Erro ao ler PDF {pdf_path}: {e}") from e

    full_text = "\n\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(f"Não foi possível extrair texto do PDF: {pdf_path}")

    return full_text


def save_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def process_source(source: dict) -> None:
    source_id = source.get("id")
    source_type = source.get("tipo")

    if not source_id or not source_type:
        raise ValueError(f"Fonte inválida: {source}")

    output_path = RAW_DIR / f"{source_id}.txt"

    print(f"[INFO] A processar: {source_id} ({source_type})")

    if source_type == "web":
        url = source.get("url")
        if not url:
            raise ValueError(f"Fonte web sem URL: {source_id}")
        text = extract_web_text(url)

    elif source_type == "pdf":
        relative_path = source.get("path")
        if not relative_path:
            raise ValueError(f"Fonte PDF sem path: {source_id}")
        pdf_path = PROJECT_ROOT / relative_path
        text = extract_pdf_text(pdf_path)

    else:
        raise ValueError(f"Tipo de fonte não suportado: {source_type}")

    save_text(output_path, text)
    print(f"[OK] Guardado em: {output_path}")


def main():
    sources = load_sources(SOURCES_FILE)

    for source in sources:
        try:
            process_source(source)
        except Exception as e:
            print(f"[ERRO] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()