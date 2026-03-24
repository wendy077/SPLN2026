import json
from pathlib import Path
from datetime import date


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "output" / "tex"


def load_sources(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_top_sentences(top_sentences: list) -> str:
    items = []
    for item in top_sentences:
        sentence = latex_escape(item["sentence"])
        items.append(f"\\item {sentence}")
    return "\n".join(items)


def format_entities(entities: dict) -> str:
    sections = []
    for label, items in entities.items():
        if not items:
            continue
        sections.append(f"\\subsection*{{{latex_escape(label)}}}")
        sections.append("\\begin{itemize}")
        for item in items:
            text = latex_escape(item["text"])
            count = item["count"]
            sections.append(f"\\item {text} ({count})")
        sections.append("\\end{itemize}")
        sections.append("")
    return "\n".join(sections)


# formata o texto original da fonte para o corpo do LaTeX
# Divide em parágrafos e escapa cada um, ignorando linhas vazias e ruído curto.
def format_source_text(raw_text: str) -> str:
    paragraphs = []
    for paragraph in raw_text.split("\n\n"):
        # limpar espaços extra dentro do parágrafo
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        joined = " ".join(lines)

        # ignorar parágrafos muito curtos (títulos soltos, ruído residual)
        if len(joined.split()) < 6:
            continue

        escaped = latex_escape(joined)
        paragraphs.append(escaped)

    return "\n\n".join(paragraphs)


def format_source_reference(source: dict) -> str:
    title = latex_escape(source.get("titulo", "Untitled source"))
    author = latex_escape(source.get("autor", "Unknown author"))
    ano = latex_escape(source.get("ano", ""))
    ano_str = f" {ano}." if ano else "."

    url = source.get("url", "")
    if url:
        return f"{author}. \\textit{{{title}}}{ano_str} Disponível em: \\url{{{url}}}."
    else:
        path = latex_escape(source.get("path", ""))
        return f"{author}. \\textit{{{title}}}{ano_str} Ficheiro PDF: {path}."

# recebe também source_text (str com o texto limpo da fonte)
def build_tex(source: dict, top_sentences: list, entities: dict, source_text: str) -> str:
    title = latex_escape(source.get("titulo", source["id"]))
    author = latex_escape(source.get("autor", "Unknown author"))
    source_type = latex_escape(source.get("tipo", "unknown"))
    
    meses = {
        "January": "janeiro", "February": "fevereiro", "March": "março",
        "April": "abril", "May": "maio", "June": "junho",
        "July": "julho", "August": "agosto", "September": "setembro",
        "October": "outubro", "November": "novembro", "December": "dezembro"
    }
    if source.get("ano"):
        data_str = source["ano"]
    else:
        hoje = date.today()
        data_str = f"Acedido em {hoje.day} de {meses[hoje.strftime('%B')]} de {hoje.year}"

    abstract_items = format_top_sentences(top_sentences)
    entities_block = format_entities(entities)
    reference = format_source_reference(source)

    # texto da fonte formatado em parágrafos LaTeX
    source_text_block = format_source_text(source_text)

    source_location = source.get("url") or source.get("path", "")

    tex = rf"""
\documentclass[12pt]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{geometry}}
\usepackage{{hyperref}}

\geometry{{margin=2.5cm}}

\title{{{title}}}
\author{{{author}}}
\date{{{data_str}}}

\begin{{document}}

\maketitle

\begin{{abstract}}
\begin{{itemize}}
{abstract_items}
\end{{itemize}}
\end{{abstract}}

\section{{Fonte}}

\begin{{itemize}}
\item \textbf{{Tipo}}: {source_type}
\item \textbf{{Título}}: {title}
\item \textbf{{Autor / organização}}: {author}
\item \textbf{{Localização}}: \url{{{source_location}}}
\end{{itemize}}

\section{{Texto da Fonte}}

% texto original limpo da fonte, incluído no corpo do artigo conforme pedido no enunciado
{source_text_block}

\section{{Entidades Nomeadas}}

{entities_block}

\section{{Observações}}

Este documento foi gerado automaticamente a partir da fonte selecionada sobre exploração espacial.
As frases do resumo foram escolhidas com base num modelo de linguagem de n-grams (trigramas) com scoring por log-probabilidade média e suavização de Laplace.
As entidades nomeadas foram extraídas com spaCy (modelo \texttt{{en\_core\_web\_lg}}) e posteriormente filtradas.

\begin{{thebibliography}}{{9}}

\bibitem{{source}}
{reference}

\end{{thebibliography}}

\end{{document}}
"""
    return tex.strip() + "\n"


def process_source(source: dict) -> None:
    source_id = source["id"]

    top_sentences_path = CLEAN_DIR / f"{source_id}_top_sentences.json"
    entities_path = CLEAN_DIR / f"{source_id}_entities_filtered.json"
    # lê também o texto limpo da fonte
    source_text_path = CLEAN_DIR / f"{source_id}.txt"
    output_path = OUTPUT_DIR / f"{source_id}.tex"

    top_sentences = read_json(top_sentences_path)
    entities = read_json(entities_path)
    source_text = read_text(source_text_path)   

    # passa source_text para build_tex
    tex = build_tex(source, top_sentences, entities, source_text)
    write_text(output_path, tex)

    print(f"[OK] TEX gerado: {output_path}")


def main():
    sources = load_sources(SOURCES_FILE)
    for source in sources:
        process_source(source)


if __name__ == "__main__":
    main()