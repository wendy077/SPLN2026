#!/bin/bash
set -e  # para se algum comando falhar

echo "=== A extrair texto das fontes ==="
python src/extract_text.py

echo "=== A limpar texto ==="
python src/clean_text.py

echo "=== A segmentar frases ==="
python src/split_sentences.py

echo "=== A construir n-grams ==="
python src/ngrams.py

echo "=== A selecionar frases ==="
python src/select_sentences.py

echo "=== A extrair entidades (NER) ==="
python src/ner.py

echo "=== A filtrar entidades ==="
python src/filter_entities.py

echo "=== A gerar ficheiros LaTeX ==="
python src/latex.py

echo "=== A compilar PDFs ==="
python -c "import json; [print(s['id']) for s in json.load(open('sources.json'))]" | while read source; do
    echo "  Compilando $source..."
    pdflatex -interaction=nonstopmode -output-directory=output/pdf output/tex/$source.tex > /dev/null
    pdflatex -interaction=nonstopmode -output-directory=output/pdf output/tex/$source.tex > /dev/null
done

echo ""
echo "=== Pipeline concluído. PDFs em output/pdf/ ==="