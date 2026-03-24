# TP1 - Scripting no Processamento de Linguagem Natural

**Unidade Curricular:** Scripting no Processamento de Linguagem Natural  
**Ano letivo:** 2025/26  
**Tema:** Espaço / Exploração Espacial

---

## Descrição

Este projeto implementa um **pipeline completo e automático de PLN** aplicado a três fontes reais sobre exploração espacial (duas páginas web e um documento PDF). O objetivo central, conforme o enunciado, é processar o texto de cada fonte e selecionar automaticamente as **3 frases mais representativas**, apresentando-as num artigo LaTeX com entidades nomeadas identificadas.

O pipeline cobre todas as fases pedidas no enunciado:

* **a)** Recolha e limpeza do texto das fontes
* **b)** Tokenização e construção de um modelo de linguagem baseado em n-grams
* **c)** Scoring de frases com base no modelo (log-probabilidade média com suavização de Laplace)
* **d)** NER com spaCy para identificação de entidades nomeadas
* **e) + f)** Geração de artigos LaTeX compilados para PDF

---

## Fontes utilizadas

| # | ID | Tipo | Título | URL / Path |
|---|---|---|---|---|
| 1 | `wiki_space_exploration` | web | Space Exploration | https://en.wikipedia.org/wiki/Space_exploration |
| 2 | `wiki_human_spaceflight` | web | Human Spaceflight | https://en.wikipedia.org/wiki/Human_spaceflight |
| 3 | `esa_terrae_novae` | pdf | Terrae Novae 2030+ Strategy Roadmap | `data/raw/terrae_novae.pdf` |

As fontes estão descritas em `sources.json` e seguem o formato `{id, tipo, url/path, titulo, autor, ano}`.

---

## Estrutura do projeto

```
TP1_SPLN/
│
├── src/
│   ├── extract_text.py       # a) extração de texto (web + PDF)
│   ├── clean_text.py         # a) limpeza e normalização
│   ├── split_sentences.py    # b) segmentação em frases com spaCy
│   ├── ngrams.py             # b) construção de bigramas e trigramas
│   ├── select_sentences.py   # c) scoring e seleção das 3 frases
│   ├── ner.py                # d) extração de entidades nomeadas
│   ├── filter_entities.py    # d) filtragem e normalização de entidades
│   └── latex.py              # e) geração dos ficheiros .tex
│
├── data/
│   ├── raw/                  # textos extraídos das fontes
│   └── clean/                # textos limpos, frases, n-grams, entidades
│
├── output/
│   ├── tex/                  # artigos LaTeX gerados
│   └── pdf/                  # PDFs compilados (resultado final)
│
├── sources.json              # configuração das fontes
├── run_pipeline.sh           # script de execução completa
├── clean_output.sh           # script de limpeza
└── README.md
```

---

## Pipeline

### 1. Extração de texto (`extract_text.py`)

Extrai o texto de cada fonte conforme o seu tipo:

* **Web** — usa `requests` para obter o HTML e `trafilatura` (com `favor_precision=True`) para extrair apenas o conteúdo editorial, removendo menus, cabeçalhos e rodapés automáticamente
* **PDF** — usa `pdfplumber` para extrair o texto página a página, com aviso para páginas sem texto extraível

Output: `data/raw/<source>.txt`

---

### 2. Limpeza (`clean_text.py`)

Aplica uma sequência de transformações para produzir texto limpo e contínuo:

1. **`normalize_basic`** — normaliza encoding (CRLF → LF, non-breaking spaces), remove caracteres Unicode fora do range Latin suportado pelo LaTeX, remove referências bibliográficas `[1]`, `[30]`, etc., e remove marcadores inline da Wikipedia como `[update]` e `[note 1]`
2. **`remove_common_noise`** — remove cabeçalhos/rodapés de PDF (e.g. "ESA UNCLASSIFIED"), legendas de figuras, e linhas de lista com `- ` (entradas de tabelas da Wikipedia)
3. **`remove_table_of_contents_lines`** — remove linhas de índice com padrão `......... 12`
4. **`fix_hyphenation`** — junta palavras partidas por hifenização de fim de linha (`explora-\ntion` → `exploration`)
5. **`join_broken_lines`** — junta linhas quebradas dentro do mesmo parágrafo, preservando quebras naturais (fim de frase, bullets, títulos curtos)
6. **`fix_joined_headings`** — corrige títulos de secção que foram colados ao parágrafo seguinte durante a extração do PDF
7. **`remove_corrupted_pdf_lines`** — remove URLs soltos, linhas com letras isoladas separadas por espaços (texto corrompido), e linhas com densidade de espaços anormalmente alta
8. **`collapse_blank_lines`** — normaliza espaçamento (máximo uma linha em branco entre parágrafos)

Output: `data/clean/<source>.txt`

---

### 3. Segmentação em frases (`split_sentences.py`)

Usa o modelo `spacy en_core_web_lg` (modelo large — melhor precisão que o small para textos técnicos) para segmentar o texto limpo em frases. Aplica filtragem adicional para remover:

* frases com menos de 8 palavras ou mais de 45 (demasiado curtas/longas para scoring útil)
* títulos de secção (linhas curtas em Title Case)
* fragmentos que começam com letra minúscula
* pronomes sem antecedente claro em frases curtas (`It`, `This`, `These`, ...)
* linhas numeradas, referências bibliográficas e perguntas

Output: `data/clean/<source>_sentences.json`

---

### 4. N-grams (`ngrams.py`)

Para cada fonte, tokeniza o texto (lowercase, apenas tokens alfabéticos e numéricos) e constrói:

* **Bigramas** — pares de tokens consecutivos `(w1, w2)` com as suas frequências
* **Trigramas** — triplos de tokens consecutivos `(w1, w2, w3)` com as suas frequências

As frequências são calculadas sobre todas as frases do corpus da fonte.

| Fonte | Palavras | Frases | Bigramas únicos | Trigramas únicos |
|---|---|---|---|---|
| wiki_space_exploration | 6646 | 278 | 4879 | 5787 |
| wiki_human_spaceflight | 6605 | 278 | 4907 | 5834 |
| esa_terrae_novae | 6230 | 188 | 4229 | 5254 |

Output: `data/clean/<source>_bigrams.json`, `data/clean/<source>_trigrams.json`

---

### 5. Seleção de frases (`select_sentences.py`)

O scoring de frases é feito com um **modelo de linguagem probabilístico baseado em n-grams com suavização de Laplace**, diretamente alinhado com a fórmula de Maximum Likelihood Estimation lecionada (fórmula 3.11 dos slides) e com o conceito de perplexidade (fórmulas 3.14–3.15).

**Modelo probabilístico:**

Para bigramas, a probabilidade condicional é estimada com suavização de Laplace (add-1):

```
P(w2 | w1) = (C(w1, w2) + 1) / (C(w1) + |V|)
```

Para trigramas, analogamente:

```
P(w3 | w1, w2) = (C(w1, w2, w3) + 1) / (C(w1, w2) + |V|)
```

onde `|V|` é o tamanho do vocabulário (número de tokens únicos no corpus de cada fonte). A suavização de Laplace garante que n-grams não observados no treino não produzem probabilidade nula.

**Score de cada frase:**

O score é a **log-probabilidade média** combinando bigramas e trigramas:

```
score = 0.6 × mean(log P_bigrama) + 0.4 × mean(log P_trigrama)
```

Frases com score mais alto (log-prob média menos negativa) são as mais "esperadas" pelo modelo — têm **menor perplexidade** — e são consideradas mais representativas do corpus. Esta relação é direta: maximizar a log-probabilidade média equivale a minimizar a perplexidade.

**Restantes critérios:**

* penalização leve para frases muito longas (> 35 tokens), que tendem a ter log-prob mais baixa simplesmente por terem mais tokens
* filtro de diversidade por **similaridade de Jaccard** (threshold 0.2) — evita selecionar frases com vocabulário muito sobreposto, garantindo que as 3 frases escolhidas cobrem aspetos distintos do texto

Output: `data/clean/<source>_top_sentences.json`

**Frases selecionadas:**

*Space Exploration:*
1. Landmarks of this era include the launch of the first human-made object to orbit Earth... (score: -5.608)
2. Venus was the first target of interplanetary flyby and lander missions... (score: -5.6497)
3. The Artemis program is an ongoing crewed spaceflight program carried out by NASA... (score: -5.7668)

*Human Spaceflight:*
1. In 1961, US President John F. Kennedy raised the stakes of the Space Race... (score: -5.6721)
2. In the first use of a launch escape system on the launchpad... Soyuz T-10a... (score: -5.7207)
3. The third lunar landing expedition, Apollo 13, in April 1970, was aborted... (score: -5.7343)

*ESA Terrae Novae:*
1. The HESAC looks forward to the vision outlined in Terrae Novae 2030+ strategy... (score: -5.376)
2. It is expected that these capabilities will secure the ESA ambition of "European roots on the Moon"... (score: -5.4205)
3. At CM22, specific decisions will be required to ensure long-term European capabilities... (score: -5.4464)

---

### 6. Extração de entidades (`ner.py`)

Usa o modelo `spacy en_core_web_lg` sobre o texto limpo de cada fonte para identificar entidades nomeadas, agrupadas por label e ordenadas por frequência. Labels extraídos: `ORG`, `LOC`, `GPE`, `PERSON`, `DATE`, `PRODUCT`.

Output: `data/clean/<source>_entities.json`

---

### 7. Filtragem de entidades (`filter_entities.py`)

Aplica pós-processamento às entidades extraídas pelo spaCy:

* **Limpeza** — remove entidades com bullets, comprimento inferior a 2 caracteres, ou padrões de texto corrompido
* **Normalização de texto** — remove artigos definidos desnecessários (e.g. "the Solar System" → "Solar System")
* **Normalização de label** — corrige classificações erradas do spaCy (e.g. planetas como "Moon", "Mars", "Earth" reclassificados como `LOC` em vez de `ORG`)
* **Deduplicação** — remove entidades duplicadas (case-insensitive)
* **Limite por categoria** — máximo 15 entidades por label (as mais frequentes)

Output: `data/clean/<source>_entities_filtered.json`

---

### 8. Geração de LaTeX (`latex.py`)

Gera automaticamente um ficheiro `.tex` por fonte, com a seguinte estrutura:

* **`\maketitle`** — título, autor e data (ou "Acedido em" para fontes web)
* **`abstract`** com `\itemize` — as 3 frases selecionadas pelo modelo
* **Secção "Fonte"** — metadados: tipo, título, autor, URL/path
* **Secção "Texto da Fonte"** — texto limpo original, dividido em parágrafos LaTeX (conforme pedido no enunciado: *"texto da fonte original incluído no corpo do LaTeX"*)
* **Secção "Entidades Nomeadas"** — entidades por categoria em `\itemize` com contagem de ocorrências
* **Secção "Observações"** — nota sobre o método de scoring usado
* **`\thebibliography`** — referência bibliográfica com URL

Output: `output/tex/<source>.tex`

---

### 9. Geração de PDFs

Compilação com `pdflatex` em duas passagens (para resolver referências internas e hiperligações).

Output: `output/pdf/<source>.pdf`

---

## Como executar

### Pipeline completo (recomendado)

```bash
bash run_pipeline.sh
```

### Passo a passo (alternativa)

```bash
python src/extract_text.py
python src/clean_text.py
python src/split_sentences.py
python src/ngrams.py
python src/select_sentences.py
python src/ner.py
python src/filter_entities.py
python src/latex.py

pdflatex -output-directory=output/pdf output/tex/wiki_space_exploration.tex
pdflatex -output-directory=output/pdf output/tex/wiki_space_exploration.tex
pdflatex -output-directory=output/pdf output/tex/wiki_human_spaceflight.tex
pdflatex -output-directory=output/pdf output/tex/wiki_human_spaceflight.tex
pdflatex -output-directory=output/pdf output/tex/esa_terrae_novae.tex
pdflatex -output-directory=output/pdf output/tex/esa_terrae_novae.tex
```

### Limpar todos os ficheiros gerados

```bash
bash clean_output.sh
```

> ⚠️ Este comando apaga todos os ficheiros em `data/raw/`, `data/clean/` e `output/`. O ficheiro `data/raw/terrae_novae.pdf` (fonte original da ESA) não é apagado.

---

## Dependências

* Python 3.10+
* spaCy com modelo large:

```bash
pip install spacy
python -m spacy download en_core_web_lg
```

* Bibliotecas Python:

```bash
pip install trafilatura requests pdfplumber
```

* LaTeX: TeX Live (ou MiKTeX no Windows)

---

## Observações

* O pipeline é totalmente automático a partir de `sources.json` — adicionar uma nova fonte implica apenas adicionar uma entrada nesse ficheiro
* O modelo de seleção de frases não utiliza semântica profunda: quando um tema domina fortemente o corpus (e.g. "Moon" em Space Exploration), as frases selecionadas tendem a pertencer a esse tema dominante, o que é comportamento esperado de um modelo n-gram
* A suavização de Laplace garante que n-grams não observados no treino não produzem probabilidade nula, tornando o modelo robusto a vocabulário novo
* O spaCy comete erros de classificação em textos técnicos — por exemplo, classifica planetas como `ORG` ou siglas técnicas como `GPE`; o passo de filtragem de entidades corrige os casos mais evidentes
* O texto da fonte ESA contém um erro tipográfico no PDF original ("European oots on the Moon" em vez de "roots") que foi preservado fielmente

---

## Conclusão

O projeto demonstra um pipeline completo de PLN aplicado a dados reais, desde a extração de texto de fontes heterogéneas (web e PDF) até à geração automática de documentos LaTeX estruturados. As principais componentes técnicas implementadas são: limpeza multi-fase de texto, segmentação de frases com spaCy, modelação de linguagem por n-grams com estimação de máxima verosimilhança e suavização de Laplace, seleção de frases por log-probabilidade média (equivalente a minimizar a perplexidade), reconhecimento e filtragem de entidades nomeadas, e geração automática de LaTeX.