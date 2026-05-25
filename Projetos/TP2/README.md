# TP2 SPLN — Music Information Retrieval & Question Answering

Projeto desenvolvido para a Parte 1 do TP2 de **Scripting e Processamento de Linguagem Natural**.

A aplicação implementa uma pipeline de **Information Retrieval** e **Question Answering** sobre um corpus musical com artistas, álbuns, géneros, festivais, prémios, eventos e movimentos musicais.

## Resumo

A pipeline contém:

- corpus com **110 documentos**;
- pré-processamento e divisão em **3982 chunks**;
- retriever lexical com **BM25**;
- retriever semântico com **SBERT** (`sentence-transformers/all-MiniLM-L6-v2`);
- retriever híbrido **BM25 + SBERT**;
- QA extrativo com **DistilBERT fine-tuned em SQuAD**;
- QA abstrativo com **FLAN-T5-base**;
- avaliação quantitativa;
- interface web em **Gradio**.

## Estrutura principal

```text
TP2/
├── app.py
├── run_project.py
├── requirements.txt
├── seeds.yaml
├── README.md
├── Relatorio_TP2.pdf
├── data/
│   ├── raw/
│   │   ├── corpus.jsonl
│   │   ├── corpus_audit.csv
│   │   └── backups/
│   ├── processed/
│   │   └── chunks.jsonl
│   ├── eval_queries.json
│   ├── eval_results.json
│   ├── eval_metrics.png
│   ├── eval_metrics_table.md
│   ├── corpus_distribution.png
│   └── embeddings_plot.png
├── models/
│   ├── sbert_chunk_embeddings.npy
│   ├── sbert_config.json
│   └── qa_finetuned/
└── src/
    ├── audit_corpus.py
    ├── build_corpus.py
    ├── build_sbert_index.py
    ├── evaluate.py
    ├── preprocess.py
    ├── qa_abstractive.py
    ├── qa_extractive.py
    ├── repair_corpus.py
    ├── retriever_bm25.py
    ├── retriever_hybrid.py
    ├── retriever_sbert.py
    ├── train_qa_squad.py
    ├── visualize_corpus_distribution.py
    ├── visualize_embeddings.py
    └── visualize_eval_metrics.py
```

## Instalação

Criar e ativar ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

## Verificar o projeto

```bash
python run_project.py check
```

Estado esperado:

```text
Corpus documents: 110
Chunks: 3982
Embeddings shape: (3982, 384)
Fine-tuned QA model: available
```

## Usar a pipeline pela linha de comandos

Pesquisar documentos com o retriever híbrido:

```bash
python run_project.py search "Who released OK Computer?" --filter-type album
```

Responder com QA extrativo:

```bash
python run_project.py ask "Who released OK Computer?" --mode extractive --filter-type album
```

Responder com QA abstrativo:

```bash
python run_project.py ask "Explain why To Pimp a Butterfly is considered important." --mode abstractive --filter-type album
```

Responder com os dois módulos:

```bash
python run_project.py ask "What genre is Nirvana associated with?" --mode both --filter-type genre
```

## Demo completo (para apresentação)

Corre uma sequência automática com QA extrativo (perguntas factuais), QA abstrativo (pergunta explicativa) e geração das visualizações:

```bash
python run_project.py demo-full
```

O comando executa perguntas factuais com o módulo extrativo, perguntas explicativas/definicionais com o módulo abstrativo, uma comparação entre os dois modos, regenera os gráficos de métricas e embeddings, e imprime as instruções para lançar a interface Gradio em `http://127.0.0.1:7860`.

Para uma demonstração mais rápida (só QA, sem visualizações):

```bash
python run_project.py demo
```

## Interface Gradio

Executar localmente:

```bash
python app.py
```

A interface corre em `http://127.0.0.1:7860`. Permite inserir perguntas, escolher filtro por tipo de documento, ajustar o peso semântico `alpha`, escolher `top-k`, selecionar o modo de QA e consultar os documentos recuperados.

## Avaliação

Correr avaliação completa:

```bash
python run_project.py evaluate --save-results data/eval_results.json
```

Correr apenas avaliação dos retrievers:

```bash
python run_project.py evaluate --no-qa --save-results data/eval_results.json
```

Gerar gráfico e tabela de métricas:

```bash
python src/visualize_eval_metrics.py
```

Outputs:

```text
data/eval_results.json
data/eval_metrics.png
data/eval_metrics_table.md
```

Resultados finais:

| Retriever | MRR | P@1 | Hit@3 |
|---|---:|---:|---:|
| BM25 | 0.8475 | 0.7500 | 0.9000 |
| SBERT | 0.9250 | 0.8500 | 1.0000 |
| Hybrid | 0.9500 | 0.9000 | 1.0000 |

| Módulo | Exact Match | F1 |
|---|---:|---:|
| Extractive QA | 45.00% | 46.43% |
| Abstractive QA | 50.00% | 52.86% |

## Visualizações

Distribuição do corpus por tipo:

```bash
python src/visualize_corpus_distribution.py
```

Output:

```text
data/corpus_distribution.png
```

Espaço de embeddings SBERT com t-SNE:

```bash
python src/visualize_embeddings.py --label-mode key --output data/embeddings_plot.png
```

Output:

```text
data/embeddings_plot.png
```

## Reconstrução dos artefactos

Se o corpus for alterado, reconstruir chunks e embeddings:

```bash
python run_project.py rebuild-index
```

Se for necessário reconstruir o corpus a partir das seeds:

```bash
python run_project.py rebuild-corpus
```

Este comando depende de Internet e pode demorar.

### Modelo extrativo fine-tuned

A entrega em zip inclui o modelo fine-tuned completo em `models/qa_finetuned/`.

No repositório GitHub, o ficheiro `models/qa_finetuned/model.safetensors` não é versionado por exceder o limite de tamanho permitido. Para reconstruir o modelo extrativo, executar:

```bash
python src/train_qa_squad.py --epochs 1 --batch-size 4 --learning-rate 3e-5
```

Este é o comando correspondente à Abordagem 1, usada como modelo principal por ter obtido melhor desempenho do que a Abordagem 2.

## Ficheiros principais

| Ficheiro | Função |
|---|---|
| `run_project.py` | Ponto de entrada principal: check, search, ask, demo, demo-full, evaluate e rebuild. |
| `app.py` | Interface Gradio. |
| `seeds.yaml` | Entidades usadas para construir o corpus. |
| `src/build_corpus.py` | Recolha inicial dos documentos. |
| `src/audit_corpus.py` | Auditoria do corpus. |
| `src/repair_corpus.py` | Reparação de documentos problemáticos. |
| `src/preprocess.py` | Limpeza e chunking. |
| `src/build_sbert_index.py` | Criação dos embeddings SBERT. |
| `src/retriever_bm25.py` | Retriever lexical. |
| `src/retriever_sbert.py` | Retriever semântico. |
| `src/retriever_hybrid.py` | Retriever híbrido BM25 + SBERT. |
| `src/qa_extractive.py` | QA extrativo. |
| `src/qa_abstractive.py` | QA abstrativo. |
| `src/evaluate.py` | Avaliação quantitativa. |
| `src/visualize_eval_metrics.py` | Gráfico e tabela das métricas. |
| `src/visualize_corpus_distribution.py` | Gráfico da distribuição do corpus. |
| `src/visualize_embeddings.py` | Projeção 2D dos embeddings. |

## Limitações conhecidas

- O corpus é pequeno e centrado em entidades pré-selecionadas.
- O corpus está em inglês, pelo que perguntas em português podem ter pior desempenho.
- O QA abstrativo usa `google/flan-t5-base`, que pode gerar respostas incompletas.
- O QA extrativo depende fortemente dos documentos recuperados.
- A heurística de metadados melhora perguntas factuais simples, mas é apenas uma camada auxiliar.
- A avaliação usa apenas 20 queries anotadas manualmente, pelo que os valores de EM/F1 e métricas de retrieval devem ser interpretados como estimativas.