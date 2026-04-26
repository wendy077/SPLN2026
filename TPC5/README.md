# TPC5 - NER com spaCy

## Objetivo

O objetivo deste trabalho foi executar o exercício da aula sobre **Named Entity Recognition (NER)** e treinar um modelo específico utilizando a biblioteca **spaCy**, recorrendo aos ficheiros fornecidos em formato **IOB**. No final, os resultados de ambas as abordagens foram comparados.

## Ficheiros Fornecidos

* `aula9.ipynb`
* `arquivo_ner_train.iob`
* `arquivo_ner_test.iob`

## Exercício da Aula

O notebook `aula9.ipynb` foi executado como *baseline*. Os melhores resultados obtidos foram:

| Epoch | Precision | Recall | F1 | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| 2 | 94.56 | 96.80 | 95.67 | 98.48 |

---

## Treino com spaCy

### 1. Conversão de Formatos
Os ficheiros `.iob` foram convertidos para o formato binário `.spacy`, nativo da biblioteca:

```bash
spacy convert arquivo_ner_train.iob ./datasets_spacy -c iob
spacy convert arquivo_ner_test.iob ./datasets_spacy -c iob
````

---

### 2. Configuração e Dependências
Foi gerado o ficheiro de configuração inicial:

```bash
spacy init config config.cfg --lang pt --pipeline ner --optimize accuracy -F
````

Como a configuração utilizava vetores do modelo pt_core_news_lg, foi necessária a sua instalação:

```bash
python -m spacy download pt_core_news_lg
````
---

### 3. Execução do Treino
O modelo foi treinado com o seguinte comando:

```bash
spacy train config.cfg --output ./output --paths.train ./datasets_spacy/arquivo_ner_train.spacy --paths.dev ./datasets_spacy/arquivo_ner_test.spacy
````

O melhor resultado observado durante o treino foi:

| Epoch | Step | Precision | Recall |    F1 |
| ----- | ---: | --------: | -----: | ----: |
| 28    | 4200 |     99.31 |  99.40 | 99.36 |


O modelo final foi guardado em: ./output/model-best.

---

### Comparação
| Modelo | Precision | Recall |    F1 |
| ------ | --------: | -----: | ----: |
| Aula9  |     94.56 |  96.80 | 95.67 |
| spaCy  |     99.31 |  99.40 | 99.36 |


O modelo treinado com spaCy obteve resultados superiores em todas as métricas principais (precision, recall e F1-score).

---

### Conclusão
Neste trabalho, compararam-se duas abordagens para o reconhecimento de entidades mencionadas. O modelo da aula serviu como ponto de partida, enquanto a otimização através do ecossistema spaCy permitiu alcançar um patamar de precisão mais elevado, culminando num F1-score de 99.36.