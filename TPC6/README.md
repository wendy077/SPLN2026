# TPC6 - Recuperação de Informação com TF-IDF

## Objetivo

Este trabalho implementa um sistema simples de recuperação de informação.
Dado um conjunto de documentos e uma query, o programa calcula quais são os documentos mais relevantes para essa query.

A relevância é calculada através de:

1. tokenização dos documentos;
2. remoção de stopwords;
3. cálculo de TF;
4. cálculo de IDF;
5. cálculo de TF-IDF;
6. vetorização dos documentos;
7. vetorização da query no mesmo espaço vetorial;
8. cálculo da similaridade cosseno;
9. ordenação dos documentos por score decrescente.

---

## Corpus usado

O corpus usado no exemplo é:

```python
CORPUS = [
    "the sky is blue",
    "the sun is bright",
    "the sun in the sky",
]
```

Foram também usadas várias queries de teste:

```python
QUERIES = [
    "the bright sun",
    "blue sky",
    "sun sky",
    "green mountain",
]
```

A última query serve para testar o comportamento do sistema quando a query contém termos que não aparecem no corpus.

---

## Fórmulas usadas

### TF

O TF mede a frequência relativa de um termo dentro de um documento.

```text
TF(t,d) = número de ocorrências de t em d / número total de tokens em d
```

### IDF

O IDF mede a importância de um termo no corpus.
Termos que aparecem em muitos documentos recebem menor peso.

```text
IDF(t) = log10(N / df(t))
```

Onde:

- `N` é o número total de documentos;
- `df(t)` é o número de documentos onde o termo `t` aparece.

### TF-IDF

O peso final de cada termo num documento é calculado por:

```text
TF-IDF(t,d) = TF(t,d) * IDF(t)
```

### Similaridade cosseno

Depois de os documentos e a query serem representados como vetores, calcula-se a similaridade cosseno entre a query e cada documento.

```text
cos(A,B) = (A . B) / (||A|| * ||B||)
```

Quanto maior o valor, mais relevante é o documento para a query.

---

## Estrutura do programa

O ficheiro `tpc6.py` contém as seguintes funções principais:

- `tokenizer(text)` - tokeniza o texto, coloca em minúsculas e remove stopwords;
- `doc_tf(doc_tokens)` - calcula o TF dos termos de um documento;
- `idf(corpus_tokens)` - calcula o IDF dos termos do corpus;
- `tf_idf(corpus_tokens)` - calcula os pesos TF-IDF dos documentos;
- `build_vocabulary(corpus_tokens)` - cria o vocabulário ordenado do corpus;
- `vectorize(tfidf_docs, vocab)` - transforma documentos em vetores;
- `query_tf_idf(query_tokens, idf_values)` - calcula os pesos TF-IDF da query;
- `vectorize_query(query, vocab, idf_values)` - vetoriza a query;
- `cosine_similarity(vector_a, vector_b)` - calcula a similaridade cosseno;
- `rank_documents(...)` - ordena os documentos por relevância;
- `print_query_analysis(query, vocab, idf_values)` - mostra os tokens e o vetor TF-IDF da query;
- `print_ranked_results(query, results)` - imprime apenas os documentos com score superior a 0.

Documentos com score 0 não são apresentados como relevantes. Quando todos os documentos têm score 0, o programa indica que não foi encontrado nenhum documento relevante.

---

## Resultado esperado

O programa imprime:

1. o corpus tokenizado;
2. o vocabulário usado nos vetores;
3. os valores de IDF;
4. os vetores TF-IDF dos documentos;
5. os tokens e o vetor TF-IDF de cada query;
6. os documentos ordenados por relevância para cada query.

Para a query:

```text
the bright sun
```

é esperado que o documento mais relevante seja:

```text
the sun is bright
```

porque contém os dois termos principais da query: `sun` e `bright`.

## Nota sobre termos desconhecidos

Se a query tiver palavras que não aparecem no corpus, essas palavras recebem peso 0.
Isto acontece porque o sistema só consegue comparar documentos e queries dentro do vocabulário construído a partir do corpus.

---

## Extra implementado

Além de imprimir os resultados no terminal, o programa também gera um ficheiro `resultados.json`.

Este ficheiro contém:

- corpus original;
- corpus tokenizado;
- vocabulário;
- valores de IDF;
- vetores TF-IDF dos documentos;
- tokens e vetor de cada query;
- ranking dos documentos relevantes para cada query.

A exportação em JSON permite reutilizar os resultados noutros programas ou analisá-los de forma estruturada.