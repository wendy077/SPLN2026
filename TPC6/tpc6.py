"""
TPC6 - Scripting e Processamento de Linguagem Natural
Recuperação de Informação com TF-IDF e similaridade cosseno.

Objetivo:
    Dado um conjunto de documentos e uma query, representar ambos no mesmo
    espaço vetorial TF-IDF e devolver os documentos ordenados por relevância.

"""

from __future__ import annotations

import math
import re
import json
from collections import Counter
from typing import Iterable


CORPUS = [
    "the sky is blue",
    "the sun is bright",
    "the sun in the sky",
]

QUERIES = [
    "the bright sun",
    "blue sky",
    "sun sky",
    "green mountain",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with"
}


def tokenizer(text: str) -> list[str]:
    """
    Normaliza e tokeniza texto.

    Passos:
      - converte para minúsculas;
      - extrai apenas palavras;
      - remove stopwords simples em inglês.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def doc_tf(doc_tokens: list[str]) -> dict[str, float]:
    """
    Calcula o TF de todos os termos de um documento.

    TF(t,d) = número de ocorrências de t em d / número total de tokens em d
    """
    if not doc_tokens:
        return {}

    total_terms = len(doc_tokens)
    counts = Counter(doc_tokens)

    return {
        term: count / total_terms
        for term, count in counts.items()
    }


def idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """
    Calcula o IDF de cada termo do corpus.

    IDF(t) = log10(N / df(t))
    onde:
      - N é o número de documentos;
      - df(t) é o número de documentos onde t aparece.
    """
    number_of_docs = len(corpus_tokens)
    vocabulary = sorted({term for doc in corpus_tokens for term in doc})

    idf_values = {}

    for term in vocabulary:
        document_frequency = sum(1 for doc in corpus_tokens if term in doc)
        idf_values[term] = math.log10(number_of_docs / document_frequency)

    return idf_values


def tf_idf(corpus_tokens: list[list[str]]) -> list[dict[str, float]]:
    """
    Constrói a representação TF-IDF dos documentos.

    O resultado é uma lista de dicionários. Cada dicionário representa um
    documento e associa cada termo ao respetivo peso TF-IDF.
    """
    idf_values = idf(corpus_tokens)
    matrix = []

    for doc_tokens in corpus_tokens:
        tf_values = doc_tf(doc_tokens)
        doc_weights = {
            term: tf_value * idf_values[term]
            for term, tf_value in tf_values.items()
        }
        matrix.append(doc_weights)

    return matrix


def build_vocabulary(corpus_tokens: list[list[str]]) -> list[str]:
    """Cria um vocabulário ordenado para garantir vetores consistentes."""
    return sorted({term for doc in corpus_tokens for term in doc})


def vectorize(tfidf_docs: list[dict[str, float]], vocab: list[str]) -> list[list[float]]:
    """
    Converte os dicionários TF-IDF dos documentos para vetores densos.

    Todos os documentos usam a mesma ordem de termos, definida pelo vocabulário.
    """
    return [
        [doc_weights.get(term, 0.0) for term in vocab]
        for doc_weights in tfidf_docs
    ]


def query_tf_idf(query_tokens: list[str], idf_values: dict[str, float]) -> dict[str, float]:
    """
    Calcula os pesos TF-IDF da query.

    Termos que não aparecem no corpus ficam com IDF 0, porque não pertencem ao
    espaço vetorial usado pelos documentos.
    """
    tf_values = doc_tf(query_tokens)

    return {
        term: tf_value * idf_values.get(term, 0.0)
        for term, tf_value in tf_values.items()
    }


def vectorize_query(query: str, vocab: list[str], idf_values: dict[str, float]) -> list[float]:
    """Vetoriza uma query no mesmo espaço vetorial dos documentos."""
    query_tokens = tokenizer(query)
    query_weights = query_tf_idf(query_tokens, idf_values)

    return [query_weights.get(term, 0.0) for term in vocab]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calcula a similaridade cosseno entre dois vetores."""
    numerator = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def rank_documents(
    query: str,
    corpus: list[str],
    doc_vectors: list[list[float]],
    vocab: list[str],
    idf_values: dict[str, float],
) -> list[dict[str, object]]:
    """Ordena os documentos por relevância relativamente à query."""
    query_vector = vectorize_query(query, vocab, idf_values)

    results = []

    for doc_id, doc_vector in enumerate(doc_vectors, start=1):
        score = cosine_similarity(query_vector, doc_vector)
        results.append({
            "doc_id": doc_id,
            "score": score,
            "document": corpus[doc_id - 1],
        })

    return sorted(results, key=lambda item: item["score"], reverse=True)


def print_tokens(title: str, token_lists: Iterable[list[str]]) -> None:
    print(title)
    for index, tokens in enumerate(token_lists, start=1):
        print(f"  Doc {index}: {tokens}")
    print()


def print_query_analysis(query: str, vocab: list[str], idf_values: dict[str, float]) -> None:
    query_tokens = tokenizer(query)
    query_vector = vectorize_query(query, vocab, idf_values)

    print(f"Query: {query!r}")
    print(f"  Tokens da query: {query_tokens}")
    print(f"  Vetor da query: {[round(value, 4) for value in query_vector]}")


def print_ranked_results(query: str, results: list[dict[str, object]]) -> None:
    print("Documentos ordenados por relevância:")

    relevant_results = [
        result for result in results
        if result["score"] > 0
    ]

    if not relevant_results:
        print("  Nenhum documento relevante encontrado.")
        print("  Todos os documentos tiveram score 0.0000.\n")
        return

    for position, result in enumerate(relevant_results, start=1):
        print(
            f"  {position}. Doc {result['doc_id']} | "
            f"score={result['score']:.4f} | "
            f"{result['document']}"
        )

    print()


def build_report(
    corpus: list[str],
    corpus_tokens: list[list[str]],
    vocabulary: list[str],
    idf_values: dict[str, float],
    doc_vectors: list[list[float]],
    queries: list[str],
) -> dict[str, object]:
    report = {
        "corpus": corpus,
        "corpus_tokens": corpus_tokens,
        "vocabulary": vocabulary,
        "idf": idf_values,
        "document_vectors": doc_vectors,
        "queries": []
    }

    for query in queries:
        query_tokens = tokenizer(query)
        query_vector = vectorize_query(query, vocabulary, idf_values)
        results = rank_documents(query, corpus, doc_vectors, vocabulary, idf_values)

        report["queries"].append({
            "query": query,
            "query_tokens": query_tokens,
            "query_vector": query_vector,
            "results": [
                result for result in results
                if result["score"] > 0
            ]
        })

    return report


def save_report_json(report: dict[str, object], filename: str = "resultados.json") -> None:
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def main() -> None:
    corpus_tokens = [tokenizer(document) for document in CORPUS]
    vocabulary = build_vocabulary(corpus_tokens)
    idf_values = idf(corpus_tokens)
    tfidf_docs = tf_idf(corpus_tokens)
    doc_vectors = vectorize(tfidf_docs, vocabulary)

    print_tokens("Corpus tokenizado:", corpus_tokens)
    print(f"Vocabulário: {vocabulary}\n")

    print("IDF:")
    for term in vocabulary:
        print(f"  {term}: {idf_values[term]:.4f}")
    print()

    print("Vetores TF-IDF dos documentos:")
    for index, vector in enumerate(doc_vectors, start=1):
        formatted_vector = [round(value, 4) for value in vector]
        print(f"  Doc {index}: {formatted_vector}")
    print()

    for query in QUERIES:
        print_query_analysis(query, vocabulary, idf_values)
        results = rank_documents(query, CORPUS, doc_vectors, vocabulary, idf_values)
        print_ranked_results(query, results)

    report = build_report(
        CORPUS,
        corpus_tokens,
        vocabulary,
        idf_values,
        doc_vectors,
        QUERIES,
    )
    save_report_json(report)
    print("Relatório guardado em resultados.json")


if __name__ == "__main__":
    main()
