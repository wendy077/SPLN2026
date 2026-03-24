import json
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def load_sources(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+", text)


def build_bigrams(tokens: list[str]) -> list[tuple]:
    return list(zip(tokens, tokens[1:]))


def build_trigrams(tokens: list[str]) -> list[tuple]:
    return list(zip(tokens, tokens[1:], tokens[2:]))


def load_bigram_freqs(path: Path) -> dict:
    data = read_json(path)
    return {tuple(item["bigram"]): item["freq"] for item in data}


def load_trigram_freqs(path: Path) -> dict:
    data = read_json(path)
    return {tuple(item["trigram"]): item["freq"] for item in data}


def build_unigram_freqs(bigram_freqs: dict) -> dict:
    """
    Reconstrói as contagens de unigramas a partir das frequências de bigramas.
    Para cada bigrama (w1, w2) com frequência f, C(w1) += f.
    Usado no denominador da estimativa MLE com suavização de Laplace.
    """
    unigram_freqs: dict[str, int] = {}
    for (w1, _w2), freq in bigram_freqs.items():
        unigram_freqs[w1] = unigram_freqs.get(w1, 0) + freq
    return unigram_freqs


def build_bigram_context_freqs(trigram_freqs: dict) -> dict:
    """
    Reconstrói as contagens de bigramas contexto a partir das frequências de trigramas.
    Para cada trigrama (w1, w2, w3) com frequência f, C(w1, w2) += f.
    Usado no denominador da estimativa trigrama com suavização de Laplace.
    """
    context_freqs: dict[tuple, int] = {}
    for (w1, w2, _w3), freq in trigram_freqs.items():
        key = (w1, w2)
        context_freqs[key] = context_freqs.get(key, 0) + freq
    return context_freqs


def laplace_bigram_logprob(
    bigram: tuple,
    bigram_freqs: dict,
    unigram_freqs: dict,
    vocab_size: int,
) -> float:
    """
    Estima log P(w2 | w1) com suavização de Laplace (add-1):

        P(w2 | w1) = (C(w1, w2) + 1) / (C(w1) + |V|)

    Retorna o logaritmo natural desta probabilidade.
    Corresponde à fórmula 3.11 dos slides (Maximum Likelihood Estimation),
    estendida com suavização de Laplace para evitar probabilidades nulas.
    """
    w1, _w2 = bigram
    count_bigram = bigram_freqs.get(bigram, 0)
    count_w1 = unigram_freqs.get(w1, 0)
    prob = (count_bigram + 1) / (count_w1 + vocab_size)
    return math.log(prob)


def laplace_trigram_logprob(
    trigram: tuple,
    trigram_freqs: dict,
    bigram_context_freqs: dict,
    vocab_size: int,
) -> float:
    """
    Estima log P(w3 | w1, w2) com suavização de Laplace (add-1):

        P(w3 | w1, w2) = (C(w1, w2, w3) + 1) / (C(w1, w2) + |V|)

    Retorna o logaritmo natural desta probabilidade.
    """
    w1, w2, _w3 = trigram
    count_trigram = trigram_freqs.get(trigram, 0)
    count_context = bigram_context_freqs.get((w1, w2), 0)
    prob = (count_trigram + 1) / (count_context + vocab_size)
    return math.log(prob)


def sentence_score(
    sentence: str,
    bigram_freqs: dict,
    trigram_freqs: dict,
    unigram_freqs: dict,
    bigram_context_freqs: dict,
    vocab_size: int,
) -> float:
    """
    Calcula o score de uma frase como log-probabilidade média com suavização de Laplace.

    O score é o negativo da perplexidade por token: frases com log-probabilidade
    média mais alta (menos negativa) são mais "esperadas" pelo modelo e consideradas
    mais representativas do corpus — são as selecionadas.

    Combina modelo de bigramas (60%) e trigramas (40%):
        score = 0.6 * mean_log_P_bigram + 0.4 * mean_log_P_trigram

    Esta abordagem está alinhada com o conceito de perplexidade lecionado
    (fórmulas 3.14 e 3.15 dos slides): minimizar perplexidade equivale a
    maximizar a log-probabilidade média.
    """
    tokens = tokenize(sentence)
    bigrams = build_bigrams(tokens)
    trigrams = build_trigrams(tokens)

    if not bigrams:
        return -math.inf

    bigram_logprob = sum(
        laplace_bigram_logprob(bg, bigram_freqs, unigram_freqs, vocab_size)
        for bg in bigrams
    ) / len(bigrams)

    if trigrams:
        trigram_logprob = sum(
            laplace_trigram_logprob(tg, trigram_freqs, bigram_context_freqs, vocab_size)
            for tg in trigrams
        ) / len(trigrams)
    else:
        trigram_logprob = bigram_logprob

    score = 0.6 * bigram_logprob + 0.4 * trigram_logprob

    # Penalização leve para frases muito longas (tendem a ter log-prob mais baixa
    # simplesmente por terem mais tokens, não por serem menos representativas)
    if len(tokens) > 35:
        score *= 0.9

    return score


def is_candidate_sentence(sentence: str) -> bool:
    s = sentence.strip()
    lower = s.lower()
    words = s.split()

    if len(words) < 8:
        return False

    if len(words) > 45:
        return False

    bad_prefixes = (
        "introduction",
        "conclusion",
        "foreword",
        "about terrae novae",
        "figure",
        "table",
    )
    if any(lower.startswith(prefix) for prefix in bad_prefixes):
        return False

    if s[0].islower():
        return False

    if re.match(r'^(It|This|These|They|He|She|We)\b', s) and len(words) < 12:
        return False

    if re.match(r"^\d+(\.\d+)*\s+", s):
        return False

    if "figure " in lower or "table " in lower:
        return False

    if s.endswith(":"):
        return False

    if s.endswith("?"):
        return False

    if lower.startswith("archived from"):
        return False
    if lower.startswith("retrieved "):
        return False
    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b", lower) and len(words) < 12:
        return False

    return True


def process_source(source: dict) -> None:
    source_id = source["id"]

    sentences_path = CLEAN_DIR / f"{source_id}_sentences.json"
    bigrams_path = CLEAN_DIR / f"{source_id}_bigrams.json"
    trigrams_path = CLEAN_DIR / f"{source_id}_trigrams.json"
    output_path = CLEAN_DIR / f"{source_id}_top_sentences.json"

    sentences = read_json(sentences_path)
    bigram_freqs = load_bigram_freqs(bigrams_path)
    trigram_freqs = load_trigram_freqs(trigrams_path)

    # Estruturas auxiliares para o modelo probabilístico com suavização de Laplace
    unigram_freqs = build_unigram_freqs(bigram_freqs)
    bigram_context_freqs = build_bigram_context_freqs(trigram_freqs)

    # Tamanho do vocabulário: número de tokens únicos vistos nos unigramas
    vocab_size = len(unigram_freqs)

    scored = []
    for sentence in sentences:
        if not is_candidate_sentence(sentence):
            continue

        score = sentence_score(
            sentence,
            bigram_freqs,
            trigram_freqs,
            unigram_freqs,
            bigram_context_freqs,
            vocab_size,
        )
        scored.append({
            "sentence": sentence,
            "score": round(score, 4)
        })

    # Ordenar por score descendente: score mais alto = log-prob média mais alta
    # = frase mais esperada pelo modelo = menor perplexidade
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Filtro de diversidade: evita selecionar frases muito semelhantes entre si
    def jaccard_sim(s1, s2):
        t1, t2 = set(tokenize(s1)), set(tokenize(s2))
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    top3 = []
    for candidate in scored:
        if len(top3) == 3:
            break
        too_similar = any(
            jaccard_sim(candidate["sentence"], sel["sentence"]) > 0.2
            for sel in top3
        )
        if not too_similar:
            top3.append(candidate)

    write_json(output_path, top3)

    print(f"[OK] {source_id}: top 3 frases selecionadas por menor perplexidade -> {output_path}")
    for item in top3:
        print(f"  score (log-prob média): {item['score']:.4f} | {item['sentence'][:80]}...")


def main():
    sources = load_sources(SOURCES_FILE)
    for source in sources:
        process_source(source)


if __name__ == "__main__":
    main()