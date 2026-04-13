import re
from pathlib import Path
from gensim.models import Word2Vec

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = [
    DATA_DIR / "pedra_filosofal.txt",
    DATA_DIR / "camara_secreta.txt",
]

STOPWORDS = {
    "a", "à", "ao", "aos", "as", "às", "o", "os",
    "de", "da", "das", "do", "dos",
    "um", "uma", "uns", "umas",
    "e", "é", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sob", "sobre",
    "que", "se", "não", "sim", "ou", "mas",
    "como", "quando", "onde", "qual", "quais",
    "quem", "porque", "porquê", "isso", "isto", "aquilo",
    "ele", "ela", "eles", "elas",
    "eu", "tu", "você", "vocês", "nós",
    "me", "te", "lhe", "nos", "vos", "lhes",
    "meu", "minha", "meus", "minhas",
    "seu", "sua", "seus", "suas",
    "era", "foi", "ser", "estar", "está", "estavam",
    "tem", "tinha", "ter", "há", "havia",
    "muito", "muita", "muitos", "muitas",
    "mais", "menos", "bem", "mal",
    "já", "ainda", "também", "só", "até",
    "num", "numa", "dum", "duma",
    "lhe", "dela", "dele", "deles", "delas",
    "disse", "estava", "vai", "então", "parecia", "agora",
    "depois", "todos", "alguma", "dizer", "pela", "nada",
    "vez", "coisa", "olhos", "cabeça", "porta", "voz",
    "pelo", "dois", "fazer", "nem", "lado", "tão", "nunca",
    "quase", "fora", "perguntou", "tinham", "tempo",
    "mesmo", "enquanto", "aqui", "casa", "grande"
}


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str, source_name: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    marker = "— CAPÍTULO UM —"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx:]

    text = text.lower()
    text = re.sub(r"\n+", "\n", text)
    return text


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[“”\"()\[\]{}]", " ", text)
    text = re.sub(r"[-—–]", " ", text)
    text = re.sub(r"[^a-záàâãéêíóôõúç\s\.\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_sentences(sentences: list[str]) -> list[list[str]]:
    corpus = []

    for sentence in sentences:
        tokens = sentence.split()
        tokens = [
            tok for tok in tokens
            if len(tok) > 2 and tok not in STOPWORDS
        ]

        if len(tokens) >= 3:
            corpus.append(tokens)

    return corpus


def build_corpus() -> list[list[str]]:
    all_sentences = []

    for path in FILES:
        if not path.exists():
            print(f"[ERRO] Ficheiro não encontrado: {path}")
            continue

        raw_text = read_text(path)
        clean = clean_text(raw_text, path.name)
        sentences = split_sentences(clean)
        tokenized = tokenize_sentences(sentences)

        all_sentences.extend(tokenized)

        print(f"{path.name}:")
        print(f"  frases: {len(sentences)}")
        print(f"  frases tokenizadas aproveitadas: {len(tokenized)}")

    return all_sentences


def train_model(corpus: list[list[str]]) -> Word2Vec:
    model = Word2Vec(
        sentences=corpus,
        vector_size=120,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=30,
    )
    return model


def safe_most_similar(model: Word2Vec, word: str, topn: int = 8):
    if word not in model.wv:
        return None
    return model.wv.most_similar(word, topn=topn)


def safe_similarity(model: Word2Vec, w1: str, w2: str):
    if w1 not in model.wv or w2 not in model.wv:
        return None
    return model.wv.similarity(w1, w2)


def safe_doesnt_match(model: Word2Vec, words: list[str]):
    valid = [w for w in words if w in model.wv]
    if len(valid) < 3:
        return None
    return model.wv.doesnt_match(valid)


def safe_analogy(model: Word2Vec, positive: list[str], negative: list[str], topn: int = 5):
    needed = positive + negative
    if any(w not in model.wv for w in needed):
        return None
    return model.wv.most_similar(positive=positive, negative=negative, topn=topn)


def save_results(model: Word2Vec):
    results_path = OUTPUT_DIR / "resultados_experiencias.txt"

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== TOP 30 PALAVRAS DO VOCABULÁRIO ===\n")
        f.write(str(model.wv.index_to_key[:30]))
        f.write("\n\n")

        f.write("=== MOST SIMILAR ===\n")
        for word in ["harry", "rony", "hermione", "hagrid", "hogwarts", "dobby", "malfoy"]:
            result = safe_most_similar(model, word)
            f.write(f"\n[{word}]\n")
            if result is None:
                f.write("Palavra não encontrada no vocabulário.\n")
            else:
                for similar_word, score in result:
                    f.write(f"{similar_word}: {score:.4f}\n")

        f.write("\n=== SIMILARITY ===\n")
        similarity_pairs = [
            ("harry", "rony"),
            ("harry", "hermione"),
            ("harry", "malfoy"),
            ("harry", "hagrid"),
            ("rony", "hermione"),
            ("dobby", "harry"),
            ("hogwarts", "hagrid"),
        ]
        for w1, w2 in similarity_pairs:
            score = safe_similarity(model, w1, w2)
            if score is None:
                f.write(f"{w1} ~ {w2}: palavra em falta\n")
            else:
                f.write(f"{w1} ~ {w2}: {score:.4f}\n")

        f.write("\n=== DOESN'T MATCH ===\n")
        groups = [
            ["harry", "rony", "hermione", "vassoura"],
            ["hagrid", "dobby", "malfoy", "hogwarts"],
            ["varinha", "feitiço", "poção", "carro"],
            ["hogwarts", "grifinória", "sonserina", "londres"],
        ]
        for group in groups:
            result = safe_doesnt_match(model, group)
            f.write(f"{group} -> {result}\n")

        f.write("\n=== ANALOGIAS ===\n")
        analogies = [
            (["harry", "rony"], ["hermione"]),
            (["malfoy", "harry"], ["rony"]),
            (["dobby", "harry"], ["hagrid"]),
            (["hogwarts", "expresso"], ["londres"]),
        ]
        for positive, negative in analogies:
            result = safe_analogy(model, positive, negative)
            f.write(f"\npositive={positive}, negative={negative}\n")
            if result is None:
                f.write("Não foi possível calcular.\n")
            else:
                for word, score in result:
                    f.write(f"{word}: {score:.4f}\n")

    print(f"Resultados guardados em: {results_path}")


def main():
    corpus = build_corpus()

    print(f"\nTotal de frases no corpus: {len(corpus)}")

    if not corpus:
        print("Corpus vazio. Verifica os ficheiros em data/.")
        return

    model = train_model(corpus)

    model_path = OUTPUT_DIR / "word2vec_hp.model"
    model.save(str(model_path))

    print(f"Modelo guardado em: {model_path}")
    print(f"Tamanho do vocabulário: {len(model.wv.index_to_key)}")
    print("Top 30 palavras do vocabulário:")
    print(model.wv.index_to_key[:30])

    save_results(model)


if __name__ == "__main__":
    main()