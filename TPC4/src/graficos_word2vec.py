from pathlib import Path
from xml.parsers.expat import model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "word2vec_hp.model"


def load_model():
    return Word2Vec.load(str(MODEL_PATH))


def plot_most_similar_bar(model, word="dobby", topn=8):
    if word not in model.wv:
        print(f"Palavra '{word}' não encontrada no vocabulário.")
        return

    similar = model.wv.most_similar(word, topn=topn)
    words = [w for w, _ in similar]
    scores = [s for _, s in similar]

    plt.figure(figsize=(10, 6))
    plt.bar(words, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Similaridade")
    plt.title(f"Palavras mais similares a '{word}'")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"most_similar_{word}.png", dpi=300)
    plt.close()

    print(f"Gráfico guardado em: {OUTPUT_DIR / f'most_similar_{word}.png'}")


def plot_pca_words(model, words, filename="pca_palavras.png"):
    valid_words = [w for w in words if w in model.wv]

    if len(valid_words) < 2:
        print("Não há palavras suficientes no vocabulário para PCA.")
        return

    vectors = [model.wv[w] for w in valid_words]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1])

    for i, word in enumerate(valid_words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]))

    plt.title("Visualização PCA dos embeddings")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

    print(f"Gráfico guardado em: {OUTPUT_DIR / filename}")


def main():
    model = load_model()

    plot_most_similar_bar(model, word="dobby", topn=8)
    plot_most_similar_bar(model, word="malfoy", topn=8)

    pca_words = [
        "harry", "rony", "hermione", "hagrid", "dobby", "malfoy",
        "dumbledore", "snape", "hogwarts", "expresso", "bruxaria",
        "varinha", "feitiço", "poção", "elfo", "doméstico"
    ]
    plot_pca_words(model, pca_words, filename="pca_palavras.png")


if __name__ == "__main__":
    main()