import json
from collections import defaultdict
from itertools import combinations

import fitz
import spacy
from spacy.pipeline import EntityRuler


def extrair_texto_pdf(caminho_pdf):
    doc = fitz.open(caminho_pdf)
    return "\n".join(page.get_text() for page in doc)


def criar_nlp():
    nlp = spacy.load("pt_core_news_sm")

    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = [
        {"label": "PER", "pattern": "Harry"},
        {"label": "PER", "pattern": "Harry Potter"},
        {"label": "PER", "pattern": "Potter"},

        {"label": "PER", "pattern": "Rony"},
        {"label": "PER", "pattern": "Rony Weasley"},
        {"label": "PER", "pattern": "Ron"},

        {"label": "PER", "pattern": "Hermione"},
        {"label": "PER", "pattern": "Hermione Granger"},

        {"label": "PER", "pattern": "Hagrid"},
        {"label": "PER", "pattern": "Rúbeo Hagrid"},

        {"label": "PER", "pattern": "Dumbledore"},
        {"label": "PER", "pattern": "Alvo Dumbledore"},
        {"label": "PER", "pattern": "Professor Dumbledore"},

        {"label": "PER", "pattern": "Snape"},
        {"label": "PER", "pattern": "Severo Snape"},
        {"label": "PER", "pattern": "Professor Snape"},

        {"label": "PER", "pattern": "McGonagall"},
        {"label": "PER", "pattern": "Minerva"},
        {"label": "PER", "pattern": "Minerva McGonagall"},
        {"label": "PER", "pattern": "Professora McGonagall"},

        {"label": "PER", "pattern": "Draco"},
        {"label": "PER", "pattern": "Malfoy"},
        {"label": "PER", "pattern": "Draco Malfoy"},

        {"label": "PER", "pattern": "Neville"},
        {"label": "PER", "pattern": "Neville Longbottom"},

        {"label": "PER", "pattern": "Fred"},
        {"label": "PER", "pattern": "Fred Weasley"},
        {"label": "PER", "pattern": "Jorge"},
        {"label": "PER", "pattern": "George"},
        {"label": "PER", "pattern": "Jorge Weasley"},
        {"label": "PER", "pattern": "Percy"},
        {"label": "PER", "pattern": "Percy Weasley"},

        {"label": "PER", "pattern": "Duda"},
        {"label": "PER", "pattern": "Dudley"},
        {"label": "PER", "pattern": "Válter"},
        {"label": "PER", "pattern": "Vernon"},
        {"label": "PER", "pattern": "Petúnia"},
        {"label": "PER", "pattern": "tia Petúnia"},
        {"label": "PER", "pattern": "tio Válter"},

        {"label": "PER", "pattern": "Quirrell"},
        {"label": "PER", "pattern": "Voldemort"},
        {"label": "PER", "pattern": "Crabbe"},
        {"label": "PER", "pattern": "Goyle"},
        {"label": "PER", "pattern": "Filch"},
        {"label": "PER", "pattern": "Olívio"},
        {"label": "PER", "pattern": "Olívio Wood"},
        {"label": "PER", "pattern": "Wood"},
        {"label": "PER", "pattern": "Nicolau Flamel"},
        {"label": "PER", "pattern": "Flamel"},

        {"label": "PER", "pattern": "Seamus"},
        {"label": "PER", "pattern": "Seamus Finnigan"},
        {"label": "PER", "pattern": "Lavender"},
        {"label": "PER", "pattern": "Lavender Brown"},
        {"label": "PER", "pattern": "Lee"},
        {"label": "PER", "pattern": "Lee Jordan"},
        {"label": "PER", "pattern": "Argus"},
        {"label": "PER", "pattern": "Argus Filch"},
        {"label": "PER", "pattern": "Lupin"},
    ]

    ruler.add_patterns(patterns)
    return nlp


MAPA_NOMES = {
    "Harry": "Harry Potter",
    "Harry Potter": "Harry Potter",
    "Potter": "Harry Potter",

    "Rony": "Rony Weasley",
    "Rony Weasley": "Rony Weasley",
    "Ron": "Rony Weasley",

    "Hermione": "Hermione Granger",
    "Hermione Granger": "Hermione Granger",

    "Hagrid": "Rúbeo Hagrid",
    "Rúbeo Hagrid": "Rúbeo Hagrid",

    "Dumbledore": "Alvo Dumbledore",
    "Alvo Dumbledore": "Alvo Dumbledore",
    "Professor Dumbledore": "Alvo Dumbledore",

    "Snape": "Severo Snape",
    "Severo Snape": "Severo Snape",
    "Professor Snape": "Severo Snape",

    "McGonagall": "Minerva McGonagall",
    "Minerva": "Minerva McGonagall",
    "Minerva McGonagall": "Minerva McGonagall",
    "Professora McGonagall": "Minerva McGonagall",

    "Draco": "Draco Malfoy",
    "Malfoy": "Draco Malfoy",
    "Draco Malfoy": "Draco Malfoy",

    "Neville": "Neville Longbottom",
    "Neville Longbottom": "Neville Longbottom",

    "Fred": "Fred Weasley",
    "Fred Weasley": "Fred Weasley",

    "Jorge": "Jorge Weasley",
    "George": "Jorge Weasley",
    "Jorge Weasley": "Jorge Weasley",

    "Percy": "Percy Weasley",
    "Percy Weasley": "Percy Weasley",

    "Duda": "Dudley Dursley",
    "Dudley": "Dudley Dursley",

    "Válter": "Vernon Dursley",
    "Vernon": "Vernon Dursley",
    "tio Válter": "Vernon Dursley",

    "Petúnia": "Petúnia Dursley",
    "tia Petúnia": "Petúnia Dursley",

    "Olívio": "Olívio Wood",
    "Olívio Wood": "Olívio Wood",
    "Wood": "Olívio Wood",

    "Flamel": "Nicolau Flamel",
    "Nicolau Flamel": "Nicolau Flamel",

    "Quirrell": "Quirrell",
    "Voldemort": "Voldemort",
    "Crabbe": "Crabbe",
    "Goyle": "Goyle",
    "Filch": "Argus Filch",

    "Seamus": "Seamus Finnigan",
    "Seamus Finnigan": "Seamus Finnigan",

    "Lavender": "Lavender Brown",
    "Lavender Brown": "Lavender Brown",

    "Lee": "Lee Jordan",
    "Lee Jordan": "Lee Jordan",

    "Argus": "Argus Filch",
    "Argus Filch": "Argus Filch",

    "Lupin": "Lupin",

}


FALSOS_POSITIVOS = {
    "Acho",
    "Capítulo",
    "Página",
    "Professor",
    "Professora",
    "Sr",
    "Sr.",
    "Sra",
    "Sra.",
    "Dursley",
    "Pedro",   
}


def normalizar_nome(nome):
    nome = nome.strip()
    nome = MAPA_NOMES.get(nome, nome)

    if nome in FALSOS_POSITIVOS:
        return None

    if len(nome) <= 1:
        return None

    return nome


def personagens_na_frase(sent):
    personagens = set()

    for ent in sent.ents:
        if ent.label_ == "PER":
            nome = normalizar_nome(ent.text)
            if nome:
                personagens.add(nome)

    return personagens


def main():
    caminho_pdf = "Harry Potter e A Pedra Filosofal.pdf"

    print("A extrair texto do PDF...")
    texto = extrair_texto_pdf(caminho_pdf)

    print("A carregar spaCy + EntityRuler...")
    nlp = criar_nlp()
    nlp.max_length = 2_000_000

    print("A processar texto...")
    doc = nlp(texto)

    frases = list(doc.sents)
    print("Número de frases:", len(frases))

    frequencias = defaultdict(int)
    relacoes = defaultdict(int)

    for sent in frases:
        pers = sorted(personagens_na_frase(sent))

        for p in pers:
            frequencias[p] += 1

        if len(pers) >= 2:
            for a, b in combinations(pers, 2):
                relacoes[(a, b)] += 1

    personagens_ordenadas = sorted(
        frequencias.items(),
        key=lambda x: x[1],
        reverse=True
    )

    relacoes_ordenadas = sorted(
        relacoes.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # filtra relações mais significativas
    relacoes_fortes = [
        ((a, b), peso)
        for (a, b), peso in relacoes_ordenadas
        if peso >= 3
    ]

    print("\nTop 20 personagens:\n")
    for nome, freq in personagens_ordenadas[:20]:
        print(f"{nome}: {freq}")

    print("\nTop 20 relações:\n")
    for (a, b), peso in relacoes_ordenadas[:20]:
        print(f"{a} -- {b}: {peso}")

    with open("personagens_final.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"personagem": nome, "frequencia": freq} for nome, freq in personagens_ordenadas],
            f,
            ensure_ascii=False,
            indent=2
        )

    with open("relacoes_final.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"source": a, "target": b, "weight": peso} for (a, b), peso in relacoes_ordenadas],
            f,
            ensure_ascii=False,
            indent=2
        )

    with open("relacoes_fortes.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"source": a, "target": b, "weight": peso} for (a, b), peso in relacoes_fortes],
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nFicheiros gerados:")
    print("- personagens_final.json")
    print("- relacoes_final.json")
    print("- relacoes_fortes.json")


if __name__ == "__main__":
    main()