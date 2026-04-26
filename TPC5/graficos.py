import matplotlib.pyplot as plt

# Comparação final entre modelos
modelos = ["Aula9", "spaCy"]
f1_scores = [95.67, 99.36]

plt.figure()
plt.bar(modelos, f1_scores)
plt.ylabel("F1 (%)")
plt.title("Comparação do F1 entre modelos")
plt.ylim(90, 100)
plt.savefig("comparacao_f1.png", bbox_inches="tight")
plt.close()


# Evolução do F1 durante o treino spaCy
epochs = [0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 6, 7, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
ents_f = [0.90, 88.00, 91.36, 93.32, 94.68, 94.62, 95.57, 96.95, 97.68, 98.18, 98.47, 98.56, 98.77, 98.89, 99.04, 99.04, 98.98, 99.30, 99.24, 99.10, 98.98, 99.36, 99.15]

plt.figure()
plt.plot(epochs, ents_f, marker="o")
plt.xlabel("Epoch")
plt.ylabel("ENTS_F (%)")
plt.title("Evolução do F1 no treino spaCy")
plt.ylim(80, 100)
plt.savefig("evolucao_spacy_f1.png", bbox_inches="tight")
plt.close()