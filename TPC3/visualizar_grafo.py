import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("personagens_final.json", encoding="utf-8") as f:
    personagens = json.load(f)

with open("relacoes_final.json", encoding="utf-8") as f:
    relacoes = json.load(f)

frequencias = {p["personagem"]: p["frequencia"] for p in personagens}

GRUPOS = {
    "Harry Potter":        "protagonistas",
    "Rony Weasley":        "protagonistas",
    "Hermione Granger":    "protagonistas",
    "Rúbeo Hagrid":        "aliados",
    "Alvo Dumbledore":     "aliados",
    "Minerva McGonagall":  "aliados",
    "Olívio Wood":         "aliados",
    "Nicolau Flamel":      "aliados",
    "Neville Longbottom":  "aliados",
    "Severo Snape":        "antagonistas",
    "Draco Malfoy":        "antagonistas",
    "Voldemort":           "antagonistas",
    "Quirrell":            "antagonistas",
    "Crabbe":              "antagonistas",
    "Goyle":               "antagonistas",
    "Vernon Dursley":      "dursleys",
    "Petúnia Dursley":     "dursleys",
    "Dudley Dursley":      "dursleys",
    "Fred Weasley":        "weasleys",
    "Jorge Weasley":       "weasleys",
    "Percy Weasley":       "weasleys",
    "Argus Filch":         "outros",
}

CORES = {
    "protagonistas": "#4C9BE8",
    "aliados":       "#2ECC71",
    "antagonistas":  "#E74C3C",
    "dursleys":      "#E67E22",
    "weasleys":      "#9B59B6",
    "outros":        "#95A5A6",
}


def cor_do_no(nome):
    grupo = GRUPOS.get(nome, "outros")
    return CORES[grupo]


def desenhar_grafo(G, titulo, ficheiro, figsize=(16, 11), k=3.5, seed=42):
    pos = nx.spring_layout(G, seed=seed, k=k)

    node_colors = [cor_do_no(n) for n in G.nodes()]
    node_sizes  = [frequencias.get(n, 10) * 3 for n in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 0.25 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4,
                           edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95, ax=ax)

    # labels ligeiramente deslocados para não sobrepor o nó
    label_pos = {n: (x, y + 0.06) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8,
                            font_color="black", font_weight="bold", ax=ax)

    patches = [mpatches.Patch(color=cor, label=grupo.capitalize())
               for grupo, cor in CORES.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.7)

    ax.set_title(titulo, fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(ficheiro, dpi=150)
    plt.close()
    print(f"Gerado: {ficheiro}")


# Grafo 1: Top 15 personagens
top15 = {p["personagem"] for p in personagens[:15]}

G1 = nx.Graph()
for r in relacoes:
    a, b, peso = r["source"], r["target"], r["weight"]
    if a in top15 and b in top15:
        G1.add_edge(a, b, weight=peso)

desenhar_grafo(
    G1,
    titulo="Rede de Relações — Top 15 Personagens",
    ficheiro="grafo_top15.png",
    k=4.0
)

# Grafo 2: Relações com peso >= 10 
G2 = nx.Graph()
for r in relacoes:
    if r["weight"] >= 10:
        G2.add_edge(r["source"], r["target"], weight=r["weight"])

desenhar_grafo(
    G2,
    titulo="Rede de Relações Fortes (peso ≥ 10)",
    ficheiro="grafo_relacoes_fortes.png",
    k=3.5
)