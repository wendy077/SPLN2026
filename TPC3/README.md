# TPC3 — Extração de personagens e relações em Harry Potter

## Objetivo

Este trabalho tem como objetivo extrair automaticamente as personagens do livro *Harry Potter e a Pedra Filosofal* usando **spaCy** e construir uma rede de relações com base em **coocorrência na mesma frase**.

Duas personagens são consideradas relacionadas quando aparecem na mesma frase. O peso da relação corresponde ao número de frases em que esse par coocorre.

---

## Metodologia

O processo foi dividido em várias etapas:

1. **Extração do texto do PDF**  
   O texto do livro foi extraído a partir do ficheiro PDF com recurso à biblioteca `PyMuPDF`.

2. **Processamento linguístico com spaCy**  
   O texto foi processado com o modelo `pt_core_news_sm` do spaCy para segmentação em frases e reconhecimento de entidades do tipo pessoa (PER).

3. **Reforço da deteção de personagens**  
   Como o reconhecimento automático nem sempre identifica corretamente nomes do universo Harry Potter, foi usado um `EntityRuler` com padrões manuais para personagens principais.

4. **Normalização de nomes**  
   Várias formas do mesmo nome foram unificadas, por exemplo:
   - `Harry`, `Potter` → `Harry Potter`
   - `Draco`, `Malfoy` → `Draco Malfoy`
   - `Minerva` → `Minerva McGonagall`

5. **Construção das relações**  
   Para cada frase, foram identificadas as personagens presentes.  
   Sempre que duas personagens aparecem na mesma frase, é registada uma coocorrência entre elas.

6. **Cálculo de frequências**
   A frequência de cada personagem corresponde ao número de frases em que ocorre, e não ao número total de menções no texto.
   Esta escolha mantém consistência com a unidade usada na construção das relações (frase).

---

## Ficheiros gerados

- `personagens_final.json` — lista de personagens e respetiva frequência (por frase)
- `relacoes_final.json` — todas as relações entre pares de personagens
- `relacoes_fortes.json` — apenas relações com peso ≥ 3 (consideradas mais significativas)

---

## Resultados

O processamento identificou personagens centrais como:

- Harry Potter
- Rony Weasley
- Hermione Granger
- Rúbeo Hagrid
- Severo Snape
- Alvo Dumbledore

As relações mais fortes incluem:

- Harry Potter — Rony Weasley
- Harry Potter — Hermione Granger
- Hermione Granger — Rony Weasley
- Harry Potter — Rúbeo Hagrid

---

## Visualizações

Foram gerados dois grafos complementares para analisar a rede de relações sob perspetivas diferentes.

**Grafo 1 — Top 15 personagens** (`grafo_top15.png`)  
Mostra as 15 personagens mais mencionadas no livro e todas as relações entre elas, incluindo as mais fracas. É útil para ter uma visão geral de quem são os protagonistas e como se interligam. Os nós são coloridos por grupo (protagonistas, aliados, antagonistas, família Dursley, família Weasley) e o seu tamanho é proporcional à frequência de aparição.

**Grafo 2 — Relações fortes** (`grafo_relacoes_fortes.png`)  
Filtra apenas as relações com peso ≥ 10, ou seja, pares de personagens que coocorrem na mesma frase pelo menos 10 vezes. Elimina ligações pouco significativas e permite identificar claramente os eixos relacionais mais relevantes da narrativa.

Em ambos os grafos, a espessura das arestas é proporcional ao peso da relação.

---

## Nota metodológica

Seguindo a definição do enunciado, dois personagens são considerados **amigos** quando
aparecem na mesma frase. O peso da relação corresponde ao número de frases em que esse
par coocorre — quanto maior o peso, mais forte a ligação.

Note-se que esta é uma definição operacional de "amizade" baseada em coocorrência textual,
não uma avaliação subjetiva das relações entre personagens.