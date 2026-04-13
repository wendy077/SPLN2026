# TPC4 - Word2Vec com Harry Potter

## Objetivo

Este trabalho teve como objetivo treinar um modelo Word2Vec a partir de um corpus textual baseado em dois livros da saga Harry Potter e realizar várias experiências sobre relações semânticas entre palavras.

Foram testadas operações típicas deste tipo de modelo, nomeadamente:

- `most_similar`
- `similarity`
- `doesnt_match`
- analogias

Foram também gerados gráficos para apoiar a análise dos resultados.

---

## Corpus utilizado

O corpus foi construído a partir dos seguintes ficheiros fornecidos no repositório da unidade curricular:

- `pedra_filosofal.txt`
- `camara_secreta.txt`

Os textos foram limpos a partir do início do primeiro capítulo, removendo conteúdo introdutório irrelevante no caso de `pedra_filosofal.txt`.

---

## Pré-processamento

O pré-processamento aplicado incluiu:

- conversão para minúsculas
- remoção de pontuação e símbolos desnecessários
- divisão em frases
- tokenização por palavras
- remoção de stopwords simples
- remoção de tokens muito curtos

No final, o corpus ficou com milhares de frases tokenizadas usadas no treino do modelo.

---

## Treino do modelo

O modelo foi treinado com `gensim`, usando Word2Vec com os seguintes parâmetros:

- `vector_size = 120`
- `window = 5`
- `min_count = 2`
- `sg = 1`
- `epochs = 30`

---

## Experiências realizadas

### 1. Most similar

Foram testadas várias palavras do universo Harry Potter.

Alguns resultados interessantes:

- `dobby` aproximou-se de palavras como `castigar`, `doméstico`, `elfo` e `libertado`
- `malfoy` apareceu próximo de `lúcio` e `desdenhoso`
- `hogwarts` surgiu relacionado com `expresso` e `bruxaria`

Estes resultados mostram que o modelo conseguiu captar algumas relações contextuais relevantes do corpus.

---

### 2. Similarity

Foram medidos vários pares de palavras.

Exemplos:

- `harry` / `rony`
- `rony` / `hermione`
- `harry` / `malfoy`

Os valores mais altos surgiram entre personagens que aparecem frequentemente em contextos semelhantes.

---

### 3. Doesn't match

Foram criados grupos de palavras para identificar o elemento menos relacionado.

Exemplos com bons resultados:

- `["harry", "rony", "hermione", "vassoura"]` → `vassoura`
- `["varinha", "feitiço", "poção", "carro"]` → `carro`

Nestes casos, o modelo conseguiu separar corretamente o elemento semanticamente menos ligado ao grupo.

---

### 4. Analogias

Também foram testadas analogias com combinações de palavras.

No entanto, os resultados foram fracos e pouco interpretáveis. Em vários casos, as palavras retornadas estavam mais relacionadas com o contexto narrativo do que com uma relação semântica clara.

Isto sugere que o corpus, apesar de útil para captar proximidade contextual, é relativamente pequeno e pouco diversificado para produzir analogias robustas.

---

## Gráficos

Foram gerados três gráficos:

- gráfico de barras com as palavras mais similares a `dobby`
- gráfico de barras com as palavras mais similares a `malfoy`
- visualização em 2D dos embeddings com PCA

Os gráficos de barras mostram exemplos claros de relações semânticas aprendidas pelo modelo, sendo particularmente evidentes nos casos de `dobby` e `malfoy`.

Já a visualização PCA permite observar algumas proximidades entre palavras do mesmo universo, embora sem separações totalmente nítidas.

No gráfico PCA foram incluídas palavras de diferentes categorias:

- personagens (harry, rony, hermione, malfoy, hagrid, snape)
- locais (hogwarts)
- objetos (varinha, feitiço, poção)
- criaturas (dobby, elfo)

Observa-se alguma proximidade entre palavras da mesma categoria, embora a separação não seja totalmente clara.

---

## Conclusões

O modelo Word2Vec conseguiu aprender algumas relações semânticas interessantes a partir de apenas dois livros da saga Harry Potter.

Os melhores resultados surgiram em palavras com contexto muito característico, como `dobby`, `malfoy` e `hogwarts`, que apresentaram associações coerentes com o seu papel no universo da narrativa. Já palavras mais frequentes e mais gerais, como `harry`, produziram vizinhanças menos claras e mais dependentes do contexto específico em que aparecem.

As experiências de `doesnt_match` revelaram-se bastante eficazes, permitindo identificar corretamente elementos fora de contexto em vários grupos de palavras. Por outro lado, as analogias foram a componente menos conseguida: em muitos casos, os resultados estavam mais relacionados com o contexto narrativo do que com relações semânticas claras.

Este comportamento evidencia uma limitação importante do modelo: o Word2Vec aprende relações com base no contexto em que as palavras aparecem, e não no seu significado real. Assim, a qualidade dos resultados depende fortemente da dimensão, diversidade e qualidade do corpus utilizado.

Neste caso, o uso de apenas dois livros limita a capacidade do modelo, especialmente em tarefas mais exigentes como analogias. Uma possível melhoria seria utilizar um corpus maior, incluindo mais livros da saga, ou aplicar técnicas adicionais de pré-processamento para reduzir ainda mais o ruído textual.

---

## Como executar

Instalar dependências:

```bash
pip install -r requirements.txt
```

Treinar o modelo:

```bash
python src/tpc4_word2vec.py
```

Gerar gráficos:

```bash
python src/graficos_word2vec.py
```