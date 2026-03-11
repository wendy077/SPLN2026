# Web Scraper – Atlas da Saúde (Doenças A-Z)

## Objetivo

Este trabalho consiste no desenvolvimento de um **web scraper em Python** que recolhe informação sobre doenças disponíveis no site **Atlas da Saúde**, percorrendo todas as páginas do índice **A-Z**.

O objetivo é construir um **ficheiro JSON estruturado** contendo informação relevante sobre cada doença.

Fonte dos dados:
https://www.atlasdasaude.pt/doencasAaZ

---

## Informação recolhida

Para cada doença são extraídos os seguintes campos:

* **nome** – nome da doença
* **data** – data da publicação/atualização do artigo
* **descricao** – descrição principal da doença
* **causas** – lista de causas identificadas no artigo
* **sintomas** – lista de sintomas
* **tratamento** – lista de tratamentos mencionados
* **descricao_pequena** – resumo da doença presente na página de listagem A-Z
* **letra** – letra do índice a que pertence
* **url** – URL da página da doença

Os dados são guardados no ficheiro:

```
atlas_doencas.json
```

---

## Estratégia de scraping

O scraper segue os seguintes passos:

1. Percorrer todas as letras do alfabeto (`a`–`z`).
2. Para cada letra, aceder à página:

```
https://www.atlasdasaude.pt/doencasaaz/<letra>
```

3. Identificar todas as doenças listadas nessa página.
4. Para cada doença:

   * recolher a descrição curta (`descricao_pequena`) da listagem
   * visitar a página individual da doença
   * extrair informação detalhada do artigo.

---

## Extração do conteúdo dos artigos

Os artigos do Atlas da Saúde podem apresentar **diferentes estruturas de conteúdo**.

### 1. Estrutura com secções explícitas

Alguns artigos apresentam cabeçalhos como:

* Causas
* Sintomas
* Tratamento

Nestes casos o scraper identifica estas secções e extrai o conteúdo correspondente.

### 2. Estruturas com texto corrido 

Outras páginas não possuem estas secções e apresentam apenas **texto contínuo** após o título.

Nesses casos o scraper percorre os elementos seguintes ao título (`h1`) e recolhe os parágrafos relevantes até encontrar elementos que indicam o final do conteúdo (ex.: *Nota*, *Site*, rodapé do Atlas da Saúde).

Este mecanismo funciona como um **fallback**, permitindo extrair a descrição mesmo quando a página não possui uma estrutura com secções claramente identificadas.

---

## Tratamento de casos especiais

Algumas páginas não permitem extrair de forma fiável a descrição longa do artigo devido a diferenças na estrutura HTML.

Nestes casos é utilizado um **fallback**:

```
descricao = descricao_pequena
```

Ou seja, quando não é possível extrair a descrição completa do artigo, utiliza-se o resumo disponível na página de listagem A-Z.

Isto garante que **nenhuma entrada no JSON fica com descrição vazia**.

---

## Robustez do scraper

Para tornar o scraping mais estável foram implementadas algumas medidas:

* utilização de **requests.Session**
* configuração de **retry automático** para erros HTTP
* **User-Agent personalizado**
* pequenos **intervalos entre pedidos** (`sleep`) para evitar sobrecarga do servidor
* remoção de duplicados e limpeza de texto.

---

## Resultado

O scraper gera um ficheiro JSON com:

* **391 doenças**
* cobertura de **todas as letras A-Z**
* **todas as entradas possuem descrição**
* dados estruturados prontos para processamento posterior.

