# Extração de conceitos de um dicionário médico (XML → JSON)

## Objetivo

Este trabalho consiste em extrair conceitos de um ficheiro **medicina.xml** (gerado através da ferramenta `pdftoxml`) e convertê-los para um **ficheiro JSON estruturado**, preservando a informação linguística presente no dicionário.

Cada conceito deve incluir:

* **id** — identificador único do conceito
* **ga** — termo em galego
* **dom** — domínios do conceito (opcional)
* **sin** — sinónimos (opcional)
* **var** — variantes (opcional)
* **nota** — notas associadas (opcional)
* **pt, es, en, la** — traduções nas respetivas línguas (opcional)

O objetivo final é produzir uma representação estruturada do dicionário que possa ser utilizada em tarefas de **Processamento de Linguagem Natural (PLN)**.

---

## Estrutura do projeto

**parse_medicina_xml.py**
Script principal responsável por ler o XML e gerar o JSON estruturado.

**medicina.xml**
Ficheiro de entrada gerado a partir do PDF original do dicionário.

**medicina_from_xml.json**
Resultado da execução do parser.

**stats_medicina.py**
Script adicional que calcula estatísticas sobre o dicionário extraído.

---

## Estratégia de extração

O ficheiro XML foi gerado automaticamente a partir de um PDF.
Isto significa que o documento **não possui estrutura semântica explícita**, contendo apenas elementos posicionais (`<text>` com coordenadas).

Desta forma, foi necessário reconstruir a estrutura lógica do dicionário a partir do layout do documento.

O processo seguido foi dividido em várias etapas.

---

## 1. Reconstrução de linhas e colunas

Cada elemento `<text>` contém coordenadas:

* `top`
* `left`

Estas coordenadas foram utilizadas para:

* reconstruir linhas de texto
* separar as duas colunas do dicionário
* manter a ordem correta dos elementos

Isto permite aproximar a estrutura original do documento.

---

## 2. Identificação das entradas

Uma nova entrada do dicionário é identificada por um padrão semelhante a:

```
id termo pos
```

por exemplo:

```
4 abdome agudo m
```

A partir desta linha são extraídos:

* identificador (`id`)
* termo galego (`ga`)
* classe gramatical (`pos`)

---

## 3. Extração dos campos

Depois de identificar o início de uma entrada, o restante conteúdo é analisado linha a linha.

São utilizados marcadores presentes no próprio dicionário:

```
SIN.-   sinónimos
VAR.-   variantes
Nota.-  notas explicativas
es      tradução espanhola
pt      tradução portuguesa
en      tradução inglesa
la      tradução latina
```

As listas são separadas utilizando o carácter `;`.

Exemplo:

```
pt abdome agudo; abdômen agudo [Br.]; abdómen agudo [Pt.]
```

é convertido para:

```
"pt": [
  "abdome agudo",
  "abdômen agudo [Br.]",
  "abdómen agudo [Pt.]"
]
```

---

## 4. Remissões

O dicionário contém também entradas remissivas, por exemplo:

```
aberración cromosómica Vid.- anomalía cromosómica
```

Estas remissões são armazenadas separadamente no JSON:

```
"remissoes": {
  "aberración cromosómica": "anomalía cromosómica"
}
```

Desta forma mantém-se a distinção entre **conceitos completos** e **entradas remissivas**.

---

## Estrutura do JSON gerado

O ficheiro final possui a seguinte estrutura:

```
{
  "entries": {
    "1": {
      "id": "1",
      "ga": "abdome agudo",
      "pos": "m",
      "dom": ["Semioloxía"],
      "sin": [...],
      "var": [...],
      "nota": "...",
      "trad": {
        "pt": [...],
        "es": [...],
        "en": [...],
        "la": [...]
      }
    }
  },
  "remissoes": {
    "termo": "alvo"
  }
}
```

O uso de **listas** permite preservar múltiplos valores para:

* traduções
* sinónimos
* variantes
* domínios

---

## Estatísticas do dicionário

O script `stats_medicina.py` permite gerar estatísticas automáticas sobre o dicionário extraído.

São calculados, por exemplo:

* número total de conceitos
* número total de remissões
* conceitos com sinónimos
* conceitos com variantes
* conceitos com nota
* número de traduções por língua
* domínios mais frequentes

Estas estatísticas ajudam a validar a cobertura da extração.

---

## Dificuldades encontradas

A principal dificuldade deste trabalho resulta do facto de o XML ter sido gerado automaticamente a partir de um PDF.

Isto provoca vários problemas:

* fragmentação de linhas
* palavras partidas entre linhas
* ausência de estrutura semântica explícita
* mistura de colunas no XML

Por esse motivo foi necessário desenvolver **heurísticas específicas** para reconstruir a estrutura lógica do dicionário.

---

## Melhorias futuras

Apesar de a extração atual recuperar corretamente a maioria da informação, ainda existem alguns casos de fragmentação, especialmente em traduções.

Por exemplo:

```
"abdômen em"
"tábua [Br.]"
```

que deveria resultar em:

```
"abdômen em tábua [Br.]"
```

Estas situações resultam de quebras de linha no PDF original.

Melhorias futuras poderiam incluir:

* heurísticas adicionais para reconstrução automática de traduções fragmentadas
* validação cruzada entre traduções em diferentes línguas

---

# Conclusão

Neste trabalho foi desenvolvido um **parser capaz de extrair automaticamente conceitos de um dicionário médico a partir de XML gerado por PDF**.

A solução implementada permite:

* reconstruir a estrutura do documento
* identificar entradas do dicionário
* extrair metadados linguísticos
* gerar uma representação estruturada em JSON

