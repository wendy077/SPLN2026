import json
import re
import string
import time
from urllib.parse import urljoin
from typing import Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.atlasdasaude.pt"
INDEX_TEMPLATE = f"{BASE_URL}/doencasaaz/{{}}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
}


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_date(soup: BeautifulSoup) -> Optional[str]:
    page_text = soup.get_text("\n", strip=True)
    m = re.search(r"\b\d{2}/\d{2}/\d{4}(?:\s*-\s*\d{2}:\d{2})?\b", page_text)
    return m.group(0) if m else None


def extract_title(soup: BeautifulSoup) -> str:
    # tenta apanhar o título principal do conteúdo e não o cabeçalho global
    breadcrumb = soup.find(string=re.compile(r"Está aqui"))
    if breadcrumb:
        for h1 in soup.find_all("h1"):
            txt = clean_text(h1.get_text())
            if txt and txt.lower() != "atlas da saúde":
                return txt

    for h1 in soup.find_all("h1"):
        txt = clean_text(h1.get_text())
        if txt and txt.lower() != "atlas da saúde":
            return txt

    return ""


def extract_article(session: requests.Session, article_url: str) -> dict:
    soup = get_soup(session, article_url)

    data = {
        "nome": extract_title(soup),
        "data": extract_date(soup),
        "descricao": "",
        "causas": [],
        "sintomas": [],
        "tratamento": [],
        "url": article_url,
    }

    body_div = soup.find("div", class_="field-name-body")

    # Primeiro tenta o formato "rico"
    if body_div:
        current_section = "descricao"

        for tag in body_div.find_all(["h2", "p", "ul"]):
            if tag.name == "h2":
                heading = clean_text(tag.get_text()).lower()

                if "causa" in heading:
                    current_section = "causas"
                elif "sintoma" in heading:
                    current_section = "sintomas"
                elif "tratamento" in heading:
                    current_section = "tratamento"
                else:
                    current_section = "descricao"

            elif tag.name == "p":
                txt = clean_text(tag.get_text(" ", strip=True))
                if not txt:
                    continue

                if current_section == "descricao":
                    data["descricao"] += (" " if data["descricao"] else "") + txt
                else:
                    data[current_section].append(txt)

            elif tag.name == "ul":
                for li in tag.find_all("li"):
                    txt = clean_text(li.get_text(" ", strip=True))
                    if not txt:
                        continue

                    if current_section == "descricao":
                        data["descricao"] += (" " if data["descricao"] else "") + txt
                    else:
                        data[current_section].append(txt)

    # Se ainda não há descrição, tenta o formato "antigo"
    if not data["descricao"]:
        h1 = None
        for candidate in soup.find_all("h1"):
            txt = clean_text(candidate.get_text())
            if txt and txt.lower() != "atlas da saúde":
                h1 = candidate
                break

        if h1:
            collected = []

            # percorre os elementos que vêm depois do título
            for elem in h1.find_all_next():
                text = clean_text(elem.get_text(" ", strip=True))

                if not text:
                    continue

                # parar quando chegar a blocos que já não pertencem ao corpo
                if text.startswith("Nota:") or text == "Nota:":
                    break
                if text.startswith("Site:") or text == "Site:":
                    break
                if "As informações e conselhos disponibilizados no Atlas da Saúde" in text:
                    break
                if text == data["nome"]:
                    continue
                if text == "Atlas da Saúde":
                    continue
                if text == "Está aqui":
                    continue
                if text == data["data"]:
                    continue

                # evita cabeçalhos de secções do site
                if text in {
                    "Atualização Diária",
                    "Formulário de procura",
                    "Pesquisar",
                    "Pesquisa Avançada",
                }:
                    break

                # só texto corrido
                if elem.name in {"p", "div"}:
                    collected.append(text)

            # remove duplicados consecutivos
            filtered = []
            for t in collected:
                if not filtered or filtered[-1] != t:
                    filtered.append(t)

            if filtered:
                data["descricao"] = " ".join(filtered)

    return data

def scrape_letter(session: requests.Session, letter: str) -> list[dict]:
    url = INDEX_TEMPLATE.format(letter)
    soup = get_soup(session, url)

    results = []
    seen_urls = set()

    for row in soup.find_all("div", class_="views-row"):
        h3 = row.find("h3")
        a = h3.find("a", href=True) if h3 else None
        if not a:
            continue

        nome = clean_text(a.get_text())
        disease_url = urljoin(BASE_URL, a["href"])

        if disease_url in seen_urls:
            continue
        seen_urls.add(disease_url)

        short_desc = ""
        desc_div = row.find("div", class_="views-field-body")
        if desc_div:
            short_desc = clean_text(desc_div.get_text(" ", strip=True))

        try:
            article = extract_article(session, disease_url)
        except Exception as e:
            print(f"[ERRO] artigo {disease_url}: {e}")
            article = {
                "nome": nome,
                "data": None,
                "descricao": "",
                "causas": [],
                "sintomas": [],
                "tratamento": [],
                "url": disease_url,
            }

        article["letra"] = letter.upper()
        article["descricao_pequena"] = short_desc

        if not article["nome"]:
            article["nome"] = nome

        if not article["descricao"] and short_desc:
            article["descricao"] = short_desc

        results.append(article)
        print(f"[{letter.upper()}] {article['nome']}")
        time.sleep(0.5)

    return results


def scrape_all() -> list[dict]:
    session = create_session()
    all_data = []

    for letter in string.ascii_lowercase:
        try:
            print(f"\n--- letra {letter.upper()} ---")
            all_data.extend(scrape_letter(session, letter))
            time.sleep(1)
        except Exception as e:
            print(f"[ERRO] letra {letter.upper()}: {e}")

    return all_data


if __name__ == "__main__":
    data = scrape_all()

    with open("atlas_doencas.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nTotal de doenças recolhidas: {len(data)}")