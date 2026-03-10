import re
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

XML_PATH = "medicina.xml"
OUT_PATH = "medicina_from_xml.json"

# Heurísticas 
COL_SPLIT_X = 330
TOP_TOL = 2
TOP_MIN = 90
TOP_MAX = 840

KNOWN_POS = {"m", "f", "a", "s", "loc", "pl", "sb", "sg"}
LANGS = {"es", "en", "pt", "la"}

LANG_TRAIL_RE = re.compile(r"\s+(es|en|pt|la)\s*$", re.IGNORECASE)

# Inclui [cult.] e mantém o resto
MARKER_ONLY_RE = re.compile(
    r"^\[(Br\.|Pt\.|Am\.|EUA|pop\.|fig\.|pej\.|cult\.|arc\.|col\.)\]$",
    re.IGNORECASE
)

# (sg)/(pl)/... sozinhos
PAREN_ONLY_RE = re.compile(r"^\((sg|pl|sb|sg\.)\)$", re.IGNORECASE)

VID_ITEM_RE = re.compile(r"^\s*Vid\.?-?\s*(.+?)\s*$", re.IGNORECASE)

# "TERMO Vid.- ALVO" numa só linha (mais permissivo)
INLINE_VID_RE = re.compile(r"^(.*?)\s+Vid\.?-?\s*(.+)$", re.IGNORECASE)

SG_ONLY_RE = re.compile(r"^\(?sg\)?$", re.IGNORECASE)

GA_TRAILING_POS_RE = re.compile(r"^(.*)\s+(m|f|a|s|loc|pl|sb|sg)$", re.IGNORECASE)

def norm_spaces(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_semicolons(s: str):
    parts = [norm_spaces(p) for p in s.split(";")]
    return [p for p in parts if p]


def dedup_list(seq):
    out, seen = [], set()
    for x in seq:
        x = norm_spaces(x)
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_header(line: str):
    m = re.match(r"^\s*(\d+)\s+(.*)$", line)
    if not m:
        return None
    cid = m.group(1)
    rest = norm_spaces(m.group(2))

    tokens = rest.split(" ")
    pos = None
    if tokens and tokens[-1] in KNOWN_POS:
        pos = tokens[-1]
        ga = norm_spaces(" ".join(tokens[:-1]))
    else:
        ga = rest

    return cid, ga, pos


def looks_like_header_even_without_bold(line: str) -> bool:
    s = norm_spaces(line)
    return re.match(r"^\d+\s+.+\s+(?:m|f|a|s|loc|pl|sb|sg)$", s) is not None


def parse_header_continuation(line: str):
    s = norm_spaces(line)
    if re.match(r"^\d+", s):
        return None
    m = re.match(r"^(.+?)\s+(m|f|a|s|loc|pl|sb|sg)$", s)
    if not m:
        return None
    ga_extra = norm_spaces(m.group(1))
    pos = m.group(2)
    return ga_extra, pos


def parse_pos_only(line: str):
    s = norm_spaces(line)
    return s if s in KNOWN_POS else None


def is_fieldish_global(s: str) -> bool:
    if s.startswith(("SIN.-", "VAR.-", "Nota.-")):
        return True
    if re.match(r"^(es|en|pt|la)\s+", s):
        return True
    return False


def should_accept_header_followup(it, header_meta) -> bool:
    if not header_meta:
        return False
    if it["page"] != header_meta["page"] or it["col"] != header_meta["col"]:
        return False
    return (it["top"] - header_meta["top"]) <= 30


def _extract_vid_targets(text: str):
    targets = []
    s = text
    while True:
        m = re.search(r"\bVid\.?-?\s*([^;]+)", s, flags=re.IGNORECASE)
        if not m:
            break
        tgt = norm_spaces(m.group(1))
        if tgt:
            targets.append(tgt)
        start, end = m.span()
        s = norm_spaces((s[:start] + " " + s[end:]).strip())
    return s, targets


def looks_like_continuation(s: str) -> bool:
    if not s:
        return False

    s = norm_spaces(s)

    if MARKER_ONLY_RE.match(s) or PAREN_ONLY_RE.match(s):
        return True

    if s.startswith(("[", "(", ",", ".", ":")):
        return True

    # fragmentos muito curtos ou que claramente continuam sintaticamente
    if re.match(r"^(de|do|da|dos|das|e|em|en|a|o|as|os)\b", s, re.IGNORECASE):
        return True

    if re.match(r"^[a-záàâãäåæçéèêëíìîïñóòôõöœúùûüýÿ]", s) and len(s) <= 40:
        return True

    return False


def clean_lang_trailing(item: str) -> str:
    item = norm_spaces(item)
    item = LANG_TRAIL_RE.sub("", item).strip()
    return item


def parse_domains_from_raw(raw: str):
    raw = raw.replace("\xa0", " ")
    parts = [p for p in re.split(r"\s{2,}", raw.strip()) if p.strip()]
    parts = [norm_spaces(p) for p in parts if norm_spaces(p)]
    return parts


def _add_list_items_with_markers(target_list: list, items: list, see: list):
    """
    Para sin/var:
      - limpa trailing lang
      - cola marcadores [..] e (sg)/(pl)/... ao item anterior
      - itens "Vid.- X"/"Vid. X" vão para see
    """
    for it in items:
        it = norm_spaces(it)
        if not it:
            continue

        mv = VID_ITEM_RE.match(it)
        if mv:
            tgt = norm_spaces(mv.group(1))
            if tgt:
                see.append(tgt)
            continue

        # "(sg)" sozinho -> cola ao anterior (ou ignora se não houver anterior)
        if SG_ONLY_RE.match(it):
            if target_list:
                target_list[-1] = norm_spaces(target_list[-1] + " (sg)")
            continue

        if (MARKER_ONLY_RE.match(it) or PAREN_ONLY_RE.match(it)) and target_list:
            target_list[-1] = norm_spaces(target_list[-1] + " " + it)
            continue

        target_list.append(it)

def add_list_continuation(target_list: list, text: str, see: list):
    text, vids = _extract_vid_targets(text)
    see.extend(vids)
    text = norm_spaces(text)
    if not text:
        return

    items = split_semicolons(text)

    if len(items) == 1 and target_list and ";" not in text and looks_like_continuation(items[0]):
        extra = clean_lang_trailing(items[0])

        if MARKER_ONLY_RE.match(extra) or PAREN_ONLY_RE.match(extra):
            target_list[-1] = norm_spaces(target_list[-1] + " " + extra)
            return

        mv = VID_ITEM_RE.match(extra)
        if mv:
            tgt = norm_spaces(mv.group(1))
            if tgt:
                see.append(tgt)
            return

        target_list[-1] = norm_spaces(target_list[-1] + " " + extra)
        return

    _add_list_items_with_markers(target_list, items, see)

def add_trad_items(trad: dict, lang: str, items: list, see: list):
    trad.setdefault(lang, [])
    for it in items:
        it = clean_lang_trailing(it)
        if not it:
            continue

        mv = VID_ITEM_RE.match(it)
        if mv:
            tgt = norm_spaces(mv.group(1))
            if tgt:
                see.append(tgt)
            continue

        if (MARKER_ONLY_RE.match(it) or PAREN_ONLY_RE.match(it)) and trad[lang]:
            trad[lang][-1] = norm_spaces(trad[lang][-1] + " " + it)
            continue

        trad[lang].append(it)


def parse_entry_lines(lines, ga_for_selfcheck: str | None = None):
    res = {"dom": [], "sin": [], "var": [], "trad": {}}
    nota_parts = []
    see = []

    current_field = None
    current_lang = None

    pending_internal_rem_src = None

    def looks_like_free_term_line(s: str) -> bool:
        if not s:
            return False

        s = norm_spaces(s)

        if s.startswith("*"):
            return True

        if re.match(r"^(es|en|pt|la)\s+", s):
            return False

        if s.startswith(("SIN.-", "VAR.-", "Nota.-")):
            return False

        if re.search(r"\bVid\.?-", s, flags=re.IGNORECASE) or re.search(r"\bVid\.", s, flags=re.IGNORECASE):
            return True

        # termo curto/limpo, sem pontuação estrutural
        if len(s) <= 50 and not re.search(r"[;:]", s):
            if re.match(r"^[A-Za-zÁÉÍÓÚÜÑÇàáâãäèéêëìíîïòóôõöùúûüçñ][A-Za-zÁÉÍÓÚÜÑÇàáâãäèéêëìíîïòóôõöùúûüçñ \*\.\-]*$", s):
                return True

        return False

    def is_fieldish_local(s: str) -> bool:
        return is_fieldish_global(s)

    # domínio: preferir primeira linha itálica
    for it in lines:
        s = norm_spaces(it["text"])
        if not s:
            continue
        if it.get("italic"):
            if not re.match(r"^(es|en|pt|la)\s+", s) and not (
                s.startswith("SIN.-") or s.startswith("VAR.-") or s.startswith("Nota.-")
            ):
                raw = it.get("raw_text", it["text"])
                doms = parse_domains_from_raw(raw)
                res["dom"] = doms if doms else [s]
                break

    for it in lines:
        raw_line = it["text"].rstrip("\n")
        if not raw_line.strip():
            continue
        s = norm_spaces(raw_line)

        # pendente + Vid => see e quebra estado
        if pending_internal_rem_src:
            if ("Vid.-" in s) or re.search(r"\bVid\.", s, flags=re.IGNORECASE):
                _, vids = _extract_vid_targets(s)
                see.extend(vids)
                pending_internal_rem_src = None
                current_field = None
                current_lang = None
                continue
            else:
                pending_internal_rem_src = None

        # inline "TERMO Vid.- ALVO"
        m_same = INLINE_VID_RE.match(s)
        if m_same and not is_fieldish_local(s):
            tgt = norm_spaces(m_same.group(2))
            if tgt:
                see.append(tgt)
            current_field = None
            current_lang = None
            continue

        # bold solto => pendente
        if it.get("bold") and (not is_fieldish_local(s)):
            if not re.match(r"^\d+\s+", s) and not (
                ("Vid.-" in s) or re.search(r"\bVid\.", s, flags=re.IGNORECASE)
            ):
                pending_internal_rem_src = s.lstrip("*").strip()
                current_field = None
                current_lang = None
                continue

        # não-bold mas parece termo solto enquanto estamos em tradução => pendente
        if (not it.get("bold")) and current_field == "lang" and current_lang:
            if looks_like_free_term_line(s):
                pending_internal_rem_src = s.lstrip("*").strip()
                current_field = None
                current_lang = None
                continue

        # Vid.* sozinho
        if re.match(r"^Vid\.?-", s, flags=re.IGNORECASE) or re.match(r"^Vid\.", s, flags=re.IGNORECASE):
            _, vids = _extract_vid_targets(s)
            see.extend(vids)
            current_field = None
            current_lang = None
            continue

        # fallback para dom
        if not res["dom"]:
            if not (s.startswith("SIN.-") or s.startswith("VAR.-") or s.startswith("Nota.-")):
                if not re.match(r"^(es|en|pt|la)\s+", s):
                    raw = it.get("raw_text", raw_line)
                    doms = parse_domains_from_raw(raw)
                    res["dom"] = doms if doms else [s]
                    continue

        if s.startswith("SIN.-"):
            current_field = "sin"
            current_lang = None
            content = norm_spaces(s[len("SIN.-"):])
            content, vids = _extract_vid_targets(content)
            see.extend(vids)
            if content:
                _add_list_items_with_markers(res["sin"], split_semicolons(content), see)
            continue

        if s.startswith("VAR.-"):
            current_field = "var"
            current_lang = None
            content = norm_spaces(s[len("VAR.-"):])
            content, vids = _extract_vid_targets(content)
            see.extend(vids)
            if content:
                _add_list_items_with_markers(res["var"], split_semicolons(content), see)
            continue

        if s.startswith("Nota.-"):
            current_field = "nota"
            current_lang = None
            content = norm_spaces(s[len("Nota.-"):])
            content, vids = _extract_vid_targets(content)
            see.extend(vids)
            if content:
                nota_parts.append(clean_lang_trailing(content))
            continue

        mlang = re.match(r"^(es|en|pt|la)\s+(.*)$", s)
        if mlang:
            lang = mlang.group(1)
            current_field = "lang"
            current_lang = lang

            content = norm_spaces(mlang.group(2))
            content, vids = _extract_vid_targets(content)
            see.extend(vids)

            if content:
                items = split_semicolons(content)
                if len(items) == 1 and res["trad"].get(lang) and ";" not in content and looks_like_continuation(items[0]):
                    if looks_like_free_term_line(items[0]):
                        add_trad_items(res["trad"], lang, items, see)
                    else:
                        last = clean_lang_trailing(res["trad"][lang][-1])
                        extra = clean_lang_trailing(items[0])
                        if (MARKER_ONLY_RE.match(extra) or PAREN_ONLY_RE.match(extra)):
                            res["trad"][lang][-1] = norm_spaces(last + " " + extra)
                        else:
                            mv = VID_ITEM_RE.match(extra)
                            if mv:
                                tgt = norm_spaces(mv.group(1))
                                if tgt:
                                    see.append(tgt)
                            else:
                                res["trad"][lang][-1] = norm_spaces(last + " " + extra)
                else:
                    add_trad_items(res["trad"], lang, items, see)
            continue

        # continuação
        if current_field == "sin":
            add_list_continuation(res["sin"], s, see)
        elif current_field == "var":
            add_list_continuation(res["var"], s, see)
        elif current_field == "nota":
            cont, vids = _extract_vid_targets(s)
            see.extend(vids)
            if cont:
                nota_parts.append(clean_lang_trailing(cont))
        elif current_field == "lang" and current_lang:
            cont, vids = _extract_vid_targets(s)
            see.extend(vids)
            if cont:
                items = split_semicolons(cont)
                if len(items) == 1 and res["trad"].get(current_lang) and ";" not in cont and looks_like_continuation(items[0]):
                    if looks_like_free_term_line(items[0]):
                        add_trad_items(res["trad"], current_lang, items, see)
                    else:
                        last = clean_lang_trailing(res["trad"][current_lang][-1])
                        extra = clean_lang_trailing(items[0])
                        if (MARKER_ONLY_RE.match(extra) or PAREN_ONLY_RE.match(extra)):
                            res["trad"][current_lang][-1] = norm_spaces(last + " " + extra)
                        else:
                            mv = VID_ITEM_RE.match(extra)
                            if mv:
                                tgt = norm_spaces(mv.group(1))
                                if tgt:
                                    see.append(tgt)
                            else:
                                res["trad"][current_lang][-1] = norm_spaces(last + " " + extra)
                else:
                    add_trad_items(res["trad"], current_lang, items, see)

    if nota_parts:
        res["nota"] = norm_spaces(" ".join(nota_parts))

    # limpeza final de traduções
    if "trad" in res:
        for lang, arr in list(res["trad"].items()):
            cleaned = []
            for x in arr:
                x = clean_lang_trailing(x)
                if not x:
                    continue
                mv = VID_ITEM_RE.match(x)
                if mv:
                    tgt = norm_spaces(mv.group(1))
                    if tgt:
                        see.append(tgt)
                    continue
                cleaned.append(x)
            res["trad"][lang] = cleaned

        res["trad"] = {k: v for k, v in res["trad"].items() if v}
        if not res["trad"]:
            res.pop("trad", None)

    # dedup refinamento
    if "sin" in res:
        res["sin"] = dedup_list(res["sin"])
        if not res["sin"]:
            res.pop("sin", None)
    if "var" in res:
        res["var"] = dedup_list(res["var"])
        if not res["var"]:
            res.pop("var", None)
    if "dom" in res:
        res["dom"] = dedup_list(res["dom"])
        if not res["dom"]:
            res.pop("dom", None)
    if "trad" in res:
        for lang in list(res["trad"].keys()):
            res["trad"][lang] = dedup_list(res["trad"][lang])
            if not res["trad"][lang]:
                res["trad"].pop(lang, None)
        if not res["trad"]:
            res.pop("trad", None)

    # remover vazios
    if not res.get("sin"):
        res.pop("sin", None)
    if not res.get("var"):
        res.pop("var", None)
    if not res.get("dom"):
        res.pop("dom", None)

    # dedup see + remover auto-referência ao ga
    if see:
        ga_norm = norm_spaces(ga_for_selfcheck or "").casefold()
        dedup, seen_set = [], set()
        for x in see:
            x = norm_spaces(x)
            if not x:
                continue
            if ga_norm and x.casefold() == ga_norm:
                continue
            if x not in seen_set:
                seen_set.add(x)
                dedup.append(x)
        if dedup:
            res["see"] = dedup

    return res


def reconstruct_lines_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pages = defaultdict(lambda: {0: [], 1: []})

    def smart_join(parts_text):
        out = ""
        for piece in parts_text:
            piece = piece.replace("\n", " ")
            piece = piece.strip()
            if not piece:
                continue

            if not out:
                out = piece
                continue

            # não meter espaço antes de pontuação/fechos
            if re.match(r"^[\]\)\.,;:]", piece):
                out += piece
            
            # não meter espaço extra depois de abertura
            elif out.endswith(("(", "[", "/")):
                out += piece
            else:
                out += " " + piece
        return out

    for page in root.findall("page"):
        pnum = int(page.attrib.get("number", "0"))
        for t in page.findall("text"):
            top = int(t.attrib.get("top", "0"))
            left = int(t.attrib.get("left", "0"))

            if top < TOP_MIN or top > TOP_MAX:
                continue

            txt = "".join(t.itertext())
            if not txt:
                continue

            is_bold = t.find("b") is not None
            is_italic = t.find("i") is not None

            col = 0 if left < COL_SPLIT_X else 1
            pages[pnum][col].append((top, left, txt, is_bold, is_italic))

    all_lines = []
    for pnum in sorted(pages.keys()):
        for col in (0, 1):
            items = sorted(pages[pnum][col], key=lambda x: (x[0], x[1]))
            if not items:
                continue

            current_top = None
            current_parts = []

            def flush():
                nonlocal current_top, current_parts
                if not current_parts:
                    return

                raw_text = smart_join([part[2] for part in current_parts])
                bold = any(part[3] for part in current_parts)
                italic = any(part[4] for part in current_parts)

                text = norm_spaces(raw_text)
                if text:
                    all_lines.append({
                        "page": pnum,
                        "col": col,
                        "top": current_top,
                        "text": text,
                        "raw_text": raw_text,
                        "bold": bold,
                        "italic": italic,
                    })

            for top, left, txt, is_bold, is_italic in items:
                if current_top is None:
                    current_top = top
                    current_parts = [(top, left, txt, is_bold, is_italic)]
                    continue

                if abs(top - current_top) <= TOP_TOL:
                    current_parts.append((top, left, txt, is_bold, is_italic))
                else:
                    flush()
                    current_top = top
                    current_parts = [(top, left, txt, is_bold, is_italic)]

            flush()

    return all_lines


def main():
    lines = reconstruct_lines_from_xml(XML_PATH)

    entries = {}
    remissoes = {}

    current_id = None
    current_header = None
    current_header_meta = None
    current_block = []

    pending_rem_src = None  # termo remissivo (bold sem id), só fora de entradas numeradas

    def flush_entry():
        nonlocal current_id, current_header, current_header_meta, current_block

        if current_id and current_header:
            cid, ga, pos = current_header

            # fallback POS extraction
            if pos is None:
                m = GA_TRAILING_POS_RE.match(ga)
                if m:
                    ga = norm_spaces(m.group(1))
                    pos = m.group(2)

            parsed = parse_entry_lines(current_block, ga_for_selfcheck=ga)

            obj = {"id": cid, "ga": ga}

            if pos:
                obj["pos"] = pos

            obj.update(parsed)
            entries[cid] = obj

        current_id = None
        current_header = None
        current_header_meta = None
        current_block = []

    def is_fieldish(s: str) -> bool:
        return is_fieldish_global(s)

    def looks_like_remissive_term(s: str) -> bool:
        if not s:
            return False
        if s.startswith("*"):
            return True
        if len(s) <= 6 and re.match(r"^[A-Za-zÁÉÍÓÚÜÑÇàáâãäèéêëìíîïòóôõöùúûüçñ]+\.?$", s):
            return True
        return False

    for it in lines:
        s = norm_spaces(it["text"])

        # Continuação de header: aceita com/sem bold, desde que colado ao header e antes de campos
        if current_id and current_header and current_header[2] is None and not is_fieldish_global(s) and not current_block:
            if should_accept_header_followup(it, current_header_meta):
                cont = parse_header_continuation(s)
                if cont:
                    ga_extra, pos = cont
                    cid, ga, _ = current_header
                    ga2 = norm_spaces(ga + " " + ga_extra) if ga_extra else ga
                    current_header = (cid, ga2, pos)
                    continue

                pos_only = parse_pos_only(s)
                if pos_only:
                    cid, ga, _ = current_header
                    current_header = (cid, ga, pos_only)
                    continue

                if s and not re.match(r"^\d+\s+", s) and "Vid" not in s:
                    cid, ga, _ = current_header
                    current_header = (cid, norm_spaces(ga + " " + s), None)
                    continue

        if pending_rem_src and ("Vid.-" in s or re.search(r"\bVid\.", s, flags=re.IGNORECASE)) and current_id is None:
            m = re.search(r"\bVid\.?-?\s*(.+)$", s, flags=re.IGNORECASE)
            if m:
                remissoes[pending_rem_src] = norm_spaces(m.group(1))
                pending_rem_src = None
                continue

        hdr = None
        if it.get("bold") or looks_like_header_even_without_bold(s):
            hdr = parse_header(s)

        if hdr:
            flush_entry()
            current_id = hdr[0]
            current_header = hdr
            current_header_meta = {"page": it["page"], "col": it["col"], "top": it["top"]}
            current_block = []
            pending_rem_src = None
            continue

        if current_id is not None and it.get("bold") and not hdr and not is_fieldish(s):
            if looks_like_remissive_term(s):
                flush_entry()
                current_id = None
                current_header = None
                current_header_meta = None
                current_block = []
                pending_rem_src = s.lstrip("*").strip()
                continue

        if current_id is None:
            if it.get("bold") and not hdr and not is_fieldish(s) and not ("Vid.-" in s or "Vid." in s):
                pending_rem_src = s.lstrip("*").strip() if s else None
                continue

            if ("Vid.-" in s or re.search(r"\bVid\.", s, flags=re.IGNORECASE)) and (not hdr) and not is_fieldish(s):
                m2 = re.match(r"^(.*?)\s+Vid\.?-?\s*(.+)$", s, flags=re.IGNORECASE)
                if m2:
                    src = norm_spaces(m2.group(1)).lstrip("*").strip()
                    tgt = norm_spaces(m2.group(2))
                    if src and tgt:
                        remissoes[src] = tgt
                    continue

        if current_id:
            current_block.append(it)

    flush_entry()

    out = {"entries": entries, "remissoes": remissoes}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()