import json
from collections import Counter

JSON_PATH = "medicina_from_xml.json"

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

entries = data["entries"]
remissoes = data["remissoes"]

total_entries = len(entries)
total_remissoes = len(remissoes)

lang_counter = Counter()
sin_count = 0
var_count = 0
nota_count = 0
dom_counter = Counter()

for e in entries.values():

    if "sin" in e:
        sin_count += 1

    if "var" in e:
        var_count += 1

    if "nota" in e:
        nota_count += 1

    if "dom" in e:
        for d in e["dom"]:
            dom_counter[d] += 1

    if "trad" in e:
        for lang in e["trad"]:
            lang_counter[lang] += 1

print("==== Estatísticas do Dicionário Médico ====\n")

print("Número de conceitos:", total_entries)
print("Número de remissões:", total_remissoes)

print("\nCampos opcionais:")
print("conceitos com sinónimos:", sin_count)
print("conceitos com variantes:", var_count)
print("conceitos com nota:", nota_count)

print("\nTraduções por língua:")
for lang, n in sorted(lang_counter.items()):
    print(lang, ":", n)

print("\nDomínios mais frequentes:")
for dom, n in dom_counter.most_common(10):
    print(dom, ":", n)