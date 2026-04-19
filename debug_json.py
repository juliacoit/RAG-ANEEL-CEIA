from pathlib import Path
import json

raiz = Path(__file__).resolve().parent
json_path = raiz / "data" / "aneel_vigentes_completo.json"

print(f"Raiz do projeto: {raiz}")
print(f"JSON path: {json_path}")
print(f"JSON existe: {json_path.exists()}")

if json_path.exists():
    with open(json_path, encoding="utf-8") as f:
        registros = json.load(f)
    print(f"Total registros: {len(registros)}")

    downloads = []
    for reg in registros:
        for pdf in reg.get("pdfs") or []:
            if pdf.get("categoria") == "texto_integral":
                downloads.append(pdf.get("arquivo"))

    print(f"PDFs texto_integral encontrados: {len(downloads)}")
    print(f"Amostra: {downloads[:3]}")
