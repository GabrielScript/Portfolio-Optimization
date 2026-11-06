"""Ask every judge the same materiality question about the 60 sampled filings; one row per (item, judge) with p, latency, tokens, cost."""
import json, random, time, sys
from pathlib import Path
import requests

D = Path(sys.argv[1])
env = dict(l.strip().split("=", 1) for l in open(sys.argv[2], encoding="utf-8") if "=" in l and not l.startswith("#"))
CHAVES = {"liquid": env["LIQUID_API_KEY"].strip("\"'"), "or": env["OPENROUTER_API_KEY"].strip("\"'")}
URLS = {"liquid": "https://api.liquid.ai/decisions/v1/systemone", "or": "https://openrouter.ai/api/alpha/decisions"}
JUIZES = [("Jev 1.14", "or", "typesafe/jev-1.14"), ("d1-ultra", "or", "liquid/d1-ultra"), ("d1.1", "or", "liquid/d1.1"),
          ("d1-ultra (Liquid direto)", "liquid", "d1-ultra"), ("d1.1 (Liquid direto)", "liquid", "d1.1")]
PERGUNTA = {"material": {"type": "noul", "instructions": (
    "Is this filing new information likely to move the company's share price, beyond routine disclosure? Routine means "
    "scheduled reports and calendars, recurring dividends or interest on equity, routine board or officer changes, "
    "internal reorganizations and debt issuance.")}}
itens = json.loads((D / "amostra.json").read_text(encoding="utf-8"))
saida = (D / "respostas.jsonl").open("w", encoding="utf-8")
sorteio = random.Random(0)
for item in itens:
    state = f"Empresa: {item['nome_companhia']}. {item['categoria']}: {item['assunto']}"
    ordem = JUIZES[:]
    sorteio.shuffle(ordem)
    for nome, rota, modelo in ordem:
        for tentativa in range(3):
            t = time.perf_counter()
            try:
                r = requests.post(URLS[rota], timeout=90, headers={"Authorization": f"Bearer {CHAVES[rota]}"},
                                  json={"model": modelo, "state": state, "questions": PERGUNTA})
                seg = time.perf_counter() - t
                if r.ok:
                    j = r.json()
                    linha = {"id": item["id"], "juiz": nome, "p": j["answers"]["material"]["noul"], "s": seg,
                             "tokens": j["usage"]["input_tokens"], "custo": j["usage"].get("cost"), "modelo": j.get("model")}
                    break
                linha = {"id": item["id"], "juiz": nome, "erro": r.status_code, "s": seg}
            except requests.RequestException as e:
                linha = {"id": item["id"], "juiz": nome, "erro": str(e)[:80], "s": time.perf_counter() - t}
            time.sleep(2 * (tentativa + 1))
        saida.write(json.dumps(linha, ensure_ascii=False) + "\n"); saida.flush()
print("done")
