"""The 05/11 materiality question (docs/juizes-2026-11-05.md) to GPT-OSS-120B on Groq and DeepSeek Flash, both at low
reasoning effort as the Sentinela's scan; same 60 filings, same labels. One row per (item, model): p, latency, tokens."""
import json, random, re, sys, time
from pathlib import Path
import requests

D = Path(sys.argv[1])
env = {k: v.strip().strip("\"'") for k, v in (l.strip().split("=", 1) for l in open(sys.argv[2], encoding="utf-8")
                                                 if "=" in l and not l.startswith("#"))}
PERGUNTA = ("Is this filing new information likely to move the company's share price, beyond routine disclosure? Routine "
            "means scheduled reports and calendars, recurring dividends or interest on equity, routine board or officer "
            "changes, internal reorganizations and debt issuance. Answer only with JSON {\"p\": <probability from 0 to 1 "
            "that it is material>}.")
MODELOS = [("GPT-OSS-120B (Groq)", "https://api.groq.com/openai/v1/chat/completions", env["GROQ_API_KEY"],
            {"model": "openai/gpt-oss-120b", "reasoning_effort": "low"}),
           ("DeepSeek Flash (low)", "https://api.deepseek.com/chat/completions", env["DEEPSEEK_API_KEY"],
            {"model": "deepseek-flash", "thinking": {"type": "enabled"}, "reasoning_effort": "low"})]
itens = json.loads((D / "amostra.json").read_text(encoding="utf-8"))
saida = (D / "respostas.jsonl").open("w", encoding="utf-8")
sorteio = random.Random(0)
for item in itens:
    state = f"Empresa: {item['nome_companhia']}. {item['categoria']}: {item['assunto']}"
    ordem = MODELOS[:]
    sorteio.shuffle(ordem)
    for nome, url, chave, extra in ordem:
        for tentativa in range(3):
            t = time.perf_counter()
            try:
                r = requests.post(url, timeout=90, headers={"Authorization": f"Bearer {chave}"},
                                  json=extra | {"messages": [{"role": "system", "content": PERGUNTA},
                                                             {"role": "user", "content": state}],
                                                "response_format": {"type": "json_object"}, "max_tokens": 2000})
                seg = time.perf_counter() - t
                if r.ok:
                    j = r.json()
                    texto = j["choices"][0]["message"]["content"] or ""
                    u = j.get("usage") or {}
                    linha = {"id": item["id"], "juiz": nome, "p": float(json.loads(re.search(r"\{.*\}", texto, re.S).group(0))["p"]),
                             "s": seg, "tokens": u.get("prompt_tokens"), "saida": u.get("completion_tokens"),
                             "modelo": j.get("model")}
                    break
                linha = {"id": item["id"], "juiz": nome, "erro": f"{r.status_code} {r.text[:80]}", "s": seg}
            except (requests.RequestException, ValueError, AttributeError, KeyError) as e:
                linha = {"id": item["id"], "juiz": nome, "erro": str(e)[:80], "s": time.perf_counter() - t}
            time.sleep(2 * (tentativa + 1))
        saida.write(json.dumps(linha, ensure_ascii=False) + "\n"); saida.flush()
print("done")
