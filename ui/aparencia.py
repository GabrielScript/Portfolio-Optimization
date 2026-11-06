"""
aparencia.py - The page's appearance (the user's design request of 2026-11-06): royal blue, graphite and silver with
chrome shine in proportion, a slow flow and subtle floating light, plus Streamlit's own light and dark.

Streamlit has no API that sets the native theme, so the native light/dark (``.streamlit/config.toml``, switched in the
menu > Settings) stays underneath and writes the text, widgets and dataframes; "Royal cromado" and "Grafite prateado"
repaint only the grounds, the panels, the hairlines and the nameplate on top of it, in a light and a dark variant that
follow the native type. "Padrão do Streamlit" adds nothing. The quote band (``fita``) is the page's signature: the
Composite's prices drifting under the nameplate.

UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

import pandas as pd

ROYAL, GRAFITE, PADRAO = "Royal cromado", "Grafite prateado", "Padrão do Streamlit"
OPCOES = (ROYAL, GRAFITE, PADRAO)
CHAVE = "aparencia"               # the widget key and the URL query parameter that keeps the choice
# one emphasis ramp for numbers (the brief's magnitude ramp): headline figure, price, ticker, change and labels
RAMPA = ("2rem", "1.25rem", "1rem", "0.8125rem")
AJUDA = ("Royal cromado e Grafite prateado seguem o claro ou escuro do Streamlit (menu ⋮ → Settings → Theme). "
         "Padrão do Streamlit mostra as cores nativas, sem efeitos.")

# the picked segment: white on the royal action color (5.5:1) in every appearance, Streamlit's default included
SELETOR_CSS = (f".st-key-{CHAVE} button[aria-checked=\"true\"] {{ background: #3A5FD9 !important; "
               "color: #FFFFFF !important; border-color: #3A5FD9 !important; }"
               f".st-key-{CHAVE} button[aria-checked=\"true\"] * {{ color: #FFFFFF !important; }}")

ALTA = {"dark": "#5BD69A", "light": "#0E6B3F"}
BAIXA = {"dark": "#FF8A7A", "light": "#A3301F"}
# the band's keyboard focus ring: at least 3:1 on every ground of its native type
FOCO_DA_FITA = {"dark": "#8FA8FF", "light": "#2A4FC8"}


@dataclass(frozen=True)
class Paleta:
    """One appearance in one native type. ``fundo``: the ground's gradient stops; ``painel``: the panel color (opaque,
    as tested; drawn with ``alfa``); ``fio``: the 1px hairline."""
    fundo: tuple[str, ...]
    painel: str
    alfa: float
    aurora: tuple[str, str]
    fio: str
    texto_secundario: str
    foco: str
    selecao: str
    alta: str
    baixa: str


PALETAS: dict[tuple[str, str], Paleta] = {
    (ROYAL, "dark"): Paleta(("#060E27", "#0A1A4D", "#0B1F5C", "#071233"), "#0E1E56", 0.62,
                            ("rgba(58,95,217,0.30)", "rgba(150,170,220,0.14)"), "rgba(201,209,224,0.18)",
                            "#B7C2DA", "#8FA8FF", "rgba(58,95,217,0.45)", ALTA["dark"], BAIXA["dark"]),
    (ROYAL, "light"): Paleta(("#F4F7FC", "#EEF2F9", "#DCE4F3", "#E8EDF7"), "#FFFFFF", 0.78,
                             ("rgba(58,95,217,0.14)", "rgba(120,140,190,0.12)"), "rgba(14,26,58,0.14)",
                             "#44506B", "#2A4FC8", "rgba(58,95,217,0.22)", ALTA["light"],
                             BAIXA["light"]),
    (GRAFITE, "dark"): Paleta(("#0E1013", "#171A1F", "#23272F", "#121418"), "#252930", 0.62,
                              ("rgba(170,180,200,0.14)", "rgba(120,130,150,0.10)"), "rgba(201,209,224,0.18)",
                              "#B8BEC9", "#C9D1E0", "rgba(170,180,200,0.35)", ALTA["dark"],
                              BAIXA["dark"]),
    (GRAFITE, "light"): Paleta(("#F3F4F6", "#E9EBEF", "#DADDE3", "#ECEEF1"), "#FFFFFF", 0.80,
                               ("rgba(120,130,150,0.14)", "rgba(90,100,120,0.10)"), "rgba(20,24,32,0.14)",
                               "#474E5C", "#2A4FC8",
                               "rgba(90,100,120,0.22)", ALTA["light"], BAIXA["light"]),
}


def tipo(valor: str | None) -> str:
    """The native theme type from ``st.context.theme.type``: ``"light"`` or ``"dark"`` (unknown reads as dark)."""
    return "light" if valor == "light" else "dark"


def paleta(nome: str, tipo_nativo: str) -> Paleta:
    return PALETAS[(nome, tipo(tipo_nativo))]


def _rgba(cor: str, alfa: float) -> str:
    r, g, b = (int(cor.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alfa})"


def css(nome: str, tipo_nativo: str) -> str:
    """The stylesheet of an appearance over the native ``tipo_nativo``; empty for Streamlit's default."""
    if nome not in (ROYAL, GRAFITE):
        return ""
    p = paleta(nome, tipo_nativo)
    painel = _rgba(p.painel, p.alfa)
    paradas = ", ".join(f"{cor} {int(i * 100 / (len(p.fundo) - 1))}%" for i, cor in enumerate(p.fundo))
    sombra = "rgba(2,6,23,0.40)" if tipo(tipo_nativo) == "dark" else "rgba(14,26,58,0.08)"
    borda_acesa = _rgba(p.texto_secundario, 0.42)          # the lit silver edge of a panel: a hairline, never a stripe
    return f"""
.stApp {{ background: linear-gradient(160deg, {paradas}) fixed; }}
.stApp::before {{
  content: ""; position: fixed; inset: -20%; pointer-events: none; z-index: 0;
  background: radial-gradient(40% 35% at 30% 20%, {p.aurora[0]}, transparent 70%),
              radial-gradient(35% 30% at 78% 62%, {p.aurora[1]}, transparent 70%);
  animation: aq-aurora 38s ease-in-out infinite;
}}
@keyframes aq-aurora {{
  0%, 100% {{ transform: translate(-4%, -2%) scale(1); }}
  50% {{ transform: translate(4%, 3%) scale(1.08); }}
}}
[data-testid="stAppViewContainer"], [data-testid="stBottomBlockContainer"] {{ background: transparent; }}
[data-testid="stHeader"] {{ background: {p.fundo[0]}; border-bottom: 1px solid {p.fio}; z-index: 999990; }}
[data-testid="stMain"] {{ position: relative; z-index: 1; }}
[class*="st-key-aq-secao-"] {{
  background: {painel}; border: 1px solid {p.fio} !important; border-top-color: {borda_acesa} !important;
  border-radius: 18px; padding: clamp(18px, 2.4vw, 32px) !important; margin-top: 1.25rem;
  box-shadow: 0 14px 36px {sombra};
}}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ max-width: 78ch; }}
.st-key-aq-nome h1 {{
  font-weight: 800; letter-spacing: -0.025em; font-size: clamp(2.6rem, 5vw, 4rem); line-height: 1.05;
}}
[data-testid="stMetricValue"] {{ font-size: {RAMPA[0]}; font-weight: 700; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{ font-size: {RAMPA[3]}; }}
[data-testid="stMetricDelta"] {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.st-key-aq-nome [data-testid="stCaptionContainer"] {{ color: {p.texto_secundario}; }}
[data-testid="stMarkdownContainer"] hr {{
  border: none; height: 1px; margin: 2.2rem 0;
  background: linear-gradient(90deg, transparent, {p.fio} 12%, {p.fio} 88%, transparent);
}}
[data-testid="stExpander"] details, [data-testid="stMetric"], [data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {{
  background: transparent; border: 1px solid {p.fio}; border-radius: 14px;
}}
[data-testid="stPlotlyChart"], [data-testid="stMetric"] {{ padding: 10px 12px; }}
[data-testid="stVegaLiteChart"] svg, [data-testid="stArrowVegaLiteChart"] svg {{ background-color: transparent !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {p.fio}; border-radius: 12px; overflow: hidden; }}
[data-testid="stCaptionContainer"] {{ color: {p.texto_secundario}; }}
::selection {{ background: {p.selecao}; }}
html, .stApp, [data-testid="stMain"] {{ scrollbar-color: {p.fio} transparent; }}
.stApp :focus-visible {{ outline: 2px solid {p.foco}; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{
  .stApp::before {{ animation: none; }}
}}
"""


def _numero(valor: float, casas: int = 2) -> str:
    """Brazilian format: 1234.5 -> '1.234,50'."""
    return f"{valor:,.{casas}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _item(ticker: str, preco: float, variacao: float, tipo_nativo: str, oculto: bool) -> str:
    partes = [f'<b class="aq-t">{html.escape(ticker)}</b>', f'<span class="aq-p">R$ {_numero(preco)}</span>']
    if not math.isnan(variacao):
        sinal, cor = ("+", ALTA[tipo_nativo]) if variacao >= 0 else ("−", BAIXA[tipo_nativo])
        partes.append(f'<span class="aq-v" style="color:{cor}">{sinal}{_numero(abs(variacao) * 100)}%</span>')
    oculta = ' aria-hidden="true"' if oculto else ""
    return f'<span class="aq-q"{oculta}>{"".join(partes)}</span>'


def fita(tabela: pd.DataFrame, tipo_nativo: str) -> str:
    """The quote band: each ``Ticker`` with its ``Cotação (R$)`` and ``Variação no dia``, drifting right to left in a
    loop (the set is drawn twice; the copy is hidden from screen readers). Paused on hover or focus, still under reduced
    motion. Empty table, no band."""
    if tabela.empty:
        return ""
    t = tipo(tipo_nativo)
    linhas = list(zip(tabela["Ticker"], tabela["Cotação (R$)"].astype(float), tabela["Variação no dia"].astype(float)))
    itens = "".join(_item(tk, p, v, t, False) for tk, p, v in linhas)
    copia = "".join(_item(tk, p, v, t, True) for tk, p, v in linhas)
    fio = PALETAS[(ROYAL, t)].fio                    # the one hairline alpha of the page
    fundo = "rgba(255,255,255,0.03)" if t == "dark" else "rgba(14,26,58,0.035)"
    segundos = max(30, 5 * len(linhas))
    return f"""<style>
.aq-fita {{ overflow: hidden; border-top: 1px solid {fio}; border-bottom: 1px solid {fio}; background: {fundo};
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
          mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); }}
.aq-trilho {{ display: flex; width: max-content; animation: aq-fluxo {segundos}s linear infinite; }}
.aq-fita:hover .aq-trilho, .aq-fita:focus-within .aq-trilho {{ animation-play-state: paused; }}
.aq-fita:focus-visible {{ outline: 2px solid {FOCO_DA_FITA[t]}; outline-offset: -2px; }}
.aq-q {{ display: inline-flex; align-items: baseline; gap: 10px; padding: 12px 24px; border-right: 1px solid {fio};
  font-variant-numeric: tabular-nums; white-space: nowrap; }}
.aq-t {{ font-family: Manrope, Inter, sans-serif; font-weight: 700; font-size: {RAMPA[2]}; }}
.aq-p {{ font-size: {RAMPA[1]}; font-weight: 500; }}
.aq-v {{ font-weight: 600; font-size: {RAMPA[3]}; }}
@keyframes aq-fluxo {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{
  .aq-fita {{ overflow-x: auto; }} .aq-trilho {{ animation: none; }}
}}
</style>
<div class="aq-fita" role="region" tabindex="0" aria-label="Cotações das ações da Composite"><div class="aq-trilho">{itens}{copia}</div></div>"""


def chave_da_secao(nome: str) -> str:
    return f"aq-secao-{nome}"


def painel(nome: str):
    """A section of the page as a lit panel (styled by ``css``; Streamlit's own bordered card in its default look)."""
    import streamlit as st

    return st.container(border=True, key=chave_da_secao(nome))


def escolher() -> str:
    """The picker, top right; the choice lives in the URL (``?aparencia=``) so a reload or a shared link keeps it."""
    import streamlit as st

    return st.segmented_control("Aparência", OPCOES, default=ROYAL, required=True, key=CHAVE, bind="query-params",
                                help=AJUDA, label_visibility="collapsed", wrap=True)   # wraps on a phone


def aplicar(nome: str) -> str:
    """Injects the appearance's stylesheet over the native theme; returns the native type it was drawn for."""
    import streamlit as st

    nativo = tipo(st.context.theme.type)
    st.html(f"<style>{SELETOR_CSS}{css(nome, nativo)}</style>")
    return nativo
