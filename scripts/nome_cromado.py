"""
nome_cromado.py - Makes the chrome "Aleph Quant" wordmark (the redesign of 2026-11-06; the finish review asked for a
produced asset instead of CSS gradient text, which read as a grey fade).

Provenance: Manrope (SIL Open Font License, github.com/google/fonts ofl/manrope) at weight 800, rendered with Pillow;
a classic horizon chrome (bright sky, a sharp dark horizon a little under the middle, a silver reflection rising to
the base), a bevel highlight on the top edges and a thin navy outline so it holds on its ground. Two variants: one for
the night grounds, a deeper one for platinum. Output: marca/nome-cromado-escuro.webp and marca/nome-cromado-claro.webp
(lossless, @2x of a ~64 px nameplate).

Usage: python scripts/nome_cromado.py path/to/Manrope.ttf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

TEXTO = "Aleph Quant"
TAMANHO = 150                    # px at @2x
MARGEM = 12
PASTA = Path(__file__).resolve().parents[1] / "marca"

# (position from the top of the letters 0..1, color): sky, horizon, reflection
CROMO = {
    "escuro": [(0.00, "#FFFFFF"), (0.28, "#E3EAF6"), (0.49, "#A9B8D6"), (0.515, "#24335E"), (0.60, "#3E5185"),
               (0.80, "#C3CEE4"), (1.00, "#F4F7FC")],
    "claro": [(0.00, "#DCE3F0"), (0.28, "#9DADCC"), (0.49, "#5A6A90"), (0.515, "#0B1638"), (0.60, "#1E2E5C"),
              (0.80, "#6E80A8"), (1.00, "#AEBBD6")],
}
CONTORNO = {"escuro": (6, 12, 36, 235), "claro": (14, 26, 58, 255)}
BRILHO = {"escuro": 150, "claro": 90}


def _rgb(cor: str) -> np.ndarray:
    return np.array([int(cor[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _gradiente(altura: int, topo: int, base: int, paradas: list[tuple[float, str]]) -> np.ndarray:
    """altura x 3: the chrome stops spread between the letters' top and base rows."""
    t = np.clip((np.arange(altura) - topo) / max(base - topo, 1), 0, 1)
    posicoes = np.array([p for p, _ in paradas])
    cores = np.stack([_rgb(c) for _, c in paradas])
    return np.stack([np.interp(t, posicoes, cores[:, canal]) for canal in range(3)], axis=1)


def wordmark(fonte: Path, variante: str) -> Image.Image:
    letra = ImageFont.truetype(str(fonte), TAMANHO)
    letra.set_variation_by_axes([800])
    caixa = letra.getbbox(TEXTO)
    largura, altura = caixa[2] - caixa[0] + 2 * MARGEM, caixa[3] + 2 * MARGEM
    mascara = Image.new("L", (largura, altura), 0)
    ImageDraw.Draw(mascara).text((MARGEM - caixa[0], MARGEM), TEXTO, font=letra, fill=255)
    maiuscula = letra.getbbox("A")             # the horizon sits on the cap height, not down to the "p"
    topo, base = MARGEM + maiuscula[1], MARGEM + maiuscula[3]
    cor = _gradiente(altura, topo, base, CROMO[variante])
    rgb = np.broadcast_to(cor[:, None, :], (altura, largura, 3)).astype(np.uint8)
    letras = Image.fromarray(rgb, "RGB").convert("RGBA")
    letras.putalpha(mascara)
    # bevel: the top edge of every stroke catches the light
    abaixo = ImageChops.offset(mascara, 0, 3)
    borda = ImageChops.subtract(mascara, abaixo).filter(ImageFilter.GaussianBlur(0.8))
    luz = Image.new("RGBA", (largura, altura), (255, 255, 255, 0))
    luz.putalpha(borda.point(lambda v: v * BRILHO[variante] // 255))
    # a thin outline under everything so the chrome holds on its ground
    contorno = Image.new("RGBA", (largura, altura), CONTORNO[variante])
    contorno.putalpha(mascara.filter(ImageFilter.MaxFilter(5)))
    saida = Image.alpha_composite(Image.alpha_composite(contorno, letras), luz)
    return saida.crop(saida.getbbox())


def main(fonte: Path) -> None:
    for variante in CROMO:
        destino = PASTA / f"nome-cromado-{variante}.webp"
        wordmark(fonte, variante).save(destino, "WEBP", lossless=True)
        print(destino)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
