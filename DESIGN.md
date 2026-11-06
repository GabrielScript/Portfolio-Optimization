---
name: Aleph Quant
description: A B3 quote board at night behind royal-blue glass, drawn over Streamlit's native light and dark themes.
colors:
  royal-action: "#3A5FD9"
  on-royal: "#FFFFFF"
  royal-link-dark: "#8FA8FF"
  royal-link-light: "#2A4FC8"
  royal-abyss: "#060E27"
  royal-midnight: "#0B1F5C"
  royal-glass-dark: "rgba(14,30,86,0.62)"
  platinum-ground: "#F4F7FC"
  platinum-mist: "#DCE4F3"
  royal-glass-light: "rgba(255,255,255,0.78)"
  graphite-ground: "#0E1013"
  graphite-slate: "#23272F"
  graphite-glass-dark: "rgba(37,41,48,0.62)"
  graphite-ground-light: "#F3F4F6"
  graphite-glass-light: "rgba(255,255,255,0.80)"
  native-text-dark: "#FAFAFA"
  native-text-light: "#31333F"
  silver-secondary: "#B7C2DA"
  slate-secondary: "#44506B"
  graphite-silver: "#B8BEC9"
  graphite-secondary-light: "#474E5C"
  hairline-dark: "rgba(201,209,224,0.18)"
  hairline-light: "rgba(14,26,58,0.14)"
  up-dark: "#5BD69A"
  up-light: "#0E6B3F"
  down-dark: "#FF8A7A"
  down-light: "#A3301F"
typography:
  display: { fontFamily: "Manrope, Inter, sans-serif", fontSize: "clamp(2.6rem, 5vw, 4rem)", fontWeight: 800, lineHeight: 1.05, letterSpacing: "-0.025em" }
  headline: { fontFamily: "Manrope, Inter, sans-serif", fontWeight: 700 }
  metric-figure: { fontFamily: "Inter, sans-serif", fontSize: "2rem", fontWeight: 700, fontFeature: "tnum" }
  quote-price: { fontFamily: "Inter, sans-serif", fontSize: "1.25rem", fontWeight: 500, fontFeature: "tnum" }
  quote-ticker: { fontFamily: "Manrope, Inter, sans-serif", fontSize: "1rem", fontWeight: 700 }
  body: { fontFamily: "Inter, sans-serif", fontSize: "15px", fontWeight: 400 }
  label: { fontFamily: "Inter, sans-serif", fontSize: "0.8125rem", fontWeight: 600, fontFeature: "tnum" }
  code: { fontFamily: "JetBrains Mono, monospace", fontWeight: 400 }
rounded:
  panel: "18px"
  frame: "14px"
  table: "12px"
spacing:
  page-top: "2.5rem"
  panel-pad: "clamp(18px, 2.4vw, 32px)"
  panel-gap: "1.25rem"
  divider: "2.2rem"
  band-item: "12px 24px"
  frame-pad: "10px 12px"
components:
  section-panel: { backgroundColor: "{colors.royal-glass-dark}", rounded: "{rounded.panel}", padding: "{spacing.panel-pad}" }
  section-panel-light: { backgroundColor: "{colors.royal-glass-light}", rounded: "{rounded.panel}", padding: "{spacing.panel-pad}" }
  picker-selected: { backgroundColor: "{colors.royal-action}", textColor: "{colors.on-royal}" }
  quote-item: { typography: "{typography.quote-price}", padding: "{spacing.band-item}" }
  quote-up: { textColor: "{colors.up-dark}", typography: "{typography.label}" }
  quote-down: { textColor: "{colors.down-dark}", typography: "{typography.label}" }
  metric: { typography: "{typography.metric-figure}", rounded: "{rounded.frame}", padding: "{spacing.frame-pad}" }
  dataframe: { rounded: "{rounded.table}" }
---

# Design System: Aleph Quant

## Overview

**Creative North Star: "The Pregão at Night"**

The page is the B3 trading floor's quote board after dark, seen through royal-blue glass: the Composite's prices drift
in one band under a solid nameplate, and every section below is a lit panel on that floor. Streamlit's native theme
(`.streamlit/config.toml`, light and dark) stays underneath and writes all text, widgets and dataframes; the
appearances in `ui/aparencia.py` repaint only grounds, panels, hairlines and the nameplate. No code API sets the native
theme, so each appearance ships a light and a dark variant and follows whichever type the native menu reports.

Three appearances, picked top right and kept in the URL (`?aparencia=`): **Royal cromado** (default), **Grafite
prateado** (the same system on graphite), and **Padrão do Streamlit** (native colors, no injected stylesheet). In the
native appearance the quote band and the bordered section panels stay, because they carry content, not decoration.
Contrast is WCAG 2.1 AA in every pairing, held by `tests/test_tema.py` and `tests/test_aparencia.py`.

**Key Characteristics:**
- Dark royal or graphite gradient ground (160deg, fixed) with a slow two-spot aurora behind the content.
- Translucent glass panels with one 1px hairline and a lit silver top edge.
- One four-step magnitude ramp for every number; tabular numerals throughout.
- Colour quarantined to the shell, data series and states; prose stays neutral.

## Colors

Night royal and graphite grounds with silver neutrals; one royal action color and a green/coral pair for direction.

### Primary
- **Royal Action** (royal-action): the native `primaryColor` and the picked segment of the appearance picker, always
  with white text (5.5:1). Links use **Royal Link** (dark / light variants), also the dark focus ring.

### Neutral
- **Royal Abyss to Royal Midnight**: the Royal dark ground, four stops (#060E27, #0A1A4D, #0B1F5C, #071233); light
  variant runs **Platinum Ground** to **Platinum Mist**. Graphite runs #0E1013 to **Graphite Slate** and back.
- **Royal / Graphite Glass**: section panels, the panel color drawn at 0.62 alpha on dark, 0.78-0.80 on light.
- **Silver / Slate Secondary**: captions and secondary text per appearance and type; also the source of the lit edge.
- **Hairline** (dark / light): every divider, panel edge, band rule, chart frame, header rule and scrollbar thumb.
- **Native Text**: Streamlit's own text colors, never overridden.

### States
- **Up / Down** (green / coral on dark, deep green / brick on light): day change in the quote band only.

### Named Rules
**The One Hairline Rule.** Every edge on the page is 1px at one alpha: 0.18 on dark, 0.14 on light. The quote band
uses the Royal hairline in every appearance. The only exception is a panel's top edge (see Elevation).

**The Quarantined Colour Rule.** Prose stays in native neutrals. Colour lives in the shell (ground, aurora, picker),
the data series and the up/down states.

**The Ten Slots Rule.** Each native type carries exactly ten `chartCategoricalColors`, one per slot of Streamlit's
Plotly template; fewer and Plotly drops the theme colors. Light: #2A4FC8, #5A6478, #137A4B, #C0392B, #6D3FC0,
#0E7C8C, #8A6A10, #A04B00, #3D4B66, #9C2F6B. Dark: #7B96F0, #C9D1E0, #4CC38A, #F08A76, #B39BF0, #4FC1D6, #D4AF37,
#E39B4B, #9AA6BF, #F2A7C3. Each holds 3:1 on its grounds. Charts never set their own color range.

## Typography

**Display Font:** Manrope (with Inter, sans-serif)
**Body Font:** Inter, 15px base
**Label/Mono Font:** JetBrains Mono for code

**Character:** A geometric, heavy Manrope for names and headings over a quiet Inter that does the reading and counting.

### Hierarchy
- **Display**: the nameplate "Aleph Quant" only, centered under the eagle.
- **Headline**: section titles; native heading weights 800, 700, 700, 600, 600, 600 (h1-h6).
- **Metric figure / Quote price / Quote ticker / Label**: the magnitude ramp, largest to smallest.
- **Body**: prose capped at 78ch.

### Named Rules
**The Magnitude Ramp Rule.** Numbers take one of four sizes: 2rem (headline figure), 1.25rem (price), 1rem
(ticker), 0.8125rem (change and labels). No fifth size; tabular numerals on all of them.

**The Solid Nameplate Rule.** The nameplate is solid type in the native text color. Gradient "chrome" text read as a
grey fade at review and was rejected.

## Layout

One long single column in Streamlit's centered layout, no sidebar, 2.5rem above the first row. Order: picker
(right-aligned row), nameplate block (eagle 104px, title, two caption lines), the full-width quote band, then one
panel per section in story order: sobre, cotacoes, backtest, comparacao, formulas, simulador, opcoes, sentinela,
(eua when enabled), mercado, assessor. Panels are 1.25rem apart with fluid padding; Markdown dividers take 2.2rem.
On a phone the picker wraps to two rows and the band keeps scrolling at full width.

## Elevation & Depth

Depth is glass over a lit ground: translucent panels let the gradient and aurora show through, plus one soft ambient
shadow per panel. Nothing lifts on interaction.

### Shadow Vocabulary
- **Panel ambient** (`box-shadow: 0 14px 36px rgba(2,6,23,0.40)` dark, `rgba(14,26,58,0.08)` light): section panels only.

### Named Rules
**The Still Panel Rule.** Panels have no hover state and never translate. The "chrome" is one lit top edge: the
secondary text color at 0.42 alpha on the 1px border, a hairline, never a stripe or pseudo-element bar.

**The Slow Flow Rule.** Motion is ambient and slow: the aurora drifts on a 38s ease-in-out loop; the band moves
linearly over max(30s, 5s per ticker), pauses on hover and keyboard focus, and both stop under reduced motion.

## Shapes

Soft, nested corners that step down with scale: panels 18px, charts/metrics/expanders 14px, dataframes 12px, native
widgets and buttons at Streamlit's `medium` radius. All borders are hairlines; no fills on inner frames.

## Components

### Appearance picker
- Native segmented control, label hidden, `bind="query-params"`, `wrap=True`. Picked segment: royal fill, white text
  and white border color in every appearance, Padrão included.

### Section panels
- Native bordered containers keyed `aq-secao-<name>`; restyled as glass with the lit top edge, ambient shadow and
  18px corners in Royal and Grafite; plain native bordered cards in Padrão.

### Quote band (signature)
- Full-width strip between hairlines on a 3% (dark) or 3.5% (light) tint, edge-masked at 6%/94%. Each item:
  ticker, "R$ 1.234,56", signed change with a true minus; items divided by hairlines. The set renders twice for the
  loop, the copy `aria-hidden`. Focusable region; scrolls by hand under reduced motion. Empty quotes, no band.

### Charts
- Altair account charts (`ui/graficos.py`): pt-BR locale, R$ axis zoomed off zero, dd/mm daily ticks, drawdown in
  percent from zero, legend at the bottom, 2px lines, 320px tall, theme colors. Plotly equity curves: theme colorway,
  highlighted curve first (first color, 2.8px solid, others 1.5px dashed), transparent paper and plot.
- Vega svgs drop the native black ground; chart frames get the hairline but no padding (it overflowed the panel).

### Tables
- Dataframes stay native (canvas; CSS cannot restyle them) inside a 12px hairline frame. Percent columns are pt-BR
  text, right-aligned, with an em dash where an account has no history for the window.

## Do's and Don'ts

### Do:
- **Do** put every new section in `aparencia.painel("<name>")` in story order.
- **Do** size numbers from the magnitude ramp only (2 / 1.25 / 1 / 0.8125 rem), with tabular numerals.
- **Do** draw edges with the one hairline alpha (0.18 dark / 0.14 light).
- **Do** let charts take the theme colorway; put the highlighted series first.
- **Do** add a light and a dark variant for any new appearance and test it for WCAG AA.

### Don't:
- **Don't** add hover states, lifts or translations to panels.
- **Don't** set gradient or clipped-background text on the nameplate or headings.
- **Don't** override native text, widget or dataframe colors; repaint grounds, panels and hairlines only.
- **Don't** shorten or lengthen a chart colorway from ten.
- **Don't** inject appearance CSS in Padrão do Streamlit beyond the picked-segment rule; the band keeps its own styles.
