"""Visual theme for the Streamlit app: papel picado banner, talavera-tile
background, fonts, and component styles. Pure CSS/SVG - no external images."""
from __future__ import annotations

import base64
import math

TEAL = "#0A8F9E"    # demographics-only model
ROSA = "#D6266F"    # immigration-aware model
INK = "#2B1B17"
MUTED = "#6B5A52"
CREAM = "#F8F3EA"
AZUL = "#34508A"    # talavera blue - interface accents only (never data)


def talavera_band_svg() -> str:
    """One square of a talavera tile frieze (muted cobalt line-art on white)."""
    s, c = 44, 22
    petals = "".join(
        f'<ellipse cx="{c}" cy="{c - 7}" rx="3.2" ry="6.5" transform="rotate({a} {c} {c})"/>' for a in range(0, 360, 90)
    )
    leaves = "".join(
        f'<ellipse cx="{c}" cy="{c - 8}" rx="1.6" ry="4" transform="rotate({a} {c} {c})"/>' for a in range(45, 360, 90)
    )
    corners = "".join(f'<circle cx="{x}" cy="{y}" r="8" fill="none"/>' for x, y in [(0, 0), (s, 0), (0, s), (s, s)])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 {s} {s}">'
        f'<rect width="{s}" height="{s}" fill="#FFFFFF"/>'
        f'<rect x="0.5" y="0.5" width="{s - 1}" height="{s - 1}" fill="none" stroke="{AZUL}" stroke-opacity=".25"/>'
        f'<g fill="none" stroke="{AZUL}" stroke-width="1.3" stroke-opacity=".75">{petals}{corners}</g>'
        f'<g fill="{AZUL}" fill-opacity=".55">{leaves}</g>'
        f'<circle cx="{c}" cy="{c}" r="2.6" fill="#C8553D" fill-opacity=".8"/></svg>'
    )


def talavera_svg() -> str:
    """Very faint, sparse talavera motif for the page background."""
    s, c = 160, 80
    petals = "".join(
        f'<ellipse cx="{c}" cy="{c - 11}" rx="5" ry="10" transform="rotate({a} {c} {c})"/>' for a in range(0, 360, 90)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 {s} {s}">'
        f'<g fill="none" stroke="{AZUL}" stroke-width="1.2" opacity="0.07">{petals}</g></svg>'
    )


def _uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, label, p, div {{
  font-family: 'Nunito', sans-serif;
}}
.stApp {{
  background-color: {CREAM};
  background-image: url("{_uri(talavera_band_svg())}"), url("{_uri(talavera_svg())}");
  background-repeat: repeat-x, repeat;
  background-position: top left, 0 44px;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 4.4rem; max-width: 1180px; }}

.hero h1 {{
  font-family: 'DM Serif Display', Georgia, serif; color: {INK}; font-size: 2.7rem; line-height: 1.1;
  margin: 0; padding: 0; font-weight: 400;
}}
.hero p {{ color: {MUTED}; font-size: 1.05rem; margin: .35rem 0 0 0; }}

.section-title {{
  display: flex; align-items: center; gap: .5rem; font-weight: 800; color: {INK};
  font-size: 1.05rem; margin: .2rem 0 .3rem 0;
}}
.section-title .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.chart-note {{ color: {MUTED}; font-size: .86rem; margin: 0 0 .4rem 0; }}

.st-key-profile_card, .st-key-imm_card, [class*="st-key-chart_"] {{
  background: rgba(255,255,255,.96); border-radius: 16px !important; border: 1px solid #E8DCCB !important;
  box-shadow: 0 4px 18px rgba(60, 40, 20, .06); padding: 1rem 1.1rem !important;
}}
[data-testid="stWidgetLabel"] p {{ font-weight: 700; color: {INK}; }}

.legend {{ display: flex; gap: 1.2rem; flex-wrap: wrap; align-items: center; font-weight: 700;
  color: {INK}; font-size: .95rem; margin: .1rem 0 .3rem 0; }}
.legend span.sw {{ display: inline-block; width: 14px; height: 14px; border-radius: 4px; margin-right: .4rem;
  vertical-align: -2px; }}
.legend-note {{ color: {MUTED}; font-size: .84rem; margin: 0 0 .7rem 0; }}

.ocard {{ background: rgba(255,255,255,.97); border: 1px solid #E8DCCB; border-radius: 16px;
  padding: 1rem 1.1rem .8rem 1.1rem; margin-bottom: .9rem; box-shadow: 0 4px 18px rgba(60,40,20,.06); }}
.ocard .top {{ display: flex; align-items: center; gap: .7rem; }}
.ocard .icon {{ width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; flex: none; }}
.ocard .icon img {{ width: 22px; height: 22px; }}
.ocard .name {{ font-weight: 800; font-size: 1.12rem; color: {INK}; line-height: 1.1; }}
.ocard .item {{ color: {MUTED}; font-size: .88rem; }}
.ocard .body {{ display: grid; grid-template-columns: minmax(170px, 1fr) 1.6fr; gap: 1.2rem;
  align-items: end; margin-top: .7rem; }}
@media (max-width: 640px) {{ .ocard .body {{ grid-template-columns: 1fr; }} }}
.ocard .klabel {{ color: {INK}; font-size: .84rem; font-weight: 700; margin-bottom: .25rem; }}
.ocard .knums {{ display: flex; align-items: baseline; gap: .55rem; flex-wrap: wrap; }}
.ocard .knum {{ font-size: 2rem; font-weight: 800; line-height: 1; }}
.ocard .arrow {{ color: {MUTED}; font-size: 1.2rem; }}
.ocard .delta {{ font-weight: 800; font-size: .9rem; padding: .15rem .5rem; border-radius: 999px; }}

.bars {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; align-items: end; }}
.bars .grp {{ display: flex; flex-direction: column; align-items: stretch; }}
.bars .pair {{ display: flex; gap: 3px; align-items: flex-end; height: 92px; justify-content: center;
  border-bottom: 1.5px solid #CDBBA2; }}
.bars .col {{ display: flex; flex-direction: column; align-items: center; justify-content: flex-end; width: 44%;
  max-width: 30px; }}
.bars .pct {{ font-size: .68rem; font-weight: 800; line-height: 1; margin-bottom: 3px; white-space: nowrap; }}
.bars .bar {{ width: 100%; border-radius: 4px 4px 0 0; }}
.bars .lab {{ text-align: center; font-size: .74rem; color: {MUTED}; margin-top: .3rem; line-height: 1.1;
  white-space: nowrap; }}
.ocard .xtitle {{ text-align: center; font-size: .76rem; font-weight: 700; color: {INK}; margin-top: .35rem; }}

.st-key-party [data-testid="stButtonGroup"] > div {{ flex-wrap: nowrap !important; }}
.st-key-party button {{ padding-left: .5rem !important; padding-right: .5rem !important; min-width: 0 !important; }}
.st-key-party button p {{ font-size: .84rem !important; white-space: nowrap; }}

.stTabs [role="tablist"] {{ gap: .5rem; border: none !important; box-shadow: none !important; }}
.stTabs [data-testid="stTab"] {{ background: rgba(255,255,255,.9); border-radius: 999px; padding: .45rem 1.2rem;
  border: 1px solid #E8DCCB; }}
.stTabs [data-testid="stTab"] p {{ font-weight: 800; color: {INK}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ background: {AZUL}; border-color: {AZUL}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] p {{ color: white !important; }}
.stTabs .react-aria-SelectionIndicator {{ display: none; }}

.stat {{ background: rgba(255,255,255,.96); border: 1px solid #E8DCCB; border-radius: 16px; padding: 1rem 1.1rem;
  height: 140px; box-sizing: border-box; margin-bottom: .9rem; }}
.stat .sub {{ color: {MUTED}; font-size: .84rem; font-weight: 600; margin-top: .15rem; }}
.stat .v {{ font-size: 2.1rem; font-weight: 800; line-height: 1; }}
.stat .l {{ color: {INK}; font-size: .9rem; font-weight: 700; margin-top: .3rem; }}
</style>
"""
