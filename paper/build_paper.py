"""Render paper/paper_template.md -> paper/paper.md and paper/paper.html.

Placeholders (all resolved from files produced by scripts/tfml_report.py):
  {{n:dotted.path:fmt}}   number from results_summary.json, e.g.
                          {{n:runs.benchmark.headline[stage=S1_filter].mean_dSharpe:+.3f}}
                          list elements are selected with [key=value] (multiple: [a=x,b=y])
  {{pct:dotted.path}}     same, formatted as a percentage
  {{table:NAME}}          paper/tables/NAME.md (optionally {{table:NAME|col1,col2}} to subset columns)
  {{fig:FILE|caption}}    paper/figures/FILE
  {{include:sections/X.md}} splice a section file (resolved first)

An unresolved placeholder is a hard error: the paper can only quote computed results.
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "results_summary.json").read_text())


def resolve(path: str):
    cur = SUMMARY
    for part in re.findall(r"[^.\[\]]+(?:\[[^\]]+\])?", path):
        m = re.match(r"([^\[]+)(?:\[([^\]]+)\])?", part)
        key, sel = m.group(1), m.group(2)
        cur = cur[int(key)] if isinstance(cur, list) else cur[key]
        if sel:
            conds = [c.split("=", 1) for c in sel.split(",")]
            hits = [r for r in cur if all(str(r.get(k)) == v for k, v in conds)]
            if len(hits) != 1:
                raise KeyError(f"{path}: selector [{sel}] matched {len(hits)} rows")
            cur = hits[0]
    return cur


def sub_n(m):
    path, _, fmt = m.group(1).partition(":")
    v = resolve(path)
    return format(v, fmt) if fmt else str(v)


def sub_pct(m):
    return f"{100 * float(resolve(m.group(1))):.0f}%"


def sub_table(m):
    name, _, cols = m.group(1).partition("|")
    if cols:
        df = pd.read_csv(HERE / "tables" / f"{name}.csv")[cols.split(",")]
        return df.to_markdown(index=False, floatfmt=".3f")
    return (HERE / "tables" / f"{name}.md").read_text()


def sub_fig(m):
    f, _, cap = m.group(1).partition("|")
    if not (HERE / "figures" / f).exists():
        raise FileNotFoundError(f)
    return f"![{cap}](figures/{f})\n\n*{cap}*"


def build():
    text = (HERE / "paper_template.md").read_text(encoding="utf-8")
    text = re.sub(r"\{\{include:([^}]+)\}\}", lambda m: (HERE / m.group(1)).read_text(encoding="utf-8"), text)
    text = re.sub(r"\{\{n:([^}]+)\}\}", sub_n, text)
    text = re.sub(r"\{\{pct:([^}]+)\}\}", sub_pct, text)
    text = re.sub(r"\{\{table:([^}]+)\}\}", sub_table, text)
    text = re.sub(r"\{\{fig:([^}]+)\}\}", sub_fig, text)
    left = re.findall(r"\{\{[^}]+\}\}", text)
    if left:
        sys.exit(f"unresolved placeholders: {left[:5]}")
    (HERE / "paper.md").write_text(text, encoding="utf-8")

    import markdown
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])

    def inline(mm):
        p = HERE / mm.group(1)
        return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
    body = re.sub(r'src="(figures/[^"]+)"', inline, body)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML vs Rule-Based Trend Following</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--accent:#2a78d6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f4f3ef;--ink2:#c3c2b7;--rule:#383835;--accent:#3987e5}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f4f3ef;--ink2:#c3c2b7;--rule:#383835;--accent:#3987e5}}
body{{background:var(--bg);color:var(--ink);font:16px/1.6 Georgia,serif;max-width:900px;margin:0 auto;padding:24px 16px}}
h1,h2,h3{{font-family:system-ui,sans-serif;line-height:1.25}} a{{color:var(--accent)}}
table{{border-collapse:collapse;font:12px/1.4 system-ui,sans-serif;display:block;overflow-x:auto;margin:1em 0}}
th,td{{border-bottom:1px solid var(--rule);padding:4px 8px;text-align:right;white-space:nowrap}} th{{color:var(--ink2)}}
td:first-child,th:first-child{{text-align:left}} img{{max-width:100%;background:#fff;border-radius:4px}}
code{{font-size:13px}} blockquote{{color:var(--ink2);border-left:3px solid var(--rule);margin:0;padding-left:12px}}
</style></head><body>{body}</body></html>"""
    (HERE / "paper.html").write_text(html, encoding="utf-8")
    print("paper.md / paper.html written")


if __name__ == "__main__":
    build()
