"""Generate readable references and thin, drift-checked agent entry points."""

from __future__ import annotations

import html
import re
from pathlib import Path

from markdown_it import MarkdownIt


def project_guides(root: str | Path, *, check: bool = False) -> list[str]:
    root = Path(root)
    source = (root / "docs" / "agent-guide.md").read_text()
    content = "<!-- Generated from docs/agent-guide.md. Edit that source, then regenerate. -->\n\n" + source
    changed = []
    for name in ("AGENTS.md", "CLAUDE.md"):
        target = root / name
        if not target.exists() or target.read_text() != content:
            changed.append(name)
            if not check:
                target.write_text(content)
    if check and changed:
        raise ValueError("Generated entry-point drift: " + ", ".join(changed))
    return changed


def build_reference(root: str | Path, destination: str | Path) -> list[str]:
    root, destination = Path(root), Path(destination)
    if destination.resolve() == root.resolve() or root.resolve() in destination.resolve().parents:
        raise ValueError("Generated reference must live outside the source checkout")
    destination.mkdir(parents=True, exist_ok=True)
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    sources = sorted((root / "docs").glob("*.md"))
    names = {s.name for s in sources}
    pages = []
    for source in sources:
        markdown = source.read_text()
        for target in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", markdown):
            if "://" not in target and Path(target).name not in names:
                raise ValueError(f"Broken documentation target in {source.name}: {target}")
        body = parser.render(markdown)
        body = re.sub(r'href="([^":#]+)\.md(#[^"]*)?"',
                      lambda m: 'href="' + m[1] + '.html' + (m[2] or '') + '"', body)
        title = markdown.splitlines()[0].lstrip("# ")
        document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + html.escape(title) + """ — Drummer</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif;line-height:1.65}
body{max-width:76ch;margin:auto;padding:1rem 1.5rem 4rem;background:#fafafa;color:#18181b}
nav{border-bottom:1px solid #71717a;padding:1rem 0}a{color:#1547a0;text-underline-offset:.2em}
a:focus-visible{outline:3px solid #a44b00;outline-offset:4px}h1,h2,h3{line-height:1.25;margin-top:1.6em}
pre{overflow:auto;padding:1rem;background:#eee}code{font-size:.92em;overflow-wrap:anywhere}
table{display:block;overflow:auto;border-collapse:collapse}th,td{text-align:left;border:1px solid #71717a;padding:.5rem}
.skip{display:inline-block;padding:.5rem}footer{margin-top:3rem;border-top:1px solid #71717a}
@media(prefers-color-scheme:dark){body{background:#18181b;color:#fafafa}a{color:#9fc4ff}pre{background:#27272a}}
</style></head><body><a class="skip" href="#content">Skip to content</a>
<nav aria-label="Documentation"><a href="index.html">Drummer reference</a> · <a href="plan.html">Plan</a> ·
<a href="protocol.html">Protocol</a> · <a href="atlas.html">Language atlas</a></nav>
<main id="content">""" + body + """</main><footer>Luke Steuber · Documentation CC BY 4.0</footer></body></html>
"""
        target = destination / (source.stem + ".html")
        target.write_text(document)
        pages.append(str(target))
    return pages
