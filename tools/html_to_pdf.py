"""Print a self-contained HTML report to PDF with headless Chromium.

The reports draw their charts with JavaScript (inline SVG), so a browser
engine is needed; this loads the page, lets the scripts run, forces the light
theme, opens collapsed <details> data tables, hides hover tooltips, and prints
A4 with page numbers and a bookmark outline.

Usage:
    python tools/html_to_pdf.py docs/report.html [out.pdf]

Setup (not a GARNET dependency; Python >= 3.9):
    python3.12 -m venv pwenv && pwenv/bin/pip install playwright
    pwenv/bin/playwright install chromium-headless-shell
Behind a TLS-intercepting proxy, point Node at the system CA bundle for the
browser download: NODE_EXTRA_CA_CERTS=/etc/ssl/ca-bundle.pem.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PRINT_CSS = """
:root, :root[data-theme] { --bg: #ffffff; }
html, body { background: #ffffff !important; }
body { font-size: 10.5pt; line-height: 1.5; }
/* combining marks (D-tilde) are missing from the default serif */
.pdf-mark { font-family: "Latin Modern Roman", serif; }
figure img { max-height: 105mm; width: auto; }
main { max-width: none; padding: 0; }
figure.wide { margin-left: 0; margin-right: 0; }
.eyebrow, header .eyebrow { margin-top: 1.6rem; }
footer { margin-top: 1rem; padding-top: 0.5rem; break-before: avoid; }
.chart .tip, details summary { display: none !important; }
.tablewrap, .eq, pre { overflow: visible !important; }
figure, table, pre, .tiles, .note, nav.toc, .eq { break-inside: avoid; }
tr { break-inside: avoid; }
h1, h2, h3, .eyebrow { break-after: avoid; }
a { text-decoration: none; }
"""

FOOTER = ('<div style="width:100%;font-size:7.5pt;color:#7c868e;'
          'font-family:Georgia,serif;text-align:center;">'
          '<span class="title"></span> &mdash; <span class="pageNumber"></span>'
          ' / <span class="totalPages"></span></div>')


def github_base(src: Path, branch: str = "main") -> str | None:
    """https://github.com/OWNER/REPO/blob/BRANCH/<dir of src>/ from the origin remote."""
    try:
        repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                            cwd=src.parent, text=True).strip())
        url = subprocess.check_output(["git", "remote", "get-url", "origin"],
                                      cwd=repo, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if not m:
        return None
    rel = src.resolve().parent.relative_to(repo).as_posix()
    return f"https://github.com/{m.group(1)}/blob/{branch}/{rel}/"


def html_to_pdf(src: Path, out: Path, base_url: str | None = None) -> None:
    base_url = base_url or github_base(src)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.emulate_media(media="print", color_scheme="light")
        page.goto(src.resolve().as_uri(), wait_until="load")
        page.evaluate("""(base) => {
            document.documentElement.setAttribute('data-theme', 'light');
            // relative links would print as local file:// paths
            document.querySelectorAll('a[href]').forEach(a => {
                const h = a.getAttribute('href');
                if (base && !/^([a-z]+:|#)/i.test(h)) a.href = new URL(h, base).href;
            });
            document.querySelectorAll('details').forEach(d => d.open = true);
            // glyphs the print fonts lack: superscript characters become <sup>,
            // base letter + combining mark goes to a font that has both
            const SUP = {'\\u2070':'0','\\u00b9':'1','\\u00b2':'2','\\u00b3':'3',
                '\\u2074':'4','\\u2075':'5','\\u2076':'6','\\u2077':'7','\\u2078':'8',
                '\\u2079':'9','\\u207a':'+','\\u207b':'\\u2212'};
            const runs = /[\\u2070\\u00b9\\u00b2\\u00b3\\u2074-\\u207b]+|\\S[\\u0300-\\u036f]+/g;
            const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            const nodes = [];
            while (walk.nextNode()) {
                const t = walk.currentNode;
                if (t.parentElement.closest('script, style, svg')) continue;
                runs.lastIndex = 0;
                if (runs.test(t.data)) nodes.push(t);
            }
            for (const t of nodes) {
                const frag = document.createDocumentFragment();
                let last = 0;
                for (const m of t.data.matchAll(runs)) {
                    frag.append(t.data.slice(last, m.index));
                    let el;
                    if (/[\\u0300-\\u036f]/.test(m[0])) {
                        el = document.createElement('span');
                        el.className = 'pdf-mark'; el.textContent = m[0];
                    } else {
                        el = document.createElement('sup');
                        el.textContent = [...m[0]].map(c => SUP[c]).join('');
                    }
                    frag.append(el);
                    last = m.index + m[0].length;
                }
                frag.append(t.data.slice(last));
                t.replaceWith(frag);
            }
        }""", base_url)
        page.add_style_tag(content=PRINT_CSS)
        page.wait_for_timeout(300)
        page.pdf(path=str(out), format="A4", print_background=True,
                 margin=dict(top="16mm", bottom="18mm", left="17mm", right="17mm"),
                 display_header_footer=True, header_template="<span></span>",
                 footer_template=FOOTER, tagged=True, outline=True)
        browser.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    html_to_pdf(src, out)
    print(out)
