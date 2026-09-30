"""Build notebooks/practice_<module>.ipynb from the PyCells of the module's posts.

`uv run practice-notebook m1 --site ../condados-site`. The code cells are extracted from
the site's MDX, not copied by hand, so a post and its notebook cannot drift: re-run this
after any edit to a cell. The notebook is written with its outputs cleared.

Three cells are added in front, because a notebook is not the page:
- pinned installs (the page runs OpenCV 4.11 in Pyodide; this runs the pip release);
- the module's images, fetched from condados.ai with a named User-Agent (Cloudflare
  refuses Python's default one);
- cv2.imshow redirected to matplotlib, since imshow opens a window in a script and does
  nothing useful in Colab or Jupyter. The page does the same redirect under the hood.
"""

import json
import re
from pathlib import Path

import click

from fundamentals_lab.config import PROJECT_ROOT

MODULES = {
    "m1": {
        "title": "OpenCV in Practice, M1: fix a badly exposed photo",
        "slugs": [
            "opencv-read-image-pixels",
            "opencv-image-histogram",
            "opencv-brightness-contrast",
            "opencv-gamma-correction",
            "opencv-histogram-equalization",
            "opencv-crop-resize-save",
            "fix-badly-exposed-photo-opencv",
        ],
        "credit": 'Frames from Mark Fairchild\'s HDR Photographic Survey, "Luxo Double Checker" '
        "(research and non-commercial use), developed by this repository's `practice-m1`.",
    },
}

CELL = re.compile(r'<PyCell\b((?:[^>"]|"[^"]*")*)>\s*```python[^\n]*\n([\s\S]*?)```\s*</PyCell>')
ATTR = re.compile(r'(\w+)="([^"]*)"')


def _cells(mdx: str):
    for m in CELL.finditer(mdx):
        yield dict(ATTR.findall(m.group(1))), m.group(2).rstrip("\n")


def _src(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build(module: str, site: Path) -> Path:
    spec = MODULES[module]
    cells: list[dict] = []

    def md(text, tags=None):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(text.strip("\n"))})

    def code(text, tags=None):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"tags": tags} if tags else {},
                "outputs": [],
                "source": _src(text.strip("\n")),
            }
        )

    posts, files = [], []
    for slug in spec["slugs"]:
        mdx = (site / "src" / "content" / "blog" / f"{slug}.mdx").read_text()
        title = re.search(r'^title: "(.*)"$', mdx, re.M).group(1)
        found = list(_cells(mdx))
        posts.append((slug, title, found))
        for attrs, _ in found:
            files += [f for f in attrs.get("files", "").split(",") if f and f not in files]

    md(f"""
# {spec["title"]}

Every code block in the posts, in order, so you can run them on your own machine. On the
site they run in the browser; here they run on the OpenCV you install below.

{spec["credit"]}
""")
    code("""
# Pinned so a Colab base-image change cannot silently move the numbers.
%pip install -q 'opencv-python>=5.0.0,<6.0.0' 'numpy>=2' 'matplotlib>=3.8'
""")
    names = ", ".join(repr(f.rsplit("/", 1)[-1]) for f in files)
    code(f"""
import os
import urllib.request

import cv2
import matplotlib.pyplot as plt

print("OpenCV", cv2.__version__)

# The module's images, served by condados.ai. Cloudflare refuses Python's
# default user agent, so the request names itself.
BASE = os.environ.get("PRACTICE_BASE", "https://condados.ai")
UA = {{"User-Agent": "fundamentals-lab/0.1 (+https://github.com/CondadosAI/fundamentals-lab)"}}
for path in {files!r}:
    name = path.rsplit("/", 1)[-1]
    if not os.path.exists(name):
        req = urllib.request.Request(BASE + path, headers=UA)
        open(name, "wb").write(urllib.request.urlopen(req).read())
print("images:", {names})
""")
    code("""
# cv2.imshow opens a window in a script and does nothing useful in a notebook.
# Draw with matplotlib instead (OpenCV is BGR, matplotlib expects RGB).
def _imshow(title, img):
    plt.figure(figsize=(10, 6))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    plt.show()

cv2.imshow = _imshow
cv2.waitKey = lambda delay=0: -1
""")
    for slug, title, found in posts:
        md(f"## {title}\n\nhttps://condados.ai/blog/{slug}")
        for attrs, src in found:
            if attrs.get("check"):
                md(f"**Your turn.** {attrs.get('hint', '')}")
                code(src)
                code(
                    "# Checks your answer; prints 'not yet' until the cell above is done.\n"
                    f"try:\n    ok = bool({attrs['check']})\nexcept Exception:\n    ok = False\n"
                    'print("correct" if ok else "not yet")'
                )
            elif attrs.get("title") == "Break it":
                # Some Break it cells raise on purpose; the tag lets Run All go on past them.
                md("**Break it.** The common mistake, run on purpose. Some raise an error.")
                code(src, tags=["raises-exception"])
            else:
                code(src)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = PROJECT_ROOT / "notebooks" / f"practice_{module}.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    return out


@click.command()
@click.argument("module")
@click.option(
    "--site", type=click.Path(exists=True, path_type=Path), default=Path.home() / "condados-site"
)
def practice_notebook(module: str, site: Path) -> None:
    """Write notebooks/practice_<MODULE>.ipynb from the site's MDX."""
    print(build(module, site))


if __name__ == "__main__":
    practice_notebook()
