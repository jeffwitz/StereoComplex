from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


class _NavParser(HTMLParser):
    def __init__(self, current_stem: str):
        super().__init__()
        self.current_stem = current_stem
        self.in_side = False
        self.depth = 0
        self.in_link = False
        self.href: str | None = None
        self.text: list[str] = []
        self.items: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attr = dict(attrs)
        if tag == "div" and "wy-menu-vertical" in attr.get("class", ""):
            self.in_side = True
            self.depth = 1
            return
        if not self.in_side:
            return
        if tag == "div":
            self.depth += 1
        if tag == "a" and "reference internal" in attr.get("class", ""):
            self.in_link = True
            self.href = attr.get("href")
            self.text = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_side and self.in_link and tag == "a":
            text = "".join(self.text).strip()
            stem = None
            if self.href == "#":
                stem = self.current_stem
            elif self.href and "#" not in self.href and self.href.endswith(".html"):
                stem = Path(self.href).stem
            if text and stem:
                self.items.append((text, stem))
            self.in_link = False
            self.href = None
            self.text = []
        if self.in_side and tag == "div":
            self.depth -= 1
            if self.depth <= 0:
                self.in_side = False

    def handle_data(self, data: str) -> None:
        if self.in_side and self.in_link:
            self.text.append(data)


def _menu(path: Path) -> list[tuple[str, str]]:
    parser = _NavParser(path.stem)
    parser.feed(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text, stem in parser.items:
        if stem not in seen:
            out.append((text, stem))
            seen.add(stem)
    return out


def main() -> int:
    html_dir = Path("docs/_build/html")
    index = html_dir / "index.html"
    if not index.exists():
        raise SystemExit("docs/_build/html/index.html is missing; build the docs first")

    source_pages = sorted(
        path.stem
        for path in Path("docs").glob("*")
        if path.suffix in {".md", ".rst"} and path.name != "index.rst"
    )
    reference = _menu(index)
    reference_stems = [stem for _, stem in reference]
    missing_sources = sorted(set(source_pages) - set(reference_stems))
    extra_menu = sorted(set(reference_stems) - set(source_pages))
    failures: list[str] = []

    if missing_sources:
        failures.append(f"source pages missing from menu: {missing_sources}")
    if extra_menu:
        failures.append(f"menu pages without source document: {extra_menu}")

    for path in sorted(html_dir.glob("*.html")):
        current = _menu(path)
        if current != reference:
            failures.append(f"{path.name} has a menu different from index.html")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"docs nav OK: {len(reference)} menu entries checked across {len(list(html_dir.glob('*.html')))} HTML pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
