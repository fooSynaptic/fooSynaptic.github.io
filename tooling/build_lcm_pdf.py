#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strip Hexo frontmatter + appendix from LCM post, fix image paths, run pandoc -> PDF."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POST = REPO / "source" / "_posts" / "LCM（Lossless Context Management） 无损上下文管理.md"
OUT_DIR = REPO / "source" / "files" / "lcm"
OUT_PDF = OUT_DIR / "lcm-lossless-context-full-text.pdf"
IMG_SRC = REPO / "source" / "images" / "lcm-lossless-context-management"
IMG_DST = OUT_DIR / "img"


def main() -> None:
    if not POST.is_file():
        print("missing", POST, file=sys.stderr)
        sys.exit(1)
    raw = POST.read_text(encoding="utf-8")
    # drop YAML frontmatter
    if raw.startswith("---"):
        end = raw.find("\n---\n", 3)
        if end != -1:
            raw = raw[end + 5 :]
    # drop appendix (keep main article only for PDF)
    idx = raw.find("\n## 附录：")
    if idx != -1:
        raw = raw[:idx].rstrip() + "\n"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if IMG_DST.is_dir():
        shutil.rmtree(IMG_DST)
    IMG_DST.mkdir(parents=True, exist_ok=True)

    def repl_img(m: re.Match[str]) -> str:
        alt, path = m.group(1), m.group(2)
        # 切勿 path.lstrip("./")：会把 "../images" 误删成 "images"
        rel = path[2:] if path.startswith("./") else path
        if rel.startswith("../images/lcm-lossless-context-management/"):
            fname = Path(rel).name
            src = IMG_SRC / fname
            if src.is_file():
                shutil.copy2(src, IMG_DST / fname)
            return f"![{alt}](img/{fname})"
        return m.group(0)

    raw = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl_img, raw)

    tmp = OUT_DIR / ".build-body.md"
    tmp.write_text(
        "---\ntitle: LCM（Lossless Context Management）无损上下文管理\n---\n\n" + raw,
        encoding="utf-8",
        newline="\n",
    )

    cmd = [
        "pandoc",
        str(tmp),
        "-o",
        str(OUT_PDF),
        "--pdf-engine=wkhtmltopdf",
        "--pdf-engine-opt=--enable-local-file-access",
        "--metadata",
        "title=LCM 无损上下文管理",
    ]
    subprocess.run(cmd, cwd=str(OUT_DIR), check=True)
    tmp.unlink(missing_ok=True)
    print("wrote", OUT_PDF, "size", OUT_PDF.stat().st_size)


if __name__ == "__main__":
    main()
