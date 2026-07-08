"""waymarks.png → waymarks.ico (PyInstaller exe 아이콘용)"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "assets" / "waymarks.png"
ICO = ROOT / "assets" / "waymarks.ico"


def main() -> int:
    if not PNG.is_file():
        print(f"아이콘 PNG 없음: {PNG}", file=sys.stderr)
        return 1

    from PIL import Image

    img = Image.open(PNG)
    if img.mode not in ("RGBA", "RGB"):
        img = img.convert("RGBA")
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    img.save(ICO, format="ICO", sizes=sizes)
    print(f"생성: {ICO.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
