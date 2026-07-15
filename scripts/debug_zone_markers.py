"""지역 상세지도에 spots/에테 좌표 마커를 그려 QC 이미지 생성."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.coordinate_service import CoordinateService


def main() -> int:
    zone_id = sys.argv[1] if len(sys.argv) > 1 else "coerthas_western_highlands"
    cs = CoordinateService()
    zone = cs.get_zone(zone_id)
    if zone is None:
        print(f"zone not found: {zone_id}")
        return 1

    path = cs.get_detail_map_path(zone_id)
    if not path.exists():
        print(f"map missing: {path}")
        return 1

    img = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    ae = (zone.get("aetherytes") or [{}])[0]
    if ae.get("x") is not None:
        px, py = cs.game_to_detail_pixel(zone, float(ae["x"]), float(ae["y"]))
        draw.ellipse((px - 12, py - 12, px + 12, py + 12), outline="red", width=4)
        print(f"ae {ae.get('name_ko')} ({ae['x']}, {ae['y']}) -> ({px}, {py})")

    for i, spot in enumerate(zone.get("spots") or [], start=1):
        t = spot.get("treasure") or {}
        if t.get("x") is None:
            continue
        gx, gy = float(t["x"]), float(t["y"])
        px, py = cs.game_to_detail_pixel(zone, gx, gy)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline="lime", width=2)
        if i <= 5:
            print(f"spot{i} ({gx}, {gy}) -> ({px}, {py})")

    out = ROOT / "debug_qc" / f"{zone_id}_markers.png"
    out.parent.mkdir(exist_ok=True)
    img.save(out)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
