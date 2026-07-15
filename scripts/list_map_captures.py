"""assets/maps/{확장팩}/{zone_id}.png ↔ zones.json 커버리지 점검"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.services.coordinate_service import CoordinateService


def main() -> None:
    cs = CoordinateService()
    zones = cs.zones
    missing: list[str] = []
    wrong_size: list[str] = []
    extra: list[str] = []

    zone_ids = {z["id"] for z in zones}
    seen_paths: set[Path] = set()

    for zone in zones:
        zid = zone["id"]
        path = cs.get_detail_map_path(zid)
        if not path.is_file():
            missing.append(f"  {zid} ({zone.get('name_ko', '?')})")
            continue
        seen_paths.add(path.resolve())
        try:
            with Image.open(path) as img:
                w, h = img.size
            if (w, h) != (1627, 1627):
                wrong_size.append(f"  {path.relative_to(cs.maps_dir)} → {w}×{h}")
        except OSError as exc:
            wrong_size.append(f"  {path.relative_to(cs.maps_dir)} → 읽기 실패: {exc}")

    for expansion in cs.EXPANSION_FOLDERS:
        folder = cs.maps_dir / expansion
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.png")):
            if path.resolve() in seen_paths:
                continue
            stem = path.stem
            if stem in zone_ids:
                continue
            extra.append(f"  {path.relative_to(cs.maps_dir)}")

    print(f"zones {len(zones)}개, 지도 있음 {len(zones) - len(missing)}개")
    if missing:
        print("지도 없는 지역:")
        print("\n".join(missing))
    if wrong_size:
        print("해상도 1627×1627 아님:")
        print("\n".join(wrong_size))
    if extra:
        print("zones에 없는 추가 파일:")
        print("\n".join(extra))
    if not missing and not wrong_size and not extra:
        print("모든 지역 지도 OK (1627×1627)")


if __name__ == "__main__":
    main()
