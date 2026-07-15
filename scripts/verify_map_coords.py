"""전 지역 상세지도 좌표 변환 점검.

실행 (프로젝트 루트):
    python scripts/verify_map_coords.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.coordinate_service import CoordinateService

MAX_ROUNDTRIP_ERR = 0.02  # 게임 좌표 왕복 허용 오차 (픽셀 양자화)


def load_zones() -> list[dict]:
    path = ROOT / "data" / "zones.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    zones: list[dict] = []
    for expansion, items in data.items():
        for zone in items:
            merged = dict(zone)
            merged.setdefault("expansion", expansion)
            zones.append(merged)
    return zones


def main() -> int:
    cs = CoordinateService()
    zones = load_zones()
    issues: list[str] = []
    ok_count = 0
    missing_map: list[str] = []
    checked: list[str] = []

    print(f"zones.json: {len(zones)}개 지역")
    print("-" * 88)

    for zone in sorted(zones, key=lambda z: str(z.get("id", ""))):
        zone_id = str(zone.get("id", ""))
        name = zone.get("name_ko") or zone.get("name_en") or zone_id
        map_path = cs.get_detail_map_path(zone_id)
        if not map_path.exists():
            missing_map.append(f"{zone_id} ({name})")
            continue

        with Image.open(map_path) as img:
            png_w, png_h = img.size

        json_w = int(zone.get("detail_width", 0) or 0)
        json_h = int(zone.get("detail_height", 0) or 0)
        size_mismatch = json_w and json_h and (json_w != png_w or json_h != png_h)

        aetherytes = zone.get("aetherytes") or []
        if not aetherytes:
            for spot in zone.get("spots") or []:
                ae = spot.get("aetheryte")
                if ae and ae.get("x") is not None:
                    aetherytes.append(ae)
                    break

        if not aetherytes:
            for spot in zone.get("spots") or []:
                tr = spot.get("treasure") or {}
                if tr.get("x") is not None and tr.get("y") is not None:
                    aetherytes.append(
                        {
                            "name_ko": "보물 spot",
                            "x": tr["x"],
                            "y": tr["y"],
                        }
                    )
                    break

        if not aetherytes:
            issues.append(f"[WARN] {zone_id}: 검증용 좌표 없음 (스킵)")
            continue

        ae = aetherytes[0]
        gx = float(ae["x"])
        gy = float(ae["y"])
        ae_name = ae.get("name_ko") or ae.get("name_en") or "?"

        px_i, py_i = cs.game_to_detail_pixel(zone, gx, gy)
        px_f, py_f = cs.game_to_detail_pixel_float(zone, gx, gy)
        rx, ry = cs.pixel_to_game(zone, px_f, py_f, png_w, png_h)

        out_of_bounds = not (0 <= px_i < png_w and 0 <= py_i < png_h)
        roundtrip_err = max(abs(rx - gx), abs(ry - gy))

        checked.append(zone_id)
        flags: list[str] = []
        if out_of_bounds:
            flags.append(f"OOB pixel=({px_i},{py_i})")
        if roundtrip_err > MAX_ROUNDTRIP_ERR:
            flags.append(f"roundtrip Δ={roundtrip_err:.3f}")
        if size_mismatch:
            issues.append(f"[META] {zone_id}: json={json_w}x{json_h} ≠ png={png_w}x{png_h}")

        status = "OK" if not (out_of_bounds or roundtrip_err > MAX_ROUNDTRIP_ERR) else "ISSUE"
        if status == "OK":
            ok_count += 1
        elif out_of_bounds or roundtrip_err > MAX_ROUNDTRIP_ERR:
            issues.append(
                f"[{status}] {zone_id} ({name}): {', '.join(flags)}"
            )

        print(
            f"{status:5s} {zone_id:<32} png={png_w}x{png_h}  "
            f"ae=({gx:.2f},{gy:.2f})→({px_i},{py_i})  rt=({rx:.2f},{ry:.2f})  {ae_name}"
        )

    print("-" * 88)
    print(f"점검: {len(checked)}개  OK: {ok_count}  이슈: {len(issues)}  지도없음: {len(missing_map)}")

    if missing_map:
        print("\n[지도 PNG 없음]")
        for line in missing_map:
            print(f"  - {line}")

    if issues:
        print("\n[이슈 상세]")
        for line in issues:
            print(f"  {line}")

    # json detail_width 일괄 불일치 요약
    mismatches: dict[str, int] = {}
    for zone in zones:
        zone_id = str(zone.get("id", ""))
        map_path = cs.get_detail_map_path(zone_id)
        if not map_path.exists():
            continue
        with Image.open(map_path) as img:
            pw, ph = img.size
        jw = int(zone.get("detail_width", 0) or 0)
        jh = int(zone.get("detail_height", 0) or 0)
        if jw and jh and (jw != pw or jh != ph):
            key = f"json {jw}x{jh} vs png {pw}x{ph}"
            mismatches[key] = mismatches.get(key, 0) + 1

    if mismatches:
        print("\n[zones.json detail_width/height vs PNG]")
        for key, count in sorted(mismatches.items(), key=lambda x: -x[1]):
            print(f"  {count}개 지역: {key}")
        print("  → PNG 실측 우선 사용 중. json 값 2048로 통일 권장.")

    real_issues = [i for i in issues if not i.startswith("[META]")]
    return 1 if any(i.startswith("[ISSUE]") for i in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
