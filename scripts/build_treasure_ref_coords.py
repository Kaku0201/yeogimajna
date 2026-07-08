"""treasure_refs PNG 파일명 → data/treasure_ref_coords/{zone_id}.json 생성"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.treasure_capture import TreasureCaptureProcessor

REFS_DIR = ROOT / "assets" / "treasure_refs"
OUT_DIR = ROOT / "data" / "treasure_ref_coords"
ZONES_PATH = ROOT / "data" / "zones.json"

COORD_RE = re.compile(r"^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")
COORD_TOL = 0.02


def _load_zones() -> dict[str, dict]:
    with ZONES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    zones: dict[str, dict] = {}
    if isinstance(data, list):
        for zone in data:
            zones[str(zone["id"])] = zone
    else:
        for items in data.values():
            for zone in items:
                zones[str(zone["id"])] = zone
    return zones


def _parse_coords(stem: str) -> tuple[float, float] | None:
    match = COORD_RE.match(stem)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _coord_near(
    x: float, y: float, sx: float, sy: float, tol: float = COORD_TOL
) -> bool:
    return abs(x - sx) < tol and abs(y - sy) < tol


def _party_sizes_for_coord(
    x: float, y: float, zone: dict | None, folder_name: str | None = None
) -> list[int] | None:
    """16-spot 지역: zones.json spots 인덱스로 1인/8인 슬롯 판별"""
    if folder_name == "solo":
        return [1]
    if folder_name == "party8":
        return [8]
    if zone is None:
        return None
    spots = zone.get("spots", [])
    if len(spots) != 16:
        return None

    parties: set[int] = set()
    for index, spot in enumerate(spots):
        treasure = spot.get("treasure", {})
        tx, ty = treasure.get("x"), treasure.get("y")
        if tx is None or ty is None:
            continue
        if _coord_near(x, y, float(tx), float(ty)):
            parties.add(1 if index < 8 else 8)
    if not parties:
        return None
    return sorted(parties)


def _detect_marker_rel(path: Path) -> tuple[float, float] | None:
    """ref PNG에서 X 마커 정규화 좌표 검출"""
    rgb = np.array(Image.open(path).convert("RGB"))
    return TreasureCaptureProcessor._find_treasure_x_marker_rel(rgb)


def build_zone_index(zone_id: str, folder: Path, zone: dict | None) -> dict[str, dict]:
    spot_coords: list[tuple[float, float]] = []
    if zone:
        for spot in zone.get("spots", []):
            treasure = spot.get("treasure", {})
            tx, ty = treasure.get("x"), treasure.get("y")
            if tx is not None and ty is not None:
                spot_coords.append((float(tx), float(ty)))

    image_paths: list[tuple[Path, str | None]] = []
    split_dirs = [p for p in sorted(folder.iterdir()) if p.is_dir() and p.name in ("solo", "party8")]
    if split_dirs:
        for subdir in split_dirs:
            for path in sorted(subdir.iterdir()):
                image_paths.append((path, subdir.name))
    else:
        for path in sorted(folder.iterdir()):
            image_paths.append((path, None))

    index: dict[str, dict] = {}
    for path, folder_name in image_paths:
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        coords = _parse_coords(path.stem)
        if coords is None:
            index_match = re.match(r"^(?:spot[_-]?)?(\d+)$", path.stem, re.IGNORECASE)
            if index_match and spot_coords:
                idx = int(index_match.group(1)) - 1
                if 0 <= idx < len(spot_coords):
                    coords = spot_coords[idx]
        if coords is None:
            print(f"  [skip] 좌표 없음: {path.name}", file=sys.stderr)
            continue
        x, y = coords
        item: dict = {"x": x, "y": y}
        party_sizes = _party_sizes_for_coord(x, y, zone, folder_name=folder_name)
        if party_sizes:
            item["party_sizes"] = party_sizes
        marker_rel = _detect_marker_rel(path)
        if marker_rel is not None:
            item["marker_rx"] = round(marker_rel[0], 4)
            item["marker_ry"] = round(marker_rel[1], 4)
        else:
            item["marker_skip"] = True
        key = f"{folder_name}/{path.name}" if folder_name else path.name
        index[key] = item
    return index


def main() -> int:
    zones = _load_zones()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for expansion_dir in sorted(REFS_DIR.iterdir()):
        if not expansion_dir.is_dir():
            continue
        for zone_dir in sorted(expansion_dir.iterdir()):
            if not zone_dir.is_dir():
                continue
            zone_id = zone_dir.name
            index = build_zone_index(zone_id, zone_dir, zones.get(zone_id))
            if not index:
                continue
            out_path = OUT_DIR / f"{zone_id}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            written += 1
            print(f"{zone_id}: {len(index)}개 → {out_path.relative_to(ROOT)}")

    legacy = REFS_DIR
    for zone_dir in sorted(legacy.iterdir()):
        if not zone_dir.is_dir() or zone_dir.name in zones:
            continue
        if zone_dir.name in ("신생", "창천", "홍련", "칠흑", "효월", "황금"):
            continue
        zone_id = zone_dir.name
        index = build_zone_index(zone_id, zone_dir, zones.get(zone_id))
        if not index:
            continue
        out_path = OUT_DIR / f"{zone_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        written += 1
        print(f"{zone_id}: {len(index)}개 → {out_path.relative_to(ROOT)}")

    print(f"완료: {written}개 지역")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
