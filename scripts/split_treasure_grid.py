"""보물지도 그리드 이미지 → assets/treasure_refs/{확장팩}/{zone_id}/ 좌표별 PNG 분할"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def find_segments(projection: np.ndarray, min_size: int = 40) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, val in enumerate(projection):
        active = val > 0
        if active and not in_seg:
            start = i
            in_seg = True
        elif not active and in_seg:
            if i - start >= min_size:
                segments.append((start, i))
            in_seg = False
    if in_seg and len(projection) - start >= min_size:
        segments.append((start, len(projection)))
    return segments


def detect_tiles(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """밝은(양피지) 영역으로 타일 bbox 탐지 — (left, top, right, bottom)"""
    rgb = np.array(image.convert("RGB"))
    bright = rgb.mean(axis=2) > 90
    row_proj = bright.sum(axis=1)
    col_proj = bright.sum(axis=0)
    rows = find_segments(row_proj, min_size=50)
    cols = find_segments(col_proj, min_size=50)

    tiles: list[tuple[int, int, int, int]] = []
    for top, bottom in rows:
        for left, right in cols:
            patch = bright[top:bottom, left:right]
            if patch.mean() < 0.15:
                continue
            tiles.append((left, top, right, bottom))

    tiles.sort(key=lambda b: (b[1], b[0]))
    return tiles


def load_zone_info(
    data_dir: Path, zone_id: str
) -> tuple[str, list[tuple[float, float]]]:
    """(확장팩 폴더명, spots 좌표 목록)"""
    path = data_dir / "zones.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        zone = next((z for z in data if z.get("id") == zone_id), None)
        if zone is None:
            raise ValueError(f"zones.json에 지역 없음: {zone_id}")
        expansion = str(zone.get("expansion", ""))
        if not expansion:
            raise ValueError(f"지역 '{zone_id}'에 expansion 필드가 없습니다.")
        spots = _extract_spots(zone)
        return expansion, spots

    for expansion, items in data.items():
        for zone in items:
            if zone.get("id") != zone_id:
                continue
            return str(expansion), _extract_spots(zone)

    raise ValueError(f"zones.json에 지역 없음: {zone_id}")


def _extract_spots(zone: dict) -> list[tuple[float, float]]:
    spots: list[tuple[float, float]] = []
    for spot in zone.get("spots", []):
        t = spot.get("treasure", {})
        x, y = t.get("x"), t.get("y")
        if x is None or y is None:
            continue
        spots.append((float(x), float(y)))
    return spots


def load_zone_spots(data_dir: Path, zone_id: str) -> list[tuple[float, float]]:
    _, spots = load_zone_info(data_dir, zone_id)
    return spots


def coord_filename(x: float, y: float) -> str:
    xs = f"{x:.2f}".rstrip("0").rstrip(".")
    ys = f"{y:.2f}".rstrip("0").rstrip(".")
    return f"{xs}_{ys}.png"


def split_grid(
    image_path: Path,
    zone_id: str,
    output_root: Path,
    data_dir: Path,
    padding: int = 2,
    limit: int | None = None,
) -> list[Path]:
    image = Image.open(image_path)
    expansion, spots = load_zone_info(data_dir, zone_id)
    tiles = detect_tiles(image)

    if limit is not None:
        if limit > len(tiles):
            raise ValueError(
                f"limit {limit} > 탐지된 타일 {len(tiles)}개 (지역: {zone_id})"
            )
        if limit > len(spots):
            raise ValueError(
                f"limit {limit} > spots {len(spots)}개 (지역: {zone_id})"
            )
        tiles = tiles[:limit]
        spots = spots[:limit]
    elif len(tiles) != len(spots):
        raise ValueError(
            f"타일 {len(tiles)}개 ≠ spots {len(spots)}개 "
            f"(지역: {zone_id}). 그리드 배열을 확인하거나 --limit 사용."
        )

    out_dir = output_root / expansion / zone_id
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for (left, top, right, bottom), (tx, ty) in zip(tiles, spots):
        crop = image.crop(
            (
                max(0, left - padding),
                max(0, top - padding),
                min(image.width, right + padding),
                min(image.height, bottom + padding),
            )
        )
        out_path = out_dir / coord_filename(tx, ty)
        crop.save(out_path, format="PNG")
        saved.append(out_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="보물지도 그리드 분할")
    parser.add_argument("image", type=Path, help="그리드 PNG 경로")
    parser.add_argument("zone_id", help="zones.json id (예: central_shroud)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "treasure_refs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞 N칸만 사용 (8인 중복 줄 제외 시)",
    )
    args = parser.parse_args()

    saved = split_grid(
        args.image,
        args.zone_id,
        args.output,
        args.data_dir,
        limit=args.limit,
    )
    expansion, _ = load_zone_info(args.data_dir, args.zone_id)
    out_dir = args.output / expansion / args.zone_id
    print(f"저장 완료: {len(saved)}장 → {out_dir}")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
