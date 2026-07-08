"""debug_query.png vs 지역 ref 전체 스코어·비교 그리드 생성.

사용법:
  $env:TR_DEBUG_QUERY='1'; python main.py   # 캡처 1회 → debug_query.png 생성
  python scripts/compare_query_refs.py thavnair 1
  python scripts/compare_query_refs.py thavnair 1 path/to/query.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.treasure_map_matcher import TreasureMapMatcher

SOLO_REFS = Path("assets/treasure_refs/효월/thavnair/solo")
OUT_DIR = ROOT / "debug_qc"


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("malgun.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = img.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (32, 32, 32))
    ox = (size[0] - fitted.width) // 2
    oy = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (ox, oy))
    return canvas


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> None:
    x, y = xy
    draw.rectangle((x, y, x + 420, y + 44), fill=(0, 0, 0))
    draw.text((x + 4, y + 2), text, fill=(255, 255, 255), font=font)


def build_grid(
    query: Image.Image,
    scored: list[tuple[float, float, float, float, object, float]],
    ref_dir: Path,
    out_path: Path,
) -> None:
    cell = (200, 150)
    cols = 3
    font = _load_font(14)
    title_font = _load_font(16)

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    tiles: list[tuple[str, Image.Image, str]] = []

    tiles.append(
        (
            "QUERY (debug_query.png)",
            _fit(query, cell),
            f"{query.width}x{query.height}",
        )
    )

    score_by_name = {entry.ref_name.split("/")[-1]: item for item in ranked for entry in [item[4]]}

    for ref_path in sorted(ref_dir.glob("*.png")):
        item = score_by_name.get(ref_path.name)
        if item is None:
            label = f"{ref_path.stem}\n(no score)"
            sub = ""
        else:
            adj, terrain, _ccorr, _ssim, _entry, marker_dist = item
            rank = next(
                i + 1
                for i, s in enumerate(ranked)
                if s[4].ref_name.endswith(ref_path.name)
            )
            label = (
                f"#{rank} {ref_path.stem}\n"
                f"adj={adj:.3f} terrain={terrain:.3f} marker={marker_dist:.3f}"
            )
            sub = "TOP1" if rank == 1 else ""
        img = _fit(Image.open(ref_path), cell)
        tiles.append((label, img, sub))

    rows = (len(tiles) + cols - 1) // cols
    pad = 8
    label_h = 48
    grid_w = cols * cell[0] + (cols + 1) * pad
    grid_h = rows * (cell[1] + label_h) + (rows + 1) * pad + 36

    canvas = Image.new("RGB", (grid_w, grid_h), (48, 48, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), "thavnair solo/ vs debug_query — adj 내림차순", fill=(255, 255, 0), font=title_font)

    y0 = pad + 28
    for idx, (label, img, badge) in enumerate(tiles):
        row, col = divmod(idx, cols)
        x = pad + col * (cell[0] + pad)
        y = y0 + row * (cell[1] + label_h + pad)
        _draw_label(draw, (x, y), label.replace("\n", " | "), font)
        canvas.paste(img, (x, y + label_h))
        if badge:
            draw.rectangle((x, y + label_h, x + 52, y + label_h + 20), fill=(0, 128, 0))
            draw.text((x + 4, y + label_h + 2), badge, fill=(255, 255, 255), font=font)

    OUT_DIR.mkdir(exist_ok=True)
    canvas.save(out_path)
    print(f"grid saved: {out_path}")


def main() -> None:
    zone_id = sys.argv[1] if len(sys.argv) > 1 else "thavnair"
    party_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    query_path = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "debug_query.png"

    if not query_path.is_file():
        print(f"MISSING: {query_path}")
        print("  $env:TR_DEBUG_QUERY='1'; python main.py")
        print("  → 프로젝트 루트에 debug_query.png 생성 후 재실행")
        sys.exit(1)

    query = Image.open(query_path).convert("RGB")
    matcher = TreasureMapMatcher()
    scored, query_rel = matcher._score_zone(
        query, zone_id, party_size, full_scan=True
    )

    if not scored:
        print("스코어 결과 없음")
        sys.exit(1)

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    print(f"query: {query_path} ({query.width}x{query.height}) query_marker={query_rel}")
    print(f"{'rank':>4}  {'ref':<28}  {'adj':>6}  {'terrain':>7}  {'marker':>7}")
    print("-" * 62)
    for i, (adj, terrain, ccorr, ssim, entry, marker_dist) in enumerate(ranked, 1):
        print(
            f"{i:4d}  {entry.ref_name:<28}  {adj:6.3f}  {terrain:7.3f}  {marker_dist:7.3f}"
        )

    ref_dir = ROOT / "assets/treasure_refs/효월/thavnair/solo"
    if party_size == 8:
        ref_dir = ROOT / "assets/treasure_refs/효월/thavnair/party8"

    out_path = OUT_DIR / f"{zone_id}_party{party_size}_compare.png"
    build_grid(query, scored, ref_dir, out_path)


if __name__ == "__main__":
    main()
