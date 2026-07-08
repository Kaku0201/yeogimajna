"""
캡처 크기별 OCR / 매칭 경계 스윕.

사용:
  python scripts/sweep_capture_sizes.py
  python scripts/sweep_capture_sizes.py debug_query.png --zone east_shroud --expect 25.66 23
  python scripts/sweep_capture_sizes.py my_capture.png --trust-frame

게임에서 로그 받을 때:
  TR_DEBUG=1 python main.py   (match()마다 debug_query.png 저장)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.services.coordinate_service import CoordinateService
from src.services.treasure_capture import TreasureCaptureProcessor
from src.services.treasure_map_matcher import TreasureMapMatcher

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s %(levelname)s %(message)s",
)
# PIL 노이즈 줄이기
logging.getLogger("PIL").setLevel(logging.WARNING)


@dataclass
class SweepRow:
    scale: float
    raw_size: tuple[int, int]
    map_window_size: tuple[int, int]
    ocr_zone: str | None
    ocr_score: float
    match_x: float | None
    match_y: float | None
    match_ref: str | None
    match_ok: bool
    note: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="캡처 크기별 OCR/매칭 스윕")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(ROOT / "debug_query.png"),
        help="테스트 이미지 (기본: 프로젝트 루트 debug_query.png)",
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="매칭 존 ID (미지정 시 OCR 결과 사용, 없으면 east_shroud)",
    )
    parser.add_argument(
        "--expect",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="정답 좌표 (오차 0.5 이내면 match_ok=True)",
    )
    parser.add_argument(
        "--trust-frame",
        action="store_true",
        help="확정 캡처 경로 — prepare(trust_frame=True)",
    )
    parser.add_argument(
        "--scales",
        default="0.35,0.45,0.55,0.65,0.75,0.85,1.0",
        help="리사이즈 비율 (쉼표 구분)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="결과 텍스트 저장 경로 (기본: logs/sweep_<이미지명>.txt)",
    )
    return parser.parse_args()


def _coords_match(
    got_x: float | None,
    got_y: float | None,
    expect: tuple[float, float] | None,
    tol: float = 0.5,
) -> bool:
    if expect is None or got_x is None or got_y is None:
        return got_x is not None and got_y is not None
    ex, ey = expect
    return abs(got_x - ex) <= tol and abs(got_y - ey) <= tol


def run_sweep(args: argparse.Namespace) -> list[SweepRow]:
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"이미지 없음: {image_path}")

    base = Image.open(image_path).convert("RGB")
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    expect = tuple(args.expect) if args.expect else None
    fallback_zone = args.zone or "east_shroud"

    capture = TreasureCaptureProcessor(CoordinateService())
    matcher = TreasureMapMatcher(data_dir=capture.coordinate_service.data_dir)

    rows: list[SweepRow] = []

    print(f"\n=== 스윕 시작: {image_path.name} 원본 {base.size[0]}x{base.size[1]} ===")
    if expect:
        print(f"    정답 좌표: {expect[0]}, {expect[1]}")
    print(f"    prepare trust_frame={args.trust_frame}")
    print()

    for scale in scales:
        w = max(40, int(round(base.width * scale)))
        h = max(40, int(round(base.height * scale)))
        raw = base.resize((w, h), Image.Resampling.LANCZOS)

        print(f"--- scale={scale:.2f} raw={w}x{h} ---")

        _focused, zone_hint, map_window = capture.prepare(
            raw,
            refocus=not args.trust_frame,
            trust_frame=args.trust_frame,
        )
        zone_obj, ocr_score, ocr_text = capture.detect_zone_from_banner_scored(
            map_window
        )
        zone_id = (
            args.zone
            or (str(zone_obj["id"]) if zone_obj else None)
            or fallback_zone
        )
        ocr_name = str(zone_obj.get("name_ko", zone_obj.get("id", ""))) if zone_obj else None

        print(
            f"  map_window={map_window.width}x{map_window.height} "
            f"OCR={ocr_name!r} score={ocr_score:.3f} text={ocr_text!r}"
        )

        match = matcher.match(map_window, zone_id, party_size=1)
        if match:
            print(
                f"  match -> {match.treasure_x}, {match.treasure_y} "
                f"({match.template_path.name})"
            )
            note = "OK"
        else:
            print("  match -> None")
            note = "match_fail"

        if ocr_score < 0.55:
            note = "ocr_low" if note == "OK" else f"{note}+ocr_low"

        ok = _coords_match(
            match.treasure_x if match else None,
            match.treasure_y if match else None,
            expect,
        )
        if match and not ok:
            note = "wrong_coord"

        rows.append(
            SweepRow(
                scale=scale,
                raw_size=(w, h),
                map_window_size=(map_window.width, map_window.height),
                ocr_zone=ocr_name,
                ocr_score=ocr_score,
                match_x=match.treasure_x if match else None,
                match_y=match.treasure_y if match else None,
                match_ref=match.template_path.name if match else None,
                match_ok=ok,
                note=note,
            )
        )
        print()

    return rows


def _format_table(rows: list[SweepRow]) -> str:
    header = (
        f"{'scale':>5} {'raw':>9} {'map_win':>9} "
        f"{'ocr_score':>9} {'match':>12} {'ref':>18} {'ok':>4} note"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        match_s = (
            f"{r.match_x},{r.match_y}"
            if r.match_x is not None
            else "None"
        )
        lines.append(
            f"{r.scale:5.2f} {r.raw_size[0]:4d}x{r.raw_size[1]:<4d} "
            f"{r.map_window_size[0]:4d}x{r.map_window_size[1]:<4d} "
            f"{r.ocr_score:9.3f} {match_s:>12} {(r.match_ref or '-'):>18} "
            f"{'Y' if r.match_ok else 'N':>4} {r.note}"
        )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    rows = run_sweep(args)
    table = _format_table(rows)

    print("=== 요약 ===")
    print(table)

    out_path = Path(args.out) if args.out else ROOT / "logs" / f"sweep_{Path(args.image).stem}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
