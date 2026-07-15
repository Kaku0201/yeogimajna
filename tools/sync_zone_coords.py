"""보물지도 CSV(공식 추출 데이터)로 data/zones.json 보물 좌표를 보정.

각 지역(zone)의 기존 보물 좌표를 CSV의 해당 지역 좌표 중 가장 가까운 값으로
스냅(치환)한다. 임계값(THRESHOLD)을 넘는 경우는 치환하지 않고 리포트만 한다.

사용법 (프로젝트 루트):
    python tools/sync_zone_coords.py --report   # 매칭 결과만 출력 (변경 없음)
    python tools/sync_zone_coords.py --apply     # zones.json 실제 수정
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZONES_PATH = _PROJECT_ROOT / "data" / "zones.json"
CSV_PATH = Path(
    r"c:\Users\JJyou\Downloads\ffxiv-kr-data-extractor-main\ffxiv-kr-data-extractor-main"
    r"\extract\SaintCoinach.Cmd\bin\Debug\net7.0\2026.06.18.0000.0000\보물지도_좌표.csv"
)

# 좌표 스냅 허용 거리(게임 좌표 단위). 이보다 멀면 치환하지 않음.
DEFAULT_THRESHOLD = 0.5


def normalize_name(name: str) -> str:
    """지역명 매칭용 정규화 — 공백 제거."""
    return name.replace(" ", "").strip()


def load_csv_coords() -> dict[str, list[tuple[float, float]]]:
    """정규화된 세부구역명 -> 고유 좌표 리스트."""
    by_zone: dict[str, set[tuple[float, float]]] = {}
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get("구역") or ""
            key = normalize_name(sub)
            if not key:
                continue
            try:
                x = float(row["지도X"])
                y = float(row["지도Y"])
            except (KeyError, ValueError):
                continue
            by_zone.setdefault(key, set()).add((round(x, 2), round(y, 2)))
    return {k: sorted(v) for k, v in by_zone.items()}


def nearest(
    coords: list[tuple[float, float]], x: float, y: float
) -> tuple[tuple[float, float], float]:
    best = coords[0]
    best_d = math.hypot(best[0] - x, best[1] - y)
    for c in coords[1:]:
        d = math.hypot(c[0] - x, c[1] - y)
        if d < best_d:
            best, best_d = c, d
    return best, best_d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="zones.json 실제 수정")
    parser.add_argument("--report", action="store_true", help="리포트만 출력")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="스냅 허용 거리(초과 시 미변경)",
    )
    args = parser.parse_args()
    threshold = args.threshold

    csv_coords = load_csv_coords()
    with open(ZONES_PATH, encoding="utf-8") as f:
        zones = json.load(f)

    total_spots = 0
    changed = 0
    unchanged_same = 0
    no_zone = 0
    over_threshold = 0
    lines: list[str] = []

    for expansion, zone_list in zones.items():
        for zone in zone_list:
            name_ko = zone.get("name_ko", "")
            zid = zone.get("id", "")
            key = normalize_name(name_ko)
            coords = csv_coords.get(key)
            spots = zone.get("spots") or []
            if coords is None:
                no_zone += len(spots)
                lines.append(f"[CSV없음] {expansion}/{zid} ({name_ko}) spots={len(spots)}")
                continue
            seen_targets: dict[tuple[float, float], tuple[float, float]] = {}
            for spot in spots:
                treasure = spot.get("treasure") or {}
                x, y = treasure.get("x"), treasure.get("y")
                if x is None or y is None:
                    continue
                total_spots += 1
                x, y = float(x), float(y)
                (nx, ny), dist = nearest(coords, x, y)
                if dist > threshold:
                    over_threshold += 1
                    lines.append(
                        f"[임계초과] {expansion}/{zid} {x:.2f},{y:.2f} "
                        f"-> 최근접 {nx:.2f},{ny:.2f} (거리 {dist:.2f}) 미변경"
                    )
                    continue
                if (nx, ny) in seen_targets:
                    prev = seen_targets[(nx, ny)]
                    lines.append(
                        f"[중복경고] {expansion}/{zid} {x:.2f},{y:.2f} 와 "
                        f"{prev[0]:.2f},{prev[1]:.2f} 가 동일 CSV {nx:.2f},{ny:.2f} 로 스냅됨"
                    )
                seen_targets[(nx, ny)] = (x, y)
                if (round(x, 2), round(y, 2)) == (nx, ny):
                    unchanged_same += 1
                    continue
                lines.append(
                    f"[변경] {expansion}/{zid} {x:.2f},{y:.2f} -> {nx:.2f},{ny:.2f} "
                    f"(거리 {dist:.2f})"
                )
                treasure["x"] = nx
                treasure["y"] = ny
                changed += 1

    print("\n".join(lines))
    print("\n===== 요약 =====")
    print(f"총 spot: {total_spots}")
    print(f"변경: {changed}")
    print(f"이미 동일: {unchanged_same}")
    print(f"임계 초과(미변경): {over_threshold}")
    print(f"CSV 매칭 지역 없음(spot): {no_zone}")

    if args.apply:
        with open(ZONES_PATH, "w", encoding="utf-8") as f:
            json.dump(zones, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\n[적용] {ZONES_PATH} 저장 완료")
    else:
        print("\n[미적용] --apply 옵션을 주면 실제로 저장합니다.")


if __name__ == "__main__":
    main()
