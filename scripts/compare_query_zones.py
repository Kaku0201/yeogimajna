"""debug_query.png vs 전 존 지형 유사도 순위.

사용법:
  $env:TR_DEBUG_QUERY='1'; python main.py   # 캡처 1회 → debug_query.png 생성
  python scripts/compare_query_zones.py
  python scripts/compare_query_zones.py path/to/query.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.treasure_map_matcher import TreasureMapMatcher


def main() -> int:
    query_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "debug_query.png"
    )
    if not query_path.is_file():
        print(f"쿼리 이미지 없음: {query_path}")
        return 1

    matcher = TreasureMapMatcher()
    matcher.preload_all()
    query = Image.open(query_path).convert("RGB")
    ranked = matcher.rank_zones_by_terrain(query)

    if not ranked:
        print("비교 가능한 ref 지역이 없습니다.")
        return 1

    print(f"query: {query_path} ({query.width}x{query.height})")
    print(f"zones with refs: {len(matcher.zone_ids_with_refs())}")
    print("-" * 72)
    for idx, item in enumerate(ranked[:15], start=1):
        margin_txt = f" margin={item.margin:.3f}" if idx == 1 else ""
        ref_txt = f" ref={item.best_ref_name}" if item.best_ref_name else ""
        print(
            f"{idx:2d}. {item.zone_id:<32} score={item.score:.3f}{margin_txt}{ref_txt}"
        )

    zone_id, score, margin = matcher.identify_zone_from_terrain(query)
    print("-" * 72)
    if zone_id:
        print(f"identify_zone_from_terrain → {zone_id} (score={score:.3f}, margin={margin:.3f})")
    else:
        print(
            f"identify_zone_from_terrain → None (best={score:.3f}, margin={margin:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
