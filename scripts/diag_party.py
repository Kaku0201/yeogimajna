"""party_size 감지 디버그 — debug_query.png에서 1/8 판별 경로 추적."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TR_DEBUG", "1")
os.environ.setdefault("TR_DEBUG_PARTY", "1")

from src.services.treasure_capture import TreasureCaptureProcessor
from src.services.treasure_map_matcher import TreasureMapMatcher
from src.services.coordinate_service import CoordinateService
from src.services.map_analyzer import MapAnalyzer

QUERY = ROOT / "debug_query.png"
ZONE = "western_thanalan"


def main() -> int:
    img = Image.open(QUERY).convert("RGB")
    cs = CoordinateService()
    cap = TreasureCaptureProcessor(cs)
    analyzer = MapAnalyzer(cs)

    print(f"query={QUERY.name} size={img.size}")
    print("-" * 60)

    for label, fn in [
        ("detect_party_size", lambda: cap.detect_party_size(img, debug_source="query")),
        ("detect_party_size_aggressive", lambda: cap.detect_party_size_aggressive(img, debug_source="aggressive")),
    ]:
        cap._party_debug_saved = False
        result = fn()
        trace = cap.last_party_trace
        print(f"{label}: result={result}")
        if trace:
            print(f"  trace: {trace.summary()}")
        print()

    # ref coarse 비교
    m = TreasureMapMatcher()
    solo, party8 = m.compare_party_folder_coarse(img, ZONE)
    print(f"ref coarse solo={solo:.3f}  party8={party8:.3f}  winner={'solo' if solo >= party8 else 'party8'}")

    party, uncertain = analyzer._resolve_party_size(ZONE, img, img)
    print(f"_resolve_party_size: party={party} uncertain={uncertain}")

    party2, unc2 = analyzer._confirm_party_with_map_refs(ZONE, img, party, uncertain)
    print(f"_confirm_party_with_map_refs: party={party2} uncertain={unc2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
