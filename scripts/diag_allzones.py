"""전 지역 self-match 변별력 집계 — 각 ref를 쿼리로 넣어 self가 top1인지,
1등-2등 마진이 얼마나 벌어지는지(변별력) 측정.

터레인 매칭 개선의 회귀 하네스. top1은 정확도, margin은 확신 가능성을 나타낸다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.treasure_map_matcher import TreasureMapMatcher


def run_zone(
    matcher: TreasureMapMatcher, zone_id: str, party: int
) -> tuple[int, int, int, list[float]]:
    entries = matcher._load_zone_entries(zone_id, party)
    if not entries:
        return 0, 0, 0, []
    top1 = 0
    confident_ok = 0
    margins: list[float] = []
    for entry in entries:
        query = Image.open(entry.path).convert("RGB")
        confident, ranked, _p = matcher.match_with_ranked(
            query, zone_id, party_size=party, top_k=2, full_scan=True
        )
        if ranked and ranked[0].ref_name == entry.ref_name:
            top1 += 1
            if len(ranked) > 1:
                margins.append(ranked[0].score - ranked[1].score)
        if confident is not None and confident.ref_name == entry.ref_name:
            confident_ok += 1
    return len(entries), top1, confident_ok, margins


def main() -> int:
    verbose = "-v" in sys.argv
    matcher = TreasureMapMatcher()
    zone_ids = matcher.zone_ids_with_refs()
    tot_n = tot_top1 = tot_conf = 0
    all_margins: list[float] = []
    if verbose:
        print(f"{'zone':<32} {'party':>5} {'n':>4} {'top1':>8} {'conf':>8} {'avgMrg':>7}")
        print("-" * 70)
    for zone_id in zone_ids:
        for party in (1, 8):
            n, top1, conf, margins = run_zone(matcher, zone_id, party)
            if n == 0:
                continue
            tot_n += n
            tot_top1 += top1
            tot_conf += conf
            all_margins.extend(margins)
            if verbose:
                avg_m = sum(margins) / len(margins) if margins else 0.0
                print(
                    f"{zone_id:<32} {party:>5} {n:>4} "
                    f"{top1:>3}/{n:<4} {conf:>3}/{n:<4} {avg_m:>7.3f}"
                )
    print("-" * 70)
    if tot_n:
        avg_margin = sum(all_margins) / len(all_margins) if all_margins else 0.0
        tight = sum(1 for m in all_margins if m < 0.06)
        print(
            f"TOTAL n={tot_n}  top1={tot_top1}/{tot_n} ({100*tot_top1/tot_n:.1f}%)  "
            f"confident={tot_conf}/{tot_n} ({100*tot_conf/tot_n:.1f}%)"
        )
        print(
            f"avg margin(top1-top2)={avg_margin:.3f}  "
            f"margin<0.06: {tight}/{len(all_margins)} "
            f"({100*tight/max(1,len(all_margins)):.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
