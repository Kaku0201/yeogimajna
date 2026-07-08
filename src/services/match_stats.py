"""인식 경로 통계 — ref DB vs 상세지도 fallback"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.services.app_paths import get_user_data_dir

logger = logging.getLogger(__name__)

SOURCE_LABELS: dict[str, str] = {
    "learned": "⚡ 학습 매칭 (ref DB)",
    "ref": "✓ ref DB 확정",
    "ref_tentative": "? ref DB 참고 (후보)",
    "detail": "🗺️ 상세지도 지형찾기",
}

REF_DB_SOURCES = frozenset({"learned", "ref", "ref_tentative"})


class MatchStatsService:
    """인식 성공 시 match_source별 누적 카운트"""

    def __init__(self, store_path: Path | None = None) -> None:
        if store_path is None:
            store_path = get_user_data_dir() / "match_stats.json"
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if not self.store_path.exists():
            return {"total": 0, "by_source": {}, "by_zone": {}}
        try:
            with self.store_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return {"total": 0, "by_source": {}, "by_zone": {}}
            raw.setdefault("total", 0)
            raw.setdefault("by_source", {})
            raw.setdefault("by_zone", {})
            return raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("매칭 통계 로드 실패: %s", exc)
            return {"total": 0, "by_source": {}, "by_zone": {}}

    def _save(self) -> None:
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def record(self, match_source: str, zone_id: str) -> None:
        source = match_source or "ref"
        self._data["total"] = int(self._data.get("total", 0)) + 1
        by_source: dict = self._data.setdefault("by_source", {})
        by_source[source] = int(by_source.get(source, 0)) + 1

        by_zone: dict = self._data.setdefault("by_zone", {})
        zone_stats = by_zone.setdefault(zone_id, {})
        zone_stats[source] = int(zone_stats.get(source, 0)) + 1

        self._save()
        logger.info("매칭 통계 +1 %s (%s)", source, zone_id)

    def clear(self) -> None:
        self._data = {"total": 0, "by_source": {}, "by_zone": {}}
        if self.store_path.exists():
            self.store_path.unlink(missing_ok=True)

    @property
    def total(self) -> int:
        return int(self._data.get("total", 0))

    def count(self, source: str) -> int:
        return int(self._data.get("by_source", {}).get(source, 0))

    def ref_db_total(self) -> int:
        return sum(self.count(s) for s in REF_DB_SOURCES)

    def detail_total(self) -> int:
        return self.count("detail")

    def pct(self, n: int) -> float:
        total = self.total
        if total <= 0:
            return 0.0
        return round(n * 100.0 / total, 1)

    def summary_lines(self) -> list[str]:
        """다이얼로그용 요약 텍스트"""
        total = self.total
        if total <= 0:
            return ["아직 기록된 인식이 없습니다.", "지도를 캡처하면 통계가 쌓입니다."]

        ref_db = self.ref_db_total()
        detail = self.detail_total()
        lines = [
            f"총 {total}회 인식",
            "",
            f"ref DB 경로 합계: {ref_db}회 ({self.pct(ref_db)}%)",
            f"  └ 상세지도 fallback: {detail}회 ({self.pct(detail)}%)",
            "",
            "── 세부 ──",
        ]
        for source in ("learned", "ref", "ref_tentative", "detail"):
            n = self.count(source)
            if n <= 0:
                continue
            label = SOURCE_LABELS.get(source, source)
            lines.append(f"{label}: {n}회 ({self.pct(n)}%)")
        return lines

    def last_source_hint(self, match_source: str) -> str:
        """결과 패널용 한 줄 힌트"""
        total = self.total
        ref_db = self.ref_db_total()
        detail = self.detail_total()
        label = SOURCE_LABELS.get(match_source, match_source)
        if total <= 0:
            return label
        return (
            f"이번: {label} | 누적 ref {self.pct(ref_db)}% · 상세지도 {self.pct(detail)}%"
        )
