"""사용자 확정 보물지도 — 지형 지문 저장·빠른 재매칭"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.services.app_paths import get_user_data_dir
from src.services.image_similarity import terrain_similarity_from_features
from src.services.treasure_capture import TreasureCaptureProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearnedRefHit:
    """학습 캐시 히트"""

    ref_name: str
    x: float
    y: float
    score: float
    hits: int


class UserRefLearnService:
    """
    사용자가 확정한 ref를 지형 지문과 함께 저장.
    같은 지역·비슷한 캡처는 전체 ref 스캔 전에 즉시 매칭.
    """

    FP_SIZE = 64
    BASE_HIT_SCORE = 0.88
    MIN_HIT_SCORE = 0.76
    MAX_FINGERPRINTS = 5
    MAX_PER_ZONE = 40

    def __init__(self, store_path: Path | None = None) -> None:
        if store_path is None:
            store_path = get_user_data_dir() / "learned_refs.json"
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, list[dict]] = self._load()
        self._migrate_entries()
        self._consolidate_all()

    def _load(self) -> dict[str, list[dict]]:
        if not self.store_path.exists():
            return {}
        try:
            with self.store_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return {
                    str(zone_id): list(entries)
                    for zone_id, entries in raw.items()
                    if isinstance(entries, list)
                }
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("학습 캐시 로드 실패: %s", exc)
        return {}

    def _save(self) -> None:
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @classmethod
    def _hit_threshold(cls, hits: int) -> float:
        """확정 횟수가 많을수록 캡처 차이를 허용"""
        bonus = min(max(hits, 1), 15) * 0.008
        return max(cls.MIN_HIT_SCORE, cls.BASE_HIT_SCORE - bonus)

    @classmethod
    def _entry_fingerprints(cls, entry: dict) -> list[str]:
        raw = entry.get("fingerprints_b64")
        if isinstance(raw, list):
            fps = [str(item) for item in raw if item]
            if fps:
                return fps[-cls.MAX_FINGERPRINTS :]
        legacy = entry.get("fingerprint_b64")
        if legacy:
            return [str(legacy)]
        return []

    def _migrate_entries(self) -> None:
        """단일 지문 → 다중 지문, hits 과다 누적 보정"""
        changed = False
        for zone_id, entries in self._data.items():
            for entry in entries:
                fps = self._entry_fingerprints(entry)
                if fps and entry.get("fingerprints_b64") != fps:
                    entry["fingerprints_b64"] = fps
                    entry["fingerprint_b64"] = fps[-1]
                    changed = True
                hits = int(entry.get("hits", 1))
                if hits > 50:
                    entry["hits"] = min(hits, 50)
                    changed = True
        if changed:
            self._save()

    def _consolidate_zone(self, zone_id: str) -> None:
        """구버전 중복 ref_name 항목을 하나로 합침"""
        entries = self._data.get(zone_id)
        if not entries:
            return

        merged: dict[str, dict] = {}
        for entry in entries:
            ref_name = str(entry.get("ref_name", ""))
            if not ref_name:
                continue
            if ref_name not in merged:
                merged[ref_name] = dict(entry)
                continue
            keep = merged[ref_name]
            # 중복 항목 hits는 max 사용 (과거 sum 누적 버그 방지)
            keep["hits"] = max(
                int(keep.get("hits", 1)), int(entry.get("hits", 1))
            )
            keep_fps = self._entry_fingerprints(keep)
            for fp in self._entry_fingerprints(entry):
                if fp not in keep_fps:
                    keep_fps.append(fp)
            keep_fps = keep_fps[-self.MAX_FINGERPRINTS :]
            keep["fingerprints_b64"] = keep_fps
            if keep_fps:
                keep["fingerprint_b64"] = keep_fps[-1]
            if int(entry.get("updated", 0)) >= int(keep.get("updated", 0)):
                keep["x"] = entry.get("x", keep.get("x"))
                keep["y"] = entry.get("y", keep.get("y"))
                keep["updated"] = entry.get("updated", keep.get("updated"))
                if entry.get("party_size") in (1, 8):
                    keep["party_size"] = entry["party_size"]

        consolidated = list(merged.values())
        consolidated.sort(
            key=lambda item: int(item.get("hits", 0)), reverse=True
        )
        self._data[zone_id] = consolidated[: self.MAX_PER_ZONE]

    def _consolidate_all(self) -> None:
        before = json.dumps(self._data, sort_keys=True)
        for zone_id in list(self._data):
            self._consolidate_zone(zone_id)
        after = json.dumps(self._data, sort_keys=True)
        if before != after:
            self._save()

    @classmethod
    def fingerprint_from_map(cls, map_window: Image.Image) -> np.ndarray:
        """매칭 파이프라인과 동일한 지형 특징 → 64×64"""
        from src.services.image_similarity import enhance_terrain_features

        _patch, gray, _center = TreasureCaptureProcessor.prepare_matching_patch(
            map_window
        )
        feat = enhance_terrain_features(gray)
        return cv2.resize(
            feat,
            (cls.FP_SIZE, cls.FP_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    @classmethod
    def _encode_fp(cls, feat: np.ndarray) -> str:
        return base64.b64encode(feat.astype(np.uint8).tobytes()).decode("ascii")

    @classmethod
    def _decode_fp(cls, encoded: str) -> np.ndarray | None:
        try:
            raw = base64.b64decode(encoded.encode("ascii"))
            expected = cls.FP_SIZE * cls.FP_SIZE
            if len(raw) != expected:
                return None
            return np.frombuffer(raw, dtype=np.uint8).reshape(
                cls.FP_SIZE, cls.FP_SIZE
            )
        except (ValueError, OSError):
            return None

    @classmethod
    def _compare_fp(cls, a: np.ndarray, b: np.ndarray) -> float:
        combined, _, _ = terrain_similarity_from_features(a, b)
        return combined

    def get_ref_hits(self, zone_id: str, ref_name: str) -> int:
        for entry in self._data.get(zone_id, []):
            if entry.get("ref_name") == ref_name:
                return int(entry.get("hits", 1))
        return 0

    def get_party_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for zone_id, entries in self._data.items():
            best_updated = -1
            best_size: int | None = None
            for entry in entries:
                party_size = entry.get("party_size")
                if party_size not in (1, 8):
                    continue
                updated = int(entry.get("updated", 0))
                if updated >= best_updated:
                    best_updated = updated
                    best_size = int(party_size)
            if best_size is not None:
                sizes[zone_id] = best_size
        return sizes

    def get_top_entry(self, zone_id: str) -> dict | None:
        entries = self._data.get(zone_id)
        if not entries:
            return None
        return max(entries, key=lambda item: int(item.get("hits", 0)))

    def try_fast_match(
        self,
        zone_id: str,
        map_window: Image.Image,
    ) -> LearnedRefHit | None:
        entries = self._data.get(zone_id)
        if not entries:
            return None

        query_fp = self.fingerprint_from_map(map_window)
        best: LearnedRefHit | None = None

        for entry in entries:
            hits = int(entry.get("hits", 1))
            threshold = self._hit_threshold(hits)
            best_score = 0.0
            for encoded in self._entry_fingerprints(entry):
                stored = self._decode_fp(encoded)
                if stored is None:
                    continue
                best_score = max(best_score, self._compare_fp(query_fp, stored))
            if best_score < threshold:
                continue
            hit = LearnedRefHit(
                ref_name=str(entry["ref_name"]),
                x=float(entry["x"]),
                y=float(entry["y"]),
                score=best_score,
                hits=hits,
            )
            if best is None or hit.score > best.score:
                best = hit

        if best is not None:
            logger.debug(
                "학습 캐시 히트 %s %s score=%.3f hits=%d",
                zone_id,
                best.ref_name,
                best.score,
                best.hits,
            )
        return best

    def confirm(
        self,
        zone_id: str,
        ref_name: str,
        x: float,
        y: float,
        map_window: Image.Image,
        party_size: int | None = None,
    ) -> int:
        """확정 저장 — 동일 ref_name이면 hits 누적, 지문은 최대 5개 보관"""
        fp = self.fingerprint_from_map(map_window)
        encoded = self._encode_fp(fp)
        zone_entries = self._data.setdefault(zone_id, [])

        for entry in zone_entries:
            if entry.get("ref_name") != ref_name:
                continue
            entry["hits"] = int(entry.get("hits", 1)) + 1
            fps = self._entry_fingerprints(entry)
            fps.append(encoded)
            entry["fingerprints_b64"] = fps[-self.MAX_FINGERPRINTS :]
            entry["fingerprint_b64"] = encoded
            entry["x"] = x
            entry["y"] = y
            entry["updated"] = int(time.time())
            if party_size in (1, 8):
                entry["party_size"] = party_size
            zone_entries.sort(
                key=lambda item: int(item.get("hits", 0)), reverse=True
            )
            self._save()
            logger.debug(
                "학습 캐시 갱신 %s %s hits=%d",
                zone_id,
                ref_name,
                entry["hits"],
            )
            return int(entry["hits"])

        zone_entries.append(
            {
                "ref_name": ref_name,
                "x": x,
                "y": y,
                "fingerprint_b64": encoded,
                "fingerprints_b64": [encoded],
                "hits": 1,
                "updated": int(time.time()),
                **({"party_size": party_size} if party_size in (1, 8) else {}),
            }
        )
        zone_entries.sort(key=lambda item: int(item.get("hits", 0)), reverse=True)
        if len(zone_entries) > self.MAX_PER_ZONE:
            del zone_entries[self.MAX_PER_ZONE :]
        self._save()
        logger.debug("학습 캐시 추가 %s %s", zone_id, ref_name)
        return 1

    def clear_all(self) -> None:
        """학습 캐시 전체 삭제"""
        self._data = {}
        if self.store_path.exists():
            self.store_path.unlink(missing_ok=True)

    def count_zone(self, zone_id: str) -> int:
        return len(self._data.get(zone_id, []))
