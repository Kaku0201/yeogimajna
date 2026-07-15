"""보물지도 참조 매칭 — 지역 내 전체 ref × terrain SSIM (2단계 가속)"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.services.image_similarity import (
    enhance_terrain_features,
    phase_align_gray,
    terrain_similarity_from_features,
)
from src.app_config import is_debug_query, is_debug
from src.services.app_paths import get_app_root
from src.services.treasure_capture import TreasureCaptureProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TreasureMapMatch:
    score: float
    ccorr_score: float
    ssim_score: float
    treasure_x: float
    treasure_y: float
    template_path: Path
    ref_name: str
    terrain_score: float = 0.0
    marker_dist: float = 1.0


@dataclass(frozen=True)
class ZoneTerrainIdentification:
    """전 존 지형 비교 결과 — spot 식별 전 1차 zone 후보"""

    zone_id: str
    score: float
    margin: float
    best_ref_name: str | None = None


@dataclass
class _RefEntry:
    ref_name: str
    path: Path
    treasure_x: float
    treasure_y: float
    gray: np.ndarray
    feat_gray: np.ndarray
    weight_center: tuple[int, int]
    tpl_weights: np.ndarray
    feat_gray_coarse: np.ndarray
    tpl_weights_coarse: np.ndarray
    marker_rel: tuple[float, float] | None = None
    source_size: tuple[int, int] = (0, 0)
    party_sizes: frozenset[int] | None = None


class _ScalePack:
    """스케일별 리사이즈 결과 (루프 밖 1회 생성)"""

    __slots__ = ("sw", "sh", "feat", "weights")

    def __init__(
        self,
        sw: int,
        sh: int,
        feat: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.sw = sw
        self.sh = sh
        self.feat = feat
        self.weights = weights


class TreasureMapMatcher:
    """
    OCR로 지역 고정 후 treasure_refs 1:1 매칭.

    1) 128px coarse 로 top-K 선별
    2) top-K만 full 해상도 정밀 비교
    """

    # 빨간 X marker 위치 보정 스위치.
    # ref_maker가 보물 좌표를 크롭 중심에 놓으므로 X는 모든 ref에서 대략 중앙 →
    # spot 구분에 변별력이 없고, 검출 실패 시 중앙 기본값(0.4831/0.4751)으로 떨어져
    # 오답 ref에 허위 보너스를 준다. 실측상 terrain 단독이 100%, marker 보정이 오히려
    # 정확도를 떨어뜨려(78~94%) 비활성화한다.
    MARKER_ADJUST_ENABLED = False

    # 확신 판정: 절대 점수 문턱 + 마진(1등-2등 격차).
    # DoG 밴드패스 특징은 양피지 노이즈를 제거하는 대신 절대 점수대가 낮아진다
    # (self-match≈1.0, 실제 인게임 캡처≈0.55~0.65). 문턱을 실측 캡처 분포에 맞춰
    # 재조정하되, 마진(경쟁 spot 대비 격차)을 주 신호로 삼아 오확정을 막는다.
    CONFIDENT_SCORE = 0.55          # 고득점 경로 절대 하한
    CONFIDENT_SCORE_HIGH = 0.68     # 이 이상이면 마진 요구 완화
    MIN_MARGIN = 0.06
    STRONG_MARGIN = 0.10
    DOMINANT_MARGIN = 0.14          # 마진 우세 경로 — 절대 점수 중간이어도 확정
    DOMINANT_SCORE_FLOOR = 0.48     # 마진 우세 경로 절대 하한(쓰레기 매칭 차단)

    COARSE_MAX_SIDE = 128
    COARSE_SCALE_STEPS = 4
    REFINE_SCALE_STEPS = 7
    REFINE_TOP_K = 12
    FULL_SCAN_REF_THRESHOLD = 8

    # 콘텐츠 스케일 탐색: 인게임 지도 창은 ref 크롭보다 더 넓은 영역을 보여줘
    # (실측 ~0.65배) 정규화만으로는 지형이 겹치지 않는다. 쿼리를 중앙에서 여러
    # 비율로 잘라 각각 채점하고, 1등-2등 마진이 가장 큰 스케일의 순위를 채택한다.
    # 참 스케일에서 정답 봉우리가 가장 날카롭게 서므로 마진이 최대가 된다.
    # factor=1.0은 self-match(ref==query) 및 이미 꽉 찬 캡처를 위해 유지.
    CONTENT_SCALE_FACTORS = (1.0, 0.82, 0.70, 0.60)
    # 이 이상으로 확신되면 다른 스케일을 더 보지 않고 조기 종료(속도).
    CONTENT_SCALE_SKIP_SCORE = 0.82
    CONTENT_SCALE_SKIP_MARGIN = 0.15
    MARKER_ALIGN_MAX_DIST = 0.12
    MARKER_BONUS_CAP = 0.12
    MARKER_MISMATCH_PENALTY = 0.50
    MARKER_TERRAIN_FLOOR = 0.62
    TERRAIN_CLUSTER_MARGIN = 0.025
    MARKER_CLUSTER_BONUS_THRESHOLD = 0.13
    MARKER_CLUSTER_BONUS_SCALE = 2.0
    MARKER_CLUSTER_PENALTY_START = 0.12
    TERRAIN_ZONE_CONFIDENT_SCORE = 0.82
    TERRAIN_ZONE_MIN_MARGIN = 0.08
    TERRAIN_ZONE_REFINE_TOP = 3

    def __init__(
        self,
        assets_dir: Path | None = None,
        data_dir: Path | None = None,
    ) -> None:
        root = get_app_root()
        if assets_dir is None:
            assets_dir = root / "assets"
        if data_dir is None:
            data_dir = root / "data"
        self.refs_dir = assets_dir / "treasure_refs"
        self._data_dir = data_dir
        self._zone_entries: dict[str, tuple[str, list[_RefEntry]]] = {}
        self._zone_ids_with_refs: list[str] | None = None
        self._ref_coords_cache: dict[str, dict[str, dict]] = {}

    def preload_all(self) -> None:
        """모든 ref 지역 gray/feature 캐시 (앱 시작 시 선택 호출)"""
        for zone_id in self.zone_ids_with_refs():
            for folder in self._resolve_refs_folders(zone_id, None):
                self._load_zone_folder(zone_id, folder)

    def preload_zone(self, zone_id: str, party_size: int | None = None) -> None:
        """OCR 지역 확정 직후 해당 지역 ref 미리 로드 (party_size 있으면 해당 폴더만)"""
        for folder in self._resolve_refs_folders(zone_id, party_size):
            self._load_zone_folder(zone_id, folder)

    def zone_ids_with_refs(self) -> list[str]:
        if self._zone_ids_with_refs is not None:
            return self._zone_ids_with_refs

        found: list[str] = []
        if not self.refs_dir.is_dir():
            self._zone_ids_with_refs = found
            return found

        for expansion_dir in sorted(self.refs_dir.iterdir()):
            if not expansion_dir.is_dir():
                continue
            for zone_dir in sorted(expansion_dir.iterdir()):
                if not zone_dir.is_dir():
                    continue
                if self._has_ref_images(zone_dir):
                    found.append(zone_dir.name)

        for zone_dir in sorted(self.refs_dir.iterdir()):
            if not zone_dir.is_dir() or zone_dir.name in found:
                continue
            if self._has_ref_images(zone_dir):
                found.append(zone_dir.name)

        self._zone_ids_with_refs = found
        return found

    def match(
        self,
        fragment: Image.Image,
        zone_id: str,
        party_size: int | None = None,
    ) -> TreasureMapMatch | None:
        confident, _ranked, _party = self.match_with_ranked(
            fragment, zone_id, party_size=party_size, top_k=3
        )
        return confident

    def has_split_party_refs(self, zone_id: str) -> bool:
        """solo/ + party8/ 폴더가 모두 있는 16칸 지역"""
        return (
            self.count_zone_refs(zone_id, 1) > 0
            and self.count_zone_refs(zone_id, 8) > 0
        )

    def compare_party_folder_coarse(
        self, fragment: Image.Image, zone_id: str
    ) -> tuple[float, float]:
        """solo/party8 ref 폴더별 coarse 최고점 — 인원 아이콘 보조 검증용"""
        solo = self._max_coarse_score(fragment, zone_id, 1)
        party8 = self._max_coarse_score(fragment, zone_id, 8)
        return solo, party8

    def _max_coarse_score(
        self,
        fragment: Image.Image,
        zone_id: str,
        party_size: int,
    ) -> float:
        entries = self._load_zone_entries(zone_id, party_size)
        if not entries:
            return -1.0

        match_fragment = self._normalize_query_to_refs(fragment, entries)
        _patch, query_gray, query_center = (
            TreasureCaptureProcessor.prepare_matching_patch(match_fragment)
        )
        query_feat = enhance_terrain_features(query_gray)
        query_weights = TreasureCaptureProcessor.build_soft_center_weights(
            query_gray.shape[0],
            query_gray.shape[1],
            query_center,
        )
        q_feat_c, q_weights_c = self._resize_pair(
            query_feat,
            query_weights,
            self.COARSE_MAX_SIDE,
        )
        coarse_query_scales = self._build_scale_packs(
            q_feat_c,
            q_weights_c,
            np.linspace(0.92, 1.08, self.COARSE_SCALE_STEPS),
        )
        best = -1.0
        for entry in entries:
            score = self._coarse_score_optimized(coarse_query_scales, entry)
            if score > best:
                best = score
        return best

    @staticmethod
    def _ref_party_size(ref_name: str) -> int | None:
        if ref_name.startswith("solo/"):
            return 1
        if ref_name.startswith("party8/"):
            return 8
        return None

    def _infer_party_from_scored(
        self,
        scored: list[tuple[float, float, float, float, _RefEntry, float]],
    ) -> int | None:
        """solo/party8 혼합 스코어에서 어느 인원 폴더가 더 맞는지 추정"""
        solo_best = max(
            (item[0] for item in scored if item[4].ref_name.startswith("solo/")),
            default=-1.0,
        )
        party8_best = max(
            (item[0] for item in scored if item[4].ref_name.startswith("party8/")),
            default=-1.0,
        )
        if solo_best < 0 and party8_best < 0:
            return None
        if solo_best < 0:
            return 8
        if party8_best < 0:
            return 1
        return 1 if solo_best >= party8_best else 8

    @staticmethod
    def _filter_scored_by_party(
        scored: list[tuple[float, float, float, float, _RefEntry, float]],
        party_size: int,
    ) -> list[tuple[float, float, float, float, _RefEntry, float]]:
        prefix = "solo/" if party_size == 1 else "party8/"
        filtered = [item for item in scored if item[4].ref_name.startswith(prefix)]
        return filtered or scored

    def match_with_ranked(
        self,
        fragment: Image.Image,
        zone_id: str,
        party_size: int | None = None,
        top_k: int = 3,
        exclude_ref_names: set[str] | None = None,
        full_scan: bool = False,
    ) -> tuple[TreasureMapMatch | None, list[TreasureMapMatch], int | None]:
        """확신 매칭 + UI용 상위 후보를 한 번의 스코어링으로 반환"""
        entry_count = self.count_zone_refs(zone_id, party_size)
        split_zone = party_size is None and self.has_split_party_refs(zone_id)
        use_full_scan = (
            full_scan
            or bool(exclude_ref_names)
            or split_zone
            or entry_count <= 8
            or entry_count > self.FULL_SCAN_REF_THRESHOLD
        )
        scored, query_rel = self._score_zone(
            fragment,
            zone_id,
            party_size,
            exclude_ref_names=exclude_ref_names,
            full_scan=use_full_scan,
        )
        if not scored:
            return None, [], party_size

        resolved_party = party_size
        if split_zone:
            inferred = self._infer_party_from_scored(scored)
            if inferred in (1, 8):
                solo_best = max(
                    (
                        item[0]
                        for item in scored
                        if item[4].ref_name.startswith("solo/")
                    ),
                    default=-1.0,
                )
                party8_best = max(
                    (
                        item[0]
                        for item in scored
                        if item[4].ref_name.startswith("party8/")
                    ),
                    default=-1.0,
                )
                resolved_party = inferred
                scored = self._filter_scored_by_party(scored, inferred)
                if is_debug():
                    logger.debug(
                        "[party] ref_infer zone=%s party_size=%s solo_best=%.3f party8_best=%.3f",
                        zone_id,
                        inferred,
                        solo_best,
                        party8_best,
                    )
                else:
                    logger.debug(
                        "party_size ref 추정 %s -> %s (solo/party8 혼합 스캔 후 필터)",
                        zone_id,
                        inferred,
                    )

        self._log_ref_top3(zone_id, scored, query_rel)
        display_scored = sorted(scored, key=lambda item: item[1], reverse=True)
        ranked = [
            self._to_match(
                (adjusted, ccorr, ssim),
                entry,
                terrain_score=combined,
                marker_dist=marker_dist,
                log_confirmed=False,
            )
            for adjusted, combined, ccorr, ssim, entry, marker_dist in display_scored[
                :top_k
            ]
        ]
        confident = self._pick_confident_match(scored, query_rel, zone_id)
        return confident, ranked, resolved_party

    def count_zone_refs(
        self,
        zone_id: str,
        party_size: int | None = None,
    ) -> int:
        return len(self._load_zone_entries(zone_id, party_size))

    def match_ranked(
        self,
        fragment: Image.Image,
        zone_id: str,
        party_size: int | None = None,
        top_k: int = 3,
    ) -> list[TreasureMapMatch]:
        """확신 여부와 무관하게 ref 상위 후보만 반환"""
        _confident, ranked, _party = self.match_with_ranked(
            fragment, zone_id, party_size=party_size, top_k=top_k
        )
        return ranked

    def identify_zone_from_terrain(
        self,
        fragment: Image.Image,
        *,
        party_size: int | None = None,
        refine_top_k: int | None = None,
    ) -> tuple[str | None, float, float]:
        """
        캡처 지형을 전 존 ref와 비교해 zone_id 추정.

        Returns:
            (zone_id, score, margin) — 확신 시 zone_id, 아니면 (None, best_score, margin)
        """
        ranked = self.rank_zones_by_terrain(
            fragment,
            party_size=party_size,
            refine_top_k=refine_top_k,
        )
        if not ranked:
            return None, 0.0, 0.0
        best = ranked[0]
        if (
            best.score >= self.TERRAIN_ZONE_CONFIDENT_SCORE
            and best.margin >= self.TERRAIN_ZONE_MIN_MARGIN
        ):
            return best.zone_id, best.score, best.margin
        return None, best.score, best.margin

    def rank_zones_by_terrain(
        self,
        fragment: Image.Image,
        *,
        party_size: int | None = None,
        refine_top_k: int | None = None,
    ) -> list[ZoneTerrainIdentification]:
        """전 존 지형 유사도 순위 — 디버그·QC용"""
        zone_ids = self.zone_ids_with_refs()
        if not zone_ids:
            return []

        query_ctx = self._build_matching_query_context(fragment)
        if query_ctx is None:
            return []

        coarse_scales, refine_scales, query_feat, query_weights = query_ctx
        t0 = time.perf_counter()

        coarse_rows: list[tuple[str, float, _RefEntry | None]] = []
        for zone_id in zone_ids:
            score, entry = self._zone_coarse_best_score(
                coarse_scales, zone_id, party_size
            )
            coarse_rows.append((zone_id, score, entry))

        coarse_rows.sort(key=lambda item: item[1], reverse=True)
        top_k = refine_top_k if refine_top_k is not None else self.TERRAIN_ZONE_REFINE_TOP

        final_rows: list[tuple[str, float, _RefEntry | None]] = []
        for zone_id, coarse_score, entry in coarse_rows[: max(1, top_k)]:
            if entry is None:
                final_rows.append((zone_id, coarse_score, None))
                continue
            refined, _ccorr, _ssim = self._refine_score_optimized(
                refine_scales,
                query_feat,
                query_weights,
                entry,
            )
            final_rows.append((zone_id, max(coarse_score, refined), entry))

        tail = coarse_rows[max(1, top_k) :]
        final_rows.extend((zone_id, score, entry) for zone_id, score, entry in tail)
        final_rows.sort(key=lambda item: item[1], reverse=True)

        if is_debug():
            logger.debug(
                "terrain_zone top5 (%.0fms): %s",
                (time.perf_counter() - t0) * 1000,
                [
                    (zone_id, round(score, 3), entry.ref_name if entry else None)
                    for zone_id, score, entry in final_rows[:5]
                ],
            )

        results: list[ZoneTerrainIdentification] = []
        for idx, (zone_id, score, entry) in enumerate(final_rows):
            if idx == 0:
                second = final_rows[1][1] if len(final_rows) > 1 else 0.0
                margin = score - max(second, 0.0)
            else:
                margin = 0.0
            results.append(
                ZoneTerrainIdentification(
                    zone_id=zone_id,
                    score=score,
                    margin=margin if idx == 0 else 0.0,
                    best_ref_name=entry.ref_name if entry else None,
                )
            )
        return results

    def _build_matching_query_context(
        self, fragment: Image.Image
    ) -> tuple[
        list[_ScalePack],
        list[_ScalePack],
        np.ndarray,
        np.ndarray,
    ] | None:
        """지형 매칭용 쿼리 feature — 존 루프 밖 1회 생성"""
        _patch, query_gray, query_center = (
            TreasureCaptureProcessor.prepare_matching_patch(fragment)
        )
        if query_gray.size == 0:
            return None

        query_feat = enhance_terrain_features(query_gray)
        query_weights = TreasureCaptureProcessor.build_soft_center_weights(
            query_gray.shape[0],
            query_gray.shape[1],
            query_center,
        )
        q_feat_c, q_weights_c = self._resize_pair(
            query_feat,
            query_weights,
            self.COARSE_MAX_SIDE,
        )
        coarse_scales = self._build_scale_packs(
            q_feat_c,
            q_weights_c,
            np.linspace(0.92, 1.08, self.COARSE_SCALE_STEPS),
        )
        refine_scales = self._build_scale_packs(
            query_feat,
            query_weights,
            np.linspace(0.88, 1.12, self.REFINE_SCALE_STEPS),
        )
        return coarse_scales, refine_scales, query_feat, query_weights

    def _zone_coarse_best_score(
        self,
        coarse_query_scales: list[_ScalePack],
        zone_id: str,
        party_size: int | None,
    ) -> tuple[float, _RefEntry | None]:
        """존 내 ref coarse 최고점 — cross-zone 식별용 (marker 보정 없음)"""
        entries = self._load_zone_entries(zone_id, party_size)
        if not entries:
            return -1.0, None

        best_score = -1.0
        best_entry: _RefEntry | None = None
        for entry in entries:
            score = self._coarse_score_optimized(coarse_query_scales, entry)
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_score, best_entry

    def _pick_confident_match(
        self,
        scored: list[tuple[float, float, float, float, _RefEntry, float]],
        query_rel: tuple[float, float] | None,
        zone_id: str,
    ) -> TreasureMapMatch | None:
        best_adj, best_combined, best_ccorr, best_ssim, best_entry, marker_dist = (
            scored[0]
        )
        second_adj = scored[1][0] if len(scored) > 1 else -1.0
        margin = best_adj - max(second_adj, 0.0)

        # 경로 A: 절대 점수 충분 + 최소 마진 (점수 높으면 마진 요구 완화)
        required_margin = (
            self.MIN_MARGIN
            if best_adj >= self.CONFIDENT_SCORE_HIGH
            else self.STRONG_MARGIN
        )
        high_score = best_adj >= self.CONFIDENT_SCORE and margin >= required_margin

        # 경로 B: 마진 우세 — 1등이 2등을 크게 앞서면 절대 점수 중간이어도 확정
        dominant = (
            margin >= self.DOMINANT_MARGIN
            and best_adj >= self.DOMINANT_SCORE_FLOOR
        )

        if high_score or dominant:
            return self._to_match(
                (best_adj, best_ccorr, best_ssim),
                best_entry,
                terrain_score=best_combined,
                marker_dist=marker_dist,
            )

        # 확신 미달 — marker rescue(비활성) / 순수 지형 마진 tie-break 시도
        if self.MARKER_ADJUST_ENABLED:
            marker_pick = self._pick_by_marker_alignment(scored, query_rel)
            if marker_pick is not None:
                return marker_pick
        terrain_pick = self._pick_by_terrain_margin(scored)
        if terrain_pick is not None:
            return terrain_pick

        logger.debug(
            "ref 확신 부족 %s adj=%.3f terrain=%.3f margin=%.3f",
            zone_id,
            best_adj,
            best_combined,
            margin,
        )
        return None

    def _log_ref_top3(
        self,
        zone_id: str,
        scored: list[tuple[float, float, float, float, _RefEntry, float]],
        query_rel: tuple[float, float] | None = None,
    ) -> None:
        logger.debug("query_rel=%s", query_rel)
        logger.debug(
            "ref ALL %s (n=%d): %s",
            zone_id,
            len(scored),
            [
                (
                    entry.ref_name,
                    round(adjusted, 3),
                    round(combined, 3),
                    round(ccorr, 3),
                    round(ssim, 3),
                    round(marker_dist, 3),
                    entry.marker_rel,
                )
                for adjusted, combined, ccorr, ssim, entry, marker_dist in scored
            ],
        )

    def _score_zone(
        self,
        fragment: Image.Image,
        zone_id: str,
        party_size: int | None,
        exclude_ref_names: set[str] | None = None,
        full_scan: bool = False,
    ) -> tuple[
        list[tuple[float, float, float, float, _RefEntry, float]],
        tuple[float, float] | None,
    ]:
        if is_debug_query():
            debug_path = self._data_dir.parent / "debug_query.png"
            try:
                fragment.save(debug_path)
                logger.debug(
                    "debug_query saved %s (%dx%d)",
                    debug_path,
                    fragment.width,
                    fragment.height,
                )
            except OSError as exc:
                logger.debug("debug_query save failed: %s", exc)

        t0 = time.perf_counter()
        entries = self._load_zone_entries(zone_id, party_size)
        if not entries:
            return [], None
        if exclude_ref_names:
            entries = [
                entry
                for entry in entries
                if entry.ref_name not in exclude_ref_names
            ]
        if not entries:
            return [], None

        # ref marker_rel은 원본 PNG 기준 — normalize 전 원본 쿼리에서 동일 좌표계로 추출
        query_rel = self._query_marker_rel(fragment)

        # 콘텐츠 스케일 탐색 — 여러 중앙 크롭 비율로 채점 후 마진 최대 스케일 채택.
        best_scored: list[tuple[float, float, float, float, _RefEntry, float]] | None = None
        best_margin = -1.0
        best_factor = 1.0
        for factor in self.CONTENT_SCALE_FACTORS:
            query_img = self._center_crop_fragment(fragment, factor)
            scored = self._score_query_image(
                query_img, entries, query_rel, full_scan
            )
            if not scored:
                continue
            margin = self._scored_margin(scored)
            if margin > best_margin:
                best_margin = margin
                best_scored = scored
                best_factor = factor
            # 첫 스케일(=1.0)만으로 이미 확신되면 나머지 생략(self-match·꽉 찬 캡처 속도).
            if (
                scored[0][1] >= self.CONTENT_SCALE_SKIP_SCORE
                and margin >= self.CONTENT_SCALE_SKIP_MARGIN
            ):
                break

        if best_scored is None:
            return [], query_rel

        logger.debug(
            "timing[_score_zone] zone=%s entries=%d full_scan=%s "
            "scale=%.2f margin=%.3f TOTAL=%.1fms",
            zone_id,
            len(entries),
            full_scan,
            best_factor,
            best_margin,
            (time.perf_counter() - t0) * 1000,
        )
        return best_scored, query_rel

    @staticmethod
    def _center_crop_fragment(
        fragment: Image.Image, factor: float
    ) -> Image.Image:
        """쿼리 중앙을 factor 비율로 크롭 — 인게임 지도가 ref보다 넓은 스케일 보정."""
        if factor >= 1.0:
            return fragment
        w, h = fragment.size
        cw = max(1, int(round(w * factor)))
        ch = max(1, int(round(h * factor)))
        left = (w - cw) // 2
        top = (h - ch) // 2
        return fragment.crop((left, top, left + cw, top + ch))

    @staticmethod
    def _scored_margin(
        scored: list[tuple[float, float, float, float, _RefEntry, float]],
    ) -> float:
        """지형 combined 기준 1등-2등 격차 (스케일 선택 신호)."""
        if not scored:
            return -1.0
        combined = sorted((item[1] for item in scored), reverse=True)
        if len(combined) < 2:
            return combined[0]
        return combined[0] - combined[1]

    def _score_query_image(
        self,
        query_img: Image.Image,
        entries: list[_RefEntry],
        query_rel: tuple[float, float] | None,
        full_scan: bool,
    ) -> list[tuple[float, float, float, float, _RefEntry, float]]:
        """단일 쿼리 이미지를 존 전체 ref와 채점 (coarse 선별 → refine)."""
        match_fragment = self._normalize_query_to_refs(query_img, entries)

        _patch, query_gray, query_center = (
            TreasureCaptureProcessor.prepare_matching_patch(match_fragment)
        )
        query_feat = enhance_terrain_features(query_gray)
        query_weights = TreasureCaptureProcessor.build_soft_center_weights(
            query_gray.shape[0],
            query_gray.shape[1],
            query_center,
        )

        q_feat_c, q_weights_c = self._resize_pair(
            query_feat,
            query_weights,
            self.COARSE_MAX_SIDE,
        )
        coarse_query_scales = self._build_scale_packs(
            q_feat_c,
            q_weights_c,
            np.linspace(0.92, 1.08, self.COARSE_SCALE_STEPS),
        )
        refine_query_scales = self._build_scale_packs(
            query_feat,
            query_weights,
            np.linspace(0.88, 1.12, self.REFINE_SCALE_STEPS),
        )

        coarse_scored: list[tuple[float, _RefEntry]] = []
        for entry in entries:
            score = self._coarse_score_optimized(coarse_query_scales, entry)
            coarse_scored.append((score, entry))
        coarse_scored.sort(key=lambda item: item[0], reverse=True)

        if full_scan:
            finalists = [entry for _score, entry in coarse_scored]
        else:
            refine_top_k = min(len(entries), self.REFINE_TOP_K)
            finalists = [entry for _score, entry in coarse_scored[:refine_top_k]]
            if query_rel is not None:
                seen = {entry.path for entry in finalists}
                by_marker = sorted(
                    entries,
                    key=lambda entry: self._marker_distance(entry, query_rel),
                )
                for entry in by_marker[:4]:
                    if entry.path in seen:
                        continue
                    if self._marker_distance(entry, query_rel) > 0.12:
                        break
                    finalists.append(entry)
                    seen.add(entry.path)

        refined_rows: list[tuple[float, float, float, _RefEntry]] = []
        for entry in finalists:
            combined, ccorr, ssim = self._refine_score_optimized(
                refine_query_scales,
                query_feat,
                query_weights,
                entry,
            )
            refined_rows.append((combined, ccorr, ssim, entry))

        terrain_clustered = self._terrain_scores_clustered(
            [combined for combined, _ccorr, _ssim, _entry in refined_rows]
        )

        scored: list[tuple[float, float, float, float, _RefEntry, float]] = []
        for combined, ccorr, ssim, entry in refined_rows:
            adjusted = self._marker_adjusted_score(
                combined,
                entry,
                query_rel,
                terrain_clustered=terrain_clustered,
            )
            marker_dist = self._marker_distance(entry, query_rel)
            scored.append((adjusted, combined, ccorr, ssim, entry, marker_dist))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _pick_by_terrain_margin(
        self,
        scored: list[tuple[float, float, float, float, _RefEntry, float]],
    ) -> TreasureMapMatch | None:
        """marker 보정 점수가 비슷할 때 순수 지형 유사도로 확정"""
        if len(scored) < 2:
            return None
        terrain_sorted = sorted(scored, key=lambda item: item[1], reverse=True)
        best_adj, best_combined, best_ccorr, best_ssim, best_entry, marker_dist = (
            terrain_sorted[0]
        )
        second_combined = terrain_sorted[1][1]
        margin = best_combined - max(second_combined, 0.0)
        if best_combined < self.CONFIDENT_SCORE - 0.04 or margin < self.STRONG_MARGIN:
            return None
        logger.debug(
            "ref terrain tie-break %s terrain=%.3f margin=%.3f",
            best_entry.path.name,
            best_combined,
            margin,
        )
        return self._to_match(
            (best_adj, best_ccorr, best_ssim),
            best_entry,
            terrain_score=best_combined,
            marker_dist=marker_dist,
        )

    def _pick_by_marker_alignment(
        self,
        scored: list[tuple[float, float, float, float, _RefEntry, float]],
        query_rel: tuple[float, float] | None,
    ) -> TreasureMapMatch | None:
        """지형 점수가 비슷할 때 X 상대 위치로 최종 후보 선택"""
        if query_rel is None:
            return None

        candidates: list[tuple[float, float, float, float, float, _RefEntry]] = []
        for adjusted, combined, ccorr, ssim, entry, dist in scored:
            if combined < self.MARKER_TERRAIN_FLOOR:
                continue
            candidates.append((dist, adjusted, combined, ccorr, ssim, entry))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_dist, best_adj, best_combined, best_ccorr, best_ssim, best_entry = (
            candidates[0]
        )
        second_dist = candidates[1][0] if len(candidates) > 1 else 1.0
        if best_dist > 0.10 or (second_dist - best_dist) < 0.04:
            return None

        logger.debug(
            "ref marker tie-break %s terrain=%.3f marker_dist=%.3f",
            best_entry.path.name,
            best_combined,
            best_dist,
        )
        return self._to_match(
            (best_adj, best_ccorr, best_ssim),
            best_entry,
            terrain_score=best_combined,
            marker_dist=best_dist,
        )

    @staticmethod
    def _to_match(
        scores: tuple[float, float, float],
        entry: _RefEntry,
        *,
        terrain_score: float = 0.0,
        marker_dist: float = 1.0,
        log_confirmed: bool = True,
    ) -> TreasureMapMatch:
        combined, ccorr, ssim = scores
        if log_confirmed:
            logger.debug(
                "ref 확정 %s score=%.3f",
                entry.path.name,
                combined,
            )
        return TreasureMapMatch(
            score=combined,
            ccorr_score=ccorr,
            ssim_score=ssim,
            treasure_x=entry.treasure_x,
            treasure_y=entry.treasure_y,
            template_path=entry.path,
            ref_name=entry.ref_name,
            terrain_score=terrain_score,
            marker_dist=marker_dist,
        )

    @staticmethod
    def _ref_target_size(entries: list[_RefEntry]) -> tuple[int, int]:
        """존 ref 원본 중 최대 크기 — 쿼리 업스케일 기준"""
        sizes = [e.source_size for e in entries if e.source_size[0] > 0]
        if not sizes:
            return (0, 0)
        return (max(w for w, _h in sizes), max(h for _w, h in sizes))

    def _normalize_query_to_refs(
        self,
        fragment: Image.Image,
        entries: list[_RefEntry],
    ) -> Image.Image:
        target_w, target_h = self._ref_target_size(entries)
        return TreasureCaptureProcessor.normalize_for_matching(
            fragment, target_w, target_h
        )

    @staticmethod
    def _query_marker_rel(fragment: Image.Image) -> tuple[float, float] | None:
        """캡처 지도 창 안 X의 정규화 위치 (0~1)"""
        rgb = np.array(fragment.convert("RGB"))
        return TreasureCaptureProcessor._find_treasure_x_marker_rel(rgb)

    def _ref_marker_rel(
        self,
        path: Path,
        *,
        ref_name: str | None = None,
        zone_id: str | None = None,
    ) -> tuple[float, float] | None:
        if ref_name and zone_id:
            item = self._ref_coords_index(zone_id).get(ref_name)
            if item is not None:
                if item.get("marker_skip"):
                    return None
                mrx = item.get("marker_rx")
                mry = item.get("marker_ry")
                if mrx is not None and mry is not None:
                    return float(mrx), float(mry)

        rgb = np.array(Image.open(path).convert("RGB"))
        return TreasureCaptureProcessor._find_treasure_x_marker_rel(rgb)

    @classmethod
    def _marker_distance(
        cls,
        entry: _RefEntry,
        query_rel: tuple[float, float] | None,
    ) -> float:
        if query_rel is None or entry.marker_rel is None:
            return 1.0
        return math.hypot(
            query_rel[0] - entry.marker_rel[0],
            query_rel[1] - entry.marker_rel[1],
        )

    @classmethod
    def _terrain_scores_clustered(cls, terrain_scores: list[float]) -> bool:
        """상위 terrain 점수가 좁게 몰리면 marker 타이브레이커 강화"""
        if len(terrain_scores) < 2:
            return False
        ordered = sorted(terrain_scores, reverse=True)
        best = ordered[0]
        if best < cls.MARKER_TERRAIN_FLOOR:
            return False
        second = ordered[1]
        if best - second > cls.TERRAIN_CLUSTER_MARGIN:
            return False
        top_band = [s for s in ordered if best - s <= cls.TERRAIN_CLUSTER_MARGIN]
        return len(top_band) >= 2

    @classmethod
    def _marker_adjusted_score(
        cls,
        terrain_score: float,
        entry: _RefEntry,
        query_rel: tuple[float, float] | None,
        *,
        terrain_clustered: bool = False,
    ) -> float:
        """지형 유사도 + X 상대 위치 일치 보정"""
        if not cls.MARKER_ADJUST_ENABLED:
            return terrain_score
        if query_rel is None or entry.marker_rel is None:
            return terrain_score
        dist = cls._marker_distance(entry, query_rel)
        if terrain_clustered:
            bonus = min(
                cls.MARKER_BONUS_CAP * 1.5,
                max(0.0, cls.MARKER_CLUSTER_BONUS_THRESHOLD - dist)
                * cls.MARKER_CLUSTER_BONUS_SCALE,
            )
            penalty = (
                max(0.0, dist - cls.MARKER_CLUSTER_PENALTY_START)
                * cls.MARKER_MISMATCH_PENALTY
            )
        else:
            bonus = min(cls.MARKER_BONUS_CAP, max(0.0, 0.10 - dist) * 1.25)
            penalty = max(0.0, dist - 0.15) * cls.MARKER_MISMATCH_PENALTY
        if terrain_score >= 0.85:
            bonus *= 0.55
            penalty *= 0.45
        return terrain_score + bonus - penalty

    @staticmethod
    def _build_scale_packs(
        feat: np.ndarray,
        weights: np.ndarray,
        scales: np.ndarray,
    ) -> list[_ScalePack]:
        fh, fw = feat.shape
        packs: list[_ScalePack] = []
        for scale in scales:
            sw = max(24, int(fw * scale))
            sh = max(24, int(fh * scale))
            packs.append(
                _ScalePack(
                    sw,
                    sh,
                    cv2.resize(feat, (sw, sh), interpolation=cv2.INTER_AREA),
                    cv2.resize(weights, (sw, sh), interpolation=cv2.INTER_AREA),
                )
            )
        return packs

    def _coarse_score_optimized(
        self,
        query_scales: list[_ScalePack],
        entry: _RefEntry,
    ) -> float:
        best = -1.0
        t_small = entry.feat_gray_coarse
        tw_small = entry.tpl_weights_coarse
        for pack in query_scales:
            ts = cv2.resize(t_small, (pack.sw, pack.sh), interpolation=cv2.INTER_AREA)
            wt = cv2.resize(tw_small, (pack.sw, pack.sh), interpolation=cv2.INTER_AREA)
            combined, _, _ = terrain_similarity_from_features(
                pack.feat,
                ts,
                ssim_weight=np.sqrt(pack.weights * wt),
            )
            best = max(best, combined)
        return best

    def _refine_score_optimized(
        self,
        query_scales: list[_ScalePack],
        query_feat: np.ndarray,
        query_weights: np.ndarray,
        entry: _RefEntry,
    ) -> tuple[float, float, float]:
        fh, fw = query_feat.shape
        best_combined = -1.0
        best_ccorr = -1.0
        best_ssim = -1.0

        for pack in query_scales:
            ts = cv2.resize(
                entry.feat_gray,
                (pack.sw, pack.sh),
                interpolation=cv2.INTER_AREA,
            )
            wt = cv2.resize(
                entry.tpl_weights,
                (pack.sw, pack.sh),
                interpolation=cv2.INTER_AREA,
            )
            combined, ccorr, ssim = terrain_similarity_from_features(
                pack.feat,
                ts,
                ssim_weight=np.sqrt(pack.weights * wt),
            )
            if combined > best_combined:
                best_combined, best_ccorr, best_ssim = combined, ccorr, ssim

        ts_full = cv2.resize(entry.feat_gray, (fw, fh), interpolation=cv2.INTER_AREA)
        wt_full = cv2.resize(entry.tpl_weights, (fw, fh), interpolation=cv2.INTER_AREA)
        combined, ccorr, ssim = terrain_similarity_from_features(
            query_feat,
            ts_full,
            ssim_weight=np.sqrt(query_weights * wt_full),
        )
        if combined > best_combined:
            best_combined, best_ccorr, best_ssim = combined, ccorr, ssim

        ts_aligned = phase_align_gray(query_feat, ts_full)
        combined_aligned, ccorr_a, ssim_a = terrain_similarity_from_features(
            query_feat,
            ts_aligned,
            ssim_weight=np.sqrt(query_weights * wt_full),
        )
        if combined_aligned > best_combined:
            best_combined, best_ccorr, best_ssim = combined_aligned, ccorr_a, ssim_a

        return best_combined, best_ccorr, best_ssim

    @staticmethod
    def _resize_pair(
        gray: np.ndarray,
        weights: np.ndarray,
        max_side: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = gray.shape
        longest = max(h, w)
        if longest <= max_side:
            return gray, weights
        scale = max_side / longest
        sw = max(24, int(w * scale))
        sh = max(24, int(h * scale))
        return (
            cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA),
            cv2.resize(weights, (sw, sh), interpolation=cv2.INTER_AREA),
        )

    def _cache_key(self, zone_id: str, folder: Path) -> str:
        return f"{zone_id}|{folder.as_posix()}"

    def _load_zone_folder(self, zone_id: str, folder: Path) -> list[_RefEntry]:
        fingerprint = self._folder_fingerprint(folder)
        cache_key = self._cache_key(zone_id, folder)
        cached = self._zone_entries.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

        t_load_start = time.perf_counter()
        entries: list[_RefEntry] = []
        spot_coords = self._zone_spot_coords(zone_id)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            ref_name = (
                f"{folder.name}/{path.name}"
                if folder.name in ("solo", "party8")
                else path.name
            )
            coords = self._resolve_ref_coords(
                path.stem,
                spot_coords,
                filename=ref_name,
                zone_id=zone_id,
            )
            if coords is None:
                logger.debug("ref 좌표 매핑 실패: %s", path.name)
                continue
            tx, ty = coords
            rgb = np.array(Image.open(path).convert("RGB"))
            source_h, source_w = rgb.shape[:2]
            marker_rel = self._ref_marker_rel(path, ref_name=ref_name, zone_id=zone_id)
            _patch, gray, weight_center = (
                TreasureCaptureProcessor.prepare_matching_patch(
                    Image.fromarray(rgb)
                )
            )
            tpl_weights = TreasureCaptureProcessor.build_soft_center_weights(
                gray.shape[0],
                gray.shape[1],
                weight_center,
            )
            feat_gray = enhance_terrain_features(gray)
            feat_gray_coarse, tpl_weights_coarse = self._resize_pair(
                feat_gray,
                tpl_weights,
                self.COARSE_MAX_SIDE,
            )
            entries.append(
                _RefEntry(
                    ref_name=ref_name,
                    path=path,
                    treasure_x=tx,
                    treasure_y=ty,
                    gray=gray,
                    feat_gray=feat_gray,
                    weight_center=weight_center,
                    tpl_weights=tpl_weights,
                    feat_gray_coarse=feat_gray_coarse,
                    tpl_weights_coarse=tpl_weights_coarse,
                    marker_rel=marker_rel,
                    source_size=(source_w, source_h),
                )
            )

        self._zone_entries[cache_key] = (fingerprint, entries)
        if not entries:
            logger.warning("ref 이미지 없음 또는 좌표 매핑 실패: %s", folder)
        logger.debug(
            "timing[_load_zone] zone=%s entries=%d load(decode+preprocess)=%.1fms (cache miss)",
            zone_id,
            len(entries),
            (time.perf_counter() - t_load_start) * 1000,
        )
        return entries

    def _load_zone_entries(
        self, zone_id: str, party_size: int | None
    ) -> list[_RefEntry]:
        folders = self._resolve_refs_folders(zone_id, party_size)
        if not folders:
            return []
        entries: list[_RefEntry] = []
        for folder in folders:
            entries.extend(self._load_zone_folder(zone_id, folder))
        return entries

    def _zone_spot_coords(self, zone_id: str) -> list[tuple[float, float]]:
        """zones.json spots treasure 좌표 목록 (그리드 순서)"""
        path = self._data_dir / "zones.json"
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        zone: dict | None = None
        if isinstance(data, list):
            zone = next((z for z in data if z.get("id") == zone_id), None)
        else:
            for items in data.values():
                for z in items:
                    if z.get("id") == zone_id:
                        zone = z
                        break
                if zone is not None:
                    break

        if zone is None:
            return []

        coords: list[tuple[float, float]] = []
        for spot in zone.get("spots", []):
            treasure = spot.get("treasure", {})
            tx = treasure.get("x")
            ty = treasure.get("y")
            if tx is None or ty is None:
                continue
            coords.append((float(tx), float(ty)))
        return coords

    def _resolve_ref_coords(
        self,
        stem: str,
        spot_coords: list[tuple[float, float]],
        filename: str | None = None,
        zone_id: str | None = None,
    ) -> tuple[float, float] | None:
        """data/treasure_ref_coords JSON → 파일명 X_Y → spot 순서"""
        if filename and zone_id:
            index = self._ref_coords_index(zone_id)
            if filename in index:
                item = index[filename]
                return float(item["x"]), float(item["y"])

        parsed = self._parse_filename_coords(stem)
        if parsed is not None:
            return parsed

        index_match = re.match(r"^(?:spot[_-]?)?(\d+)$", stem, re.IGNORECASE)
        if index_match and spot_coords:
            index = int(index_match.group(1)) - 1
            if 0 <= index < len(spot_coords):
                return spot_coords[index]

        return None

    def _ref_coords_index(self, zone_id: str) -> dict[str, dict]:
        cached = self._ref_coords_cache.get(zone_id)
        if cached is not None:
            return cached

        path = self._data_dir / "treasure_ref_coords" / f"{zone_id}.json"
        if not path.exists():
            self._ref_coords_cache[zone_id] = {}
            return {}

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            self._ref_coords_cache[zone_id] = {}
            return {}

        self._ref_coords_cache[zone_id] = data
        return data

    @staticmethod
    def _folder_fingerprint(folder: Path) -> str:
        parts: list[str] = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]

    COORD_MATCH_TOL = 0.02

    def _resolve_refs_folder(self, zone_id: str) -> Path | None:
        expansion = self._lookup_expansion(zone_id)
        if expansion:
            folder = self.refs_dir / expansion / zone_id
            if folder.is_dir():
                return folder
        legacy = self.refs_dir / zone_id
        if legacy.is_dir():
            return legacy
        return None

    def _resolve_refs_folders(
        self, zone_id: str, party_size: int | None
    ) -> list[Path]:
        base = self._resolve_refs_folder(zone_id)
        if base is None:
            return []

        solo = base / "solo"
        party8 = base / "party8"
        has_split = solo.is_dir() or party8.is_dir()
        if not has_split:
            return [base] if self._has_ref_images(base) else []

        if party_size == 1:
            return [solo] if solo.is_dir() and self._has_ref_images(solo) else []
        if party_size == 8:
            return [party8] if party8.is_dir() and self._has_ref_images(party8) else []

        folders: list[Path] = []
        if solo.is_dir() and self._has_ref_images(solo):
            folders.append(solo)
        if party8.is_dir() and self._has_ref_images(party8):
            folders.append(party8)
        return folders

    def _lookup_expansion(self, zone_id: str) -> str | None:
        path = self._data_dir / "zones.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            zone = next((z for z in data if z.get("id") == zone_id), None)
            exp = zone.get("expansion") if zone else None
            return str(exp) if exp else None
        for expansion, items in data.items():
            for zone in items:
                if zone.get("id") == zone_id:
                    return str(expansion)
        return None

    def _allowed_coords(
        self,
        zone_id: str,
        party_size: int | None,
    ) -> set[tuple[float, float]] | None:
        if party_size not in (1, 8):
            return None

        path = self._data_dir / "zones.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        zone: dict | None = None
        if isinstance(data, list):
            zone = next((z for z in data if z.get("id") == zone_id), None)
        else:
            for items in data.values():
                for z in items:
                    if z.get("id") == zone_id:
                        zone = z
                        break
                if zone is not None:
                    break

        if zone is None:
            return None

        spots = zone.get("spots", [])
        count = len(spots)
        if count == 8:
            return None
        if count == 16:
            half = spots[:8] if party_size == 1 else spots[8:]
            return {
                (float(s["treasure"]["x"]), float(s["treasure"]["y"]))
                for s in half
                if s.get("treasure", {}).get("x") is not None
            }
        return None

    @staticmethod
    def _parse_filename_coords(stem: str) -> tuple[float, float] | None:
        m = re.match(r"^([\d.]+)[_-]([\d.]+)$", stem)
        if not m:
            return None
        return float(m.group(1)), float(m.group(2))

    @staticmethod
    def _has_ref_images(folder: Path) -> bool:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                return True
            if p.is_dir():
                for child in p.iterdir():
                    if child.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        return True
        return False
