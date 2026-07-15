"""보물지도 캡처 → OCR → ref 1:1 매칭 → (실패 시) detail fallback"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mss
from PIL import Image

from src.app_config import is_debug
from src.models.recognition_result import RecognitionResult
from src.models.ref_candidate import RefCandidate
from src.services.coordinate_service import CoordinateService
from src.services.detail_map_locator import DetailLocateResult, DetailMapLocator
from src.services.map_pack_service import MapPackService
from src.services.match_stats import MatchStatsService
from src.services.treasure_capture import TreasureCaptureProcessor
from src.services.treasure_map_matcher import TreasureMapMatch, TreasureMapMatcher
from src.services.user_ref_learn import UserRefLearnService

logger = logging.getLogger(__name__)


@dataclass
class RematchContext:
    """동일 캡처로 ref 재검색할 때 사용"""

    map_window: Image.Image
    zone_hint: dict
    party_size: int | None
    capture_image: Image.Image | None = None
    party_size_locked: bool = False


class MapAnalyzer:
    """
    [1] OCR 지역 확정
    [2] TreasureMapMatcher — ref DB 1:1 비교 (좌표는 ref 파일명/JSON)
    [3] ref 후보 없을 때만 DetailMapLocator fallback
    """

    MIN_FOCUS_SIZE = 80
    BANNER_MIN_SCORE = 0.55
    REF_MATCH_TOP_K = 5
    TERRAIN_ZONE_MIN_SCORE = TreasureMapMatcher.TERRAIN_ZONE_CONFIDENT_SCORE
    TERRAIN_ZONE_MIN_MARGIN = TreasureMapMatcher.TERRAIN_ZONE_MIN_MARGIN

    def __init__(
        self,
        coordinate_service: CoordinateService,
        map_pack: MapPackService | None = None,
    ) -> None:
        self.coordinate_service = coordinate_service
        self.map_pack = map_pack or MapPackService(
            data_dir=coordinate_service.data_dir,
            bundled_maps_dir=coordinate_service.maps_dir,
        )
        self.capture_processor = TreasureCaptureProcessor(coordinate_service)
        self.detail_locator = DetailMapLocator(coordinate_service)
        self.map_matcher = TreasureMapMatcher(
            assets_dir=coordinate_service.maps_dir.parent,
            data_dir=coordinate_service.data_dir,
        )
        self.user_learn = UserRefLearnService()
        self.match_stats = MatchStatsService()
        self._rematch_context: RematchContext | None = None
        self._party_size_by_zone: dict[str, int] = {}

    def capture_region(self, rect: tuple[int, int, int, int]) -> Image.Image:
        """지정 영역 스크린샷 캡처 (x, y, width, height)"""
        x, y, w, h = rect
        with mss.mss() as sct:
            monitor = {"left": x, "top": y, "width": w, "height": h}
            shot = sct.grab(monitor)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def analyze(
        self,
        rect: tuple[int, int, int, int],
        manual_zone: Optional[str] = None,
        manual_coords: Optional[tuple[float, float]] = None,
        *,
        trust_frame: bool = True,
    ) -> RecognitionResult:
        t_start = time.perf_counter()
        raw = self.capture_region(rect)
        t_capture = time.perf_counter()

        if manual_zone and manual_coords:
            zone = self.coordinate_service.get_zone(manual_zone)
            if zone is None:
                zone = self.coordinate_service.find_zone_by_name(manual_zone)
            if zone is None:
                raise ValueError(f"알 수 없는 지역: {manual_zone}")
            return self.coordinate_service.build_result(
                zone, manual_coords[0], manual_coords[1]
            )

        if not self.map_pack.is_ready():
            raise ValueError(
                "지도 데이터가 설치되지 않았습니다.\n"
                "프로그램을 다시 실행해 지도 데이터 설치를 완료하세요."
            )

        # 1) 지역명 OCR — recognition_box 확정 캡처는 고정 슬롯(trust_frame)만 사용
        locked_zone: Optional[dict] = None
        locked_score = 0.0
        locked_text: Optional[str] = None
        best_candidate = raw
        best_map_window: Image.Image | None = raw

        if trust_frame:
            locked_zone, locked_score, locked_text, best_map_window = (
                self.capture_processor.resolve_banner_zone(
                    raw, refocus=False, trust_frame=True
                )
            )
            if best_map_window is None:
                best_map_window = raw
            t_candidates = t_capture
        else:
            candidates = list(self.capture_processor.extract_map_candidates(raw))
            t_candidates = time.perf_counter()
            raw_area = raw.width * raw.height
            best_map_window = None

            for idx, candidate in enumerate(candidates):
                tight = self.capture_processor._capture_is_tight_map_frame(candidate)
                already_extracted = candidate is not raw or tight
                refocus = (
                    not already_extracted
                    and not tight
                    and candidate.width * candidate.height >= raw_area * 0.92
                )
                zone, score, text, map_window = (
                    self.capture_processor.resolve_banner_zone(
                        candidate, refocus=refocus
                    )
                )
                if zone is not None and score > locked_score:
                    locked_zone = zone
                    locked_score = score
                    locked_text = text
                    best_candidate = candidate
                    if map_window is not None:
                        best_map_window = map_window
                    if locked_score >= TreasureCaptureProcessor.BANNER_OCR_EARLY_EXIT:
                        break
                if idx > 0 and locked_score >= 0.72:
                    break

        t_banner = time.perf_counter()
        logger.debug(
            "timing[analyze] capture=%.1fms candidates=%.1fms banner_ocr=%.1fms "
            "locked_zone=%s locked_score=%.3f trust_frame=%s",
            (t_capture - t_start) * 1000,
            (t_candidates - t_capture) * 1000,
            (t_banner - t_candidates) * 1000,
            locked_zone.get("id") if locked_zone else None,
            locked_score,
            trust_frame,
        )

        if locked_zone is None or locked_score < self.BANNER_MIN_SCORE:
            import numpy as np

            probe_rgb = np.array(best_candidate.convert("RGB"))
            map_probe = best_map_window or best_candidate
            terrain_zone = self._try_resolve_zone_from_terrain(map_probe)
            if terrain_zone is not None:
                locked_zone = terrain_zone
                locked_score = self.TERRAIN_ZONE_MIN_SCORE
                logger.info(
                    "지형 매칭으로 지역 확정: %s",
                    locked_zone.get("id"),
                )
            elif not self.capture_processor._top_header_has_title_ink(probe_rgb):
                raise ValueError(
                    "상단 지역명이 잘렸습니다.\n"
                    "• 네모를 위로 조금 올려 지역명(예: 라비린토스)이 상단에 보이게 해 주세요\n"
                    "• 빨간 X와 하단 1/8 아이콘도 함께 들어오게 맞춰 주세요"
                )
            if not self.capture_processor.ocr_available:
                raise ValueError(
                    "지역명 OCR을 사용할 수 없습니다.\n"
                    "• 프로그램 설치가 손상되었을 수 있습니다. zip을 다시 압축 해제해 보세요"
                )
            raise ValueError(
                "지역명을 읽지 못했습니다.\n"
                "• 상단 지역명 글자가 보이는데 인식만 실패한 경우, 프로그램을 완전히 종료 후 다시 실행해 보세요\n"
                "• 네모 안에는 보물지도 양피지 창만 들어오게 해 주세요 (소지품·다이얼로그 제외)\n"
                "• 상단 지역명·빨간 X·하단 1/8 아이콘이 선명하게 보이는지 확인해 주세요"
            )

        if trust_frame:
            map_for_match = best_map_window or raw
            map_for_ocr = map_for_match
        else:
            raw_area = raw.width * raw.height
            refocus_best = (
                not self.capture_processor._capture_is_tight_map_frame(best_candidate)
                and best_candidate is raw
                and best_candidate.width * best_candidate.height >= raw_area * 0.92
            )
            if best_map_window is not None:
                map_for_ocr = best_map_window
            else:
                map_for_ocr = self.capture_processor.localize_for_zone_ocr(
                    best_candidate, refocus=refocus_best
                )
            working_best = (
                self.capture_processor.focus_map_window(best_candidate)
                if refocus_best
                else best_candidate
            )
            map_for_match = self.capture_processor._refine_map_window_crop(
                working_best, map_for_ocr
            )
            map_for_match = self.capture_processor._trim_map_window_margins(
                map_for_match
            )

        # 2) 파티 인원
        party_size, party_uncertain = self._resolve_party_size(
            locked_zone["id"], raw, map_for_ocr
        )
        party_size, party_uncertain = self._confirm_party_with_map_refs(
            locked_zone["id"],
            map_for_match,
            party_size,
            party_uncertain,
        )
        t_party = time.perf_counter()
        logger.debug(
            "timing[analyze] party_size=%.1fms zone=%s party=%s uncertain=%s",
            (t_party - t_banner) * 1000,
            locked_zone["id"],
            party_size,
            party_uncertain,
        )

        # 3) 이미지 매칭
        self.map_matcher.preload_zone(locked_zone["id"], party_size=party_size)
        t_preload = time.perf_counter()
        result = self._analyze_once(
            raw,
            locked_zone=locked_zone,
            locked_party_size=party_size,
            locked_party_uncertain=party_uncertain,
            locked_map_window=map_for_match,
            strict_quality=False,
            refocus=False,
            trust_frame=trust_frame,
        )
        t_end = time.perf_counter()
        logger.debug(
            "timing[analyze] preload=%.1fms analyze_once=%.1fms TOTAL=%.1fms zone=%s",
            (t_preload - t_party) * 1000,
            (t_end - t_preload) * 1000,
            (t_end - t_start) * 1000,
            locked_zone["id"],
        )
        if result is not None:
            return result
        zone_name = locked_zone.get("name_ko", locked_zone["id"])
        raise ValueError(
            f"'{zone_name}'(으)로 인식했지만 좌표를 찾지 못했습니다.\n"
            "• 상단 지역명·빨간 X·하단 1/8 아이콘이 잘 보이는지 확인하세요\n"
            "• 네모 안에 보물지도 양피지 창 전체가 들어오도록 맞춰 주세요"
        )

    def _resolve_zone_from_banner(
        self,
        candidates: list[Image.Image],
        raw_area: int,
    ) -> tuple[Optional[dict], float, Optional[str]]:
        best_zone: Optional[dict] = None
        best_score = 0.0
        best_text: Optional[str] = None

        for candidate in candidates:
            refocus = candidate.width * candidate.height >= raw_area * 0.92
            _zone, score, _text, _map_window = self.capture_processor.resolve_banner_zone(
                candidate, refocus=refocus
            )
            if _zone is not None and score > best_score:
                best_zone = _zone
                best_score = score
                best_text = _text
                if best_score >= TreasureCaptureProcessor.BANNER_OCR_EARLY_EXIT:
                    return best_zone, best_score, best_text

        return best_zone, best_score, best_text

    def _try_resolve_zone_from_terrain(
        self, map_window: Image.Image
    ) -> Optional[dict]:
        """OCR 실패 시 전 존 지형 비교로 zone 후보 확정"""
        zone_id, score, margin = self.map_matcher.identify_zone_from_terrain(
            map_window
        )
        if zone_id is None:
            logger.debug(
                "terrain_zone fallback 실패: score=%.3f margin=%.3f",
                score,
                margin,
            )
            return None
        zone = self.coordinate_service.get_zone(zone_id)
        if zone is None:
            return None
        logger.debug(
            "terrain_zone fallback 성공: %s score=%.3f margin=%.3f",
            zone_id,
            score,
            margin,
        )
        return zone

    def _analyze_once(
        self,
        raw: Image.Image,
        *,
        locked_zone: dict | None,
        locked_party_size: int | None = None,
        locked_party_uncertain: bool | None = None,
        locked_map_window: Image.Image | None = None,
        strict_quality: bool,
        refocus: bool = True,
        trust_frame: bool = False,
    ) -> RecognitionResult:
        t0 = time.perf_counter()
        if locked_map_window is not None and locked_zone is not None:
            map_window = locked_map_window
            zone_hint = locked_zone
            if not trust_frame:
                map_window = self.capture_processor._strip_rows_above_banner(
                    map_window
                )
            focused = self.capture_processor.extract_map_content(map_window)
        else:
            focused, zone_hint, map_window = self.capture_processor.prepare(
                raw,
                refocus=refocus,
                trust_frame=trust_frame,
                skip_banner_ocr=locked_zone is not None,
            )
        t_prepare = time.perf_counter()
        logger.debug(
            "prepare: raw=%dx%d -> map_window=%dx%d (%.1fms)",
            raw.width,
            raw.height,
            map_window.width,
            map_window.height,
            (t_prepare - t0) * 1000,
        )

        if min(map_window.width, map_window.height) < self.MIN_FOCUS_SIZE:
            raise ValueError(
                "보물지도 UI가 너무 작게 캡처되었습니다.\n"
                "• 보물지도 양피지 창 전체가 선택 영역에 들어오도록 드래그하세요"
            )

        t_banner = t_prepare
        if locked_zone is not None:
            zone_hint = locked_zone
        else:
            zone_hint, banner_score, _text = (
                self.capture_processor.detect_zone_from_banner_scored(map_window)
            )
            t_banner = time.perf_counter()
            if zone_hint is None or banner_score < self.BANNER_MIN_SCORE:
                if strict_quality:
                    if not self.capture_processor.ocr_available:
                        raise ValueError(
                            "지역명 OCR을 사용할 수 없습니다.\n"
                            "• 프로그램 설치가 손상되었을 수 있습니다. zip을 다시 압축 해제해 보세요"
                        )
                    raise ValueError(
                        "지역명을 읽지 못했습니다.\n"
                        "• 상단 지역명(예: 검은장막 숲 동부 삼림)이 잘 보이게 맞춰 주세요"
                    )
                raise ValueError("지역명 OCR 신뢰도 부족")

        if locked_party_size in (1, 8):
            party_size = locked_party_size
            party_uncertain = bool(locked_party_uncertain)
        else:
            party_size, party_uncertain = self._resolve_party_size(
                zone_hint["id"], raw, map_window
            )
        t_party = time.perf_counter()
        if is_debug():
            logger.debug(
                "[party] analyze_once zone=%s party_size=%s uncertain=%s map_window=%dx%d",
                zone_hint["id"],
                party_size,
                party_uncertain,
                map_window.width,
                map_window.height,
            )
        else:
            logger.debug(
                "party_size 확정 %s -> %s (uncertain=%s)",
                zone_hint["id"],
                party_size,
                party_uncertain,
            )
        self._rematch_context = RematchContext(
            map_window=map_window.copy(),
            zone_hint=zone_hint,
            party_size=party_size,
            capture_image=raw.copy(),
            party_size_locked=party_size in (1, 8) and not party_uncertain,
        )
        if party_size in (1, 8) and not party_uncertain:
            self._store_confirmed_party_size(zone_hint["id"], party_size)
        result = self._resolve_coordinates(
            map_window,
            zone_hint,
            party_size,
            party_uncertain=party_uncertain,
        )
        t_resolve = time.perf_counter()
        logger.debug(
            "timing[_analyze_once] zone=%s prepare=%.1fms banner_ocr=%.1fms "
            "party_size=%.1fms resolve_coordinates=%.1fms result=%s",
            zone_hint["id"],
            (t_prepare - t0) * 1000,
            (t_banner - t_prepare) * 1000,
            (t_party - t_banner) * 1000,
            (t_resolve - t_party) * 1000,
            "ok" if result is not None else "none",
        )
        if result is not None:
            return result

        zone_name = zone_hint.get("name_ko", zone_hint["id"])
        raise ValueError(
            f"'{zone_name}'(으)로 인식했지만 좌표를 찾지 못했습니다.\n"
            "• 상단 지역명·빨간 X·하단 1/8 아이콘이 잘 보이는지 확인하세요"
        )

    def _pick_best_candidate(
        self,
        candidates: list[Image.Image],
        raw_area: int,
    ) -> Image.Image:
        best_candidate = candidates[0]
        best_score = -1.0
        for candidate in candidates:
            refocus = candidate.width * candidate.height >= raw_area * 0.92
            _zone, score, _text, _map_window = self.capture_processor.resolve_banner_zone(
                candidate, refocus=refocus
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _analyze_best_candidate(
        self,
        candidates: list[Image.Image],
        zone: dict,
        raw_area: int,
    ) -> RecognitionResult | None:
        candidate = self._pick_best_candidate(candidates, raw_area)
        refocus = candidate.width * candidate.height >= raw_area * 0.92
        _focused, _zone, map_window = self.capture_processor.prepare(
            candidate, refocus=refocus
        )

        party_size, party_uncertain = self._resolve_party_size(
            zone["id"], candidate, map_window
        )
        return self._resolve_coordinates(
            map_window,
            zone,
            party_size,
            party_uncertain=party_uncertain,
        )

    def _resolve_party_size(
        self,
        zone_id: str,
        capture_image: Image.Image,
        map_window: Image.Image | None = None,
    ) -> tuple[int | None, bool]:
        """
        1/8 파티 인원 — 지역명까지 확정된 map_window를 우선 사용.

        반환: (party_size, uncertain) — uncertain이면 캐시/추정값이라 재시도 허용.
        """
        sources: list[tuple[str, Image.Image]] = []
        if map_window is not None:
            sources.append(("map_window", map_window))
        if capture_image is not None and capture_image is not map_window:
            sources.append(("capture", capture_image))

        for label, source in sources:
            detected = self.capture_processor.detect_party_size(
                source, debug_source=label
            )
            if detected in (1, 8):
                self._party_size_by_zone[zone_id] = detected
                if is_debug():
                    logger.debug(
                        "[party] resolve zone=%s party_size=%s uncertain=False via=%s image=%dx%d",
                        zone_id,
                        detected,
                        label,
                        source.width,
                        source.height,
                    )
                else:
                    logger.debug("party_size 인식 %s -> %s", zone_id, detected)
                return detected, False

        party_probe = map_window if map_window is not None else capture_image
        if party_probe is not None:
            detected = self.capture_processor.detect_party_size_aggressive(
                party_probe, debug_source="aggressive"
            )
            if detected in (1, 8):
                self._party_size_by_zone[zone_id] = detected
                if is_debug():
                    logger.debug(
                        "[party] resolve zone=%s party_size=%s uncertain=False via=aggressive image=%dx%d",
                        zone_id,
                        detected,
                        party_probe.width,
                        party_probe.height,
                    )
                else:
                    logger.debug("party_size 강화 OCR %s -> %s", zone_id, detected)
                return detected, False

        if capture_image is not None and capture_image is not party_probe:
            detected = self.capture_processor.detect_party_size_aggressive(
                capture_image, debug_source="aggressive_capture"
            )
            if detected in (1, 8):
                self._party_size_by_zone[zone_id] = detected
                return detected, False

        cached = self._party_size_by_zone.get(zone_id)
        if cached in (1, 8):
            if is_debug():
                logger.debug(
                    "[party] resolve zone=%s party_size=%s uncertain=True via=session_cache",
                    zone_id,
                    cached,
                )
            else:
                logger.debug("party_size 캐시 %s -> %s (재감지 실패)", zone_id, cached)
            return cached, True

        learned_hint = self.user_learn.get_party_sizes().get(zone_id)
        if learned_hint in (1, 8):
            if is_debug():
                logger.debug(
                    "[party] resolve zone=%s party_size=%s uncertain=True via=learn_hint",
                    zone_id,
                    learned_hint,
                )
            return learned_hint, True

        if is_debug():
            logger.debug(
                "[party] resolve zone=%s party_size=None uncertain=True via=none",
                zone_id,
            )
        return None, True

    def _store_confirmed_party_size(self, zone_id: str, party_size: int | None) -> None:
        if party_size in (1, 8):
            self._party_size_by_zone[zone_id] = party_size

    def _confirm_party_with_map_refs(
        self,
        zone_id: str,
        map_window: Image.Image | None,
        party_size: int | None,
        uncertain: bool,
    ) -> tuple[int | None, bool]:
        """16칸 지역 — 아이콘 감지 후 solo/party8 ref coarse로 인원 재확인"""
        if map_window is None or not self.map_matcher.has_split_party_refs(zone_id):
            return party_size, uncertain

        solo_score, party8_score = self.map_matcher.compare_party_folder_coarse(
            map_window, zone_id
        )
        if solo_score < 0 and party8_score < 0:
            return party_size, uncertain

        margin = 0.035
        trace = self.capture_processor.last_party_trace
        icon_reason = trace.reason if trace else ""

        if solo_score - party8_score >= margin:
            if party_size != 1:
                logger.debug(
                    "party ref probe %s -> 1 (solo=%.3f party8=%.3f icon=%s)",
                    party_size,
                    solo_score,
                    party8_score,
                    icon_reason,
                )
            self._store_confirmed_party_size(zone_id, 1)
            return 1, False
        if party8_score - solo_score >= margin:
            if party_size != 8:
                logger.debug(
                    "party ref probe %s -> 8 (solo=%.3f party8=%.3f icon=%s)",
                    party_size,
                    solo_score,
                    party8_score,
                    icon_reason,
                )
            self._store_confirmed_party_size(zone_id, 8)
            return 8, False

        if party_size == 8 and solo_score > party8_score + 0.015:
            logger.debug(
                "party ref probe weak8 -> 1 (solo=%.3f party8=%.3f icon=%s)",
                solo_score,
                party8_score,
                icon_reason,
            )
            self._store_confirmed_party_size(zone_id, 1)
            return 1, False
        if party_size == 1 and party8_score > solo_score + 0.015:
            logger.debug(
                "party ref probe weak1 -> 8 (solo=%.3f party8=%.3f)",
                solo_score,
                party8_score,
            )
            self._store_confirmed_party_size(zone_id, 8)
            return 8, False

        if uncertain and party_size not in (1, 8):
            chosen = 1 if solo_score >= party8_score else 8
            self._store_confirmed_party_size(zone_id, chosen)
            return chosen, False

        return party_size, uncertain

    def rematch_refs(self, excluded_ref_names: list[str]) -> RecognitionResult:
        """이미 표시한 ref 후보를 제외하고 동일 캡처로 재검색"""
        ctx = self._rematch_context
        if ctx is None:
            raise ValueError("재검색할 캡처가 없습니다. 먼저 보물지도를 인식하세요.")

        exclude_set = set(excluded_ref_names)
        if ctx.party_size_locked and ctx.party_size in (1, 8):
            party_size = ctx.party_size
        elif ctx.party_size in (1, 8):
            party_size = ctx.party_size
        else:
            party_size, _party_uncertain = self._resolve_party_size(
                ctx.zone_hint["id"],
                ctx.capture_image or ctx.map_window,
                ctx.map_window,
            )
            ctx.party_size = party_size

        confident, ranked, _resolved_party = self.map_matcher.match_with_ranked(
            ctx.map_window,
            ctx.zone_hint["id"],
            party_size=party_size,
            top_k=self.REF_MATCH_TOP_K,
            exclude_ref_names=exclude_set,
            full_scan=True,
        )
        if not ranked:
            raise ValueError(
                "더 이상 표시할 ref 후보가 없습니다.\n"
                "후보 중 하나를 눌러 상세 지도로 확인해 주세요."
            )

        result = self._finalize_ref_result(
            ctx.zone_hint,
            party_size,
            confident,
            ranked,
            excluded_ref_names=list(excluded_ref_names),
        )
        logger.debug(
            "ref 재검색 %s party=%s 후보 %s (제외 %d개)",
            ctx.zone_hint["id"],
            party_size,
            [m.ref_name for m in ranked],
            len(exclude_set),
        )
        return result

    def confirm_user_ref(self, candidate: RefCandidate) -> tuple[RecognitionResult, int]:
        """사용자가 고른 ref를 학습 캐시에 저장"""
        ctx = self._rematch_context
        if ctx is None:
            raise ValueError("캡처 정보가 없습니다. 다시 인식해 주세요.")

        hits = self.user_learn.confirm(
            ctx.zone_hint["id"],
            candidate.ref_name,
            candidate.x,
            candidate.y,
            ctx.map_window,
            party_size=ctx.party_size,
        )
        zone = self.coordinate_service.get_effective_zone(ctx.zone_hint["id"])
        if zone is None:
            zone = ctx.zone_hint

        result = self.coordinate_service.build_result(zone, candidate.x, candidate.y)
        result.match_score = candidate.score
        result.match_source = "ref"
        result.confirmed_ref_name = candidate.ref_name
        result.learn_hits = hits
        logger.debug(
            "사용자 확정 %s %s (%.2f, %.2f) hits=%d",
            ctx.zone_hint["id"],
            candidate.ref_name,
            candidate.x,
            candidate.y,
            hits,
        )
        return result, hits

    def _try_learned_from_ref_ranking(
        self,
        zone_hint: dict,
        ranked: list[TreasureMapMatch],
    ) -> RecognitionResult | None:
        """
        지문 매칭 실패 시 — ref 1순위가 학습 확정 ref와 같으면
        후보 UI 없이 학습 좌표로 바로 반환.
        """
        top_entry = self.user_learn.get_top_entry(zone_hint["id"])
        if top_entry is None or not ranked:
            return None

        hits = int(top_entry.get("hits", 0))
        learned_name = str(top_entry.get("ref_name", ""))
        if hits < 2 or not learned_name:
            return None

        best = ranked[0]
        if not self._same_ref_name(best.ref_name, learned_name):
            return None

        from src.services.user_ref_learn import LearnedRefHit

        hit = LearnedRefHit(
            ref_name=learned_name,
            x=float(top_entry["x"]),
            y=float(top_entry["y"]),
            score=float(best.score),
            hits=hits,
        )
        logger.debug(
            "학습+ref 일치 %s %s hits=%d score=%.3f",
            zone_hint["id"],
            learned_name,
            hits,
            hit.score,
        )
        return self._build_result_from_learned(zone_hint, hit)

    def _resolve_coordinates(
        self,
        map_window: Image.Image,
        zone_hint: dict,
        party_size: int | None,
        excluded_ref_names: list[str] | None = None,
        *,
        party_uncertain: bool = False,
    ) -> RecognitionResult | None:
        """ref DB 우선 — 학습 캐시 → ref 스캔 → detail(최후)"""
        exclude_set = set(excluded_ref_names or [])

        scan_party = party_size
        zone_id = zone_hint["id"]
        if (
            party_uncertain
            and scan_party in (1, 8)
            and self.map_matcher.has_split_party_refs(zone_id)
        ):
            # 8인 테스트 후 1인 지도 — 캐시 오염 시 solo/party8 동시 스캔
            scan_party = None
            logger.debug(
                "party_size 불확실 → solo/party8 혼합 ref 스캔 %s",
                zone_id,
            )

        if not exclude_set:
            learned = self.user_learn.try_fast_match(
                zone_hint["id"], map_window, party_size=scan_party or party_size
            )
            if learned is not None:
                result = self._build_result_from_learned(zone_hint, learned)
                logger.debug(
                    "학습 캐시 즉시 확정 %s %s score=%.3f",
                    zone_hint["id"],
                    learned.ref_name,
                    learned.score,
                )
                return result

        full_scan = bool(exclude_set)
        confident_match, ranked, resolved_party = self.map_matcher.match_with_ranked(
            map_window,
            zone_hint["id"],
            party_size=scan_party,
            top_k=self.REF_MATCH_TOP_K,
            exclude_ref_names=exclude_set or None,
            full_scan=full_scan,
        )
        selected_party_size = resolved_party if resolved_party in (1, 8) else party_size
        if selected_party_size in (1, 8):
            if exclude_set:
                selected_party_size = party_size if party_size in (1, 8) else selected_party_size
            else:
                self._party_size_by_zone[zone_hint["id"]] = selected_party_size
                if self._rematch_context is not None:
                    self._rematch_context.party_size = selected_party_size

        alternate_party = self._alternate_party_size(selected_party_size)
        # 혼합 스캔으로 이미 폴더를 골랐으면 점수만으로 반대 폴더로 바꾸지 않음
        if (
            alternate_party is not None
            and party_uncertain
            and scan_party is not None
        ):
            alt_confident, alt_ranked, alt_resolved = self.map_matcher.match_with_ranked(
                map_window,
                zone_hint["id"],
                party_size=alternate_party,
                top_k=self.REF_MATCH_TOP_K,
                exclude_ref_names=exclude_set or None,
                full_scan=full_scan,
            )
            use_alternate = False
            if not ranked and alt_ranked:
                use_alternate = True
            elif alt_confident is not None and confident_match is None:
                use_alternate = True
            elif alt_ranked and (
                not ranked or alt_ranked[0].score > ranked[0].score + 0.02
            ):
                use_alternate = True

            if use_alternate:
                logger.debug(
                    "party_size 재시도 채택 %s: %s -> %s (primary=%s alt=%s)",
                    zone_hint["id"],
                    selected_party_size,
                    alternate_party,
                    round(ranked[0].score, 3) if ranked else None,
                    round(alt_ranked[0].score, 3) if alt_ranked else None,
                )
                confident_match = alt_confident
                ranked = alt_ranked
                selected_party_size = alt_resolved if alt_resolved in (1, 8) else alternate_party
                if self._rematch_context is not None:
                    self._rematch_context.party_size = selected_party_size
                self._party_size_by_zone[zone_hint["id"]] = selected_party_size

        if ranked and not exclude_set:
            assisted = self._try_learned_from_ref_ranking(zone_hint, ranked)
            if assisted is not None:
                return assisted

        if ranked:
            return self._finalize_ref_result(
                zone_hint,
                selected_party_size,
                confident_match,
                ranked,
                excluded_ref_names=list(exclude_set),
            )

        if self.map_matcher.count_zone_refs(zone_hint["id"], selected_party_size) > 0:
            logger.debug("ref 후보 없음(전부 제외됨) %s", zone_hint["id"])
            return None

        logger.debug("ref DB 없음 → detail fallback %s", zone_hint["id"])
        detail_result = self._locate_detail(map_window, zone_hint)
        if detail_result is not None:
            detail_result.match_source = "detail"
            detail_result.can_rematch = False
            logger.debug(
                "좌표 확정: detail %s (%.2f, %.2f) score=%.3f",
                zone_hint["id"],
                detail_result.x,
                detail_result.y,
                detail_result.match_score or 0.0,
            )
            return detail_result

        return None

    @staticmethod
    def _alternate_party_size(party_size: int | None) -> int | None:
        if party_size == 1:
            return 8
        if party_size == 8:
            return 1
        return None

    def _finalize_ref_result(
        self,
        zone_hint: dict,
        party_size: int | None,
        confident_match: TreasureMapMatch | None,
        ranked: list[TreasureMapMatch],
        excluded_ref_names: list[str],
    ) -> RecognitionResult:
        ref_candidates = self._build_ref_candidates(zone_hint, ranked)
        shown_names = [candidate.ref_name for candidate in ref_candidates]
        cumulative_excluded = list(
            dict.fromkeys(excluded_ref_names + shown_names)
        )
        total_refs = self.map_matcher.count_zone_refs(
            zone_hint["id"], party_size
        )
        can_rematch = len(cumulative_excluded) < total_refs

        if confident_match is not None:
            result = self._build_result_from_match(zone_hint, confident_match)
            result.match_source = "ref"
            self._attach_ref_candidates(result, ref_candidates, confident_match)
        else:
            result = self._build_result_from_match(zone_hint, ranked[0])
            result.match_source = "ref_tentative"
            self._attach_ref_candidates(result, ref_candidates, None)
            logger.debug(
                "ref 1순위(참고) %s (%.2f, %.2f) — 후보에서 확인 권장",
                zone_hint["id"],
                result.x,
                result.y,
            )

        result.excluded_ref_names = cumulative_excluded
        result.can_rematch = can_rematch
        return result

    def _build_result_from_learned(
        self,
        zone_hint: dict,
        learned,
    ) -> RecognitionResult:
        zone = self.coordinate_service.get_effective_zone(zone_hint["id"])
        if zone is None:
            zone = zone_hint

        result = self.coordinate_service.build_result(zone, learned.x, learned.y)
        result.match_score = learned.score
        result.match_source = "learned"
        result.confirmed_ref_name = learned.ref_name
        result.learn_hits = learned.hits
        result.can_rematch = False

        ref_path = self._resolve_ref_path(zone_hint["id"], learned.ref_name)
        if ref_path is not None:
            result.ref_candidates = [
                RefCandidate(
                    rank=1,
                    x=learned.x,
                    y=learned.y,
                    score=learned.score,
                    terrain_score=learned.score,
                    marker_dist=0.0,
                    ref_name=learned.ref_name,
                    ref_image_path=str(ref_path),
                )
            ]
            result.auto_candidate_rank = 1
        else:
            result.ref_candidates = []
        return result

    def _resolve_ref_path(self, zone_id: str, ref_name: str) -> Path | None:
        for folder in self.map_matcher._resolve_refs_folders(zone_id, None):
            if "/" in ref_name:
                path = folder.parent / ref_name
            else:
                path = folder / ref_name
            if path.is_file():
                return path
        return None

    def _build_ref_candidates(
        self,
        zone_hint: dict,
        ranked: list[TreasureMapMatch],
    ) -> list[RefCandidate]:
        candidates: list[RefCandidate] = []
        for index, match in enumerate(ranked, start=1):
            gx, gy = match.treasure_x, match.treasure_y
            candidates.append(
                RefCandidate(
                    rank=index,
                    x=gx,
                    y=gy,
                    score=match.score,
                    terrain_score=match.terrain_score,
                    marker_dist=match.marker_dist,
                    ref_name=match.ref_name,
                    ref_image_path=str(match.template_path),
                )
            )
        return candidates

    @staticmethod
    def _attach_ref_candidates(
        result: RecognitionResult,
        candidates: list[RefCandidate],
        confident_match: TreasureMapMatch | None,
    ) -> None:
        result.ref_candidates = candidates
        if confident_match is None:
            result.auto_candidate_rank = None
            return
        for candidate in candidates:
            if MapAnalyzer._same_ref_name(
                candidate.ref_name, confident_match.ref_name
            ):
                result.auto_candidate_rank = candidate.rank
                return

    @staticmethod
    def _same_ref_name(left: str, right: str) -> bool:
        if left == right:
            return True
        return Path(left).name == Path(right).name

    def _match_reference(
        self,
        image: Image.Image,
        zone_hint: dict,
        party_size: int | None,
    ) -> RecognitionResult | None:
        match = self.map_matcher.match(
            image, zone_hint["id"], party_size=party_size
        )
        if match is None:
            return None
        return self._build_result_from_match(zone_hint, match)

    def _locate_detail(
        self,
        map_window: Image.Image,
        zone_hint: dict,
    ) -> RecognitionResult | None:
        located = self.detail_locator.locate(map_window, zone_hint["id"])
        if located is None:
            return None
        return self._build_result_from_detail(zone_hint, located)

    def _build_result_from_detail(
        self,
        zone_hint: dict,
        located: DetailLocateResult,
    ) -> RecognitionResult:
        zone = self.coordinate_service.get_effective_zone(zone_hint["id"])
        if zone is None:
            zone = zone_hint
        result = self.coordinate_service.build_result(
            zone, located.game_x, located.game_y
        )
        result.match_score = located.score
        return result

    def _build_result_from_match(
        self,
        zone_hint: dict,
        match: TreasureMapMatch,
    ) -> RecognitionResult:
        zone_id = zone_hint["id"]
        zone = self.coordinate_service.get_effective_zone(zone_id)
        if zone is None:
            zone = zone_hint

        gx, gy = match.treasure_x, match.treasure_y
        result = self.coordinate_service.build_result(zone, gx, gy)
        result.match_score = match.score
        return result
