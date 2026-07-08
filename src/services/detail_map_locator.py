"""보물지도 조각 → zone detail PNG 위치 탐색 → 게임 좌표"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.services.coordinate_service import CoordinateService
from src.services.feature_matcher import DetailFeatureMatcher
from src.services.treasure_capture import TreasureCaptureProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetailLocateResult:
    score: float
    game_x: float
    game_y: float
    detail_x: float
    detail_y: float
    scale: float
    method: str = "template"


@dataclass(frozen=True)
class _MatchCandidate:
    score: float
    loc: tuple[int, int]
    scale: float
    tpl_size: tuple[int, int]


class DetailMapLocator:
    """
    OCR로 확정한 지역의 detail 지도 1장 위에서 보물지도 지형을 찾는다.

    1순위 SIFT homography, 2순위 matchTemplate, spots snap.
    """

    MIN_SCORE = 0.38
    MIN_SIFT_SCORE = 0.22
    SCALE_MIN = 0.40
    SCALE_MAX = 1.65
    SCALE_STEPS = 13
    EDGE_MARGIN = 0.07

    def __init__(self, coordinate_service: CoordinateService) -> None:
        self.coordinate_service = coordinate_service
        self.feature_matcher = DetailFeatureMatcher()
        self._detail_cache: dict[str, tuple[str, np.ndarray, int, int]] = {}

    def locate(
        self,
        map_window: Image.Image,
        zone_id: str,
    ) -> DetailLocateResult | None:
        zone = self.coordinate_service.get_effective_zone(zone_id)
        if zone is None:
            zone = self.coordinate_service.get_zone(zone_id)
        if zone is None:
            return None

        detail_zone_id = self.coordinate_service.resolve_detail_zone_id(zone_id)
        detail_path = self.coordinate_service.get_detail_map_path(detail_zone_id)
        if not detail_path.is_file():
            logger.debug("detail 지도 없음: %s", detail_path)
            return None

        detail_gray, detail_w, detail_h = self._load_detail_gray(detail_path)
        terrain, bbox, marker = TreasureCaptureProcessor.extract_terrain_for_detail_match(
            map_window
        )
        th, tw = terrain.shape
        if tw < 24 or th < 24:
            return None

        left, top, _right, _bottom = bbox
        if marker is not None:
            mx, my = marker
            marker_rel = (mx - left, my - top)
        else:
            logger.debug("detail 매칭: 빨간 X 미검출 %s", zone_id)
            marker_rel = (
                tw * 0.5,
                th * TreasureCaptureProcessor.MATCH_CENTER_Y_RATIO,
            )

        sift_result = self.feature_matcher.locate_marker(
            terrain,
            detail_gray,
            marker_rel,
            cache_key=str(detail_path.resolve()),
        )
        if sift_result is not None and sift_result.score >= self.MIN_SIFT_SCORE:
            located = self._finalize_coords(
                zone,
                zone_id,
                sift_result.detail_x,
                sift_result.detail_y,
                detail_w,
                detail_h,
                score=sift_result.score,
                scale=sift_result.scale,
                method="sift",
                snap_dist=2.5 if sift_result.score >= 0.35 else 1.5,
            )
            if located is not None:
                return located

        candidates = self._collect_template_candidates(
            terrain, detail_gray, detail_w, detail_h
        )
        for candidate in candidates:
            result = self._template_candidate_to_result(
                candidate,
                terrain,
                bbox,
                marker,
                zone,
                zone_id,
                detail_w,
                detail_h,
            )
            if result is not None:
                return result

        if candidates:
            logger.debug(
                "detail 매칭 실패 %s best=%.3f",
                zone_id,
                candidates[0].score,
            )
        return None

    def _finalize_coords(
        self,
        zone: dict,
        zone_id: str,
        detail_x: float,
        detail_y: float,
        detail_w: int,
        detail_h: int,
        *,
        score: float,
        scale: float,
        method: str,
        snap_dist: float,
    ) -> DetailLocateResult | None:
        gx, gy = self.coordinate_service.pixel_to_game(
            zone,
            detail_x,
            detail_y,
            detail_w,
            detail_h,
        )

        if not self.coordinate_service.validate_treasure_coords(
            zone_id, gx, gy, max_spot_dist=6.5
        ):
            logger.debug(
                "detail 기각 %s method=%s xy=(%.1f,%.1f)",
                zone_id,
                method,
                gx,
                gy,
            )
            return None

        gx, gy = self.coordinate_service.refine_treasure_coords(
            zone_id,
            gx,
            gy,
            max_dist=snap_dist,
        )

        logger.debug(
            "detail ok %s method=%s score=%.3f scale=%.2f xy=(%.2f,%.2f) px=(%.0f,%.0f)",
            zone_id,
            method,
            score,
            scale,
            gx,
            gy,
            detail_x,
            detail_y,
        )

        return DetailLocateResult(
            score=score,
            game_x=gx,
            game_y=gy,
            detail_x=detail_x,
            detail_y=detail_y,
            scale=scale,
            method=method,
        )

    def _collect_template_candidates(
        self,
        terrain: np.ndarray,
        detail_gray: np.ndarray,
        detail_w: int,
        detail_h: int,
    ) -> list[_MatchCandidate]:
        th, tw = terrain.shape
        scored: list[_MatchCandidate] = []

        for scale in np.linspace(self.SCALE_MIN, self.SCALE_MAX, self.SCALE_STEPS):
            sw = max(24, int(tw * scale))
            sh = max(24, int(th * scale))
            if sw >= detail_w or sh >= detail_h:
                continue
            tpl = cv2.resize(terrain, (sw, sh), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(detail_gray, tpl, cv2.TM_CCOEFF_NORMED)
            flat = result.reshape(-1)
            top_k = min(3, flat.size)
            if top_k <= 0:
                continue
            indices = np.argpartition(flat, -top_k)[-top_k:]
            for idx in indices:
                raw_score = float(flat[idx])
                if raw_score < self.MIN_SCORE:
                    continue
                y_idx, x_idx = divmod(int(idx), result.shape[1])
                cx = x_idx + sw * 0.5
                cy = y_idx + sh * 0.5
                penalty = self._edge_penalty(cx, cy, detail_w, detail_h)
                scored.append(
                    _MatchCandidate(
                        score=raw_score - penalty,
                        loc=(x_idx, y_idx),
                        scale=float(scale),
                        tpl_size=(sw, sh),
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:12]

    def _template_candidate_to_result(
        self,
        candidate: _MatchCandidate,
        terrain: np.ndarray,
        bbox: tuple[int, int, int, int],
        marker: tuple[float, float] | None,
        zone: dict,
        zone_id: str,
        detail_w: int,
        detail_h: int,
    ) -> DetailLocateResult | None:
        th, tw = terrain.shape
        left, top, _right, _bottom = bbox
        sw, sh = candidate.tpl_size

        if marker is not None:
            mx, my = marker
            rel_x = mx - left
            rel_y = my - top
        else:
            rel_x = tw * 0.5
            rel_y = th * TreasureCaptureProcessor.MATCH_CENTER_Y_RATIO

        sx = sw / max(tw, 1)
        sy = sh / max(th, 1)
        detail_x = candidate.loc[0] + rel_x * sx
        detail_y = candidate.loc[1] + rel_y * sy

        snap_dist = 2.0 if candidate.score >= 0.52 else 1.0
        return self._finalize_coords(
            zone,
            zone_id,
            detail_x,
            detail_y,
            detail_w,
            detail_h,
            score=candidate.score,
            scale=candidate.scale,
            method="template",
            snap_dist=snap_dist,
        )

    def _edge_penalty(
        self,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> float:
        mx = width * self.EDGE_MARGIN
        my = height * self.EDGE_MARGIN
        if cx < mx or cx > width - mx or cy < my or cy > height - my:
            return 0.12
        return 0.0

    def _load_detail_gray(self, path: Path) -> tuple[np.ndarray, int, int]:
        key = str(path.resolve())
        mtime = str(path.stat().st_mtime_ns)
        cached = self._detail_cache.get(key)
        if cached is not None and cached[0] == mtime:
            gray, w, h = cached[1], cached[2], cached[3]
            return gray, w, h

        rgb = np.array(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = TreasureCaptureProcessor.neutralize_parchment(gray, rgb)
        h, w = gray.shape
        self._detail_cache[key] = (mtime, gray, w, h)
        return gray, w, h
