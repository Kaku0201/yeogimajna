"""SIFT 특징점 매칭 — 보물지도 조각 → detail 지도 위치"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureLocateResult:
    score: float
    detail_x: float
    detail_y: float
    inliers: int
    scale: float
    method: str


class DetailFeatureMatcher:
    """
    SIFT + RANSAC homography로 양피지 지형 조각을 detail 지도에 정합.

    matchTemplate보다 스케일·질감 차이에 강하고, 지형 윤곽 특징을 활용한다.
    """

    MIN_INLIERS = 10
    MIN_INLIER_RATIO = 0.28
    RATIO_TEST = 0.72
    RANSAC_REPROJ = 4.0
    SCALE_MIN = 0.42
    SCALE_MAX = 1.72
    SCALE_STEPS = 9
    MIN_SCALE_FROM_H = 0.25
    MAX_SCALE_FROM_H = 3.5

    def __init__(self) -> None:
        self._sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.04)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self._detail_features: dict[
            str, tuple[str, np.ndarray, list[cv2.KeyPoint], np.ndarray]
        ] = {}

    def locate_marker(
        self,
        query_gray: np.ndarray,
        detail_gray: np.ndarray,
        marker_rel: tuple[float, float],
        *,
        cache_key: str | None = None,
    ) -> FeatureLocateResult | None:
        """query 내 marker_rel 위치를 detail 지도 좌표로 변환"""
        if query_gray.size == 0 or detail_gray.size == 0:
            return None

        query_base = self._enhance_for_features(query_gray)
        detail_kp, detail_des = self._detail_descriptors(
            self._enhance_for_features(detail_gray),
            cache_key,
        )
        if detail_des is None or len(detail_kp) < 12:
            return None

        best: FeatureLocateResult | None = None
        qh, qw = query_gray.shape[:2]
        mx, my = marker_rel

        for scale in np.linspace(self.SCALE_MIN, self.SCALE_MAX, self.SCALE_STEPS):
            sw = max(32, int(qw * scale))
            sh = max(32, int(qh * scale))
            scaled = cv2.resize(query_base, (sw, sh), interpolation=cv2.INTER_AREA)
            query_kp, query_des = self._sift.detectAndCompute(scaled, None)
            if query_des is None or len(query_kp) < 8:
                continue

            matches = self._ratio_matches(query_des, detail_des)
            if len(matches) < self.MIN_INLIERS:
                continue

            src_pts = np.float32(
                [query_kp[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [detail_kp[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.RANSAC_REPROJ,
            )
            if homography is None or mask is None:
                continue

            inliers = int(mask.ravel().sum())
            inlier_ratio = inliers / max(len(matches), 1)
            if inliers < self.MIN_INLIERS or inlier_ratio < self.MIN_INLIER_RATIO:
                continue

            if not self._homography_scale_ok(homography):
                continue

            marker_scaled = np.float32([[[mx * scale, my * scale]]])
            mapped = cv2.perspectiveTransform(marker_scaled, homography)
            detail_x = float(mapped[0, 0, 0])
            detail_y = float(mapped[0, 0, 1])

            dh, dw = detail_gray.shape[:2]
            if not (0 <= detail_x < dw and 0 <= detail_y < dh):
                continue

            score = inlier_ratio * min(1.0, inliers / 18.0)
            candidate = FeatureLocateResult(
                score=score,
                detail_x=detail_x,
                detail_y=detail_y,
                inliers=inliers,
                scale=float(scale),
                method="sift",
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None:
            logger.debug(
                "SIFT ok score=%.3f inliers=%d scale=%.2f px=(%.0f,%.0f)",
                best.score,
                best.inliers,
                best.scale,
                best.detail_x,
                best.detail_y,
            )
        return best

    def _detail_descriptors(
        self,
        detail_gray: np.ndarray,
        cache_key: str | None,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if cache_key is not None:
            cached = self._detail_features.get(cache_key)
            if cached is not None and cached[0] == str(detail_gray.shape):
                return cached[2], cached[3]

        kp, des = self._sift.detectAndCompute(detail_gray, None)
        if cache_key is not None and des is not None:
            self._detail_features[cache_key] = (
                str(detail_gray.shape),
                detail_gray,
                kp,
                des,
            )
        return kp, des

    def _ratio_matches(
        self,
        query_des: np.ndarray,
        detail_des: np.ndarray,
    ) -> list[cv2.DMatch]:
        try:
            pairs = self._matcher.knnMatch(query_des, detail_des, k=2)
        except cv2.error:
            return []

        good: list[cv2.DMatch] = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.RATIO_TEST * second.distance:
                good.append(first)
        return good

    def _homography_scale_ok(self, homography: np.ndarray) -> bool:
        sx = float(np.hypot(homography[0, 0], homography[1, 0]))
        sy = float(np.hypot(homography[0, 1], homography[1, 1]))
        scale = (sx + sy) * 0.5
        return self.MIN_SCALE_FROM_H <= scale <= self.MAX_SCALE_FROM_H

    @staticmethod
    def _enhance_for_features(gray: np.ndarray) -> np.ndarray:
        """빗금 지형 윤곽을 SIFT에 강조"""
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 35, 110)
        return cv2.addWeighted(blur, 0.5, edges, 0.5, 0)
