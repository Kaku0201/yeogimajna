"""그레이스케일 이미지 유사도 — TM_CCORR_NORMED + SSIM"""

from __future__ import annotations

import cv2
import numpy as np

CCORR_WEIGHT = 0.6
SSIM_WEIGHT = 0.4
TERRAIN_CCORR_WEIGHT = 0.20
TERRAIN_SSIM_WEIGHT = 0.80


def align_gray(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """같은 크기로 맞춘 그레이스케일 쌍 반환"""
    if len(a.shape) == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if len(b.shape) == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h < 8 or w < 8:
        return a[:h, :w], b[:h, :w]

    a_crop = a[:h, :w]
    b_crop = b[:h, :w]
    if a_crop.shape != b_crop.shape:
        b_crop = cv2.resize(
            b_crop,
            (a_crop.shape[1], a_crop.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return a_crop, b_crop


def ccorr_score(a: np.ndarray, b: np.ndarray) -> float:
    """cv2.matchTemplate TM_CCORR_NORMED (동일 크기)"""
    a_aligned, b_aligned = align_gray(a, b)
    if a_aligned.size == 0:
        return -1.0
    return float(
        cv2.matchTemplate(a_aligned, b_aligned, cv2.TM_CCORR_NORMED)[0, 0]
    )


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index (0~1)"""
    a_aligned, b_aligned = align_gray(a, b)
    if a_aligned.size == 0:
        return -1.0

    img1 = a_aligned.astype(np.float64)
    img2 = b_aligned.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, kernel, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, kernel, sigma) - mu12

    numerator = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def ssim_score_weighted(
    a: np.ndarray,
    b: np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    """가중 SSIM — weight가 큰 영역을 더 반영"""
    a_aligned, b_aligned = align_gray(a, b)
    if a_aligned.size == 0:
        return -1.0

    img1 = a_aligned.astype(np.float64)
    img2 = b_aligned.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, kernel, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, kernel, sigma) - mu12

    numerator = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)

    if weight is not None:
        w = weight.astype(np.float64)
        if w.shape != ssim_map.shape:
            w = cv2.resize(
                w,
                (ssim_map.shape[1], ssim_map.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        total = float(w.sum())
        if total <= 1e-8:
            return float(np.clip(ssim_map.mean(), 0.0, 1.0))
        return float(np.clip((ssim_map * w).sum() / total, 0.0, 1.0))

    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def combined_similarity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    ssim_weight: np.ndarray | None = None,
    ccorr_w: float = CCORR_WEIGHT,
    ssim_w: float = SSIM_WEIGHT,
    align: bool = True,
) -> tuple[float, float, float]:
    """CCORR + SSIM 결합 → (combined, ccorr, ssim). align=True면 위치 보정 후 비교."""
    a_aligned, b_aligned = align_gray(a, b)
    if align:
        b_aligned = _phase_align(a_aligned, b_aligned)

    score_ccorr = ccorr_score(a_aligned, b_aligned)
    score_ssim = (
        ssim_score_weighted(a_aligned, b_aligned, ssim_weight)
        if ssim_weight is not None
        else ssim_score(a_aligned, b_aligned)
    )

    if score_ccorr < 0 and score_ssim < 0:
        return -1.0, score_ccorr, score_ssim
    if score_ccorr < 0:
        return score_ssim, score_ccorr, score_ssim
    if score_ssim < 0:
        return score_ccorr, score_ccorr, score_ssim
    combined = score_ccorr * ccorr_w + score_ssim * ssim_w
    return combined, score_ccorr, score_ssim


def enhance_terrain_features(gray: np.ndarray) -> np.ndarray:
    """등고선·빗금 윤곽 강조 — 양피지 질감 영향 축소"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 35, 110)
    return cv2.addWeighted(blur, 0.30, edges, 0.70, 0)


def terrain_similarity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    ssim_weight: np.ndarray | None = None,
    ccorr_w: float = TERRAIN_CCORR_WEIGHT,
    ssim_w: float = TERRAIN_SSIM_WEIGHT,
) -> tuple[float, float, float]:
    """
    ref 1:1 매칭용 — edge 강조 + align 금지.

    CCORR만 쓰면 양피지 질감이 비슷한 오답 조각도 0.9+까지 올라간다.
    """
    return terrain_similarity_from_features(
        enhance_terrain_features(a),
        enhance_terrain_features(b),
        ssim_weight=ssim_weight,
        ccorr_w=ccorr_w,
        ssim_w=ssim_w,
    )


def terrain_similarity_from_features(
    a_feat: np.ndarray,
    b_feat: np.ndarray,
    *,
    ssim_weight: np.ndarray | None = None,
    ccorr_w: float = TERRAIN_CCORR_WEIGHT,
    ssim_w: float = TERRAIN_SSIM_WEIGHT,
) -> tuple[float, float, float]:
    """이미 edge 강조된 배열끼리 비교 (ref 캐시용)"""
    return combined_similarity(
        a_feat,
        b_feat,
        ssim_weight=ssim_weight,
        ccorr_w=ccorr_w,
        ssim_w=ssim_w,
        align=False,
    )


def _phase_align(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    """phaseCorrelate로 other를 base에 맞게 평행 이동 (spot별 X 위치 차 보정)"""
    if base.size == 0 or base.shape != other.shape:
        return other
    try:
        shift, _response = cv2.phaseCorrelate(
            base.astype(np.float32),
            other.astype(np.float32),
        )
        dx, dy = float(shift[0]), float(shift[1])
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return other
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            other,
            matrix,
            (other.shape[1], other.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        return other
