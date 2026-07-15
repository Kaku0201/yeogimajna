"""그레이스케일 이미지 유사도 — TM_CCORR_NORMED + SSIM"""

from __future__ import annotations

import cv2
import numpy as np

CCORR_WEIGHT = 0.6
SSIM_WEIGHT = 0.4
TERRAIN_CCORR_WEIGHT = 0.45
TERRAIN_SSIM_WEIGHT = 0.55


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


def ccorr_score(
    a: np.ndarray,
    b: np.ndarray,
    method: int = cv2.TM_CCORR_NORMED,
) -> float:
    """cv2.matchTemplate 정규화 상관 (동일 크기).

    method=TM_CCOEFF_NORMED면 평균을 빼고 비교 → 양피지 공통 밝기 성분이 제거돼
    지형 패턴 차이가 더 크게 벌어진다(변별력↑).
    """
    a_aligned, b_aligned = align_gray(a, b)
    if a_aligned.size == 0:
        return -1.0
    return float(cv2.matchTemplate(a_aligned, b_aligned, method)[0, 0])


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
    ccorr_method: int = cv2.TM_CCORR_NORMED,
) -> tuple[float, float, float]:
    """CCORR + SSIM 결합 → (combined, ccorr, ssim). align=True면 위치 보정 후 비교.

    ccorr_method=TM_CCOEFF_NORMED면 제로평균 상관을 쓰고 음수(불일치)는 0으로 클램프.
    """
    a_aligned, b_aligned = align_gray(a, b)
    if align:
        b_aligned = _phase_align(a_aligned, b_aligned)

    score_ccorr = ccorr_score(a_aligned, b_aligned, ccorr_method)
    if ccorr_method == cv2.TM_CCOEFF_NORMED and score_ccorr > -1.0:
        # CCOEFF는 -1~1 → 불일치(음수)는 0으로 눌러 mismatch 신호로만 사용
        score_ccorr = max(0.0, score_ccorr)
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


_BANDPASS_LOW_SIGMA = 4.0    # 이보다 미세한 고주파(양피지 구김·글자·빗금) 제거
_BANDPASS_HIGH_SIGMA = 16.0  # 이보다 완만한 초저주파(비네팅·전역 밝기 편차) 제거


def enhance_terrain_features(gray: np.ndarray) -> np.ndarray:
    """지형 구조(해안선·지형 경계)만 남기는 밴드패스 표현.

    인게임 캡처의 양피지 구김/글자/빗금은 고주파 노이즈라 Canny 엣지로 잡으면
    실제 해안선보다 노이즈가 지배해 매칭이 어긋난다. 대신 두 가우시안 블러의 차
    (Difference of Gaussians)로 밴드패스를 만들어, 비네팅 같은 초저주파와
    구김 같은 고주파를 모두 제거하고 해안선 스케일 구조만 남긴다.
    """
    g = gray.astype(np.float32)
    low = cv2.GaussianBlur(g, (0, 0), sigmaX=_BANDPASS_LOW_SIGMA)
    high = cv2.GaussianBlur(g, (0, 0), sigmaX=_BANDPASS_HIGH_SIGMA)
    band = low - high
    band = cv2.normalize(band, None, 0.0, 255.0, cv2.NORM_MINMAX)
    return band.astype(np.uint8)


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


_TERRAIN_CCORR_METHOD = cv2.TM_CCOEFF_NORMED


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
        ccorr_method=_TERRAIN_CCORR_METHOD,
    )


def phase_align_gray(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    """phaseCorrelate로 other를 base에 맞게 평행 이동 (spot별 X 위치 차 보정)"""
    return _phase_align(base, other)


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
