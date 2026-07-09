import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from src.app_config import is_debug, is_debug_party
from src.services.coordinate_service import CoordinateService
from src.services.ocr_engine import OcrEngine
from src.services.tesseract_config import has_korean_language

logger = logging.getLogger(__name__)


@dataclass
class PartyDetectTrace:
    """TR_DEBUG=1 시 party 인식 단계 추적용"""

    source: str = ""
    image_size: tuple[int, int] = (0, 0)
    icon_size: tuple[int, int] | None = None
    crop_box: tuple[int, int, int, int] | None = None
    ocr_attempts: list[str] | None = None
    ocr_raw: str | None = None
    ocr_digit: int | None = None
    left_bright: float | None = None
    right_ratio: float | None = None
    heuristic: int | None = None
    topology_holes: int | None = None
    result: int | None = None
    reason: str = ""

    def summary(self) -> str:
        parts = [
            f"source={self.source}",
            f"image={self.image_size[0]}x{self.image_size[1]}",
        ]
        if self.icon_size:
            parts.append(f"icon={self.icon_size[0]}x{self.icon_size[1]}")
        if self.crop_box:
            parts.append(f"crop={self.crop_box}")
        if self.ocr_attempts:
            parts.append(f"ocr_tries=[{'; '.join(self.ocr_attempts)}]")
        if self.ocr_raw is not None:
            parts.append(f"ocr_raw={self.ocr_raw!r}")
        if self.ocr_digit is not None:
            parts.append(f"ocr_digit={self.ocr_digit}")
        if self.left_bright is not None:
            parts.append(f"left_bright={self.left_bright:.3f}")
        if self.right_ratio is not None:
            parts.append(f"right_ratio={self.right_ratio:.3f}")
        if self.heuristic is not None:
            parts.append(f"heuristic={self.heuristic}")
        if self.topology_holes is not None:
            parts.append(f"topology_holes={self.topology_holes}")
        parts.append(f"party_size={self.result}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        return " ".join(parts)


@dataclass(frozen=True)
class CaptureReadiness:
    """드래그 프레임 실시간 품질 점수 — OCR/매칭 없이 경량 검사만"""

    score: int
    max_score: int
    ready: bool
    hint: str
    has_marker: bool
    has_parchment: bool
    has_banner: bool
    has_banner_aligned: bool
    has_party_icon: bool
    suggested_crop: tuple[int, int, int, int] | None = None
    banner_zone: tuple[int, int, int, int] | None = None
    banner_target_zone: tuple[int, int, int, int] | None = None


class TreasureCaptureProcessor:
    """인게임 보물지도 UI 캡처 전처리 (양피지 창·배너·아이콘 제외)"""

    MAP_ASPECT = 1.30
    MARKER_X_RATIO = 0.46
    MARKER_Y_RATIO = 0.40
    BANNER_TOP_EXPAND = 0.48
    BANNER_FROM_MARKER_RATIO = 0.42
    BANNER_BAND_RATIO = 0.15
    PARCHMENT_MIN_AREA_RATIO = 0.005
    PARCHMENT_MAX_AREA_RATIO = 0.995
    READINESS_MIN_SCORE = 2
    READINESS_MAX_SCORE = 2
    BANNER_TARGET_RATIO = 0.17
    BANNER_ALIGN_LEFT_BRIGHT_MIN = 0.02
    PARCHMENT_ASPECT_MIN = 0.50
    PARCHMENT_ASPECT_MAX = 1.90
    # 300% UI(1629×1360) 보물지도 창 기준 고정 템플릿 — HUD 배율과 무관한 상대 비율
    TEMPLATE_REF_WIDTH = 468
    TEMPLATE_REF_HEIGHT = 360
    TEMPLATE_MIN_WIDTH = 252
    TEMPLATE_BANNER_HEIGHT_RATIO = 0.22
    BANNER_OCR_FIXED_TOP_RATIO = 0.24
    # 배너는 지도 상단에만 — 중간 지형 해칭 오탐 방지
    BANNER_MAX_Y1_RATIO = 0.22
    BANNER_MAX_Y2_RATIO = 0.32
    TEMPLATE_BANNER_ALIGN_MIN_OVERLAP = 0.30
    TEMPLATE_SLOT_LEFT_BRIGHT_MIN = 0.035
    TEMPLATE_SLOT_DARK_MIN = 0.22
    TEMPLATE_PARCHMENT_MIN_AREA = 0.12
    MARKER_DETECT_REF_WIDTH = TEMPLATE_REF_WIDTH
    CAPTURE_EDGE_MASK_PX = 8
    MARKER_AREA_MIN = 30
    MARKER_AREA_MAX = 2500
    MARKER_ON_PARCHMENT_MIN = 0.12
    # recognition_box.py paintEvent 오버레이와 동기화 — 실시간 캡처 역보정용
    GUIDE_FRAME_TINT_RGB = (0, 140, 220)
    GUIDE_FRAME_TINT_ALPHA = 52 / 255.0
    GUIDE_SLOT_TINT_RGB = (255, 210, 50)
    GUIDE_SLOT_TINT_ALPHA = 90 / 255.0
    GUIDE_READY_TINT_RGB = (100, 230, 130)
    GUIDE_READY_TINT_ALPHA = 72 / 255.0

    def __init__(self, coordinate_service: CoordinateService) -> None:
        self.coordinate_service = coordinate_service
        self._ocr = OcrEngine()
        self._tesseract_available = self._check_tesseract()
        self._last_party_trace: PartyDetectTrace | None = None
        self._party_debug_saved = False
        self._zone_lexicon: tuple[frozenset[str], str, list[tuple[dict, str]]] | None = (
            None
        )

    @property
    def last_party_trace(self) -> PartyDetectTrace | None:
        return self._last_party_trace

    def _check_tesseract(self) -> bool:
        if not self._ocr.initialize():
            return False
        if not has_korean_language():
            return False
        logger.info("OCR 사용 가능 (backend=%s)", self._ocr.backend_name)
        return True

    @property
    def ocr_available(self) -> bool:
        return self._tesseract_available

    def prepare(
        self,
        image: Image.Image,
        *,
        refocus: bool = True,
        trust_frame: bool = False,
        skip_banner_ocr: bool = False,
    ) -> tuple[Image.Image, Optional[dict], Image.Image]:
        """
        인게임 캡처 → (지도 중심 영역, zones.json 지역 또는 None, 보물지도 창 크롭)

        trust_frame=True: 고정 프레임 확정 캡처 — 재크롭 파이프라인 생략
        skip_banner_ocr=True: 지역이 이미 다른 경로(배너 선-확정 등)로 정해져
            있어 여기서 배너 OCR을 다시 돌릴 필요가 없을 때 (zone=None 반환)
        """
        logger.debug(
            "prepare start: %dx%d refocus=%s trust_frame=%s skip_banner_ocr=%s",
            image.width,
            image.height,
            refocus,
            trust_frame,
            skip_banner_ocr,
        )

        if trust_frame:
            map_window = image
            logger.debug(
                "  trust_frame: map_window=%dx%d (re-crop skipped)",
                map_window.width,
                map_window.height,
            )
            zone = None if skip_banner_ocr else self.detect_zone_from_banner(map_window)
            focused = self.extract_map_content(map_window)
            return focused, zone, map_window

        working = self.focus_map_window(image) if refocus else image
        logger.debug("  after focus: %dx%d", working.width, working.height)
        map_window = self.locate_treasure_map_window(working)
        logger.debug("  after locate: %dx%d", map_window.width, map_window.height)
        map_window = self._refine_map_window_crop(working, map_window)
        logger.debug("  after refine: %dx%d", map_window.width, map_window.height)
        map_window = self._trim_map_window_margins(map_window)
        logger.debug("  after trim: %dx%d", map_window.width, map_window.height)
        map_window = self._strip_rows_above_banner(map_window)
        logger.debug("  after strip: %dx%d", map_window.width, map_window.height)
        zone = None if skip_banner_ocr else self.detect_zone_from_banner(map_window)
        focused = self.extract_map_content(map_window)
        return focused, zone, map_window

    def localize_for_zone_ocr(
        self, image: Image.Image, *, refocus: bool = True
    ) -> Image.Image:
        """지역명 OCR 전용 — 배너를 자르지 않고 양피지 창만 추출"""
        working = image
        if refocus and not self._capture_is_tight_map_frame(image):
            working = self.focus_map_window(image)

        if self._capture_is_tight_map_frame(working):
            return working

        map_window = self.locate_treasure_map_window(working)
        return map_window

    def _capture_is_tight_map_frame(self, image: Image.Image) -> bool:
        """사용자가 프레임에 맞춘 캡처 — refocus/locate가 배너를 자를 수 있음"""
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if width < 80 or height < 80:
            return False

        marker = self._find_treasure_x_marker_relaxed(rgb)
        if marker is None:
            return False

        if not self._top_header_has_title_ink(rgb):
            return False

        parchment = self._parchment_bounds(rgb)
        if parchment is None:
            return True

        px1, py1, px2, py2 = parchment
        area_ratio = ((px2 - px1) * (py2 - py1)) / max(width * height, 1)
        return area_ratio >= 0.42

    def _align_map_window_to_title(self, image: Image.Image) -> Image.Image:
        """상단 다이얼로그·배경 여백을 제거하고 지역명 배너가 상단에 오도록 정렬"""
        rgb = np.array(image.convert("RGB"))
        band = self._find_title_text_band(rgb)
        if band is None:
            return image
        _x1, y1, _x2, y2 = band
        top = max(0, y1 - 2)
        if top < 4:
            return image
        cropped = image.crop((0, top, image.width, image.height))
        if cropped.height < 80:
            return image
        return cropped

    def _find_title_text_band(
        self, rgb: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """상단 흰 글씨 지역명 띠 — UI 슬롯(가로 전체 밝음)과 구분"""
        height, width = rgb.shape[:2]
        if height < 30 or width < 40:
            return None

        scan_end = max(24, int(height * 0.38))
        row_infos: list[tuple[int, int, int, int, float]] = []

        for y in range(scan_end):
            row = rgb[y]
            white = (
                (row[:, 0] > 200)
                & (row[:, 1] > 190)
                & (row[:, 2] > 155)
            )
            xs = np.where(white)[0]
            count = int(xs.size)
            if count < 18 or count > 160:
                continue
            span = (int(xs.max()) - int(xs.min()) + 1) / max(width, 1)
            if span < 0.10 or span > 0.50:
                continue
            score = min(count, 90) * span
            row_infos.append((y, int(xs.min()), int(xs.max()) + 1, count, score))

        if not row_infos:
            return None

        best: tuple[int, int, int, int, float] | None = None
        idx = 0
        while idx < len(row_infos):
            start = row_infos[idx]
            j = idx + 1
            while j < len(row_infos) and row_infos[j][0] == row_infos[j - 1][0] + 1:
                j += 1
            run = row_infos[idx:j]
            if len(run) < 3:
                idx = j
                continue
            y1 = run[0][0]
            y2 = run[-1][0] + 1
            x1 = min(r[1] for r in run)
            x2 = max(r[2] for r in run)
            mean_span = float(np.mean([(r[2] - r[1]) / max(width, 1) for r in run]))
            mean_count = float(np.mean([r[3] for r in run]))
            # 지역명은 좁은 흰 글씨 띠 — UI 슬롯(가로로 넓은 밝음)보다 span이 작음
            span_score = max(0.0, 0.46 - mean_span) * 120.0
            count_score = min(mean_count, 72.0) * 0.18
            pos_bonus = 6.0 if y1 >= int(height * 0.12) else 0.0
            score = span_score + count_score + len(run) * 0.35 + pos_bonus
            if mean_span > 0.44:
                score *= 0.55
            if y1 > height * 0.30:
                idx = j
                continue
            if best is None or score > best[4]:
                best = (x1, y1, x2, y2, score)
            idx = j

        if best is None:
            return None
        x1, y1, x2, y2 = best[0], best[1], best[2], best[3]
        pad_x = max(4, int(width * 0.04))
        # 위·아래 패딩 — 글자 획이 잘리면 OCR이 깨짐
        # (안티에일리어싱된 획 끝부분은 흰 픽셀 수가 적어 band 탐지에서 누락되기 쉬움 —
        #  아래쪽과 동일하게 넉넉히 잡아 ㄹ 등의 위쪽 삐침이 잘리지 않도록 함)
        pad_top = max(8, int(height * 0.045))
        pad_bottom = max(6, int(height * 0.035))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_top),
            min(width, x2 + pad_x),
            min(height, y2 + pad_bottom),
        )

    def focus_map_window(self, image: Image.Image) -> Image.Image:
        """[고정 규격 최적화] 사용자가 고정 틀에 맞춰 배치한 이미지를 완벽한 규격으로 가공"""
        rgb = np.array(image.convert("RGB"))
        marker = self._find_treasure_x_marker(rgb)

        if marker is not None:
            return self.auto_extract_by_marker(image)

        found = self._find_parchment_window(image)
        if found is not None:
            return found

        return image

    def assess_capture_readiness(
        self, image: Image.Image, *, bare_capture: bool = False
    ) -> CaptureReadiness:
        """실시간 가이드 프레임 — X 존재 + 배너 띠 포함만 확인 (OCR·슬롯 정렬은 캡처 후)"""
        rgb = np.array(image.convert("RGB"))
        rgb = self._mask_capture_frame_edges(rgb)
        height, width = rgb.shape[:2]
        if width < 40 or height < 40:
            return CaptureReadiness(
                score=0,
                max_score=self.READINESS_MAX_SCORE,
                ready=False,
                hint="가이드 칸이 올바르게 로드되지 않았습니다",
                has_marker=False,
                has_parchment=False,
                has_banner=False,
                has_banner_aligned=False,
                has_party_icon=False,
            )

        banner_target_zone = self.fixed_banner_target_zone(width, height)
        if not bare_capture and self._slot_has_guide_overlay_tint(
            rgb, banner_target_zone
        ):
            rgb = self._decontaminate_guide_overlay(rgb, banner_target_zone)
            rgb = self._mask_guide_label_chrome(rgb, banner_target_zone)

        marker = self._find_treasure_x_marker_relaxed(rgb)
        has_marker = marker is not None

        has_banner = False
        has_banner_aligned = False
        has_party_icon = False
        banner_zone: tuple[int, int, int, int] | None = None
        suggested: tuple[int, int, int, int] | None = None
        band: tuple[int, int, int, int] | None = None

        if has_marker and marker is not None:
            band = self._find_banner_band_bounds(rgb, marker=marker)
            has_banner = self._banner_above_marker(rgb, marker, band)
            if has_banner and not self._top_header_has_title_ink(rgb):
                has_banner = False
            if has_banner and band is not None:
                banner_zone = band
                has_banner_aligned = self._band_overlaps_slot(
                    band, banner_target_zone, min_slot_coverage=0.15
                )
            suggested = (0, 0, width, height)
            has_party_icon = self._detect_party_icon_pattern(rgb)

        score = int(has_marker) + int(has_banner)
        ready = has_marker and has_banner
        hint = self._readiness_hint(
            has_marker=has_marker,
            has_banner=has_banner,
            has_banner_aligned=has_banner_aligned,
            has_party_icon=has_party_icon,
            ready=ready,
        )

        return CaptureReadiness(
            score=score,
            max_score=self.READINESS_MAX_SCORE,
            ready=ready,
            hint=hint,
            has_marker=has_marker,
            has_parchment=False,
            has_banner=has_banner,
            has_banner_aligned=has_banner_aligned,
            has_party_icon=has_party_icon,
            suggested_crop=suggested,
            banner_zone=banner_zone,
            banner_target_zone=banner_target_zone,
        )

    @classmethod
    def _mask_capture_frame_edges(
        cls, rgb: np.ndarray, margin: int | None = None
    ) -> np.ndarray:
        """실시간 품질 검사 캡처에 섞인 오버레이 테두리·코너 핸들 제거"""
        height, width = rgb.shape[:2]
        m = margin if margin is not None else cls.CAPTURE_EDGE_MASK_PX
        m = min(m, width // 4, height // 4)
        if m <= 0:
            return rgb

        out = rgb.copy()
        inner = rgb[m : height - m, m : width - m]
        if inner.size == 0:
            return rgb
        fill = np.median(inner.reshape(-1, 3), axis=0).astype(np.uint8)
        out[:m, :] = fill
        out[height - m :, :] = fill
        out[:, :m] = fill
        out[:, width - m :] = fill
        return out

    @classmethod
    def _decontaminate_guide_overlay(
        cls,
        rgb: np.ndarray,
        banner_target_zone: tuple[int, int, int, int],
    ) -> np.ndarray:
        """실시간 품질 검사 캡처에 합쳐진 시안/노란(또는 초록) 가이드 오버레이 색 역제거"""
        height, width = rgb.shape[:2]
        tx1, ty1, tx2, ty2 = banner_target_zone
        out = rgb.astype(np.float32)

        ya = cls.GUIDE_SLOT_TINT_ALPHA
        yellow = np.array(cls.GUIDE_SLOT_TINT_RGB, dtype=np.float32)
        y1, y2 = max(0, ty1), min(height, ty2)
        x1, x2 = max(0, tx1), min(width, tx2)
        if y2 > y1 and x2 > x1:
            slot = out[y1:y2, x1:x2]
            out[y1:y2, x1:x2] = np.clip(
                (slot - yellow * ya) / max(1e-3, 1.0 - ya), 0, 255
            )

        def _reverse_tint(
            arr: np.ndarray,
            tint_rgb: tuple[int, int, int],
            alpha: float,
        ) -> np.ndarray:
            tint = np.array(tint_rgb, dtype=np.float32)
            return np.clip(
                (arr - tint * alpha) / max(1e-3, 1.0 - alpha), 0, 255
            )

        out = _reverse_tint(out, cls.GUIDE_FRAME_TINT_RGB, cls.GUIDE_FRAME_TINT_ALPHA)

        if y2 > y1 and x2 > x1:
            sample = out[y1:y2, x1:x2]
            if float(sample[:, :, 1].mean()) > float(sample[:, :, 0].mean()) + 8:
                out = _reverse_tint(
                    out, cls.GUIDE_READY_TINT_RGB, cls.GUIDE_READY_TINT_ALPHA
                )

        return out.astype(np.uint8)

    @classmethod
    def _slot_has_guide_overlay_tint(
        cls,
        rgb: np.ndarray,
        banner_target_zone: tuple[int, int, int, int],
    ) -> bool:
        """실시간 캡처에 노란/시안 가이드 오버레이가 합쳐졌는지"""
        height, width = rgb.shape[:2]
        tx1, ty1, tx2, ty2 = banner_target_zone
        y1, y2 = max(0, ty1), min(height, ty2)
        x1, x2 = max(0, tx1), min(width, tx2)
        if y2 <= y1 or x2 <= x1:
            return False
        strip = rgb[y1:y2, x1:x2]
        r = strip[:, :, 0].astype(np.int16)
        g = strip[:, :, 1].astype(np.int16)
        b = strip[:, :, 2].astype(np.int16)
        yellow = (r > 190) & (g > 165) & (b < 145)
        cyan = (b > r + 12) & (g > 95)
        return float(yellow.mean()) > 0.05 or float(cyan.mean()) > 0.10

    @classmethod
    def _mask_guide_label_chrome(
        cls,
        rgb: np.ndarray,
        banner_target_zone: tuple[int, int, int, int],
    ) -> np.ndarray:
        """실시간 캡처에 찍힌 「지역명」 가이드 라벨 영역 제거"""
        height, width = rgb.shape[:2]
        tx1, ty1, tx2, ty2 = banner_target_zone
        y1, y2 = max(0, ty1), min(height, ty2)
        x1, x2 = max(0, tx1), min(width, tx2)
        if y2 <= y1 or x2 <= x1:
            return rgb

        out = rgb.copy()
        slot_h = y2 - y1
        slot_w = x2 - x1
        label_h = min(slot_h, max(12, int(slot_h * 0.42)))
        label_w = min(slot_w, max(48, int(slot_w * 0.24)))
        ref_y2 = min(y2, y1 + label_h + max(4, slot_h // 3))
        ref_x1 = min(x2, x1 + label_w + 4)
        ref = out[ref_y2:y2, ref_x1:x2]
        if ref.size == 0:
            return out
        fill = np.median(ref.reshape(-1, 3), axis=0).astype(np.uint8)
        out[y1 : y1 + label_h, x1 : x1 + label_w] = fill
        return out

    def _map_parchment_present(
        self,
        rgb: np.ndarray,
        marker: tuple[float, float],
        slot: tuple[int, int, int, int],
    ) -> bool:
        """양피지 지도 UI가 슬롯 아래에 실제로 있는지 (배경만 있는 프레임 제외)"""
        parchment = self._parchment_bounds_near_marker(rgb, marker)
        if parchment is None:
            parchment = self._parchment_bounds(rgb)
        if parchment is None:
            return False

        px1, py1, px2, py2 = parchment
        mx, my = marker
        if not (px1 <= mx < px2 and py1 <= my < py2):
            return False

        height, width = rgb.shape[:2]
        box_w = px2 - px1
        box_h = py2 - py1
        aspect = box_w / max(box_h, 1)
        area_ratio = (box_w * box_h) / max(width * height, 1)
        if not (
            self.PARCHMENT_ASPECT_MIN <= aspect <= self.PARCHMENT_ASPECT_MAX
            and 0.08 <= area_ratio <= 0.92
        ):
            return False

        if py1 > slot[3] + int(height * 0.10):
            return False

        patch = rgb[py1:py2, px1:px2]
        if patch.size == 0:
            return False
        return float(self._build_parchment_mask(patch).mean()) / 255.0 >= 0.10

    def _parchment_fill_below_slot(
        self,
        rgb: np.ndarray,
        slot: tuple[int, int, int, int],
        *,
        min_ratio: float = 0.14,
    ) -> bool:
        """슬롯 아래 양피지 면적 — 지도 UI 없이 배경만 있으면 False"""
        height, width = rgb.shape[:2]
        below = rgb[slot[3] :, :]
        if below.size == 0:
            return False
        ratio = float(self._build_parchment_mask(below).mean()) / 255.0
        if ratio < min_ratio:
            return False
        box = self._parchment_bounds(rgb)
        if box is None:
            return False
        _x1, _y1, _x2, _y2 = box
        area_ratio = ((_x2 - _x1) * (_y2 - _y1)) / max(width * height, 1)
        return area_ratio >= 0.08

    @classmethod
    def fixed_banner_target_zone(
        cls, width: int, height: int
    ) -> tuple[int, int, int, int]:
        """프레임 상단 고정 지역명 슬롯 (300% UI 기준 비율)"""
        slot_h = max(8, int(height * cls.TEMPLATE_BANNER_HEIGHT_RATIO))
        return (0, 0, width, min(height, slot_h))

    @classmethod
    def template_size_for_screen(cls, screen_w: int, screen_h: int) -> tuple[int, int]:
        """기본 템플릿 박스 크기 — 최대 UI 기준, 화면에 맞게 축소"""
        w = min(cls.TEMPLATE_REF_WIDTH, int(screen_w * 0.52))
        w = max(cls.TEMPLATE_MIN_WIDTH, w)
        h = max(80, int(w / cls.MAP_ASPECT))
        if h > int(screen_h * 0.72):
            h = int(screen_h * 0.72)
            w = max(cls.TEMPLATE_MIN_WIDTH, int(h * cls.MAP_ASPECT))
        return w, h

    @staticmethod
    def _band_overlaps_slot(
        band: tuple[int, int, int, int],
        slot: tuple[int, int, int, int],
        *,
        min_slot_coverage: float = 0.40,
    ) -> bool:
        bx1, by1, bx2, by2 = band
        tx1, ty1, tx2, ty2 = slot
        ix1 = max(bx1, tx1)
        iy1 = max(by1, ty1)
        ix2 = min(bx2, tx2)
        iy2 = min(by2, ty2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter = (ix2 - ix1) * (iy2 - iy1)
        slot_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        return inter / slot_area >= min_slot_coverage

    @staticmethod
    def _marker_in_frame(
        rgb: np.ndarray,
        marker: tuple[float, float],
    ) -> bool:
        """마커가 프레임 안쪽에 있는지 (슬롯·양피지 위치 무관)"""
        mx, my = marker
        height, width = rgb.shape[:2]
        margin = max(4, int(min(width, height) * 0.02))
        return (
            margin <= mx <= width - margin
            and margin <= my <= height - margin
        )

    def _marker_in_map_context(
        self,
        rgb: np.ndarray,
        marker: tuple[float, float],
        slot: tuple[int, int, int, int],
    ) -> bool:
        """캡처 후 전처리용 — 슬롯 아래 지도 영역 여부"""
        return self._marker_in_frame(rgb, marker)

    @classmethod
    def _is_banner_band_in_header(
        cls,
        band: tuple[int, int, int, int],
        height: int,
    ) -> bool:
        """지역명 배너는 프레임 상단에만 있어야 함 (지형 해칭 오탐 제외)"""
        _x1, y1, _x2, y2 = band
        if y1 > height * cls.BANNER_MAX_Y1_RATIO:
            return False
        if y2 > height * cls.BANNER_MAX_Y2_RATIO:
            return False
        return (y2 - y1) >= 6

    @classmethod
    def _top_header_has_title_ink(cls, rgb: np.ndarray) -> bool:
        """상단 배너 슬롯에 글자(밝은 획)가 실제로 있는지 — 잘림 감지용"""
        height, width = rgb.shape[:2]
        top_h = max(12, int(height * cls.TEMPLATE_BANNER_HEIGHT_RATIO))
        top = rgb[:top_h, : max(8, int(width * 0.88))]
        if top.size == 0:
            return False

        white = (
            (top[:, :, 0] > 165)
            & (top[:, :, 1] > 155)
            & (top[:, :, 2] > 125)
        )
        if float(white.mean()) >= 0.010:
            return True

        deficit = (
            (top[:, :, 0].astype(np.int16) + top[:, :, 1].astype(np.int16)) // 2
            - top[:, :, 2].astype(np.int16)
        )
        ink = white | (deficit > 22)
        return float(ink.mean()) >= 0.014

    def _banner_above_marker(
        self,
        rgb: np.ndarray,
        marker: tuple[float, float],
        band: tuple[int, int, int, int] | None,
    ) -> bool:
        """X보다 위에 어두운 가로 배너 띠 + 흰 글자가 있는지"""
        if band is None:
            return False

        bx1, by1, bx2, by2 = band
        mx, my = marker
        height, width = rgb.shape[:2]

        if not self._is_banner_band_in_header(band, height):
            return False

        if by2 > my + max(20, int(height * 0.10)):
            return False
        if by1 >= my:
            return False

        band_rgb = rgb[by1:by2, bx1:bx2]
        if band_rgb.size == 0:
            return False

        left_w = max(8, int((bx2 - bx1) * 0.55))
        left_bright = float(
            (
                (band_rgb[:, :left_w, 0] > 120)
                & (band_rgb[:, :left_w, 1] > 120)
                & (band_rgb[:, :left_w, 2] > 120)
            ).mean()
        )
        dark = float(
            (
                (band_rgb[:, :, 0] < 120)
                & (band_rgb[:, :, 1] < 110)
                & (band_rgb[:, :, 2] < 105)
            ).mean()
        )
        width_ratio = (bx2 - bx1) / max(width, 1)
        horizontally_near = bx1 <= mx <= bx2 or abs((bx1 + bx2) / 2 - mx) <= width * 0.35
        return (
            left_bright >= 0.02
            and dark >= 0.10
            and width_ratio >= 0.18
            and horizontally_near
        )

    def _marker_on_parchment(
        self,
        rgb: np.ndarray,
        marker: tuple[float, float],
    ) -> bool:
        return TreasureCaptureProcessor._marker_inside_map_window(rgb, marker)

    def _banner_aligns_with_slot(
        self,
        rgb: np.ndarray,
        banner_target_zone: tuple[int, int, int, int],
    ) -> bool:
        """지역명 배너가 노란 슬롯 안·경계에 걸치는 정렬 (살짝 걸침 포함)"""
        band = self._find_banner_band_bounds(rgb, allow_fallback=True)
        if band is None:
            return False

        height, width = rgb.shape[:2]
        bx1, by1, bx2, by2 = band
        _tx1, _ty1, _tx2, slot_bottom = banner_target_zone

        in_slot = self._band_overlaps_slot(
            band, banner_target_zone, min_slot_coverage=0.22
        )
        straddles = (
            by1 < slot_bottom + max(6, int(height * 0.02))
            and by2 > slot_bottom - max(4, int(height * 0.015))
            and by1 <= slot_bottom + int(height * 0.04)
        )
        if not in_slot and not straddles:
            return False

        band_rgb = rgb[by1:by2, bx1:bx2]
        if band_rgb.size == 0:
            return False
        left_w = max(8, int((bx2 - bx1) * 0.55))
        left_bright = float(
            (
                (band_rgb[:, :left_w, 0] > 120)
                & (band_rgb[:, :left_w, 1] > 120)
                & (band_rgb[:, :left_w, 2] > 120)
            ).mean()
        )
        dark = float(
            (
                (band_rgb[:, :, 0] < 120)
                & (band_rgb[:, :, 1] < 110)
                & (band_rgb[:, :, 2] < 105)
            ).mean()
        )
        width_ratio = (bx2 - bx1) / max(width, 1)
        return left_bright >= 0.02 and dark >= 0.12 and width_ratio >= 0.22

    def _parchment_near_marker_valid(
        self,
        rgb: np.ndarray,
        marker: tuple[float, float],
        slot: tuple[int, int, int, int],
    ) -> bool:
        parchment = self._parchment_bounds_near_marker(rgb, marker)
        if parchment is None:
            return False

        px1, py1, px2, py2 = parchment
        mx, my = marker
        if not (px1 <= mx < px2 and py1 <= my < py2):
            return False

        height, width = rgb.shape[:2]
        box_w = px2 - px1
        box_h = py2 - py1
        aspect = box_w / max(box_h, 1)
        area_ratio = (box_w * box_h) / max(width * height, 1)
        if not (
            self.PARCHMENT_ASPECT_MIN <= aspect <= self.PARCHMENT_ASPECT_MAX
            and self.TEMPLATE_PARCHMENT_MIN_AREA <= area_ratio <= 0.92
        ):
            return False

        if py1 > slot[3] + int(height * 0.06):
            return False

        below = rgb[slot[3] :, :]
        if below.size == 0:
            return False
        below_parchment = float(self._build_parchment_mask(below).mean()) / 255.0
        return below_parchment >= 0.10

    def is_banner_slot_ready(
        self,
        rgb: np.ndarray,
        banner_target_zone: tuple[int, int, int, int],
        *,
        marker: tuple[float, float] | None = None,
    ) -> bool:
        """사용자가 고정 틀 상단에 지역명을 맞췄을 때 배너 슬롯 유효성 경량 검증"""
        height, width = rgb.shape[:2]
        tx1, ty1, tx2, ty2 = banner_target_zone
        x1 = max(0, min(tx1, width - 1))
        y1 = max(0, min(ty1, height - 1))
        x2 = max(x1 + 8, min(tx2, width))
        y2 = max(y1 + 6, min(ty2, height))
        strip = rgb[y1:y2, x1:x2]
        if strip.size == 0:
            return False

        if marker is not None and marker[1] <= y2 + 2:
            return False

        row_dark = (
            (strip[:, :, 0] < 100)
            & (strip[:, :, 1] < 92)
            & (strip[:, :, 2] < 88)
        ).mean(axis=1)
        row_dark_hits = int((row_dark >= 0.25).sum())
        if row_dark_hits < max(2, int(strip.shape[0] * 0.26)):
            return False

        bright = (
            (strip[:, :, 0] > 120)
            & (strip[:, :, 1] > 120)
            & (strip[:, :, 2] > 120)
        )
        total_bright = int(bright.sum())
        if total_bright < 6:
            return False

        left_cut = max(1, int(strip.shape[1] * 0.50))
        if float(bright[:, :left_cut].sum()) / total_bright < 0.25:
            return False

        dark = float(
            (
                (strip[:, :, 0] < 120)
                & (strip[:, :, 1] < 110)
                & (strip[:, :, 2] < 105)
            ).mean()
        )
        if dark < 0.15:
            return False

        return True

    def _slot_has_banner_structure(
        self,
        rgb: np.ndarray,
        banner_target_zone: tuple[int, int, int, int],
        *,
        marker: tuple[float, float] | None = None,
    ) -> bool:
        """소형 HUD — 글자 좌우 분포는 생략하고 배너 띠 구조만 확인"""
        height, width = rgb.shape[:2]
        tx1, ty1, tx2, ty2 = banner_target_zone
        x1 = max(0, min(tx1, width - 1))
        y1 = max(0, min(ty1, height - 1))
        x2 = max(x1 + 8, min(tx2, width))
        y2 = max(y1 + 6, min(ty2, height))
        strip = rgb[y1:y2, x1:x2]
        if strip.size == 0:
            return False

        if marker is not None and marker[1] <= y2 + 2:
            return False

        row_dark = (
            (strip[:, :, 0] < 100)
            & (strip[:, :, 1] < 92)
            & (strip[:, :, 2] < 88)
        ).mean(axis=1)
        if int((row_dark >= 0.25).sum()) < max(2, int(strip.shape[0] * 0.26)):
            return False

        bright = (
            (strip[:, :, 0] > 120)
            & (strip[:, :, 1] > 120)
            & (strip[:, :, 2] > 120)
        )
        if int(bright.sum()) < 6:
            return False

        dark = float(
            (
                (strip[:, :, 0] < 120)
                & (strip[:, :, 1] < 110)
                & (strip[:, :, 2] < 105)
            ).mean()
        )
        return dark >= 0.15

    def _is_banner_aligned(
        self,
        width: int,
        height: int,
        marker: tuple[float, float],
        banner_zone: tuple[int, int, int, int],
        left_bright: float,
        banner_target_zone: tuple[int, int, int, int] | None,
    ) -> bool:
        """지역명 배너가 고정 슬롯(프레임 상단)에 맞게 들어왔는지"""
        x1, y1, x2, y2 = banner_zone
        _mx, my = marker

        if left_bright < self.BANNER_ALIGN_LEFT_BRIGHT_MIN:
            return False
        if y2 > my - 6:
            return False
        if x1 > int(width * 0.18):
            return False
        if x2 < int(width * 0.35):
            return False

        slot = banner_target_zone or self.fixed_banner_target_zone(width, height)
        tx1, ty1, tx2, ty2 = slot
        ix1 = max(x1, tx1)
        iy1 = max(y1, ty1)
        ix2 = min(x2, tx2)
        iy2 = min(y2, ty2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False

        inter = (ix2 - ix1) * (iy2 - iy1)
        banner_area = max(1, (x2 - x1) * (y2 - y1))
        target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        overlap_banner = inter / banner_area
        overlap_target = inter / target_area
        if overlap_banner < self.TEMPLATE_BANNER_ALIGN_MIN_OVERLAP:
            return False
        if overlap_target < 0.25:
            return False
        return True

    @staticmethod
    def _readiness_hint(
        *,
        has_marker: bool,
        has_banner: bool,
        has_banner_aligned: bool,
        has_party_icon: bool,
        ready: bool,
    ) -> str:
        if not has_marker:
            return "빨간 X 마커가 박스 안에 보이도록 맞춰주세요"
        if not has_banner:
            return "지역명 배너(상단 검은 띠)가 프레임 안에 보이도록 맞춰주세요"
        if ready and not has_banner_aligned:
            return "✓ 캡처 가능 — 지역명을 노란 칸 근처에 두면 OCR이 더 잘 됩니다"
        if ready and not has_party_icon:
            return "✓ 캡처 가능 — 하단 1/8 아이콘까지 넣으면 더 좋아요"
        if ready:
            return "✓ 캡처 가능 — 지도·X·배너 확인됨"
        return "보물지도 창을 네모 안에 맞춰주세요"

    def suggest_crop_box(
        self, image: Image.Image
    ) -> tuple[int, int, int, int] | None:
        """마커 기준 추천 크롭 (이미지 좌표 x1,y1,x2,y2) — 가이드 오버레이용"""
        box = self._estimate_map_crop_box(image)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        if x2 - x1 < 80 or y2 - y1 < 80:
            return None
        return box

    def _estimate_map_crop_box(
        self, image: Image.Image
    ) -> tuple[int, int, int, int] | None:
        rgb = np.array(image.convert("RGB"))
        marker = self._find_treasure_x_marker(rgb)
        if marker is None:
            return None

        width, height = image.size
        mx, my = marker

        mask = self._build_parchment_mask(rgb)
        ys, xs = np.where(mask > 0)
        if len(ys) >= 55:
            valid = (
                (xs >= mx - int(width * 0.55))
                & (xs <= mx + int(width * 0.55))
                & (ys >= my - int(height * 0.45))
                & (ys <= my + int(height * 0.65))
            )
            xs_f, ys_f = xs[valid], ys[valid]
            if len(ys_f) >= 50:
                x1, x2 = int(xs_f.min()), int(xs_f.max()) + 1
                y1, y2 = int(ys_f.min()), int(ys_f.max()) + 1
                actual_parchment_h = y2 - y1
                banner_h = int(actual_parchment_h * self.BANNER_FROM_MARKER_RATIO)
                if (my - y1) < banner_h:
                    y1 = max(0, int(my - banner_h))
                else:
                    y1 = max(0, y1 - int(actual_parchment_h * 0.12))
                pad_w = max(2, int((x2 - x1) * 0.03))
                x1 = max(0, x1 - pad_w)
                x2 = min(width, x2 + pad_w)
                y2 = min(height, y2 + int(actual_parchment_h * 0.05))
                band = self._find_banner_band_bounds(rgb, allow_fallback=False)
                if band is not None:
                    y1 = min(y1, max(0, band[1] - 2))
                if self._crop_contains_marker(
                    image.crop((x1, y1, x2, y2)), marker, x1, y1
                ):
                    return (x1, y1, x2, y2)

        for scale_ratio in (0.96, 0.85, 0.75):
            test_w = int(width * scale_ratio)
            test_h = int(test_w / self.MAP_ASPECT)
            left = int(mx - test_w * self.MARKER_X_RATIO)
            top = int(my - test_h * 0.46)
            left = max(0, left)
            top = max(0, top)
            right = min(width, left + test_w)
            bottom = min(height, top + test_h)
            if right - left < 80 or bottom - top < 80:
                continue
            if self._crop_contains_marker(
                image.crop((left, top, right, bottom)), marker, left, top
            ):
                return (left, top, right, bottom)
        return None

    @staticmethod
    def _parchment_bounds_near_marker(
        rgb: np.ndarray,
        marker: tuple[float, float],
    ) -> tuple[int, int, int, int] | None:
        """X가 속한 양피지 지도 창 bbox — 작은 조각(마커 주변) 오탐 제외"""
        height, width = rgb.shape[:2]
        mxi = int(np.clip(round(marker[0]), 0, width - 1))
        myi = int(np.clip(round(marker[1]), 0, height - 1))
        mask = TreasureCaptureProcessor._build_parchment_mask(rgb)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )
        min_area = max(600, int(width * height * 0.06))

        best: tuple[int, int, int, int] | None = None
        best_area = 0
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x1 = int(stats[idx, cv2.CC_STAT_LEFT])
            y1 = int(stats[idx, cv2.CC_STAT_TOP])
            x2 = x1 + int(stats[idx, cv2.CC_STAT_WIDTH])
            y2 = y1 + int(stats[idx, cv2.CC_STAT_HEIGHT])
            if not (x1 <= mxi < x2 and y1 <= myi < y2):
                continue
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is not None:
            return best

        mx, my = marker
        ys, xs = np.where(closed > 0)
        if len(ys) < 40:
            return None
        valid = (
            (xs >= mx - int(width * 0.55))
            & (xs <= mx + int(width * 0.55))
            & (ys >= my - int(height * 0.45))
            & (ys <= my + int(height * 0.65))
        )
        xs_f, ys_f = xs[valid], ys[valid]
        if len(ys_f) < 40:
            return None
        return (
            int(xs_f.min()),
            int(ys_f.min()),
            int(xs_f.max()) + 1,
            int(ys_f.max()) + 1,
        )

    def _detect_party_icon_pattern(self, rgb: np.ndarray) -> bool:
        """하단 1/8 아이콘 영역 — OCR 없이 밝은 숫자/아이콘 패턴"""
        height, width = rgb.shape[:2]
        if height < 80:
            return False

        icon = rgb[int(height * 0.78) :, : int(width * 0.30)]
        if icon.size == 0:
            return False

        gray = cv2.cvtColor(icon, cv2.COLOR_RGB2GRAY)
        num_area = gray[:, int(gray.shape[1] * 0.38) :]
        if num_area.size == 0:
            return False

        bright_ratio = float((num_area > 175).mean())
        mid_bright = float((gray > 150).mean())
        return bright_ratio >= 0.010 or (mid_bright >= 0.04 and bright_ratio >= 0.006)

    def _map_window_has_banner(self, image: Image.Image) -> bool:
        """크롭에 지역명 배너 밴드(충분한 높이·좌측 흰 글자)가 포함됐는지 확인"""
        rgb = np.array(image.convert("RGB"))
        if self._find_treasure_x_marker(rgb) is None:
            return False
        band = self._find_banner_band_bounds(rgb)
        if band is None:
            return False
        return (band[3] - band[1]) >= 8

    def _pick_best_map_window(
        self,
        raw: Image.Image,
        options: list[Image.Image],
    ) -> Image.Image:
        """배너·마커·양피지 비율로 최적 보정본 선택 (면적 최소 ≠ 정답)"""
        best = raw
        best_score = -1.0
        for option in options:
            if option.width < 80 or option.height < 80:
                continue
            score = self._score_map_window_candidate(raw, option)
            if score > best_score:
                best_score = score
                best = option
        return best

    def _score_map_window_candidate(
        self,
        raw: Image.Image,
        candidate: Image.Image,
    ) -> float:
        rgb = np.array(candidate.convert("RGB"))
        height, width = rgb.shape[:2]
        if self._find_treasure_x_marker(rgb) is None:
            return 0.0

        score = 0.45

        band = self._find_banner_band_bounds(rgb)
        if band is None:
            score -= 0.2
        else:
            _x1, y1, _x2, y2 = band
            if y1 <= height * 0.2:
                score += 0.2
            else:
                score -= 0.25
            band_h = max(1, y2 - y1)
            banner_rows = rgb[y1:y2, :, :]
            dark_ratio = float(
                (
                    (banner_rows[:, :, 0] < 100)
                    & (banner_rows[:, :, 1] < 92)
                    & (banner_rows[:, :, 2] < 88)
                ).mean()
            )
            score += min(0.25, dark_ratio * 0.45 * min(1.0, 12.0 / band_h))

        parchment_ratio = float(self._build_parchment_mask(rgb).mean() / 255.0)
        score += min(0.12, parchment_ratio * 0.18)

        area_ratio = (candidate.width * candidate.height) / max(
            1, raw.width * raw.height
        )
        if 0.08 <= area_ratio <= 0.98:
            score += 0.08
        elif area_ratio < 0.05:
            score -= 0.15

        aspect = candidate.width / max(candidate.height, 1)
        if 0.75 <= aspect <= 1.55:
            score += 0.05

        return score

    def extract_map_candidates(self, image: Image.Image) -> list[Image.Image]:
        """프레임 내 위치가 달라도 순차 시도할 보정본 (드래그 캡처 우선)"""
        seen: set[tuple[int, int, bytes]] = set()
        candidates: list[Image.Image] = []

        def add(img: Image.Image) -> None:
            if img.width < 40 or img.height < 40:
                return
            key = (img.width, img.height, img.tobytes())
            if key in seen:
                return
            seen.add(key)
            candidates.append(img)

        add(image)
        tight = self._capture_is_tight_map_frame(image)
        if not tight:
            focused = self.focus_map_window(image)
            if focused.size[0] * focused.size[1] >= image.size[0] * image.size[1] * 0.88:
                add(focused)
            marker_crop = self.auto_extract_by_marker(image)
            if marker_crop is not image:
                if marker_crop.size[0] * marker_crop.size[1] >= image.size[0] * image.size[1] * 0.88:
                    add(marker_crop)
            elif self._find_treasure_x_marker(np.array(image.convert("RGB"))) is not None:
                add(
                    self.auto_extract_landscape(
                        image,
                        marker_x_ratio=self.MARKER_X_RATIO,
                    )
                )
        return candidates

    def auto_extract_landscape(
        self,
        image: Image.Image,
        *,
        marker_x_ratio: float | None = None,
        marker_y_ratio: float | None = None,
    ) -> Image.Image:
        """보물지도 가로형 비율로 빨간 X 기준 창 추출 (프레임 내 위치 무관)"""
        rgb = np.array(image.convert("RGB"))
        marker = self._find_treasure_x_marker(rgb)
        if marker is None:
            return image

        width, height = image.size
        mx_ratio = marker_x_ratio if marker_x_ratio is not None else self.MARKER_X_RATIO
        my_ratio = marker_y_ratio if marker_y_ratio is not None else self.MARKER_Y_RATIO

        mask = self._build_parchment_mask(rgb)
        ys, xs = np.where(mask > 0)
        if len(ys) >= 60:
            box_h = int(ys.max()) - int(ys.min()) + 1
            box_w = int(xs.max()) - int(xs.min()) + 1
            map_h = min(height, max(155, int(box_h * 1.42)))
            map_w = min(width, max(200, int(box_w * 1.10)))
        else:
            map_h = min(height, max(155, int(height * 0.78)))
            map_w = min(width, max(200, int(map_h * self.MAP_ASPECT)))
        map_w = min(map_w, width)
        map_h = min(map_h, height)

        left = int(marker[0] - map_w * mx_ratio)
        top = int(marker[1] - map_h * my_ratio)
        left = max(0, min(left, width - map_w))
        top = max(0, min(top, height - map_h))
        cropped = image.crop((left, top, left + map_w, top + map_h))

        if cropped.width < 80 or cropped.height < 80:
            return image
        if cropped.width * cropped.height >= width * height * 0.995:
            return image
        return cropped

    def auto_extract_by_marker(self, image: Image.Image) -> Image.Image:
        """
        X 마커 + 양피지 실측으로 보물지도 창 크롭.
        고정 틀 안에서 지도 UI 위치·크기가 달라도 배너·양피지 기준으로 맞춘다.
        """
        rgb = np.array(image.convert("RGB"))
        marker = self._find_treasure_x_marker(rgb)
        if marker is None:
            return image

        width, height = image.size
        mx, my = marker

        parchment = self._parchment_bounds_near_marker(rgb, marker)
        if parchment is not None:
            px1, py1, px2, py2 = parchment
            area_ratio = ((px2 - px1) * (py2 - py1)) / max(width * height, 1)
            if area_ratio > 0.85:
                found = self._find_parchment_window(image)
                if found is not None:
                    return found

            x1, y1, x2, y2 = px1, py1, px2, py2

            band = self._find_banner_band_bounds(rgb, allow_fallback=True)
            if band is not None:
                y1 = min(y1, max(0, band[1] - 2))
                x1 = min(x1, band[0])
                x2 = max(x2, band[2])
            else:
                slot_h = int(height * self.TEMPLATE_BANNER_HEIGHT_RATIO)
                y1 = min(y1, max(0, slot_h - int(height * 0.04)))

            pad_x = max(2, int((x2 - x1) * 0.02))
            x1 = max(0, x1 - pad_x)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, py2 + max(4, int((py2 - py1) * 0.05)))

            cropped = image.crop((x1, y1, x2, y2))
            if (
                cropped.width >= 80
                and cropped.height >= 60
                and self._crop_contains_marker(cropped, marker, x1, y1)
            ):
                return cropped

        estimated_map_w = width
        estimated_map_h = int(estimated_map_w / self.MAP_ASPECT)

        left = int(mx - estimated_map_w * self.MARKER_X_RATIO)
        top = int(my - estimated_map_h * 0.46)
        right = left + estimated_map_w
        bottom = top + estimated_map_h

        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if right > width:
            right = width
            left = max(0, right - estimated_map_w)
        if bottom > height:
            bottom = height
            top = max(0, bottom - estimated_map_h)

        cropped = image.crop((left, top, right, bottom))
        if cropped.width >= 80 and cropped.height >= 80:
            return cropped

        return image

    @staticmethod
    def _crop_contains_marker(
        cropped: Image.Image,
        marker: tuple[float, float],
        offset_x: int,
        offset_y: int,
    ) -> bool:
        mx, my = marker
        return (
            0 <= mx - offset_x < cropped.width
            and 0 <= my - offset_y < cropped.height
        )

    def locate_treasure_map_window(self, image: Image.Image) -> Image.Image:
        """캡처 영역 안에서 보물지도 양피지 창 자동 검출 (실패 시 원본)"""
        width, height = image.size
        if width < 40 or height < 40:
            return image

        if width / max(height, 1) > 1.15:
            trimmed = self._trim_side_panels(image)
            found = self._find_parchment_window(trimmed)
            if found is not None:
                return found

        found = self._find_parchment_window(image)
        return found if found is not None else image

    def _trim_side_panels(self, image: Image.Image) -> Image.Image:
        """소지품 등 우측 어두운 패널이 포함된 캡처에서 보물지도 쪽만 남김"""
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if width < 80:
            return image

        cutoff = width
        for x in range(width - 1, int(width * 0.35), -1):
            col = rgb[:, x]
            dark_ratio = float(
                ((col[:, 0] < 85) & (col[:, 1] < 85) & (col[:, 2] < 90)).mean()
            )
            if dark_ratio > 0.52:
                cutoff = x
                break

        if cutoff < int(width * 0.88):
            return image.crop((0, 0, max(80, cutoff), height))

        return image.crop((0, 0, int(width * 0.78), height))

    def _find_parchment_window(self, image: Image.Image) -> Image.Image | None:
        """contour 기반 백업 크롭 (마커 있으면 X 포함 contour만)"""
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]

        marker = self._find_treasure_x_marker(rgb)
        parchment = self._build_parchment_mask(rgb)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(parchment, cv2.MORPH_CLOSE, kernel, iterations=2)

        img_area = height * width
        best_rect: tuple[int, int, int, int] | None = None
        best_area = 0

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            area = box_w * box_h
            if area < img_area * self.PARCHMENT_MIN_AREA_RATIO:
                continue

            if marker is not None:
                mx, my = marker
                if not (x <= mx < x + box_w and y <= my < y + box_h):
                    continue

            aspect = box_w / max(box_h, 1)
            if aspect < 0.50 or aspect > 1.80:
                continue

            if area > best_area:
                best_area = area
                best_rect = (x, y, x + box_w, y + box_h)

        if best_rect is None:
            return None

        x1, y1, x2, y2 = best_rect
        box_w = x2 - x1
        box_h = y2 - y1
        left = max(0, x1 - max(2, int(box_w * 0.02)))
        top = max(0, y1 - max(4, int(box_h * self.BANNER_FROM_MARKER_RATIO)))
        right = min(width, x2 + max(2, int(box_w * 0.02)))
        bottom = min(height, y2 + max(4, int(box_h * 0.10)))
        return image.crop((left, top, right, bottom))

    def _refine_map_window_crop(
        self, raw: Image.Image, map_window: Image.Image
    ) -> Image.Image:
        """캡처에 배경이 많을 때 양피지 창만 더 타이트하게 잘라냄"""
        if map_window.size == raw.size:
            found = self._find_parchment_window(raw)
            if found is not None:
                map_window = found

        rgb = np.array(map_window.convert("RGB"))
        marker = self._find_treasure_x_marker(rgb)
        mask = self._build_parchment_mask(rgb)
        ys, xs = np.where(mask > 0)
        if len(ys) < 60:
            return map_window

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        box_h = y2 - y1
        y1 = max(0, y1 - int(box_h * self.BANNER_TOP_EXPAND))

        pad = 3
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(map_window.width, x2 + pad)
        y2 = min(map_window.height, y2 + pad)

        if marker is not None:
            mx, my = marker
            if not (x1 <= mx < x2 and y1 <= my < y2):
                return map_window

        refined = map_window.crop((x1, y1, x2, y2))
        if refined.width < 80 or refined.height < 80:
            return map_window
        return refined

    def _trim_map_window_margins(self, image: Image.Image) -> Image.Image:
        """양피지·배너·X 기준으로 하단/측면 여백(비·숲 배경) 제거"""
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        marker = self._find_treasure_x_marker(rgb)
        mask = self._build_parchment_mask(rgb)
        ys, xs = np.where(mask > 0)
        if len(ys) < 60:
            return image

        box_h = int(ys.max()) - int(ys.min()) + 1
        x1 = max(0, int(xs.min()) - max(4, int(width * 0.04)))
        y1 = max(0, int(ys.min()) - int(box_h * self.BANNER_TOP_EXPAND))
        x2 = min(width, int(xs.max()) + max(4, int(width * 0.04)) + 1)
        y2 = min(height, int(ys.max()) + max(4, int(box_h * 0.14)) + 1)

        band = self._find_banner_band_bounds(rgb)
        if band is not None:
            _bx1, by1, _bx2, by2 = band
            # 상단 근처 실제 어두운 배너일 때만 y1을 올려 배너 위 여백 제거.
            # 중간 오탐 밴드(밝은 양피지 맵)에 y1을 올리면 지역명 배너가 잘린다.
            if by1 <= max(12, int(height * 0.12)):
                y1 = max(y1, max(0, by1 - 2))
            y2 = max(y2, min(height, by2 + 2))

        if marker is not None:
            mx, my = marker
            if not (x1 <= mx < x2 and y1 <= my < y2):
                return image

        cropped = image.crop((x1, y1, x2, y2))
        if cropped.width < 80 or cropped.height < 80:
            return image
        return cropped

    def _strip_rows_above_banner(self, image: Image.Image) -> Image.Image:
        """배너 위쪽 오버레이·배경 행 제거 (배너가 crop 상단에 오도록)"""
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if height < 40:
            return image

        band = self._find_banner_band_bounds(rgb)
        if band is not None and (band[3] - band[1]) >= 8 and band[1] > 4:
            top = max(0, band[1] - 1)
            if top >= height - 40:
                return image
            return image.crop((0, top, width, height))

        parchment = self._parchment_bounds(rgb)
        if parchment is None:
            return image
        py_top = parchment[1]
        if py_top <= 10:
            return image
        banner_h = max(10, min(int((parchment[3] - py_top) * self.BANNER_BAND_RATIO), 24))
        top = max(0, py_top - banner_h)
        if top >= height - 40:
            return image
        return image.crop((0, top, width, height))

    def _find_banner_band_bounds(
        self,
        rgb: np.ndarray,
        *,
        marker: tuple[float, float] | None = None,
        allow_fallback: bool = True,
    ) -> tuple[int, int, int, int] | None:
        """X 마커 위쪽 어두운 지역명 배너 — 금 테두리(py_top=0) 오탐 방지"""
        height, width = rgb.shape[:2]
        if height < 30 or width < 40:
            return None

        if marker is None:
            marker = self._find_treasure_x_marker(rgb)
        scan_start = 0
        if marker is not None:
            scan_end = min(height, int(marker[1]) + max(16, int(height * 0.06)))
        else:
            scan_end = max(30, int(height * 0.50))

        parchment = self._parchment_bounds(rgb)
        x_hint = (
            (parchment[0], parchment[2]) if parchment is not None else (0, width)
        )

        best: tuple[int, int, int, int, float] | None = None
        y = scan_start
        while y < scan_end:
            row = rgb[y]
            dark_ratio = float(
                ((row[:, 0] < 100) & (row[:, 1] < 92) & (row[:, 2] < 88)).mean()
            )
            if dark_ratio < 0.38:
                y += 1
                continue

            start = y
            run_scores: list[float] = []
            while y < scan_end:
                row = rgb[y]
                dr = float(
                    ((row[:, 0] < 100) & (row[:, 1] < 92) & (row[:, 2] < 88)).mean()
                )
                if dr < 0.30:
                    break
                run_scores.append(dr)
                y += 1

            if len(run_scores) < 3:
                continue

            y1 = max(0, start - 1)
            y2 = min(height, start + len(run_scores) + 2)
            band_h = y2 - y1
            if band_h < 8 or band_h > int(height * 0.38):
                continue

            band = rgb[y1:y2]
            left_w = max(8, int(width * 0.58))
            left_band = band[:, :left_w, :]
            left_bright = float(
                (
                    (left_band[:, :, 0] > 130)
                    & (left_band[:, :, 1] > 130)
                    & (left_band[:, :, 2] > 130)
                ).mean()
            )
            if left_bright < 0.025:
                continue

            bright = float(
                (
                    (band[:, :, 0] > 130)
                    & (band[:, :, 1] > 130)
                    & (band[:, :, 2] > 130)
                ).mean()
            )

            mid_x1 = max(0, int(width * 0.28))
            mid_x2 = min(width, int(width * 0.72))
            if mid_x2 > mid_x1 + 8:
                mid_band = band[:, mid_x1:mid_x2, :]
                mid_bright = float(
                    (
                        (mid_band[:, :, 0] > 130)
                        & (mid_band[:, :, 1] > 130)
                        & (mid_band[:, :, 2] > 130)
                    ).mean()
                )
                if mid_bright > left_bright * 1.25 and mid_bright > 0.05:
                    continue
                if left_bright < mid_bright * 0.75:
                    continue

            col_dark = (
                (band[:, :, 0] < 110)
                & (band[:, :, 1] < 100)
                & (band[:, :, 2] < 95)
            ).mean(axis=0)
            xs = np.where(col_dark > 0.22)[0]
            if len(xs) < max(8, int(width * 0.18)):
                continue

            x1 = max(0, int(xs.min()) - 2)
            x2 = min(width, int(xs.max()) + 3)
            width_ratio = (x2 - x1) / max(width, 1)
            if width_ratio < 0.28:
                continue

            score = (
                left_bright * 0.50
                + float(np.mean(run_scores)) * 0.25
                + bright * 0.10
                + width_ratio * 0.10
            )
            if marker is not None and y2 <= marker[1] + 12:
                score += 0.12
            if left_bright >= 0.08:
                score += 0.18
            if x1 <= x_hint[0] + 6:
                score += 0.05

            if best is None or score > best[4]:
                best = (x1, y1, x2, y2, score)

        if best is not None:
            band = best[0], best[1], best[2], best[3]
            if self._is_banner_band_in_header(band, height):
                return band

        # 밝은 양피지 배너: 어두운 런이 없어도 상단에서 갈색 글씨(블루 결핍) 띠 탐색
        parch_best: tuple[int, int, int, int, float] | None = None
        scan_h = min(scan_end, max(24, int(height * 0.28)))
        y = 0
        while y < scan_h:
            row = rgb[y]
            bright_ratio = float(
                ((row[:, 0] > 150) & (row[:, 1] > 140) & (row[:, 2] > 100)).mean()
            )
            if bright_ratio < 0.45:
                y += 1
                continue
            start = y
            run = 0
            while y < scan_h:
                row = rgb[y]
                br = float(
                    ((row[:, 0] > 150) & (row[:, 1] > 140) & (row[:, 2] > 100)).mean()
                )
                if br < 0.35:
                    break
                run += 1
                y += 1
            # 상단명 배너는 보통 얇음 — 맵 텍스처까지 포함한 긴 런 제외
            if run < 6 or run > max(14, int(height * 0.16)):
                continue
            if start > max(8, int(height * 0.10)):
                continue
            y1b = max(0, start - 1)
            y2b = min(height, start + min(run, max(10, int(height * 0.14))) + 1)
            band = rgb[y1b:y2b]
            deficit = (
                (band[:, :, 0].astype(np.int16) + band[:, :, 1].astype(np.int16)) // 2
                - band[:, :, 2].astype(np.int16)
            )
            ink_ratio = float((deficit > 28).mean())
            if ink_ratio < 0.02:
                continue
            col_ink = (deficit > 28).mean(axis=0)
            xs_ink = np.where(col_ink > 0.08)[0]
            if len(xs_ink) < max(8, int(width * 0.15)):
                continue
            x1b = max(0, int(xs_ink.min()) - 2)
            x2b = min(width, int(xs_ink.max()) + 3)
            if (x2b - x1b) / max(width, 1) < 0.25:
                continue
            score = ink_ratio * 0.55 + bright_ratio * 0.25 + ((x2b - x1b) / max(width, 1)) * 0.20
            if parch_best is None or score > parch_best[4]:
                parch_best = (x1b, y1b, x2b, y2b, score)
        if parch_best is not None:
            band = parch_best[0], parch_best[1], parch_best[2], parch_best[3]
            if self._is_banner_band_in_header(band, height):
                return band

        if not allow_fallback:
            return None

        if parchment is not None and marker is not None:
            px1, _py_top, px2, _py_bottom = parchment
            my = int(marker[1])
            banner_h = max(12, int((my) * self.BANNER_FROM_MARKER_RATIO))
            y1 = max(0, my - banner_h)
            y2 = min(height, my - max(8, int(banner_h * 0.55)))
            if y2 - y1 >= 8:
                return (max(0, px1 - 2), y1, min(width, px2 + 2), y2)

        return None

    @staticmethod
    def _parchment_bounds(
        rgb: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        """양피지 본체 bbox (x1, y1, x2, y2)"""
        mask = TreasureCaptureProcessor._build_parchment_mask(rgb)
        ys, xs = np.where(mask > 0)
        if len(ys) < 40:
            return None
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )

    @staticmethod
    def _build_parchment_mask(rgb: np.ndarray) -> np.ndarray:
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        parchment = (
            (r >= 115)
            & (r <= 235)
            & (g >= 70)
            & (g <= 200)
            & (b >= 28)
            & (b <= 150)
            & (r > g + 8)
            & (g > b + 4)
        )
        return (parchment.astype(np.uint8)) * 255

    @classmethod
    def normalize_for_detection(cls, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """작은 캡처를 마커 검출 기준 폭으로 업스케일 (절대 픽셀 area 임계값 보정)"""
        height, width = rgb.shape[:2]
        ref_w = cls.MARKER_DETECT_REF_WIDTH
        if width >= ref_w:
            return rgb, 1.0
        scale = ref_w / max(width, 1)
        new_w = ref_w
        new_h = max(1, int(round(height * scale)))
        scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return scaled, scale

    @classmethod
    def _marker_inside_map_window(
        cls,
        rgb: np.ndarray,
        marker: tuple[float, float],
    ) -> bool:
        """X가 양피지 지도 창 안·슬롯 아래에 있는지 (지형 위 X 포함)"""
        parchment = cls._parchment_bounds_near_marker(rgb, marker)
        if parchment is None:
            parchment = cls._parchment_bounds(rgb)
        if parchment is None:
            return False

        mx, my = marker
        px1, py1, px2, py2 = parchment
        if not (px1 <= mx < px2 and py1 <= my < py2):
            return False

        height, width = rgb.shape[:2]
        slot_bottom = int(height * cls.TEMPLATE_BANNER_HEIGHT_RATIO)
        if my <= slot_bottom + max(4, int(height * 0.02)):
            return False

        box_w = px2 - px1
        box_h = py2 - py1
        if box_h < max(50, int(height * 0.20)):
            return False
        aspect = box_w / max(box_h, 1)
        if not (cls.PARCHMENT_ASPECT_MIN <= aspect <= cls.PARCHMENT_ASPECT_MAX):
            return False
        return True

    @classmethod
    def _marker_on_parchment_at(
        cls,
        rgb: np.ndarray,
        marker: tuple[float, float],
    ) -> bool:
        return cls._marker_inside_map_window(rgb, marker)

    @classmethod
    def _build_treasure_x_marker_mask(cls, rgb: np.ndarray) -> np.ndarray:
        """빨간 X 마스크 — RGB + BGR 이중 조건 (화면 캡처 채널 뒤집힘 대응)"""
        c1 = rgb[:, :, 0].astype(np.int16)
        c2 = rgb[:, :, 1].astype(np.int16)
        c3 = rgb[:, :, 2].astype(np.int16)
        mask_rgb = (c1 > 120) & (c2 < 125) & (c3 < 125) & (c1 > c2 + 25)
        mask_bgr = (c3 > 120) & (c2 < 125) & (c1 < 125) & (c3 > c2 + 25)
        return ((mask_rgb | mask_bgr).astype(np.uint8)) * 255

    @classmethod
    def _collect_treasure_x_marker_candidates(
        cls, rgb: np.ndarray
    ) -> list[tuple[float, float, float]]:
        """빨간 X 후보 (cx, cy, score) — score 내림차순"""
        mask = cls._build_treasure_x_marker_mask(rgb)

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        height, width = rgb.shape[:2]
        border_margin = max(4, int(min(width, height) * 0.01))
        slot_bottom = int(height * cls.TEMPLATE_BANNER_HEIGHT_RATIO)
        upper_band = height * 0.28

        candidates: list[tuple[float, float, float]] = []

        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < cls.MARKER_AREA_MIN or area > cls.MARKER_AREA_MAX:
                continue
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if w < 5 or h < 5:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.6 or aspect > 1.6:
                continue

            cx = float(centroids[idx][0])
            cy = float(centroids[idx][1])
            if (
                cx < border_margin
                or cy < border_margin
                or cx > width - border_margin
                or cy > height - border_margin
            ):
                continue

            # area 자체를 base score로 쓰면 X와 무관한 큰 얼룩(장식·그림자)이
            # 크기만으로 진짜 X(작고 일정한 크기)를 이겨버린다.
            # base는 "이상적 크기(25~90)에 얼마나 가까운가"로만 결정한다.
            if 25 <= area <= 90:
                score = 120.0
            else:
                over = max(area - 90, 25 - area, 0)
                score = -over * 1.5

            if cy <= slot_bottom + 5:
                score -= 600.0
            elif cy < upper_band:
                score -= 200.0
            elif cy > slot_bottom + 8:
                score += 150.0

            candidates.append((cx, cy, score))

        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    @classmethod
    def _find_treasure_x_marker_raw(cls, rgb: np.ndarray) -> tuple[float, float] | None:
        candidates = cls._collect_treasure_x_marker_candidates(rgb)
        if not candidates:
            return None
        return (candidates[0][0], candidates[0][1])

    @classmethod
    def _find_treasure_x_marker_relaxed(
        cls, rgb: np.ndarray
    ) -> tuple[float, float] | None:
        """실시간 품질 게이트 — 프레임 안 빨간 X만 확인 (슬롯·양피지 비율 무관)"""
        for cx, cy, _score in cls._collect_treasure_x_marker_candidates(rgb):
            marker = (cx, cy)
            if cls._marker_in_frame(rgb, marker):
                return marker

        scaled, scale = cls.normalize_for_detection(rgb)
        if scale != 1.0:
            for cx, cy, _score in cls._collect_treasure_x_marker_candidates(scaled):
                marker = (cx / scale, cy / scale)
                if cls._marker_in_frame(rgb, marker):
                    return marker

        return None

    @classmethod
    def _find_treasure_x_marker(cls, rgb: np.ndarray) -> tuple[float, float] | None:
        """보물지도 빨간 X 중심 — 캡처 후 크롭용, 양피지 창 안 후보 우선"""
        for cx, cy, _score in cls._collect_treasure_x_marker_candidates(rgb):
            marker = (cx, cy)
            if cls._marker_inside_map_window(rgb, marker):
                return marker

        scaled, scale = cls.normalize_for_detection(rgb)
        if scale != 1.0:
            for cx, cy, _score in cls._collect_treasure_x_marker_candidates(scaled):
                marker = (cx / scale, cy / scale)
                if cls._marker_inside_map_window(rgb, marker):
                    return marker

        return None

    @classmethod
    def _find_treasure_x_marker_rel(cls, rgb: np.ndarray) -> tuple[float, float] | None:
        """정규화 X 위치 (0~1) — RGB blob 검출"""
        marker = cls._find_treasure_x_marker_relaxed(rgb)
        if marker is None:
            return None
        height, width = rgb.shape[:2]
        if width < 1 or height < 1:
            return None
        return (marker[0] / width, marker[1] / height)

    def detect_zone_from_banner(self, image: Image.Image) -> Optional[dict]:
        """상단 '야크텔 밀림' 같은 지역명 배너에서 zones.json 매칭"""
        zone, score, _text = self.detect_zone_from_banner_scored(image)
        if zone is not None and score >= 0.55:
            return zone
        return None

    def detect_zone_from_banner_scored(
        self, image: Image.Image
    ) -> tuple[Optional[dict], float, Optional[str]]:
        """배너 OCR → (지역, 신뢰도, OCR 원문)"""
        if not self._tesseract_available:
            return None, 0.0, None

        best_zone: Optional[dict] = None
        best_score = 0.0
        best_text: Optional[str] = None

        ocr_calls = 0
        max_calls_per_crop = 12

        banner_crops = self._banner_crops(image)
        if not banner_crops:
            logger.debug(
                "banner_ocr 크롭 없음: image=%dx%d (배너/지역명 띠 미검출)",
                image.width,
                image.height,
            )
            return best_zone, best_score, best_text

        logger.debug(
            "banner_ocr 시작: image=%dx%d crops=%d slot_h=%d",
            image.width,
            image.height,
            len(banner_crops),
            self.fixed_banner_target_zone(image.width, image.height)[3],
        )

        for crop_idx, banner in enumerate(banner_crops):
            crop_calls = 0
            for text in self._ocr_banner_candidates(banner):
                ocr_calls += 1
                crop_calls += 1
                if self._is_overlay_ocr_noise(text):
                    logger.debug("banner_ocr noise 필터됨: %r", text)
                    continue
                compact = re.sub(r"[\s·、．.,\-_'\"]", "", text)
                hangul_len = len(re.findall(r"[가-힣]", compact))
                if hangul_len < 2 or hangul_len > 18:
                    logger.debug(
                        "banner_ocr hangul_len 필터됨(%d): %r",
                        hangul_len,
                        text,
                    )
                    continue
                zone, score = self._match_banner_text_scored_with_runs(text)
                logger.debug(
                    "banner_ocr candidate text=%r -> zone=%s score=%.3f",
                    text,
                    zone.get("id") if zone else None,
                    score,
                )
                if zone is not None and score > best_score:
                    best_score = score
                    best_zone = zone
                    best_text = text
                    if best_score >= self.BANNER_OCR_EARLY_EXIT:
                        return best_zone, best_score, best_text
                if best_score >= self.BANNER_OCR_EARLY_EXIT:
                    return best_zone, best_score, best_text
                if crop_calls >= max_calls_per_crop and best_score < 0.55:
                    logger.debug(
                        "banner_ocr crop %d 조기 중단: crop_calls=%d best_score=%.3f",
                        crop_idx,
                        crop_calls,
                        best_score,
                    )
                    break
            if best_score >= self.BANNER_OCR_EARLY_EXIT:
                return best_zone, best_score, best_text
            if crop_idx >= 1 and best_score >= 0.72:
                return best_zone, best_score, best_text

        return best_zone, best_score, best_text

    def resolve_banner_zone(
        self, image: Image.Image, *, refocus: bool = True
    ) -> tuple[Optional[dict], float, Optional[str], Image.Image | None]:
        """캡처 보정 후 배너에서 지역명 우선 판별 — (zone, score, text, OCR용 map_window)"""
        best_zone: Optional[dict] = None
        best_score = 0.0
        best_text: Optional[str] = None
        best_map_window: Image.Image | None = None

        attempts: list[Image.Image] = []
        if self._capture_is_tight_map_frame(image):
            attempts.append(image)
        elif refocus:
            focused = self.focus_map_window(image)
            attempts.append(focused)
        else:
            attempts.append(image)

        seen: set[tuple[int, int]] = set()
        for attempt in attempts:
            key = (attempt.width, attempt.height)
            if key in seen:
                continue
            seen.add(key)

            map_window = self.localize_for_zone_ocr(attempt, refocus=False)
            zone, score, text = self.detect_zone_from_banner_scored(map_window)
            if zone is not None and score > best_score:
                best_zone = zone
                best_score = score
                best_text = text
                best_map_window = map_window
                if best_score >= self.BANNER_OCR_EARLY_EXIT:
                    break
            elif best_map_window is None:
                best_map_window = map_window

        return best_zone, best_score, best_text, best_map_window

    def _banner_crops(self, image: Image.Image) -> list[Image.Image]:
        """배너 OCR용 크롭 — 초록 슬롯(상단 22%) + 좌측 지역명 영역 우선"""
        width, height = image.size
        crops: list[Image.Image] = []
        seen: set[tuple[int, int, int, int]] = set()

        def add_crop(box: tuple[int, int, int, int]) -> None:
            x1, y1, x2, y2 = box
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 8, min(x2, width))
            y2 = max(y1 + 6, min(y2, height))
            key = (x1, y1, x2, y2)
            if key in seen:
                return
            seen.add(key)
            cropped_img = image.crop(key)
            padded_img = ImageOps.expand(cropped_img, border=6, fill="black")
            crops.append(padded_img)

        slot = self.fixed_banner_target_zone(width, height)
        _sx1, _sy1, _sx2, sy2 = slot
        rgb = np.array(image.convert("RGB"))
        band = self._find_banner_band_bounds(rgb)
        # 지역명은 좌측 — 우측 파티 아이콘·지형 노이즈 제외
        text_left = int(width * 0.04)
        text_right = int(width * 0.62)
        crop_bottom = min(height, sy2)

        # 1) 초록 슬롯 — y1=0 고정 (밴드 y1은 글자 꼭대기보다 아래라 잘림 유발)
        add_crop((text_left, 0, text_right, crop_bottom))
        add_crop((0, 0, int(width * 0.72), crop_bottom))

        # 2) 어두운 배너 — 가로만 밴드에 맞추고 세로는 슬롯 전체(y1=0)
        if band is not None:
            x1, _y1, x2, y2 = band
            add_crop(
                (
                    max(0, x1),
                    0,
                    min(x2, text_right),
                    min(height, max(crop_bottom, y2 + 2)),
                )
            )

        # 3) 밝은 양피지(라비린토스 등) — 밴드 미검출 시 슬롯 전체 너비
        if band is None:
            add_crop((0, 0, width, crop_bottom))

        return crops[:3]

    def _crop_banner_region(self, image: Image.Image) -> Image.Image | None:
        """지도 창 어두운 지역명 배너만 잘라 OCR 정확도 개선"""
        rgb = np.array(image.convert("RGB"))
        band = self._find_banner_band_bounds(rgb)
        if band is None:
            return None
        x1, y1, x2, y2 = band
        if y2 - y1 < 6:
            return None
        return image.crop((x1, y1, x2, y2))

    _OVERLAY_OCR_KEYWORDS: tuple[str, ...] = (
        "드래그",
        "모서리",
        "크기조절",
        "캡처",
        "캡저",
        "이동",
        "인식실패",
        "지역명을읽",
        "다시캡처",
        "맞춤필요",
        "소지품",
    )

    @classmethod
    def _is_overlay_ocr_noise(cls, text: str) -> bool:
        """캡처 UI·오류 다이얼로그 안내 문구 OCR 결과 제외"""
        compact = re.sub(r"[\s·、．.,\-_'\"]", "", text)
        if "드래그" in compact and ("캡처" in compact or "캡처" in text or "모서리" in compact):
            return True
        # 오류/가이드 다이얼로그 — 예시로 적힌 지역명이 오매칭되지 않게
        for key in (
            "인식실패",
            "지역명을읽",
            "다시캡처",
            "여기맞나",
            "맞춤필요",
        ):
            if key in compact:
                return True
        # 오류 다이얼로그 예시 문구(검은장막 숲 동부 삼림 등) 오매칭 방지
        example_markers = (
            "지역명",
            "상단지역명",
            "빨간x",
            "파티",
            "1/8",
            "캡처영역",
            "잘보이",
        )
        if sum(1 for key in example_markers if key in compact) >= 2:
            return True
        if "검은장막" in compact and ("동부" in compact or "삼림" in compact):
            if len(compact) > 18:
                return True
        hits = sum(1 for key in cls._OVERLAY_OCR_KEYWORDS if key in compact)
        return hits >= 2

    @staticmethod
    def _extract_banner_hangul_runs(compact: str) -> list[str]:
        """OCR 잡음 뒤에 붙은 한글을 제외하고 지역명 후보 구간만 추출"""
        return re.findall(r"[가-힣]{3,8}", compact)

    def _match_banner_text_scored_with_runs(
        self, text: str
    ) -> tuple[Optional[dict], float]:
        """전체 문자열 + 연속 한글 구간 각각 매칭 시도"""
        zone, score = self._match_banner_text_scored(text)
        if zone is not None:
            return zone, score

        compact = re.sub(r"[\s·、．.,\-_'\"~‥…]", "", text)
        for run in self._extract_banner_hangul_runs(compact):
            z, s = self._match_banner_text_scored(run)
            if z is not None and s > score:
                zone, score = z, s
        return zone, score

    def _zone_name_lexicon(
        self,
    ) -> tuple[frozenset[str], str, list[tuple[dict, str]]]:
        """zones.json name_ko 기반 한글 글자집·OCR whitelist·지역명 목록"""
        if self._zone_lexicon is not None:
            return self._zone_lexicon

        chars: set[str] = set()
        entries: list[tuple[dict, str]] = []
        seen_names: set[str] = set()

        for zone in self.coordinate_service.zones:
            compact = re.sub(
                r"[\s·、．.,\-_'\"]",
                "",
                str(zone.get("name_ko", "")),
            )
            if len(compact) < 2 or compact in seen_names:
                continue
            seen_names.add(compact)
            entries.append((zone, compact))
            chars.update(compact)

        for keywords in self._SHROUD_DIRECTIONS.values():
            for token in keywords:
                if re.fullmatch(r"[가-힣]+", token):
                    chars.update(token)

        whitelist = "".join(sorted(chars))
        self._zone_lexicon = (frozenset(chars), whitelist, entries)
        return self._zone_lexicon

    @staticmethod
    def _subsequence_coverage(needle: str, haystack: str) -> float:
        """지역명 글자가 OCR 한글열에 순서대로 포함되는 비율"""
        if not needle:
            return 0.0
        idx = 0
        for ch in haystack:
            if idx < len(needle) and ch == needle[idx]:
                idx += 1
        return idx / len(needle)

    @staticmethod
    def _best_window_similarity(needle: str, haystack: str) -> float:
        """OCR 한글열 슬라이딩 윈도우 ↔ 지역명 유사도"""
        from difflib import SequenceMatcher

        if not needle or not haystack:
            return 0.0
        if needle in haystack:
            return 1.0

        best = 0.0
        n = len(needle)
        max_w = min(len(haystack), n + 3)
        for width in range(max(2, n - 2), max_w + 1):
            for start in range(0, len(haystack) - width + 1):
                window = haystack[start : start + width]
                best = max(best, SequenceMatcher(None, needle, window).ratio())
        return best

    def _match_banner_lexicon(
        self, ocr_ko: str
    ) -> tuple[Optional[dict], float]:
        """OCR 한글열을 zones.json 지역명 글자집과 대조"""
        if len(ocr_ko) < 2:
            return None, 0.0

        _chars, _whitelist, entries = self._zone_name_lexicon()
        best_zone: Optional[dict] = None
        best_score = 0.0
        second_score = 0.0

        for zone, name in entries:
            if len(name) < 2:
                continue

            if name in ocr_ko:
                inline = min(
                    1.0,
                    0.78 + 0.22 * len(name) / max(len(ocr_ko), len(name)),
                )
            else:
                inline = 0.0

            sub = self._subsequence_coverage(name, ocr_ko)
            win = self._best_window_similarity(name, ocr_ko)
            score = max(inline, sub * 0.94, win * 0.90)

            if score > best_score:
                second_score = best_score
                best_score = score
                best_zone = zone
            elif score > second_score:
                second_score = score

        if best_zone is None or best_score < 0.72:
            return None, 0.0
        if second_score > 0.0 and best_score - second_score < 0.08:
            return None, 0.0

        zone_ko = re.sub(
            r"[\s·、．.,\-_'\"]",
            "",
            str(best_zone.get("name_ko", "")),
        )
        if abs(len(zone_ko) - len(ocr_ko)) > 6 and best_score < 0.88:
            return None, 0.0
        return best_zone, best_score

    def _match_banner_text(self, text: str) -> Optional[dict]:
        zone, _ = self._match_banner_text_scored(text)
        return zone

    _SHROUD_DIRECTIONS: dict[str, tuple[str, ...]] = {
        "east_shroud": ("동부", "east"),
        "central_shroud": ("중부", "central"),
        "south_shroud": ("남부", "south"),
        "north_shroud": ("북부", "north"),
    }

    def _match_shroud_subzone(self, text: str) -> tuple[Optional[dict], float]:
        """검은장막 숲 동부/중부/남부/북부 구분"""
        compact = re.sub(r"[\s·、．.,\-_'\"]", "", text)
        shroud_hint = any(
            token in compact
            for token in ("검은장막", "검은", "장막", "삼림", "숲")
        )
        if not shroud_hint:
            return None, 0.0
        # 다이얼로그 안내처럼 긴 문장은 제외
        if len(compact) > 24:
            return None, 0.0

        for zone_id, keywords in self._SHROUD_DIRECTIONS.items():
            if any(keyword in compact or keyword in text for keyword in keywords):
                zone = self.coordinate_service.get_zone(zone_id)
                if zone is not None:
                    return zone, 0.88
        return None, 0.0

    def _match_banner_text_scored(self, text: str) -> tuple[Optional[dict], float]:
        normalized = re.sub(r"[\s·、．.,\-_'\"]", "", text)
        if len(normalized) < 2:
            return None, 0.0

        shroud_zone, shroud_score = self._match_shroud_subzone(text)
        if shroud_zone is not None:
            return shroud_zone, shroud_score

        ocr_ko = "".join(re.findall(r"[가-힣]", text))
        if len(ocr_ko) >= 2:
            lex_zone, lex_score = self._match_banner_lexicon(ocr_ko)
            if lex_zone is not None and lex_score >= 0.72:
                return lex_zone, lex_score

        korean_count = len(ocr_ko)
        if korean_count < 3 or korean_count > 18:
            return None, 0.0

        best_zone: Optional[dict] = None
        best_score = 0.0
        second_score = 0.0

        for zone in self.coordinate_service.zones:
            for key in ("name_ko", "name_en"):
                candidate = str(zone.get(key, "")).replace(" ", "")
                if len(candidate) < 2:
                    continue
                score = self._zone_name_similarity(normalized, candidate)
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_zone = zone
                elif score > second_score:
                    second_score = score

        if best_zone is not None and best_score >= 0.55:
            if second_score > 0.0 and best_score - second_score < 0.08:
                return None, 0.0
            zone_ko = re.sub(r"[\s·、．.,\-_'\"]", "", str(best_zone.get("name_ko", "")))
            ocr_ko_count = len(re.findall(r"[가-힣]", normalized))
            if abs(len(zone_ko) - ocr_ko_count) > 5:
                return None, 0.0
            if len(zone_ko) <= 4 and best_score < 0.82:
                return None, 0.0
            if ocr_ko_count < 4 and best_score < 0.78:
                return None, 0.0
            # 짧고 확실한 후보(길이 일치 + 높은 유사)는 0.55 허용
            if (
                best_score < 0.72
                and "shroud" not in str(best_zone.get("id", ""))
                and not (
                    abs(len(zone_ko) - ocr_ko_count) <= 1
                    and best_score >= 0.58
                    and second_score < best_score - 0.10
                )
            ):
                return None, 0.0
            return best_zone, best_score

        # OCR이 1~2글자만 틀려도 SequenceMatcher로 재검토
        if best_zone is not None and best_score >= 0.50:
            zone_ko = re.sub(r"[\s·、．.,\-_'\"]", "", str(best_zone.get("name_ko", "")))
            ocr_ko_chars = re.findall(r"[가-힣]", normalized)
            if abs(len(zone_ko) - len(ocr_ko_chars)) <= 1 and best_score - second_score >= 0.12:
                return best_zone, max(best_score, 0.58)

        fallback = self.coordinate_service.find_zone_by_name(text)
        if fallback is not None and best_score >= 0.50:
            return fallback, 0.58
        return None, 0.0

    @staticmethod
    def _zone_name_similarity(ocr_text: str, zone_name: str) -> float:
        """OCR 문자열과 지역명 유사도 (0~1)"""
        from difflib import SequenceMatcher

        ocr = re.sub(r"[\s·、．.,\-_'\"]", "", ocr_text.lower().replace("'", ""))
        name = re.sub(r"[\s·、．.,\-_'\"]", "", zone_name.lower().replace("'", ""))
        if not ocr or not name:
            return 0.0
        if name in ocr or ocr in name:
            return min(1.0, len(name) / max(len(ocr), len(name)) + 0.35)

        # 한글만 추출해 길이 비슷한 경우 우선
        ocr_ko = "".join(re.findall(r"[가-힣]", ocr))
        name_ko = "".join(re.findall(r"[가-힣]", name))
        if ocr_ko and name_ko and abs(len(ocr_ko) - len(name_ko)) <= 2:
            seq = SequenceMatcher(None, ocr_ko, name_ko).ratio()
            common = sum(1 for ch in name_ko if ch in ocr_ko)
            coverage = common / len(name_ko)
            return max(seq, coverage * 0.95)

        common = sum(1 for ch in name if ch in ocr)
        coverage = common / len(name)
        if coverage >= 0.65:
            return coverage
        return coverage * 0.85

    _COORD_TEXT = re.compile(
        r"X\s*[:：]\s*([\d.]+)\s*[-–—~]\s*Y\s*[:：]\s*([\d.]+)",
        re.IGNORECASE,
    )

    def detect_coords_from_map(self, image: Image.Image) -> tuple[float, float] | None:
        """보물지도에 X: 24.05 - Y: 10.79 형식 좌표가 보이면 OCR로 추출"""
        if not self._tesseract_available:
            return self._parse_coords_from_text("")

        width, height = image.size
        # 상단 중앙 (좌표 숫자 표시 영역)
        strip = image.crop(
            (
                int(width * 0.18),
                int(height * 0.08),
                int(width * 0.82),
                int(height * 0.28),
            )
        )

        for text in self._ocr_coord_candidates(strip):
            parsed = self._parse_coords_from_text(text)
            if parsed is not None:
                return parsed
        return None

    def _parse_coords_from_text(self, text: str) -> tuple[float, float] | None:
        m = self._COORD_TEXT.search(text.replace(",", "."))
        if not m:
            return None
        try:
            x = float(m.group(1))
            y = float(m.group(2))
        except ValueError:
            return None
        if not (0.0 <= x <= 42.0 and 0.0 <= y <= 42.0):
            return None
        return round(x, 2), round(y, 2)

    def _ocr_coord_candidates(self, strip: Image.Image) -> list[str]:
        if not self._ocr.available:
            return []

        scale = max(2, int(320 / max(strip.width, 1)))
        scaled = strip.resize(
            (strip.width * scale, strip.height * scale),
            Image.Resampling.LANCZOS,
        )
        rgb = np.array(scaled.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        images: list[Image.Image] = []
        white = (rgb[:, :, 0] > 160) & (rgb[:, :, 1] > 160) & (rgb[:, :, 2] > 160)
        images.append(Image.fromarray(np.where(white, 0, 255).astype(np.uint8)))
        images.append(
            ImageOps.invert(ImageEnhance.Contrast(scaled.convert("L")).enhance(2.0))
        )

        texts: list[str] = []
        seen: set[str] = set()
        for img in images:
            for psm in (7, 6, 11):
                text = self._ocr.read_text(img, lang="eng", psm=psm)
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
                    parsed = self._parse_coords_from_text(text)
                    if parsed is not None:
                        return texts
        return texts

    def extract_map_content(self, image: Image.Image) -> Image.Image:
        """양피지 창 테두리·상단 배너·하단 UI를 제외한 지도 영역"""
        width, height = image.size
        left = int(width * 0.08)
        top = int(height * 0.14)
        right = int(width * 0.92)
        bottom = int(height * 0.88)
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _parse_party_digit(text: str) -> int | None:
        cleaned = text.strip()
        if cleaned == "1":
            return 1
        if cleaned == "8":
            return 8
        if "1" in cleaned and "8" not in cleaned:
            return 1
        if "8" in cleaned and "1" not in cleaned:
            return 8
        return None

    def _save_party_topology_debug(
        self,
        icon: Image.Image,
        digit_roi: np.ndarray,
        binary: np.ndarray,
    ) -> None:
        """TR_DEBUG_PARTY=1 — 위상 판별용 digit ROI·이진화 이미지 저장 (호출당 1회)"""
        if not is_debug_party() or self._party_debug_saved:
            return
        self._party_debug_saved = True

        out_dir = Path.cwd() / "debug_party"
        out_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)

        icon.save(out_dir / f"debug_party_{ts}_crop.png")
        Image.fromarray(digit_roi).save(out_dir / f"debug_party_{ts}_digit.png")
        Image.fromarray(binary).save(out_dir / f"debug_party_{ts}_binary.png")
        logger.debug("[party] debug 이미지 저장 %s (crop, digit, binary)", out_dir)

    @staticmethod
    def _party_banner_gray(icon: Image.Image) -> np.ndarray:
        """배지 줄 — 하단(양피지 좌하단) 우선, 없으면 상단"""
        width, height = icon.size
        banner_h = max(20, int(height * 0.48))
        # 1/8 아이콘은 보통 크롭의 하단쪽에 있음
        bottom = icon.crop((0, max(0, height - banner_h), width, height))
        bottom_rgb = np.array(bottom.convert("RGB"))
        bottom_bright = float(
            (
                (bottom_rgb[:, :, 0] > 180)
                & (bottom_rgb[:, :, 1] > 180)
                & (bottom_rgb[:, :, 2] > 180)
            ).mean()
        )
        if bottom_bright >= 0.01:
            return cv2.cvtColor(bottom_rgb, cv2.COLOR_RGB2GRAY)

        top = icon.crop((0, 0, width, banner_h))
        return cv2.cvtColor(np.array(top.convert("RGB")), cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _party_digit_roi(gray: np.ndarray) -> np.ndarray | None:
        """밝은 배지 영역에서 아이콘(왼쪽) 제외 후 숫자 ROI"""
        if gray.size == 0:
            return None

        height, width = gray.shape[:2]

        # 양피지 배경(~150)과 UI 흰색(~200+) 분리
        bright_th = max(190, int(float(np.percentile(gray, 93))))
        mask = gray > bright_th
        if int(mask.sum()) < 12:
            bright_th = max(170, int(float(np.percentile(gray, 88))))
            mask = gray > bright_th

        if int(mask.sum()) >= 12:
            ys, xs = np.where(mask)
            badge = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            split = max(1, int(badge.shape[1] * 0.42))
            roi = badge[:, split:]
            if roi.size and float((roi > bright_th - 15).mean()) >= 0.02:
                return roi

        # fallback: 연결요소 두 번째 blob 또는 고정 비율
        _, bright = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        )
        _n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            bright, connectivity=8
        )
        min_area = max(12, int(width * height * 0.008))
        blobs: list[tuple[int, int, int, int, int]] = []
        for i in range(1, _n):
            x, y, w, h, area = (
                int(stats[i, 0]),
                int(stats[i, 1]),
                int(stats[i, 2]),
                int(stats[i, 3]),
                int(stats[i, 4]),
            )
            if area >= min_area and h >= 4 and w >= 3:
                blobs.append((x, y, w, h, area))

        if len(blobs) >= 2:
            blobs.sort(key=lambda b: b[0])
            x, y, w, h, _area = blobs[1]
            pad = 2
            roi = gray[
                max(0, y - pad) : min(height, y + h + pad),
                max(0, x - pad) : min(width, x + w + pad),
            ]
            if roi.size:
                return roi

        x0 = int(width * 0.55)
        if x0 >= width - 4:
            return None
        return gray[:, x0:]

    @staticmethod
    def _count_digit_holes(digit_gray: np.ndarray) -> tuple[int | None, np.ndarray | None]:
        """Otsu 이진화 후 CCOMP 계층에서 구멍(자식 컨투어) 개수"""
        if digit_gray.size == 0:
            return None, None
        h, w = digit_gray.shape[:2]
        if h < 8 or w < 4:
            return None, None
        if float((digit_gray > 170).mean()) < 0.01:
            return None, None

        scale = max(4, int(64 / max(h, w)))
        scaled = cv2.resize(
            digit_gray,
            (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        blur = cv2.GaussianBlur(scaled, (3, 3), 0)
        _otsu, binary = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if float(binary.mean()) > 127:
            binary = 255 - binary

        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None or len(contours) == 0:
            return None, binary

        img_area = binary.shape[0] * binary.shape[1]
        min_hole_area = max(6, img_area * 0.001)

        holes = 0
        for i in range(len(contours)):
            if hierarchy[0][i][3] < 0:
                continue
            if cv2.contourArea(contours[i]) >= min_hole_area:
                holes += 1
        return holes, binary

    def _topology_party_digit(
        self,
        icon: Image.Image,
        trace: PartyDetectTrace | None = None,
    ) -> int | None:
        """숫자 ROI 위상(구멍 개수)으로 1/8 판별 — 1=구멍0, 8=구멍≥1"""
        if icon.height < 20 or icon.width < 30:
            return None

        gray = self._party_banner_gray(icon)
        digit_roi = self._party_digit_roi(gray)
        if digit_roi is None:
            return None

        holes, binary = self._count_digit_holes(digit_roi)
        if binary is not None:
            self._save_party_topology_debug(icon, digit_roi, binary)

        if holes is None:
            return None

        if trace is not None:
            trace.topology_holes = holes

        if is_debug():
            logger.debug("[party] topology holes=%d digit_roi=%dx%d", holes, *digit_roi.shape[::-1])

        left_bright = self._party_left_bright(icon)
        if trace is not None:
            trace.left_bright = left_bright
        if holes == 0:
            # 구멍 0만으로 1 확정하지 않음 — 잘린 ROI·노이즈에서 8이 1로 오판됨
            return None
        if holes >= 1 and self._confirm_topology_eight(left_bright, holes):
            return 8
        if holes >= 1 and is_debug():
            logger.debug(
                "[party] topology 8 rejected left_bright=%.3f holes=%d -> defer",
                left_bright,
                holes,
            )
        return None

    @staticmethod
    def _confirm_topology_eight(left_bright: float, holes: int) -> bool:
        """8 판정 — 8은 위 구멍(holes≥1)이 핵심, 사람 아이콘 밝기는 보조"""
        if holes >= 2:
            return True
        if holes == 1:
            return left_bright >= 0.04
        return False

    def _save_party_debug_images(
        self,
        icon: Image.Image,
        ocr_images: list[tuple[str, Image.Image]],
    ) -> None:
        """TR_DEBUG_PARTY=1 — tesseract 입력 crop/전처리 이미지 저장 (호출당 1회)"""
        if not is_debug_party() or self._party_debug_saved:
            return
        self._party_debug_saved = True

        out_dir = Path.cwd() / "debug_party"
        out_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)

        icon.save(out_dir / f"debug_party_{ts}_crop.png")
        for variant, img in ocr_images:
            img.save(out_dir / f"debug_party_{ts}_{variant}.png")

        logger.debug(
            "[party] debug 이미지 저장 %s (crop + %s)",
            out_dir,
            ", ".join(v for v, _ in ocr_images),
        )

    def _prepare_party_digit_ocr_image(
        self, icon: Image.Image
    ) -> Image.Image | None:
        """숫자 ROI만 확대 — full 아이콘 OCR보다 훨씬 빠름"""
        gray = self._party_banner_gray(icon)
        digit_roi = self._party_digit_roi(gray)
        if digit_roi is None or digit_roi.size == 0:
            return None
        digit_img = Image.fromarray(digit_roi).convert("L")
        digit_img = ImageOps.expand(digit_img, border=8, fill=0)
        scale = max(6, int(96 / max(digit_roi.shape[0], digit_roi.shape[1], 1)))
        return digit_img.resize(
            (digit_img.width * scale, digit_img.height * scale),
            Image.Resampling.LANCZOS,
        )

    def _ocr_party_digit_roi(
        self,
        icon: Image.Image,
        trace: PartyDetectTrace | None = None,
    ) -> int | None:
        """숫자 ROI만 OCR — 2~4회 시도로 1/8 판별"""
        if not self._tesseract_available:
            return None

        digit = self._prepare_party_digit_ocr_image(icon)
        if digit is None:
            return None

        collect_trace = trace is not None and is_debug()
        if collect_trace and trace.ocr_attempts is None:
            trace.ocr_attempts = []

        digit_arr = np.array(digit)
        attempts: list[tuple[str, Image.Image]] = [
            ("digit", digit),
            ("digit_inv", Image.fromarray(255 - digit_arr)),
        ]

        for variant, ocr_image in attempts:
            for psm in (10, 7):
                try:
                    text = self._ocr.read_text(
                        ocr_image,
                        lang="eng",
                        psm=psm,
                        whitelist="18",
                    )
                    parsed = self._parse_party_digit(text)
                    if collect_trace:
                        trace.ocr_attempts.append(
                            f"{variant}/psm{psm} raw={text!r} parsed={parsed}"
                        )
                    if parsed is not None:
                        if trace is not None:
                            trace.ocr_raw = text.strip()
                            trace.ocr_digit = parsed
                        return parsed
                except Exception as exc:
                    if collect_trace:
                        trace.ocr_attempts.append(
                            f"{variant}/psm{psm} error={exc!s}"
                        )
        return None

    def _ocr_party_icon_full(
        self,
        icon: Image.Image,
        trace: PartyDetectTrace | None = None,
    ) -> int | None:
        """full 아이콘 OCR — digit ROI 실패 시 fallback"""
        if not self._tesseract_available:
            return None

        collect_trace = trace is not None and is_debug()
        scale = max(5, int(120 / max(icon.width, icon.height, 1)))
        scaled = icon.resize(
            (max(60, icon.width * scale), max(60, icon.height * scale)),
            Image.Resampling.LANCZOS,
        )
        gray_arr = np.array(scaled.convert("L"))
        ocr_images: list[tuple[str, Image.Image]] = [
            ("full", scaled),
            ("full_inv", Image.fromarray(255 - gray_arr)),
        ]
        enhanced = np.array(
            ImageEnhance.Contrast(scaled.convert("L")).enhance(2.5)
        )
        ocr_images.append(("full_contrast", Image.fromarray(255 - enhanced)))
        self._save_party_debug_images(icon, ocr_images)

        for variant, ocr_image in ocr_images:
            for psm in (10, 7):
                try:
                    text = self._ocr.read_text(
                        ocr_image,
                        lang="eng",
                        psm=psm,
                        whitelist="18",
                    )
                    parsed = self._parse_party_digit(text)
                    if collect_trace and trace.ocr_attempts is not None:
                        trace.ocr_attempts.append(
                            f"{variant}/psm{psm} raw={text!r} parsed={parsed}"
                        )
                    if parsed is not None:
                        if trace is not None:
                            trace.ocr_raw = text.strip()
                            trace.ocr_digit = parsed
                        return parsed
                except Exception as exc:
                    if collect_trace and trace.ocr_attempts is not None:
                        trace.ocr_attempts.append(
                            f"{variant}/psm{psm} error={exc!s}"
                        )
        return None

    def _ocr_party_icon(
        self,
        icon: Image.Image,
        trace: PartyDetectTrace | None = None,
    ) -> int | None:
        """하위 호환 — digit ROI 우선, 실패 시 full 아이콘"""
        self._party_debug_saved = False
        result = self._ocr_party_digit_roi(icon, trace)
        if result in (1, 8):
            return result
        return self._ocr_party_icon_full(icon, trace)

    @staticmethod
    def _party_left_bright(icon: Image.Image) -> float:
        gray = cv2.cvtColor(np.array(icon.convert("RGB")), cv2.COLOR_RGB2GRAY)
        width = gray.shape[1]
        left = gray[:, : max(1, int(width * 0.38))]
        return float((left > 140).mean()) if left.size else 0.0

    @staticmethod
    def _party_icon_metrics(icon: Image.Image) -> tuple[float, float]:
        gray = cv2.cvtColor(np.array(icon.convert("RGB")), cv2.COLOR_RGB2GRAY)
        width = gray.shape[1]
        left_bright = TreasureCaptureProcessor._party_left_bright(icon)
        num_area = gray[:, int(width * 0.38) :]
        right_ratio = float((num_area > 185).mean()) if num_area.size else 0.0
        return left_bright, right_ratio

    def _heuristic_party_digit(
        self,
        icon: Image.Image,
        trace: PartyDetectTrace | None = None,
    ) -> int | None:
        """OCR 실패·오인식 보정 — 8인은 왼쪽 그룹 아이콘이 넓게 밝게 잡힘"""
        if icon.height < 28 or icon.width < 40:
            return None

        left_bright, right_ratio = self._party_icon_metrics(icon)
        if trace is not None:
            trace.left_bright = left_bright
            trace.right_ratio = right_ratio

        if left_bright >= 0.12:
            return 8
        return None

    def _resolve_party_digit(
        self,
        icon: Image.Image,
        trace: PartyDetectTrace | None = None,
    ) -> int | None:
        """OCR → 위상(구멍) → full OCR → 휴리스틱 순으로 1/8 판별"""
        left_bright = self._party_left_bright(icon)
        if trace is not None:
            trace.left_bright = left_bright
            trace.icon_size = icon.size

        ocr = self._ocr_party_digit_roi(icon, trace)
        if ocr in (1, 8):
            if trace is not None:
                trace.result = ocr
                trace.reason = f"ocr{ocr}"
            return ocr

        topology = self._topology_party_digit(icon, trace)
        if topology == 8:
            if trace is not None:
                trace.result = 8
                trace.reason = "topology8"
            return 8
        if topology == 1:
            if trace is not None:
                trace.result = 1
                trace.reason = "topology1"
            return 1

        ocr_full = self._ocr_party_icon_full(icon, trace)
        if ocr_full in (1, 8):
            if trace is not None:
                trace.result = ocr_full
                trace.reason = f"ocr_full{ocr_full}"
            return ocr_full

        heuristic = self._heuristic_party_digit(icon, trace)
        if trace is not None:
            trace.heuristic = heuristic

        if heuristic in (1, 8):
            if trace is not None:
                trace.result = heuristic
                trace.reason = "heuristic_only"
            return heuristic

        # 마지막 — digit ROI 밝기만으로 1 추정 (8은 OCR/구멍에서 잡혀야 함)
        _left, right_ratio = self._party_icon_metrics(icon)
        if trace is not None:
            trace.right_ratio = right_ratio
        if right_ratio > 0.04 and right_ratio < 0.22:
            if trace is not None:
                trace.result = 1
                trace.reason = "digit_bright_solo"
            return 1

        if trace is not None:
            trace.result = None
            trace.reason = "no_signal"
        return None

    def _log_party_trace(self, trace: PartyDetectTrace) -> None:
        if is_debug():
            logger.debug("[party] %s", trace.summary())

    def _party_icon_crop(
        self, image: Image.Image
    ) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
        width, height = image.size
        if width < 40 or height < 40:
            return None

        rgb = np.array(image.convert("RGB"))
        parchment = self._parchment_bounds(rgb)
        if parchment is not None:
            px1, _py1, px2, py2 = parchment
            parch_w = max(1, px2 - px1)
            parch_h = max(1, py2 - parchment[1])
            icon_h = max(24, int(parch_h * 0.32))
            # 양피지 좌하단 전체 — 사람 아이콘 + 숫자(8)까지 포함
            x1 = max(0, px1 - 2)
            y1 = max(0, py2 - icon_h)
            x2 = min(width, px2 + 4)
            y2 = min(height, py2 + 4)
            if x2 - x1 >= 30 and y2 - y1 >= 18:
                box = (x1, y1, x2, y2)
                return image.crop(box), box

        icon_top = int(height * 0.78)
        right = int(width * 0.38)
        box = (0, icon_top, right, height)
        return image.crop(box), box

    def detect_party_size(
        self,
        image: Image.Image,
        *,
        debug_source: str = "capture",
    ) -> int | None:
        """하단 좌측 1/8 아이콘 숫자 (1=솔로, 8=8인)"""
        self._party_debug_saved = False
        cropped = self._party_icon_crop(image)
        if cropped is None:
            trace = PartyDetectTrace(
                source=debug_source,
                image_size=image.size,
                result=None,
                reason="image_too_small",
            )
            self._last_party_trace = trace
            self._log_party_trace(trace)
            return None

        icon, box = cropped
        trace = PartyDetectTrace(
            source=debug_source,
            image_size=image.size,
            crop_box=box,
        )
        result = self._resolve_party_digit(icon, trace)
        self._last_party_trace = trace
        self._log_party_trace(trace)
        return result

    def detect_party_size_aggressive(
        self,
        image: Image.Image,
        *,
        debug_source: str = "aggressive",
    ) -> int | None:
        """OCR 전처리·크롭 변형을 더 시도하는 1/8 인원 인식"""
        width, height = image.size
        if width < 40 or height < 40:
            return None

        icon_tops = (0.80, 0.78, 0.76)
        width_ratios = (0.28, 0.30)
        seen: set[tuple[int, int, int, int]] = set()
        results: list[tuple[int, PartyDetectTrace]] = []
        for top_ratio in icon_tops:
            for width_ratio in width_ratios:
                box = (
                    0,
                    int(height * top_ratio),
                    int(width * width_ratio),
                    height,
                )
                if box in seen:
                    continue
                seen.add(box)
                icon = image.crop(box)
                if icon.width < 20 or icon.height < 16:
                    continue
                trace = PartyDetectTrace(
                    source=f"{debug_source}@{top_ratio:.2f}x{width_ratio:.2f}",
                    image_size=image.size,
                    crop_box=box,
                )
                parsed = self._resolve_party_digit(icon, trace)
                if parsed is None:
                    if is_debug():
                        logger.debug("[party] %s (skip)", trace.summary())
                    continue
                results.append((parsed, trace))

        if results:
            eights = [item for item in results if item[0] == 8]
            ones = [item for item in results if item[0] == 1]
            if eights:
                parsed, trace = eights[0]
            elif ones:
                parsed, trace = ones[0]
            else:
                parsed, trace = results[0]
            self._last_party_trace = trace
            self._log_party_trace(trace)
            return parsed

        return self.detect_party_size(image, debug_source=f"{debug_source}/fallback")

    def _ocr_banner_candidates(self, banner: Image.Image):
        """배너 OCR — 고성공률 전처리·PSM부터 순차 시도 (조기 yield)"""
        if not self._ocr.available:
            return

        if is_debug():
            debug_dir = Path.cwd() / "debug_banner"
            debug_dir.mkdir(exist_ok=True)
            banner.save(debug_dir / f"crop_{time.time_ns()}.png")

        scale = max(4, int(520 / max(banner.width, 1)), int(360 / max(banner.height, 1)))
        scale = min(scale, 12)
        scaled = banner.resize(
            (banner.width * scale, banner.height * scale),
            Image.Resampling.LANCZOS,
        )
        rgb = np.array(scaled.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        strict_white = Image.fromarray(
            np.where(
                (rgb[:, :, 0] > 200)
                & (rgb[:, :, 1] > 190)
                & (rgb[:, :, 2] > 150),
                0,
                255,
            ).astype(np.uint8)
        )
        white = Image.fromarray(
            np.where(
                (rgb[:, :, 0] > 150)
                & (rgb[:, :, 1] > 150)
                & (rgb[:, :, 2] > 150),
                0,
                255,
            ).astype(np.uint8)
        )
        bright = Image.fromarray(
            np.where(
                (rgb[:, :, 0] > 130)
                & (rgb[:, :, 1] > 130)
                & (rgb[:, :, 2] > 130),
                0,
                255,
            ).astype(np.uint8)
        )
        enhanced = np.array(ImageEnhance.Contrast(scaled.convert("L")).enhance(2.5))
        contrast_inv = Image.fromarray(255 - enhanced)
        invert = ImageOps.invert(scaled.convert("L"))

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bg = cv2.GaussianBlur(gray, (0, 0), 5)
        diff = cv2.subtract(bg, gray)
        _, ink = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink_img = Image.fromarray(np.where(ink > 0, 0, 255).astype(np.uint8))

        deficit = (
            (rgb[:, :, 0].astype(np.int16) + rgb[:, :, 1].astype(np.int16)) // 2
            - rgb[:, :, 2].astype(np.int16)
        )
        deficit = np.clip(deficit, 0, 255).astype(np.uint8)
        _, dbin = cv2.threshold(deficit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        deficit_img = Image.fromarray(np.where(dbin > 0, 0, 255).astype(np.uint8))

        gray_scaled = scaled.convert("L")

        _chars, zone_whitelist, _entries = self._zone_name_lexicon()

        # 흰 글씨(어두운 배너) 우선 → zones.json 글자 whitelist로 노이즈 억제
        stages: list[tuple[Image.Image, tuple[int, ...]]] = [
            (strict_white, (7, 8)),
            (white, (7, 8)),
            (gray_scaled, (7, 8)),
            (contrast_inv, (7,)),
            (Image.fromarray(binary), (7, 13)),
            (ink_img, (7,)),
            (deficit_img, (7,)),
            (bright, (7,)),
            (invert, (7, 13)),
        ]

        seen: set[str] = set()
        for img, psm_modes in stages:
            padded = ImageOps.expand(img, border=16, fill=0)
            for psm in psm_modes:
                text = self._ocr.read_text(
                    padded,
                    lang="kor",
                    psm=psm,
                    whitelist=zone_whitelist or None,
                )
                if not text or text in seen:
                    continue
                seen.add(text)
                yield text

    @staticmethod
    def mask_ingame_ui(gray: np.ndarray) -> np.ndarray:
        """매칭용: 인게임 UI 영역 중립색 처리 (전체 캡처 기준)"""
        masked = gray.copy()
        height, width = masked.shape
        fill = int(np.median(masked))

        masked[0 : int(height * 0.16), :] = fill
        masked[0 : int(height * 0.22), int(width * 0.78) :] = fill
        masked[int(height * 0.84) :, :] = fill

        inset = max(4, int(min(width, height) * 0.04))
        masked[:inset, :] = fill
        masked[-inset:, :] = fill
        masked[:, :inset] = fill
        masked[:, -inset:] = fill

        return masked

    TERRAIN_BORDER_RATIO = 0.12

    @staticmethod
    def mask_fragment_for_matching(
        gray: np.ndarray,
        border_ratio: float | None = None,
    ) -> np.ndarray:
        """매칭용: 테두리·하단 UI를 중립색으로 — 내부 지형만 비교"""
        masked = gray.copy()
        height, width = masked.shape
        fill = int(np.median(masked))

        masked[int(height * 0.88) :, :] = fill

        ratio = (
            border_ratio
            if border_ratio is not None
            else TreasureCaptureProcessor.TERRAIN_BORDER_RATIO
        )
        inset_x = max(4, int(width * ratio))
        inset_y = max(4, int(height * ratio))
        masked[:inset_y, :] = fill
        masked[-inset_y:, :] = fill
        masked[:, :inset_x] = fill
        masked[:, -inset_x:] = fill

        return masked

    @staticmethod
    def neutralize_parchment(
        gray: np.ndarray,
        rgb: np.ndarray | None = None,
    ) -> np.ndarray:
        """양피지 배경은 중립화, 갈색 지형 선만 남김"""
        if rgb is not None:
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)
            terrain = (
                (r > 60)
                & (r < 200)
                & (g < 150)
                & (b < 130)
                & (r > g + 10)
            )
            fill = int(np.median(gray[terrain])) if terrain.any() else 128
            out = np.full_like(gray, fill)
            out[terrain] = np.clip(
                gray[terrain].astype(np.float32) * 0.65 + 35, 0, 255
            ).astype(np.uint8)
            return out

        fill = int(np.median(gray))
        out = gray.copy()
        out[gray > 178] = fill
        return out

    @staticmethod
    def crop_terrain_core(
        gray: np.ndarray,
        border_ratio: float | None = None,
    ) -> tuple[np.ndarray, int, int]:
        """테두리 제거 후 내부 지형 영역만 반환 (offset_x, offset_y 포함)"""
        ratio = (
            border_ratio
            if border_ratio is not None
            else TreasureCaptureProcessor.TERRAIN_BORDER_RATIO
        )
        height, width = gray.shape
        inset_x = max(4, int(width * ratio))
        inset_y = max(4, int(height * ratio))

        if width <= inset_x * 2 + 8 or height <= inset_y * 2 + 8:
            return gray, 0, 0

        core = gray[inset_y : height - inset_y, inset_x : width - inset_x]
        return core, inset_x, inset_y

    # 매칭용 — 양피지 테두리·X·UI 제외, 등고선 지형 코어만
    MATCH_TERRAIN_INSET = (0.10, 0.16, 0.90, 0.84)
    MATCH_CORE_BORDER = 0.10
    MATCH_CENTER_Y_RATIO = 0.44
    MATCH_EDGE_WEIGHT = 0.58
    MATCH_RADIAL_SIGMA = 0.58
    BANNER_OCR_EARLY_EXIT = 0.80

    @classmethod
    def prepare_matching_patch(
        cls,
        image: Image.Image,
    ) -> tuple[Image.Image, np.ndarray, tuple[int, int]]:
        """
        매칭용 지형 코어 — 양피지 외곽·빨간 X·배너·하단 UI 제외.

        Returns:
            (RGB PIL, gray for SSIM, soft-weight 중심 (cx, cy))
        """
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        left = int(w * cls.MATCH_TERRAIN_INSET[0])
        top = int(h * cls.MATCH_TERRAIN_INSET[1])
        right = int(w * cls.MATCH_TERRAIN_INSET[2])
        bottom = int(h * cls.MATCH_TERRAIN_INSET[3])
        patch_rgb = rgb[top:bottom, left:right]
        if patch_rgb.size == 0:
            patch_rgb = rgb.copy()

        patch_rgb, gray = cls.neutralize_red_marker(
            patch_rgb,
            cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY),
        )
        assert gray is not None
        gray = cls.neutralize_parchment(gray, patch_rgb)
        gray = cls.mask_fragment_for_matching(gray, border_ratio=0.12)
        gray = cls._mask_coord_strip(gray)
        gray, off_x, off_y = cls.crop_terrain_core(
            gray,
            border_ratio=cls.MATCH_CORE_BORDER,
        )
        patch_rgb = patch_rgb[
            off_y : off_y + gray.shape[0],
            off_x : off_x + gray.shape[1],
        ]

        th, tw = gray.shape[:2]
        weight_center = (tw // 2, max(0, int(th * cls.MATCH_CENTER_Y_RATIO)))
        return Image.fromarray(patch_rgb), gray, weight_center

    @staticmethod
    def normalize_for_matching(
        image: Image.Image,
        target_w: int,
        target_h: int,
    ) -> Image.Image:
        """쿼리를 ref 크기에 맞춤 — 비율 유지 후 letterbox 패딩 (찌그러짐 방지)"""
        if target_w < 1 or target_h < 1:
            return image
        w, h = image.size
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if new_w == w and new_h == h and new_w == target_w and new_h == target_h:
            return image
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if new_w == target_w and new_h == target_h:
            return resized

        rgb = np.array(resized.convert("RGB"))
        fill = tuple(int(v) for v in np.median(rgb.reshape(-1, 3), axis=0))
        canvas = Image.new("RGB", (target_w, target_h), fill)
        ox = (target_w - new_w) // 2
        oy = (target_h - new_h) // 2
        canvas.paste(resized, (ox, oy))
        return canvas

    @staticmethod
    def upscale_for_matching(
        image: Image.Image,
        target_w: int,
        target_h: int,
    ) -> Image.Image:
        """하위 호환 — 작을 때만 확대 (normalize_for_matching 권장)"""
        if target_w < 1 or target_h < 1:
            return image
        w, h = image.size
        if w >= target_w and h >= target_h:
            return image
        scale = max(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @classmethod
    def _mask_coord_strip(cls, gray: np.ndarray) -> np.ndarray:
        """상단 좌표 OCR 영역 중립화 — X: 24.0 - Y: 10.0 텍스트 노이즈 제거"""
        masked = gray.copy()
        fill = int(np.median(masked))
        top = max(1, int(masked.shape[0] * 0.22))
        masked[:top, :] = fill
        return masked

    @classmethod
    def extract_terrain_for_detail_match(
        cls,
        image: Image.Image,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[float, float] | None]:
        """
        detail 지도 매칭용 지형 gray + map_window 내 bbox + 빨간 X 위치.

        Returns:
            (terrain_gray, (left, top, right, bottom), marker_xy or None)
        """
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        marker = cls._find_treasure_x_marker(rgb)
        left = int(w * cls.MATCH_TERRAIN_INSET[0])
        top = int(h * cls.MATCH_TERRAIN_INSET[1])
        right = int(w * cls.MATCH_TERRAIN_INSET[2])
        bottom = int(h * cls.MATCH_TERRAIN_INSET[3])
        terrain_rgb = rgb[top:bottom, left:right]
        if terrain_rgb.size == 0:
            terrain_rgb = rgb.copy()
            left, top, right, bottom = 0, 0, w, h

        terrain_rgb, gray = cls.neutralize_red_marker(
            terrain_rgb,
            cv2.cvtColor(terrain_rgb, cv2.COLOR_RGB2GRAY),
        )
        assert gray is not None
        gray = cls.neutralize_parchment(gray, terrain_rgb)
        gray = cls.mask_fragment_for_matching(gray, border_ratio=0.12)
        gray = cls._mask_coord_strip(gray)
        gray, off_x, off_y = cls.crop_terrain_core(
            gray,
            border_ratio=cls.MATCH_CORE_BORDER,
        )
        if marker is not None:
            marker = (marker[0] - left - off_x, marker[1] - top - off_y)
        return gray, (left, top, right, bottom), marker

    @staticmethod
    def neutralize_red_marker(
        rgb: np.ndarray,
        gray: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """빨간 X는 모든 지도에 있으므로 매칭 특징에서 제거"""
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        red = (
            (r > 150)
            & (g < 110)
            & (b < 110)
            & (r.astype(np.int16) > g.astype(np.int16) + 35)
        )
        out_rgb = rgb.copy()
        out_gray = gray.copy() if gray is not None else None

        if not red.any():
            return out_rgb, out_gray

        if (~red).any():
            fill_rgb = np.median(out_rgb[~red], axis=0).astype(np.uint8)
            fill_gray = int(np.median(out_gray[~red])) if out_gray is not None else 128
        else:
            fill_rgb = np.array([128, 128, 128], dtype=np.uint8)
            fill_gray = 128

        out_rgb[red] = fill_rgb
        if out_gray is not None:
            out_gray[red] = fill_gray
        return out_rgb, out_gray

    @classmethod
    def build_soft_center_weights(
        cls,
        height: int,
        width: int,
        center: tuple[int, int],
    ) -> np.ndarray:
        """가장자리 edge_weight ~ 중심 1.0 (가장자리 완전 제외 없음)"""
        cx, cy = center
        cx = int(np.clip(cx, 0, max(0, width - 1)))
        cy = int(np.clip(cy, 0, max(0, height - 1)))
        yy, xx = np.ogrid[:height, :width]
        dist2 = ((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        sigma = max(8.0, min(height, width) * cls.MATCH_RADIAL_SIGMA)
        radial = np.exp(-dist2 / (2.0 * sigma * sigma))
        edge = cls.MATCH_EDGE_WEIGHT
        return (edge + (1.0 - edge) * radial).astype(np.float32)
