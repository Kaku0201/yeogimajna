"""상세지도 ref 크롭·X 마커·PNG 저장 UI."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.services.app_paths import get_app_root
from src.services.coordinate_service import CoordinateService

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CROP_W = 218
DEFAULT_CROP_H = 188
DEFAULT_ZOOM = 2.0
PAN_STEP = 8.0
# 인게임 보물지도 창 실측 (300% UI, 501×430)
INGAME_FRAME_W = 501
INGAME_FRAME_H = 430
INGAME_FRAME_ASPECT = INGAME_FRAME_W / INGAME_FRAME_H
# 기존 ref DB 호환 비율 (218×188)
REF_CROP_ASPECT = DEFAULT_CROP_W / DEFAULT_CROP_H
# 상세지도 다이얼로그(MapDetailDialog.TREASURE_MARKER_VISIBLE)와 동일
TREASURE_MARKER_VISIBLE = 36


def treasure_marker_path() -> Path:
    return get_app_root() / "assets" / "treasure_marker.png"


def crop_height_for_width(width: int, aspect: float = REF_CROP_ASPECT) -> int:
    return max(32, int(round(width / aspect)))


def crop_width_for_height(height: int, aspect: float = REF_CROP_ASPECT) -> int:
    return max(32, int(round(height * aspect)))


def crop_size_for_ingame_zoom(zoom: float) -> tuple[int, int]:
    """인게임 창 비율 유지 — 줌 2.0이면 약 250×215, 2.30이면 약 218×187."""
    z = max(0.25, zoom)
    w = max(32, int(round(INGAME_FRAME_W / z)))
    h = max(32, int(round(w / INGAME_FRAME_ASPECT)))
    return w, h


def round_game_coord(value: float) -> float:
    return round(value, 2)


def round_game_coords(x: float, y: float) -> tuple[float, float]:
    return round_game_coord(x), round_game_coord(y)


def coord_filename(x: float, y: float) -> str:
    rx, ry = round_game_coords(x, y)
    return f"{rx:.2f}_{ry:.2f}.png"


def parse_game_coord_text(text: str) -> tuple[float, float] | None:
    """'22.61, 15.62' / '22.61 15.62' 형식 파싱 (소수점 둘째 자리)."""
    cleaned = text.strip().replace(",", " ")
    parts = [p for p in cleaned.split() if p]
    if len(parts) < 2:
        return None
    try:
        return round_game_coords(float(parts[0]), float(parts[1]))
    except ValueError:
        return None


def format_game_coords(x: float, y: float) -> str:
    rx, ry = round_game_coords(x, y)
    return f"{rx:.2f}, {ry:.2f}"


def treasure_spots_from_zone(zone: dict) -> list[tuple[float, float]]:
    """zones.json spots[] 보물 좌표 (중복 제거, 정렬)."""
    seen: set[tuple[float, float]] = set()
    spots: list[tuple[float, float]] = []
    for spot in zone.get("spots") or []:
        treasure = spot.get("treasure") or {}
        x, y = treasure.get("x"), treasure.get("y")
        if x is None or y is None:
            continue
        pair = round_game_coords(float(x), float(y))
        if pair in seen:
            continue
        seen.add(pair)
        spots.append(pair)
    spots.sort(key=lambda p: (p[0], p[1]))
    return spots


def pil_to_qimage(img: Image.Image) -> QImage:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
    return qimg.copy()


# 인게임 보물지도 창에서의 X 크기 (상세지도 UI 520px 기준 36px → 501px 창)
INGAME_MARKER_PX = TREASURE_MARKER_VISIBLE * INGAME_FRAME_W / 520
MARKER_SIZE_MIN = 4
MARKER_SIZE_MAX = 128


def marker_size_for_crop(crop_w: int, crop_h: int | None = None) -> int:
    """ref 크롭 해상도에 맞는 보물 마커 픽셀 크기 (인게임 창 대비)."""
    h = crop_h if crop_h is not None else crop_w
    size = int(round(INGAME_MARKER_PX * crop_w / INGAME_FRAME_W))
    cap = min(crop_w, h) // 2
    return max(6, min(size, cap) if cap > 0 else size)


def draw_treasure_marker_on_image(
    img: Image.Image,
    marker: Image.Image,
    local_x: float,
    local_y: float,
    size: int,
) -> Image.Image:
    """크롭 이미지 기준 로컬 좌표에 treasure_marker.png 합성."""
    out = img.convert("RGBA")
    scaled = marker.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    half = size // 2
    px = int(round(local_x)) - half
    py = int(round(local_y)) - half
    out.alpha_composite(scaled, (px, py))
    return out


class MapCanvas(QWidget):
    """확대·이동·크롭 프레임·X 마커 표시."""

    coords_changed = pyqtSignal(float, float)
    center_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 520)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._source: QImage | None = None
        self._source_size = (0, 0)
        self._zone: dict | None = None
        self._coord_service: CoordinateService | None = None

        self.center_x = 512.0
        self.center_y = 512.0
        self.zoom = DEFAULT_ZOOM
        self.crop_w = DEFAULT_CROP_W
        self.crop_h = DEFAULT_CROP_H
        self.marker_size_px = marker_size_for_crop(DEFAULT_CROP_W, DEFAULT_CROP_H)
        # 인게임 보물지도 대비 시각 보정 (원본 지도 픽셀, Y 음수=위로)
        self.marker_offset_x = 0.0
        self.marker_offset_y = -10.0

        self._marker_local: tuple[float, float] | None = None
        self._explicit_game_coords: tuple[float, float] | None = None
        self._treasure_marker = self._load_treasure_marker()
        self._treasure_marker_pil = self._load_treasure_marker_pil()
        self._dragging = False
        self._drag_start: QPointF | None = None
        self._drag_center: tuple[float, float] | None = None

    @staticmethod
    def _load_treasure_marker() -> QPixmap:
        path = treasure_marker_path()
        if not path.exists():
            return QPixmap()
        marker = QPixmap(str(path))
        return marker if not marker.isNull() else QPixmap()

    @staticmethod
    def _load_treasure_marker_pil() -> Image.Image | None:
        path = treasure_marker_path()
        if not path.exists():
            return None
        try:
            return Image.open(path).convert("RGBA")
        except OSError:
            return None

    def _marker_crop_size(self) -> int:
        return self.marker_size_px

    def set_marker_size(self, px: int) -> None:
        self.marker_size_px = max(MARKER_SIZE_MIN, min(MARKER_SIZE_MAX, px))
        self.update()

    def set_marker_offset(self, ox: float, oy: float) -> None:
        self.marker_offset_x = ox
        self.marker_offset_y = oy
        self._reapply_marker_from_explicit_coords()
        self.update()

    def _reapply_marker_from_explicit_coords(self) -> None:
        """좌표 입력 배치 후 오프셋만 바꿀 때 마커 위치 재계산."""
        if (
            self._explicit_game_coords is None
            or self._zone is None
            or self._coord_service is None
        ):
            return
        gx, gy = self._explicit_game_coords
        px, py = self._coord_service.game_to_detail_pixel_float(self._zone, gx, gy)
        lx = px - (self.center_x - self.crop_w / 2.0) + self.marker_offset_x
        ly = py - (self.center_y - self.crop_h / 2.0) + self.marker_offset_y
        self._marker_local = (lx, ly)

    def set_crop_size(self, w: int, h: int) -> None:
        self.crop_w = max(16, w)
        self.crop_h = max(16, h)
        self.update()

    def set_map(
        self,
        qimage: QImage,
        zone: dict,
        coord_service: CoordinateService,
    ) -> None:
        self._source = qimage
        self._source_size = (qimage.width(), qimage.height())
        self._zone = zone
        self._coord_service = coord_service
        self.center_x = qimage.width() / 2.0
        self.center_y = qimage.height() / 2.0
        self._marker_local = None
        self._explicit_game_coords = None
        self.center_changed.emit()
        self.update()

    def clear_map(self) -> None:
        self._source = None
        self._marker_local = None
        self._explicit_game_coords = None
        self.update()

    def set_zoom(self, z: float) -> None:
        self.zoom = max(0.25, min(8.0, z))
        self.update()

    def go_to_game_coords(self, gx: float, gy: float) -> bool:
        if self._zone is None or self._coord_service is None:
            return False
        px, py = self._coord_service.game_to_detail_pixel(self._zone, gx, gy)
        self.center_x = float(px)
        self.center_y = float(py)
        self._clamp_center()
        self._marker_local = None
        self.center_changed.emit()
        self.update()
        return True

    def place_marker_at_game_coords(self, gx: float, gy: float) -> bool:
        """게임 좌표로 이동하고 보물 마커를 해당 위치에 배치."""
        if self._zone is None or self._coord_service is None:
            return False
        game_coords = round_game_coords(gx, gy)
        px, py = self._coord_service.game_to_detail_pixel_float(
            self._zone, *game_coords
        )
        self.center_x = px
        self.center_y = py
        self._clamp_center()
        lx = px - (self.center_x - self.crop_w / 2.0) + self.marker_offset_x
        ly = py - (self.center_y - self.crop_h / 2.0) + self.marker_offset_y
        self._marker_local = (lx, ly)
        self._explicit_game_coords = game_coords
        self.center_changed.emit()
        self.coords_changed.emit(*game_coords)
        self.update()
        return True

    def marker_game_coords(self) -> tuple[float, float] | None:
        if self._marker_local is None:
            return None
        if self._explicit_game_coords is not None:
            return self._explicit_game_coords
        if self._zone is None or self._coord_service is None:
            return None
        sx, sy = self._source_from_crop_local(*self._marker_local)
        return round_game_coords(
            *self._coord_service.pixel_to_game(
                self._zone,
                sx - self.marker_offset_x,
                sy - self.marker_offset_y,
                self._source_size[0],
                self._source_size[1],
            )
        )

    def _source_from_crop_local(self, lx: float, ly: float) -> tuple[float, float]:
        sx = self.center_x - self.crop_w / 2.0 + lx
        sy = self.center_y - self.crop_h / 2.0 + ly
        return sx, sy

    def _screen_to_source(self, wx: float, wy: float) -> tuple[float, float]:
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        sx = self.center_x + (wx - cx) / self.zoom
        sy = self.center_y + (wy - cy) / self.zoom
        return sx, sy

    def _crop_rect_screen(self) -> tuple[float, float, float, float]:
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        half_w = self.crop_w * self.zoom / 2.0
        half_h = self.crop_h * self.zoom / 2.0
        return cx - half_w, cy - half_h, self.crop_w * self.zoom, self.crop_h * self.zoom

    def _inside_crop_screen(self, wx: float, wy: float) -> bool:
        x, y, w, h = self._crop_rect_screen()
        return x <= wx <= x + w and y <= wy <= y + h

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 32))

        if self._source is None:
            painter.setPen(QColor(160, 160, 170))
            painter.setFont(QFont("Malgun Gothic", 12))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "지역을 선택하거나 PNG를 열어주세요",
            )
            return

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        top_left_x = cx - self.center_x * self.zoom
        top_left_y = cy - self.center_y * self.zoom
        src_w = self._source.width()
        src_h = self._source.height()
        painter.drawImage(
            QRectF(top_left_x, top_left_y, src_w * self.zoom, src_h * self.zoom),
            self._source,
            QRectF(0, 0, src_w, src_h),
        )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rx, ry, rw, rh = self._crop_rect_screen()
        painter.setPen(QPen(QColor(255, 210, 60), 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QColor(255, 210, 60, 25))
        painter.drawRect(int(rx), int(ry), int(rw), int(rh))

        if self._marker_local is not None:
            mx = rx + self._marker_local[0] * self.zoom
            my = ry + self._marker_local[1] * self.zoom
            size = int(self._marker_crop_size() * self.zoom)
            half = size // 2
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            if not self._treasure_marker.isNull():
                scaled = self._treasure_marker.scaled(
                    size,
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(int(mx - half), int(my - half), scaled)
            else:
                pen = QPen(QColor(240, 50, 50), max(2, size // 12))
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(mx - half, my - half),
                    QPointF(mx + half, my + half),
                )
                painter.drawLine(
                    QPointF(mx + half, my - half),
                    QPointF(mx - half, my + half),
                )

        painter.setPen(QColor(200, 200, 210))
        painter.setFont(QFont("Consolas", 9))
        info = (
            f"줌 {self.zoom:.2f}x  |  크롭 {self.crop_w}×{self.crop_h}  |  "
            f"중심 ({self.center_x:.0f}, {self.center_y:.0f})"
        )
        painter.drawText(8, self.height() - 8, info)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1 / 1.12
        self.set_zoom(self.zoom * factor)
        self.center_changed.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._source is None:
            return
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if self._inside_crop_screen(pos.x(), pos.y()):
                rx, ry, _, _ = self._crop_rect_screen()
                lx = (pos.x() - rx) / self.zoom
                ly = (pos.y() - ry) / self.zoom
                self._marker_local = (lx, ly)
                self._explicit_game_coords = None
                coords = self.marker_game_coords()
                if coords is not None:
                    self.coords_changed.emit(coords[0], coords[1])
                self.update()
            else:
                self._dragging = True
                self._drag_start = pos
                self._drag_center = (self.center_x, self.center_y)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = True
            self._drag_start = pos
            self._drag_center = (self.center_x, self.center_y)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging or self._drag_start is None or self._drag_center is None:
            return
        delta = event.position() - self._drag_start
        self.center_x = self._drag_center[0] - delta.x() / self.zoom
        self.center_y = self._drag_center[1] - delta.y() / self.zoom
        self._clamp_center()
        if self._marker_local is not None:
            self._explicit_game_coords = None
        self.center_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._dragging = False
            self._drag_start = None
            self._drag_center = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = PAN_STEP
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step *= 4
        key = event.key()
        moved = False
        if key == Qt.Key.Key_Left:
            self.center_x -= step
            moved = True
        elif key == Qt.Key.Key_Right:
            self.center_x += step
            moved = True
        elif key == Qt.Key.Key_Up:
            self.center_y -= step
            moved = True
        elif key == Qt.Key.Key_Down:
            self.center_y += step
            moved = True
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.set_zoom(self.zoom * 1.1)
            moved = True
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.set_zoom(self.zoom / 1.1)
            moved = True
        if moved:
            self._clamp_center()
            if self._marker_local is not None:
                self._explicit_game_coords = None
            self.center_changed.emit()
            self.update()
        else:
            super().keyPressEvent(event)

    def _clamp_center(self) -> None:
        if self._source is None:
            return
        w, h = self._source_size
        half_w = self.crop_w / 2.0
        half_h = self.crop_h / 2.0
        self.center_x = max(half_w, min(w - half_w, self.center_x))
        self.center_y = max(half_h, min(h - half_h, self.center_y))

    def export_crop(self, draw_marker: bool) -> Image.Image | None:
        if self._source is None:
            return None
        pil = Image.open(self._get_current_map_path()).convert("RGBA")
        zw = int(round(pil.width * self.zoom))
        zh = int(round(pil.height * self.zoom))
        scaled = pil.resize((zw, zh), Image.Resampling.LANCZOS)
        sw = int(round(self.crop_w * self.zoom))
        sh = int(round(self.crop_h * self.zoom))
        left = int(round((self.center_x - self.crop_w / 2.0) * self.zoom))
        top = int(round((self.center_y - self.crop_h / 2.0) * self.zoom))
        left = max(0, min(max(0, zw - sw), left))
        top = max(0, min(max(0, zh - sh), top))
        region = scaled.crop((left, top, left + sw, top + sh))
        crop = region.resize(
            (self.crop_w, self.crop_h),
            Image.Resampling.LANCZOS,
        )
        if draw_marker and self._marker_local is not None:
            if self._treasure_marker_pil is not None:
                crop = draw_treasure_marker_on_image(
                    crop,
                    self._treasure_marker_pil,
                    *self._marker_local,
                    self._marker_crop_size(),
                )
        return crop

    _current_map_path: Path | None = None

    def set_current_map_path(self, path: Path) -> None:
        self._current_map_path = path

    def _get_current_map_path(self) -> Path:
        if self._current_map_path is None:
            raise RuntimeError("지도 경로가 설정되지 않았습니다.")
        return self._current_map_path


class RefMakerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ref 이미지 제작기")
        self.resize(1100, 760)

        self._coord = CoordinateService()
        self._current_zone: dict | None = None
        self._current_map_path: Path | None = None
        self._crop_sync_guard = False
        self._marker_size_guard = False
        self._spot_sync_guard = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        self._canvas = MapCanvas()
        self._canvas.coords_changed.connect(self._on_coords_changed)
        self._canvas.center_changed.connect(self._sync_zoom_spin)
        layout.addWidget(self._canvas, stretch=1)

        side = QVBoxLayout()
        layout.addLayout(side)

        zone_box = QGroupBox("지역")
        zone_form = QFormLayout(zone_box)
        self._zone_combo = QComboBox()
        self._zone_combo.setMinimumWidth(220)
        self._populate_zones()
        self._zone_combo.currentIndexChanged.connect(self._on_zone_changed)
        zone_form.addRow("지역:", self._zone_combo)
        self._map_label = QLabel("—")
        self._map_label.setWordWrap(True)
        zone_form.addRow("지도:", self._map_label)
        side.addWidget(zone_box)

        open_btn = QPushButton("PNG 직접 열기…")
        open_btn.clicked.connect(self._open_custom_png)
        side.addWidget(open_btn)

        crop_box = QGroupBox("크롭 / 줌")
        crop_form = QFormLayout(crop_box)
        self._crop_w_spin = QSpinBox()
        self._crop_w_spin.setRange(32, 1024)
        self._crop_w_spin.setValue(DEFAULT_CROP_W)
        self._crop_w_spin.valueChanged.connect(self._on_crop_w_changed)
        crop_form.addRow("너비:", self._crop_w_spin)

        self._crop_h_spin = QSpinBox()
        self._crop_h_spin.setRange(32, 1024)
        self._crop_h_spin.setValue(DEFAULT_CROP_H)
        self._crop_h_spin.valueChanged.connect(self._on_crop_h_changed)
        crop_form.addRow("높이:", self._crop_h_spin)

        self._lock_crop_aspect = QCheckBox("인게임 창 비율 고정 (501:430)")
        self._lock_crop_aspect.setChecked(True)
        crop_form.addRow(self._lock_crop_aspect)

        self._crop_aspect_label = QLabel(
            f"비율 {INGAME_FRAME_ASPECT:.3f}  |  ref 기본 {DEFAULT_CROP_W}×{DEFAULT_CROP_H}"
        )
        self._crop_aspect_label.setWordWrap(True)
        crop_form.addRow(self._crop_aspect_label)

        sync_crop_btn = QPushButton("줌 → 크롭 크기 맞춤")
        sync_crop_btn.clicked.connect(self._sync_crop_size_from_zoom)
        crop_form.addRow(sync_crop_btn)

        self._zoom_spin = QDoubleSpinBox()
        self._zoom_spin.setRange(0.25, 8.0)
        self._zoom_spin.setSingleStep(0.1)
        self._zoom_spin.setValue(DEFAULT_ZOOM)
        self._zoom_spin.valueChanged.connect(self._on_zoom_changed)
        crop_form.addRow("줌:", self._zoom_spin)
        side.addWidget(crop_box)

        goto_box = QGroupBox("게임 좌표 (빨간 X)")
        goto_form = QFormLayout(goto_box)

        self._spot_combo = QComboBox()
        self._spot_combo.setMinimumWidth(220)
        self._spot_combo.currentIndexChanged.connect(self._on_spot_selected)
        goto_form.addRow("spots:", self._spot_combo)

        self._coord_paste = QLineEdit()
        self._coord_paste.setPlaceholderText("22.61, 15.62")
        self._coord_paste.returnPressed.connect(self._place_marker_from_input)
        goto_form.addRow("붙여넣기:", self._coord_paste)

        self._goto_x = QDoubleSpinBox()
        self._goto_x.setRange(1.0, 41.0)
        self._goto_x.setDecimals(2)
        self._goto_x.setSingleStep(0.01)
        self._goto_x.setValue(20.00)
        self._goto_x.valueChanged.connect(self._sync_coord_paste_from_spin)
        goto_form.addRow("X:", self._goto_x)
        self._goto_y = QDoubleSpinBox()
        self._goto_y.setRange(1.0, 41.0)
        self._goto_y.setDecimals(2)
        self._goto_y.setSingleStep(0.01)
        self._goto_y.setValue(20.00)
        self._goto_y.valueChanged.connect(self._sync_coord_paste_from_spin)
        goto_form.addRow("Y:", self._goto_y)

        place_btn = QPushButton("마커 배치")
        place_btn.clicked.connect(self._place_marker_from_input)
        goto_form.addRow(place_btn)

        pan_btn = QPushButton("지도만 이동")
        pan_btn.clicked.connect(self._goto_coords)
        goto_form.addRow(pan_btn)
        side.addWidget(goto_box)

        marker_box = QGroupBox("보물 마커")
        marker_layout = QVBoxLayout(marker_box)
        self._coord_label = QLabel("크롭 안을 클릭해 보물 마커를 배치하세요")
        self._coord_label.setWordWrap(True)
        marker_layout.addWidget(self._coord_label)

        marker_form = QFormLayout()
        self._marker_size_spin = QSpinBox()
        self._marker_size_spin.setRange(MARKER_SIZE_MIN, MARKER_SIZE_MAX)
        self._marker_size_spin.setValue(22)
        self._marker_size_spin.setToolTip("크롭 대비 X 마커 픽셀 크기 (미리보기·저장 공통)")
        self._marker_size_spin.valueChanged.connect(self._on_marker_size_changed)
        marker_form.addRow("크기 (px):", self._marker_size_spin)

        self._marker_off_x = QSpinBox()
        self._marker_off_x.setRange(-80, 80)
        self._marker_off_x.setValue(0)
        self._marker_off_x.setToolTip("양수=오른쪽 (원본 지도 px)")
        self._marker_off_x.valueChanged.connect(self._on_marker_offset_changed)
        marker_form.addRow("X 보정:", self._marker_off_x)

        self._marker_off_y = QSpinBox()
        self._marker_off_y.setRange(-80, 80)
        self._marker_off_y.setValue(-10)
        self._marker_off_y.setToolTip("음수=위로 (인게임보다 아래면 -10~-20)")
        self._marker_off_y.valueChanged.connect(self._on_marker_offset_changed)
        marker_form.addRow("Y 보정:", self._marker_off_y)

        marker_layout.addLayout(marker_form)

        auto_marker_btn = QPushButton("크기 자동 (크롭 비율)")
        auto_marker_btn.clicked.connect(self._sync_marker_size_auto)
        marker_layout.addWidget(auto_marker_btn)

        self._draw_x_check = QCheckBox("저장 시 마커 포함")
        self._draw_x_check.setChecked(True)
        marker_layout.addWidget(self._draw_x_check)
        clear_btn = QPushButton("마커 제거")
        clear_btn.clicked.connect(self._clear_marker)
        marker_layout.addWidget(clear_btn)
        side.addWidget(marker_box)

        save_box = QGroupBox("저장")
        save_layout = QVBoxLayout(save_box)
        self._party_combo = QComboBox()
        self._party_combo.addItem("solo (1인)", "solo")
        self._party_combo.addItem("party8 (8인)", "party8")
        save_layout.addWidget(self._party_combo)
        self._save_path_label = QLabel("")
        self._save_path_label.setWordWrap(True)
        save_layout.addWidget(self._save_path_label)
        save_btn = QPushButton("PNG 저장…")
        save_btn.clicked.connect(self._save_png)
        save_layout.addWidget(save_btn)
        side.addWidget(save_box)

        hint = QLabel(
            "드래그: 이동  |  휠: 줌  |  방향키: 미세 이동\n"
            "크롭 안 클릭 또는 좌표 입력: 보물 마커 배치"
        )
        hint.setWordWrap(True)
        side.addWidget(hint)
        side.addStretch()

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("준비")

        if self._zone_combo.count() > 0:
            self._load_zone_index(0)
        self._sync_coord_paste_from_spin()
        self._canvas.set_marker_size(self._marker_size_spin.value())
        self._canvas.set_marker_offset(
            float(self._marker_off_x.value()),
            float(self._marker_off_y.value()),
        )

    def _populate_zones(self) -> None:
        zones = sorted(
            self._coord.zones,
            key=lambda z: (
                str(z.get("expansion", "")),
                str(z.get("name_ko", z.get("id", ""))),
            ),
        )
        for zone in zones:
            name = zone.get("name_ko") or zone.get("name_en") or zone.get("id")
            expansion = zone.get("expansion", "")
            label = f"[{expansion}] {name}"
            self._zone_combo.addItem(label, zone)

    def _zone_at(self, index: int) -> dict | None:
        if index < 0:
            return None
        data = self._zone_combo.itemData(index)
        return data if isinstance(data, dict) else None

    def _load_zone_index(self, index: int) -> None:
        zone = self._zone_at(index)
        if zone is None:
            return
        zone_id = str(zone.get("id", ""))
        path = self._coord.get_detail_map_path(zone_id)
        if not path.exists():
            QMessageBox.warning(
                self,
                "지도 없음",
                f"상세 지도를 찾을 수 없습니다:\n{path}",
            )
            return
        self._load_map(path, zone)

    def _on_zone_changed(self, index: int) -> None:
        self._load_zone_index(index)

    def _load_map(self, path: Path, zone: dict | None) -> None:
        try:
            pil = Image.open(path)
        except OSError as exc:
            QMessageBox.critical(self, "로드 실패", str(exc))
            return
        qimg = pil_to_qimage(pil)
        self._current_map_path = path
        self._current_zone = zone
        self._canvas.set_current_map_path(path)
        if zone is not None:
            self._canvas.set_map(qimg, zone, self._coord)
        else:
            self._canvas._source = qimg
            self._canvas._source_size = (qimg.width(), qimg.height())
            self._canvas._zone = None
            self._canvas.center_x = qimg.width() / 2.0
            self._canvas.center_y = qimg.height() / 2.0
            self._canvas._marker_local = None
            self._canvas.update()
        self._map_label.setText(str(path.relative_to(get_app_root())))
        self._status.showMessage(f"로드: {path.name} ({pil.width}×{pil.height})")
        self._populate_spot_combo(zone)
        self._update_save_hint()

    def _populate_spot_combo(self, zone: dict | None) -> None:
        self._spot_sync_guard = True
        self._spot_combo.clear()
        self._spot_combo.addItem("— spots에서 선택 —", None)
        if zone is not None:
            spots = treasure_spots_from_zone(zone)
            for gx, gy in spots:
                self._spot_combo.addItem(format_game_coords(gx, gy), (gx, gy))
            if spots:
                self._status.showMessage(
                    f"spots {len(spots)}개 · {zone.get('name_ko', zone.get('id', ''))}"
                )
        self._spot_combo.setEnabled(self._spot_combo.count() > 1)
        self._spot_sync_guard = False

    def _on_spot_selected(self, index: int) -> None:
        if self._spot_sync_guard or index <= 0:
            return
        data = self._spot_combo.itemData(index)
        if not isinstance(data, tuple) or len(data) != 2:
            return
        gx, gy = float(data[0]), float(data[1])
        self._goto_x.blockSignals(True)
        self._goto_y.blockSignals(True)
        self._goto_x.setValue(gx)
        self._goto_y.setValue(gy)
        self._goto_x.blockSignals(False)
        self._goto_y.blockSignals(False)
        self._sync_coord_paste_from_spin()
        self._place_marker_from_input()

    def _open_custom_png(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "상세 지도 PNG",
            str(get_app_root() / "assets" / "maps"),
            "PNG (*.png)",
        )
        if not path:
            return
        zone = self._zone_at(self._zone_combo.currentIndex())
        self._load_map(Path(path), zone)

    def _set_crop_spin_values(self, w: int, h: int) -> None:
        self._crop_sync_guard = True
        self._crop_w_spin.setValue(w)
        self._crop_h_spin.setValue(h)
        self._crop_sync_guard = False
        self._canvas.set_crop_size(w, h)

    def _on_crop_w_changed(self, value: int) -> None:
        if self._crop_sync_guard:
            return
        h = (
            crop_height_for_width(value, INGAME_FRAME_ASPECT)
            if self._lock_crop_aspect.isChecked()
            else self._crop_h_spin.value()
        )
        self._set_crop_spin_values(value, h)

    def _on_crop_h_changed(self, value: int) -> None:
        if self._crop_sync_guard:
            return
        w = (
            crop_width_for_height(value, INGAME_FRAME_ASPECT)
            if self._lock_crop_aspect.isChecked()
            else self._crop_w_spin.value()
        )
        self._set_crop_spin_values(w, value)

    def _sync_marker_size_auto(self) -> None:
        px = marker_size_for_crop(
            self._crop_w_spin.value(),
            self._crop_h_spin.value(),
        )
        self._marker_size_guard = True
        self._marker_size_spin.setValue(px)
        self._marker_size_guard = False
        self._canvas.set_marker_size(px)
        self._status.showMessage(f"마커 크기 자동: {px}px")

    def _on_marker_size_changed(self, value: int) -> None:
        if self._marker_size_guard:
            return
        self._canvas.set_marker_size(value)

    def _on_marker_offset_changed(self) -> None:
        self._canvas.set_marker_offset(
            float(self._marker_off_x.value()),
            float(self._marker_off_y.value()),
        )

    def _sync_crop_size_from_zoom(self) -> None:
        w, h = crop_size_for_ingame_zoom(self._zoom_spin.value())
        self._set_crop_spin_values(w, h)
        self._status.showMessage(f"크롭 {w}×{h} (줌 {self._zoom_spin.value():.2f})")

    def _on_zoom_changed(self, value: float) -> None:
        self._canvas.set_zoom(value)

    def _sync_zoom_spin(self) -> None:
        self._zoom_spin.blockSignals(True)
        self._zoom_spin.setValue(self._canvas.zoom)
        self._zoom_spin.blockSignals(False)

    def _sync_coord_paste_from_spin(self) -> None:
        self._coord_paste.blockSignals(True)
        self._coord_paste.setText(
            format_game_coords(self._goto_x.value(), self._goto_y.value())
        )
        self._coord_paste.blockSignals(False)

    def _read_game_coords(self) -> tuple[float, float] | None:
        parsed = parse_game_coord_text(self._coord_paste.text())
        if parsed is not None:
            gx, gy = parsed
            if not (1.0 <= gx <= 41.0 and 1.0 <= gy <= 41.0):
                return None
            self._goto_x.blockSignals(True)
            self._goto_y.blockSignals(True)
            self._goto_x.setValue(round_game_coord(gx))
            self._goto_y.setValue(round_game_coord(gy))
            self._goto_x.blockSignals(False)
            self._goto_y.blockSignals(False)
            return gx, gy
        return round_game_coords(self._goto_x.value(), self._goto_y.value())

    def _place_marker_from_input(self) -> None:
        coords = self._read_game_coords()
        if coords is None:
            QMessageBox.warning(
                self,
                "좌표 오류",
                "게임 좌표를 입력하세요.\n예: 22.61, 15.62 (범위 1.00~41.00)",
            )
            return
        gx, gy = coords
        ok = self._canvas.place_marker_at_game_coords(gx, gy)
        if not ok:
            QMessageBox.information(self, "배치 불가", "지역·지도가 로드되어 있어야 합니다.")
            return
        self._on_coords_changed(gx, gy)

    def _goto_coords(self) -> None:
        coords = self._read_game_coords()
        if coords is None:
            QMessageBox.warning(
                self,
                "좌표 오류",
                "게임 좌표를 입력하세요.\n예: 22.61, 15.62 (범위 1.00~41.00)",
            )
            return
        ok = self._canvas.go_to_game_coords(*coords)
        if not ok:
            QMessageBox.information(self, "이동 불가", "지역·지도가 로드되어 있어야 합니다.")

    def _on_coords_changed(self, gx: float, gy: float) -> None:
        self._coord_label.setText(f"게임 좌표: {format_game_coords(gx, gy)}")
        self._goto_x.blockSignals(True)
        self._goto_y.blockSignals(True)
        self._goto_x.setValue(round_game_coord(gx))
        self._goto_y.setValue(round_game_coord(gy))
        self._goto_x.blockSignals(False)
        self._goto_y.blockSignals(False)
        self._coord_paste.blockSignals(True)
        self._coord_paste.setText(format_game_coords(gx, gy))
        self._coord_paste.blockSignals(False)
        self._spot_sync_guard = True
        idx = self._spot_combo.findText(format_game_coords(gx, gy))
        self._spot_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._spot_sync_guard = False
        self._update_save_hint(gx, gy)
        self._status.showMessage(f"마커 좌표: {format_game_coords(gx, gy)}")

    def _clear_marker(self) -> None:
        self._canvas._marker_local = None
        self._canvas._explicit_game_coords = None
        self._coord_label.setText("크롭 안을 클릭해 보물 마커를 배치하세요")
        self._update_save_hint()
        self._canvas.update()

    def _default_save_dir(self) -> Path | None:
        if self._current_zone is None:
            return None
        expansion = str(self._current_zone.get("expansion", "unknown"))
        zone_id = str(self._current_zone.get("id", ""))
        party = str(self._party_combo.currentData() or "solo")
        return (
            get_app_root()
            / "assets"
            / "treasure_refs"
            / expansion
            / zone_id
            / party
        )

    def _update_save_hint(self, gx: float | None = None, gy: float | None = None) -> None:
        save_dir = self._default_save_dir()
        if save_dir is None:
            self._save_path_label.setText("지역 선택 후 저장 경로가 표시됩니다")
            return
        if gx is None or gy is None:
            coords = self._canvas.marker_game_coords()
            if coords is not None:
                gx, gy = coords
        if gx is not None and gy is not None:
            rel = save_dir / coord_filename(gx, gy)
            self._save_path_label.setText(
                f"권장: assets/treasure_refs/…/{rel.name}"
            )
        else:
            self._save_path_label.setText(f"폴더: {save_dir.relative_to(get_app_root())}")

    def _save_png(self) -> None:
        if self._canvas._source is None:
            QMessageBox.information(self, "저장 불가", "먼저 지도를 로드하세요.")
            return
        crop = self._canvas.export_crop(self._draw_x_check.isChecked())
        if crop is None:
            return

        coords = self._canvas.marker_game_coords()
        default_name = "ref.png"
        default_dir = str(get_app_root())
        if coords is not None:
            default_name = coord_filename(coords[0], coords[1])
        save_dir = self._default_save_dir()
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            default_dir = str(save_dir)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "ref PNG 저장",
            str(Path(default_dir) / default_name),
            "PNG (*.png)",
        )
        if not path:
            return
        try:
            crop.save(path, "PNG")
        except OSError as exc:
            QMessageBox.critical(self, "저장 실패", str(exc))
            return

        saved = Path(path)
        self._status.showMessage(f"저장 완료: {saved.name}", 8000)
        if coords is not None:
            zone_id = self._current_zone.get("id", "") if self._current_zone else ""
            self._save_path_label.setText(
                f"저장됨: {saved}\n"
                f"좌표: {format_game_coords(coords[0], coords[1])}\n"
                f"좌표 DB: python scripts/build_treasure_ref_coords.py {zone_id}"
            )
        else:
            self._save_path_label.setText(f"저장됨: {saved}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("ref_maker")
    win = RefMakerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
