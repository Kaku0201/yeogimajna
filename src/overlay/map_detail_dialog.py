from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.models.recognition_result import RecognitionResult
from src.services.app_paths import get_app_icon_path
from src.services.aetheryte_icon_service import AetheryteIconService
from src.services.app_paths import get_app_root
from src.services.coordinate_service import CoordinateService
from src.services.settings_service import SettingsService


class MapDetailDialog(QDialog):
    """좌표 클릭 시 상세 지도 + 마커 이미지 표시 (프레임리스·드래그 이동)"""

    confirmed = pyqtSignal()

    FONT_INFO = 12
    LEGEND_ICON = 22
    DISPLAY_SIZE = 520
    AE_ICON_VISIBLE = 44
    TREASURE_MARKER_VISIBLE = 36

    @classmethod
    def treasure_marker_path(cls) -> Path:
        return get_app_root() / "assets" / "treasure_marker.png"

    def __init__(
        self,
        result: RecognitionResult,
        coordinate_service: CoordinateService,
        settings: SettingsService | None = None,
        parent=None,
        *,
        show_confirm: bool = False,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.coordinate_service = coordinate_service
        self.settings = settings
        self._show_confirm = show_confirm
        self._icon_service = AetheryteIconService()
        self._treasure_marker = self._load_treasure_marker()
        self._drag_pos: QPoint | None = None
        self._bg_opacity = (
            settings.detail_map_bg_opacity if settings is not None else 0.5
        )
        icon_path = get_app_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setMinimumSize(self.DISPLAY_SIZE + 40, self.DISPLAY_SIZE + 80)
        self._setup_ui()
        self.set_background_opacity(self._bg_opacity)

    @classmethod
    def _marker_px(cls, map_size: int, visible_px: int) -> int:
        """축소 표시 후 visible_px가 되도록 원본 지도 픽셀 크기 계산"""
        return max(visible_px, int(visible_px * map_size / cls.DISPLAY_SIZE))

    @staticmethod
    def _visible_bg_alpha(opacity: float) -> int:
        return max(1, int(opacity * 255))

    def set_background_opacity(self, opacity: float) -> None:
        """배경만 투명도 조절 — 글자는 항상 불투명"""
        self._bg_opacity = max(0.0, min(1.0, opacity))
        bg_alpha = self._visible_bg_alpha(self._bg_opacity)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: rgba(0, 0, 0, {bg_alpha});
            }}
            QLabel {{
                color: #f0f0f0;
                background: transparent;
            }}
            """
        )

    def _load_treasure_marker(self) -> QPixmap:
        marker_path = self.treasure_marker_path()
        if not marker_path.exists():
            return QPixmap()
        marker = QPixmap(str(marker_path))
        return marker if not marker.isNull() else QPixmap()

    def _info_font(self) -> QFont:
        return QFont("Segoe UI", self.FONT_INFO)

    def _pipe_label(self) -> QLabel:
        label = QLabel("|")
        label.setFont(self._info_font())
        return label

    def _icon_label(self, pixmap: QPixmap, size: int) -> QLabel:
        label = QLabel()
        label.setFixedSize(size, size)
        label.setScaledContents(True)
        if not pixmap.isNull():
            label.setPixmap(pixmap)
        return label

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        info_row = QHBoxLayout()
        info_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_row.setSpacing(8)

        zone_text = QLabel(
            f"{self.result.zone_name} | {self.result.coordinate_text}"
        )
        zone_text.setFont(self._info_font())
        info_row.addWidget(zone_text)
        info_row.addWidget(self._pipe_label())

        ae_icon_path = self._icon_service.resolve_icon_path(
            self.result.nearest_aetheryte,
            self.result.nearest_aetheryte_icon,
        )
        ae_pixmap = (
            AetheryteIconService.load_qpixmap(ae_icon_path, self.LEGEND_ICON)
            if ae_icon_path
            else QPixmap()
        )
        info_row.addWidget(self._icon_label(ae_pixmap, self.LEGEND_ICON))

        ae_text = QLabel(self.result.nearest_aetheryte)
        ae_text.setFont(self._info_font())
        info_row.addWidget(ae_text)
        layout.addLayout(info_row)

        map_label = QLabel()
        map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        map_label.setPixmap(self._build_map_pixmap())
        layout.addWidget(map_label, stretch=1)

        layout.addWidget(self._build_legend_row())

        if self._show_confirm:
            self.confirm_btn = QPushButton("✓ 이 좌표로 확정 (다음부터 빠르게 찾기)")
            self.confirm_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            self.confirm_btn.setStyleSheet(
                """
                QPushButton {
                    color: #ffffff;
                    background-color: rgba(56, 142, 60, 220);
                    border: 1px solid rgba(129, 199, 132, 240);
                    border-radius: 6px;
                    padding: 8px 14px;
                }
                QPushButton:hover {
                    background-color: rgba(76, 175, 80, 240);
                }
                """
            )
            self.confirm_btn.clicked.connect(self._on_confirm)
            layout.addWidget(self.confirm_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _on_confirm(self) -> None:
        self.confirmed.emit()
        self.close()

    def _build_legend_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        legend_font = QFont("Segoe UI", self.FONT_INFO)

        if not self._treasure_marker.isNull():
            treasure_icon = self._icon_label(
                self._treasure_marker.scaled(
                    self.LEGEND_ICON,
                    self.LEGEND_ICON,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ),
                self.LEGEND_ICON,
            )
            layout.addWidget(treasure_icon)

        treasure_label = QLabel("보물 위치")
        treasure_label.setFont(legend_font)
        layout.addWidget(treasure_label)

        layout.addWidget(self._pipe_label())

        ae_icon_path = self._icon_service.resolve_icon_path(
            self.result.nearest_aetheryte,
            self.result.nearest_aetheryte_icon,
        )
        if ae_icon_path:
            layout.addWidget(
                self._icon_label(
                    AetheryteIconService.load_qpixmap(ae_icon_path, self.LEGEND_ICON),
                    self.LEGEND_ICON,
                )
            )

        ae_label = QLabel("가장 가까운 에테라이트")
        ae_label.setFont(legend_font)
        layout.addWidget(ae_label)

        return row

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.close()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is None:
            return
        self.move(event.globalPosition().toPoint() - self._drag_pos)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None:
            self._drag_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _build_map_pixmap(self) -> QPixmap:
        map_path = self.coordinate_service.get_detail_map_path(
            self.result.zone_id, self.result.map_index
        )
        detail_id = self.coordinate_service.resolve_detail_zone_id(self.result.zone_id)
        zone = self.coordinate_service.get_effective_zone(self.result.zone_id)

        if map_path.exists() and zone:
            base = QPixmap(str(map_path))
            map_w = base.width()
            ae_size = self._marker_px(map_w, self.AE_ICON_VISIBLE)
            treasure_size = self._marker_px(map_w, self.TREASURE_MARKER_VISIBLE)

            marked = QPixmap(base.size())
            marked.fill(Qt.GlobalColor.transparent)
            painter = QPainter(marked)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            painter.drawPixmap(0, 0, base)

            treasure_px, treasure_py = self.coordinate_service.game_to_detail_pixel(
                zone, self.result.x, self.result.y, self.result.map_index
            )

            if (
                self.result.nearest_aetheryte_x is not None
                and self.result.nearest_aetheryte_y is not None
            ):
                ae_px, ae_py = self.coordinate_service.game_to_detail_pixel(
                    zone,
                    self.result.nearest_aetheryte_x,
                    self.result.nearest_aetheryte_y,
                    self.result.map_index,
                )
                self._draw_route_line(painter, treasure_px, treasure_py, ae_px, ae_py)
                self._draw_aetheryte_marker(
                    painter, ae_px, ae_py, ae_size, self.result.nearest_aetheryte
                )

            self._draw_treasure_marker(painter, treasure_px, treasure_py, treasure_size)
            painter.end()

            return marked.scaled(
                self.DISPLAY_SIZE,
                self.DISPLAY_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        placeholder = QPixmap(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        placeholder.fill(QColor(45, 45, 48))
        painter = QPainter(placeholder)
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(
            placeholder.rect(),
            Qt.AlignmentFlag.AlignCenter,
            f"{self.result.zone_name}\n(상세 지도 없음)\n"
            f"maps/{{확장팩}}/{detail_id}.png\n"
            f"파일을 추가하세요",
        )
        painter.end()
        return placeholder

    def _draw_route_line(
        self,
        painter: QPainter,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        pen = QPen(QColor(255, 200, 80, 160))
        pen.setWidth(max(2, int(2 * x1 / self.DISPLAY_SIZE)))
        pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawLine(x1, y1, x2, y2)

    def _draw_treasure_marker(
        self, painter: QPainter, px: int, py: int, size: int
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if not self._treasure_marker.isNull():
            scaled = self._treasure_marker.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            half = size // 2
            painter.drawPixmap(px - half, py - half, scaled)
            return

        half = size // 2
        pen = QPen(QColor(220, 40, 40))
        pen.setWidth(max(2, size // 12))
        painter.setPen(pen)
        painter.drawLine(px - half, py - half, px + half, py + half)
        painter.drawLine(px + half, py - half, px - half, py + half)

    def _draw_aetheryte_marker(
        self,
        painter: QPainter,
        px: int,
        py: int,
        size: int,
        name: str,
    ) -> None:
        icon_path = self._icon_service.resolve_icon_path(
            name,
            self.result.nearest_aetheryte_icon,
        )
        if icon_path:
            icon = AetheryteIconService.load_qpixmap(icon_path, size)
            if not icon.isNull():
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                half = size // 2
                painter.drawPixmap(px - half, py - half, icon)

        font_size = max(22, (size // 8) * 2)
        font = QFont("Segoe UI", font_size, QFont.Weight.Bold)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text_w = metrics.horizontalAdvance(name)
        text_h = metrics.height()
        label_x = px - text_w // 2
        label_y = py + size // 2 + 6

        bg = QColor(20, 20, 28, 200)
        painter.fillRect(
            label_x - 4,
            label_y - 2,
            text_w + 8,
            text_h + 4,
            bg,
        )
        painter.setPen(QColor(255, 220, 120))
        painter.drawText(
            label_x,
            label_y + metrics.ascent(),
            name,
        )
