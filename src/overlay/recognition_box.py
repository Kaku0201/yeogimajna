from collections.abc import Callable

from PyQt6.QtCore import QPoint, Qt, QTimer, QRect, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QIcon, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from src.overlay.frame_quality_worker import FrameQualityWorker
from src.services.app_paths import get_app_icon_path
from src.services.settings_service import SettingsService
from src.services.treasure_capture import CaptureReadiness, TreasureCaptureProcessor


class RecognitionBox(QWidget):
    """고정 비율 템플릿 + 지역명 슬롯 — 보물지도 창 맞춤 캡처"""

    confirmed = pyqtSignal(tuple)
    cancelled = pyqtSignal()

    MIN_WIDTH = TreasureCaptureProcessor.TEMPLATE_MIN_WIDTH
    MIN_HEIGHT = 80
    HANDLE_SIZE = 8
    MOVE_HIT_MARGIN = 6
    QUALITY_INTERVAL_MS = 100
    TEMPLATE_ASPECT = TreasureCaptureProcessor.MAP_ASPECT

    def __init__(
        self,
        settings: SettingsService | None = None,
        capture_processor: TreasureCaptureProcessor | None = None,
        capture_fn: Callable[[tuple[int, int, int, int]], object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.capture_processor = capture_processor
        self.capture_fn = capture_fn
        self._frame_rect = QRect()
        self._drag_offset: QPoint | None = None
        self._resize_handle: str | None = None
        self._select_origin: QPoint | None = None
        self._confirm_btn: QPushButton | None = None
        self._locked = False
        self._readiness: CaptureReadiness | None = None
        self._guide_rect = QRect()
        self._banner_target_rect = QRect()
        self._banner_detected_rect = QRect()
        self._quality_worker: FrameQualityWorker | None = None
        self._pending_quality_rect: tuple[int, int, int, int] | None = None
        self._resize_anchor: str | None = None
        self._bare_quality_capture = False

        self._quality_timer = QTimer(self)
        self._quality_timer.setSingleShot(True)
        self._quality_timer.timeout.connect(self._run_quality_check)

        icon_path = get_app_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def start_selection(self) -> None:
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        if screen is None:
            return

        self.setGeometry(screen.geometry())
        self._locked = False
        self._drag_offset = None
        self._resize_handle = None
        self._select_origin = None
        self._frame_rect = QRect()
        self._readiness = None
        self._guide_rect = QRect()
        self._banner_target_rect = QRect()
        self._banner_detected_rect = QRect()
        self._pending_quality_rect = None
        self._quality_timer.stop()
        self._stop_quality_worker()
        self._remove_confirm_button()

        center = QCursor.pos()
        local = self.mapFromGlobal(center)
        self._place_default_template(local)
        self._locked = True
        self._sync_template_overlay_rects()
        self._schedule_quality_check()
        self._update_confirm_button()

        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        self.update()

    def _clamp_rect(self, rect: QRect) -> QRect:
        bounded = rect.normalized()
        bounded.setLeft(max(0, bounded.left()))
        bounded.setTop(max(0, bounded.top()))
        bounded.setRight(min(self.width(), bounded.right()))
        bounded.setBottom(min(self.height(), bounded.bottom()))
        return bounded

    def _is_valid_size(self, rect: QRect) -> bool:
        return (
            rect.width() >= self.MIN_WIDTH
            and rect.height() >= self.MIN_HEIGHT
        )

    def _place_default_template(self, center: QPoint) -> None:
        w, h = TreasureCaptureProcessor.template_size_for_screen(
            self.width(), self.height()
        )
        x = center.x() - w // 2
        y = center.y() - h // 2
        self._frame_rect = self._clamp_rect(QRect(x, y, w, h))

    def _sync_template_overlay_rects(self) -> None:
        if not self._is_valid_size(self._frame_rect):
            self._banner_target_rect = QRect()
            return
        slot_h = max(
            8,
            int(
                self._frame_rect.height()
                * TreasureCaptureProcessor.TEMPLATE_BANNER_HEIGHT_RATIO
            ),
        )
        self._banner_target_rect = QRect(
            self._frame_rect.left(),
            self._frame_rect.top(),
            self._frame_rect.width(),
            slot_h,
        )

    def _enforce_template_aspect(self, rect: QRect, anchor: str) -> QRect:
        aspect = self.TEMPLATE_ASPECT
        w = max(self.MIN_WIDTH, rect.width())
        h = max(self.MIN_HEIGHT, int(w / aspect))
        w = max(self.MIN_WIDTH, int(h * aspect))

        fixed = QRect(rect)
        if anchor in ("br", "tr"):
            fixed.setWidth(w)
            fixed.setHeight(h)
        elif anchor == "bl":
            fixed.setLeft(fixed.right() - w)
            fixed.setHeight(h)
        else:
            fixed.setRight(fixed.left() + w)
            fixed.setBottom(fixed.top() + h)
        return self._clamp_rect(fixed)

    def _global_frame_rect(self) -> tuple[int, int, int, int] | None:
        if self._frame_rect.isNull() or not self._is_valid_size(self._frame_rect):
            return None
        return (
            self._frame_rect.x() + self.x(),
            self._frame_rect.y() + self.y(),
            self._frame_rect.width(),
            self._frame_rect.height(),
        )

    def _schedule_quality_check(self) -> None:
        if self.capture_processor is None or self.capture_fn is None:
            return
        if not self._is_valid_size(self._frame_rect):
            self._readiness = None
            self._guide_rect = QRect()
            self._update_confirm_button()
            self.update()
            return
        self._quality_timer.start(self.QUALITY_INTERVAL_MS)

    def _run_quality_check(self) -> None:
        rect = self._global_frame_rect()
        if rect is None or self.capture_fn is None:
            return
        if self._quality_worker is not None and self._quality_worker.isRunning():
            self._pending_quality_rect = rect
            return

        # 오버레이(노란 슬롯·시안 덮개) 없이 게임 픽셀만 캡처
        self._bare_quality_capture = True
        self.update()
        QApplication.processEvents()

        try:
            shot = self.capture_fn(rect)
        finally:
            self._bare_quality_capture = False
            self.update()

        self._quality_worker = FrameQualityWorker(
            self.capture_processor,
            rect,
            shot=shot,
            bare_capture=True,
        )
        self._quality_worker.finished_check.connect(self._on_quality_checked)
        self._quality_worker.start()

    def _stop_quality_worker(self) -> None:
        if self._quality_worker is not None:
            try:
                self._quality_worker.finished_check.disconnect(self._on_quality_checked)
            except TypeError:
                pass
            if self._quality_worker.isRunning():
                self._quality_worker.wait(200)
            self._quality_worker = None

    def _on_quality_checked(
        self,
        readiness: CaptureReadiness,
        rect: tuple[int, int, int, int],
    ) -> None:
        worker = self.sender()
        if worker is not self._quality_worker:
            return

        current = self._global_frame_rect()
        if current is None or current != rect:
            self._maybe_run_pending_quality()
            return

        self._readiness = readiness
        self._guide_rect = QRect()
        self._banner_detected_rect = QRect()
        self._sync_template_overlay_rects()

        if readiness.banner_zone is not None:
            bx1, by1, bx2, by2 = readiness.banner_zone
            self._banner_detected_rect = QRect(
                self._frame_rect.left() + bx1,
                self._frame_rect.top() + by1,
                bx2 - bx1,
                by2 - by1,
            )
        self._update_confirm_button()
        self.update()
        self._maybe_run_pending_quality()

    def _maybe_run_pending_quality(self) -> None:
        pending = self._pending_quality_rect
        self._pending_quality_rect = None
        if pending is None:
            return
        current = self._global_frame_rect()
        if current is None or current != pending:
            return
        self._quality_timer.start(0)

    def _is_interacting(self) -> bool:
        return self._drag_offset is not None or self._resize_handle is not None

    def _is_ready(self) -> bool:
        return self._readiness is not None and self._readiness.ready

    def _frame_border_color(self) -> QColor:
        if self._readiness is None:
            return QColor(0, 220, 255)
        if self._readiness.ready:
            return QColor(80, 220, 120)
        if self._readiness.has_marker:
            return QColor(0, 200, 255)
        return QColor(0, 180, 255)

    def _status_hint(self) -> str:
        if self._readiness is not None:
            return self._readiness.hint
        if self._locked:
            return "프레임 품질 확인 중…"
        return "보물지도 창을 네모 안에 맞춰주세요"

    def _move_hit_rect(self) -> QRect:
        return self._frame_rect.adjusted(
            -self.MOVE_HIT_MARGIN,
            -self.MOVE_HIT_MARGIN,
            self.MOVE_HIT_MARGIN,
            self.MOVE_HIT_MARGIN,
        )

    def _handle_at(self, pos: QPoint) -> str | None:
        if not self._locked or self._frame_rect.isNull():
            return None

        frame = self._frame_rect
        hs = self.HANDLE_SIZE
        handles = {
            "tl": QRect(frame.left() - hs, frame.top() - hs, hs * 2, hs * 2),
            "tr": QRect(frame.right() - hs, frame.top() - hs, hs * 2, hs * 2),
            "bl": QRect(frame.left() - hs, frame.bottom() - hs, hs * 2, hs * 2),
            "br": QRect(frame.right() - hs, frame.bottom() - hs, hs * 2, hs * 2),
        }
        for name, hit in handles.items():
            if hit.contains(pos):
                return name
        return None

    def _cursor_for_handle(self, handle: str | None) -> Qt.CursorShape:
        mapping = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
        }
        return mapping.get(handle or "", Qt.CursorShape.CrossCursor)

    def _resize_rect(self, handle: str, pos: QPoint) -> QRect:
        frame = QRect(self._frame_rect)
        if handle == "tl":
            frame.setTopLeft(pos)
        elif handle == "tr":
            frame.setTopRight(pos)
        elif handle == "bl":
            frame.setBottomLeft(pos)
        elif handle == "br":
            frame.setBottomRight(pos)
        return self._enforce_template_aspect(frame, handle)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._frame_rect.isNull() or self._frame_rect.width() < 2:
            if not self._locked:
                painter.fillRect(self.rect(), QColor(0, 0, 0, 1))
                painter.setPen(QColor(255, 255, 255, 210))
                painter.setFont(QFont("Segoe UI", 11))
                painter.drawText(
                    self.rect().adjusted(0, 40, 0, 0),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    "보물지도 창을 네모 안에 맞춰주세요",
                )
            return

        frame = self._frame_rect
        interacting = self._is_interacting()
        ready = self._is_ready()

        if self._bare_quality_capture and self._locked and not interacting:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Clear
            )
            painter.fillRect(frame, Qt.GlobalColor.transparent)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            return

        if self._locked and not interacting:
            dim = QColor(0, 0, 0, 100)
            full = self.rect()
            painter.fillRect(
                QRect(full.left(), full.top(), full.width(), frame.top()),
                dim,
            )
            painter.fillRect(
                QRect(
                    full.left(),
                    frame.bottom(),
                    full.width(),
                    full.height() - frame.bottom(),
                ),
                dim,
            )
            painter.fillRect(
                QRect(full.left(), frame.top(), frame.left(), frame.height()),
                dim,
            )
            painter.fillRect(
                QRect(
                    frame.right(),
                    frame.top(),
                    full.width() - frame.right(),
                    frame.height(),
                ),
                dim,
            )

        # 노란 지역명 슬롯 — 이동 중에도 항상 표시 (덮개 위에 그림)
        def draw_banner_slot() -> None:
            if (
                self._banner_target_rect.isNull()
                or not self._is_valid_size(self._frame_rect)
            ):
                return
            target = self._banner_target_rect.intersected(frame)
            painter.fillRect(target, QColor(255, 210, 50, 90))
            painter.setPen(QPen(QColor(255, 220, 80, 250), 2, Qt.PenStyle.DashLine))
            painter.drawRect(target)
            painter.setPen(QColor(255, 255, 220, 250))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(
                target.adjusted(6, 2, -6, -2),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                "지역명 (권장)",
            )

        if interacting:
            draw_banner_slot()
            painter.setPen(QPen(QColor(255, 255, 255, 140), 1, Qt.PenStyle.SolidLine))
            painter.drawRect(frame)
            return

        # 정지 시: 프레임 전체 연초록/연청록 덮개 (빨강은 X 마커와 색 충돌)
        if ready:
            painter.fillRect(frame, QColor(100, 230, 130, 72))
        else:
            painter.fillRect(frame, QColor(0, 140, 220, 52))

        draw_banner_slot()

        border = (
            QColor(80, 210, 110)
            if ready
            else self._frame_border_color()
        )
        painter.setPen(QPen(border, 2, Qt.PenStyle.SolidLine))
        painter.drawRect(frame)

        painter.setBrush(border)
        painter.setPen(Qt.PenStyle.NoPen)
        for corner in (
            frame.topLeft(),
            frame.topRight(),
            frame.bottomLeft(),
            frame.bottomRight(),
        ):
            painter.drawRect(
                corner.x() - self.HANDLE_SIZE // 2,
                corner.y() - self.HANDLE_SIZE // 2,
                self.HANDLE_SIZE,
                self.HANDLE_SIZE,
            )

        painter.setPen(QColor(255, 255, 255, 230))
        painter.setFont(QFont("Segoe UI", 9))
        hint_rect = QRect(
            frame.left(),
            max(4, frame.top() - 26),
            frame.width(),
            22,
        )
        painter.drawText(
            hint_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            self._status_hint(),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._cancel()
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        if not self._locked:
            self._select_origin = event.pos()
            self._frame_rect = QRect(self._select_origin, self._select_origin)
            self._resize_anchor = "br"
            self.update()
            event.accept()
            return

        handle = self._handle_at(event.pos())
        if handle is not None:
            self._resize_handle = handle
            self.setCursor(QCursor(self._cursor_for_handle(handle)))
            event.accept()
            return

        if self._move_hit_rect().contains(event.pos()):
            self._drag_offset = event.pos() - self._frame_rect.topLeft()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._select_origin is not None and not self._locked:
            raw = self._clamp_rect(QRect(self._select_origin, event.pos()))
            self._frame_rect = self._enforce_template_aspect(
                raw, self._resize_anchor or "br"
            )
            self._sync_template_overlay_rects()
            self._schedule_quality_check()
            self.update()
            event.accept()
            return

        if self._resize_handle is not None:
            self._frame_rect = self._resize_rect(self._resize_handle, event.pos())
            self._sync_template_overlay_rects()
            self._schedule_quality_check()
            self._update_confirm_button()
            self.update()
            event.accept()
            return

        if self._drag_offset is not None:
            top_left = event.pos() - self._drag_offset
            top_left.setX(
                max(0, min(top_left.x(), self.width() - self._frame_rect.width()))
            )
            top_left.setY(
                max(0, min(top_left.y(), self.height() - self._frame_rect.height()))
            )
            self._frame_rect.moveTopLeft(top_left)
            self._sync_template_overlay_rects()
            self._schedule_quality_check()
            self._update_confirm_button()
            self.update()
            event.accept()
            return

        if self._locked:
            handle = self._handle_at(event.pos())
            if handle is not None:
                self.setCursor(QCursor(self._cursor_for_handle(handle)))
            elif self._move_hit_rect().contains(event.pos()):
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._select_origin is not None and not self._locked:
            self._select_origin = None
            if self._is_valid_size(self._frame_rect):
                self._locked = True
                self._sync_template_overlay_rects()
                self._schedule_quality_check()
                self._update_confirm_button()
            else:
                self._frame_rect = QRect()
                self._readiness = None
            self.update()
            event.accept()
            return

        if self._resize_handle is not None:
            self._resize_handle = None
            if not self._is_valid_size(self._frame_rect):
                self._locked = False
                self._readiness = None
                self._remove_confirm_button()
            self._schedule_quality_check()
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            event.accept()
            return

        if self._drag_offset is not None:
            self._drag_offset = None
            self._schedule_quality_check()
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            return

        if not self._locked or self._frame_rect.isNull():
            super().keyPressEvent(event)
            return

        step = 8 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 2
        moved = False
        top_left = self._frame_rect.topLeft()

        if event.key() == Qt.Key.Key_Left:
            top_left.setX(top_left.x() - step)
            moved = True
        elif event.key() == Qt.Key.Key_Right:
            top_left.setX(top_left.x() + step)
            moved = True
        elif event.key() == Qt.Key.Key_Up:
            top_left.setY(top_left.y() - step)
            moved = True
        elif event.key() == Qt.Key.Key_Down:
            top_left.setY(top_left.y() + step)
            moved = True
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_confirm()
            return

        if moved:
            top_left.setX(
                max(0, min(top_left.x(), self.width() - self._frame_rect.width()))
            )
            top_left.setY(
                max(0, min(top_left.y(), self.height() - self._frame_rect.height()))
            )
            self._frame_rect.moveTopLeft(top_left)
            self._sync_template_overlay_rects()
            self._schedule_quality_check()
            self._update_confirm_button()
            self.update()
            event.accept()
            return

        super().keyPressEvent(event)

    def _update_confirm_button(self) -> None:
        if not self._is_valid_size(self._frame_rect):
            self._remove_confirm_button()
            return
        self._show_confirm_button()

    def _show_confirm_button(self) -> None:
        if not self._is_valid_size(self._frame_rect):
            self._remove_confirm_button()
            return

        ready = self._readiness.ready if self._readiness is not None else False

        if self._confirm_btn is None:
            btn = QPushButton("✓ 캡처 · 지도 분석", self)
            btn.clicked.connect(self._on_confirm)
            self._confirm_btn = btn
        else:
            btn = self._confirm_btn

        if ready:
            btn.setText("✓ 캡처 · 지도 분석")
            btn.setEnabled(True)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #43A047;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #2E7D32; }
                """
            )
        else:
            btn.setText("맞춤 필요")
            btn.setEnabled(False)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #616161;
                    color: #E0E0E0;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                """
            )

        btn_w = 156 if ready else 120
        btn_h = 36
        x = self._frame_rect.right() - btn_w
        y = self._frame_rect.bottom() + 8
        if y + btn_h > self.height():
            y = self._frame_rect.top() - btn_h - 8
        btn.setGeometry(max(8, x), max(8, y), btn_w, btn_h)
        btn.show()
        btn.raise_()

    def _remove_confirm_button(self) -> None:
        if self._confirm_btn:
            self._confirm_btn.hide()

    def _on_confirm(self) -> None:
        if self._frame_rect.isNull() or not self._is_valid_size(self._frame_rect):
            return
        if self._readiness is not None and not self._readiness.ready:
            return

        global_rect = (
            self._frame_rect.x() + self.x(),
            self._frame_rect.y() + self.y(),
            self._frame_rect.width(),
            self._frame_rect.height(),
        )
        self.hide()
        self._locked = False
        self._select_origin = None
        self._readiness = None
        self._guide_rect = QRect()
        self._banner_target_rect = QRect()
        self._banner_detected_rect = QRect()
        self._quality_timer.stop()
        self._stop_quality_worker()
        self._remove_confirm_button()
        self.confirmed.emit(global_rect)

    def _cancel(self) -> None:
        self.hide()
        self._frame_rect = QRect()
        self._drag_offset = None
        self._resize_handle = None
        self._select_origin = None
        self._locked = False
        self._readiness = None
        self._guide_rect = QRect()
        self._banner_target_rect = QRect()
        self._banner_detected_rect = QRect()
        self._quality_timer.stop()
        self._stop_quality_worker()
        self._remove_confirm_button()
        self.cancelled.emit()
