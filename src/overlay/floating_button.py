import sys
from pathlib import Path

from PyQt6.QtCore import QPoint, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCursor, QGuiApplication, QIcon, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from src.models.recognition_result import RecognitionResult
from src.models.ref_candidate import RefCandidate
from src.app_config import configure_qt_app
from src.services.app_paths import get_app_icon_path, get_app_icon_png_path, get_app_icon_path
from src.overlay.analyze_worker import AnalyzeWorker, RematchWorker
from src.overlay.manual_input_dialog import ManualInputDialog
from src.overlay.map_detail_dialog import MapDetailDialog
from src.overlay.recognition_box import RecognitionBox
from src.overlay.result_panel import ResultPanel
from src.overlay.map_pack_setup_dialog import MapPackSetupDialog
from src.overlay.match_stats_dialog import MatchStatsDialog
from src.services.coordinate_service import CoordinateService
from src.services.map_analyzer import MapAnalyzer
from src.services.map_pack_service import MapPackService
from src.services.settings_service import SettingsService
from src.services.tesseract_config import configure_tesseract


class FloatingButton(QWidget):
    """항상 위에 표시되는 플로팅 버튼 오버레이"""

    exit_requested = pyqtSignal()

    BUTTON_SIZE = 52
    @classmethod
    def icon_path(cls) -> Path | None:
        return get_app_icon_png_path()

    def __init__(
        self,
        settings: SettingsService,
        analyzer: MapAnalyzer,
        coordinate_service: CoordinateService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.analyzer = analyzer
        self.coordinate_service = coordinate_service
        self._drag_pos: QPoint | None = None
        self._btn_press_pos: QPoint | None = None
        self._current_result: RecognitionResult | None = None
        self._detail_dialog: MapDetailDialog | None = None
        self._analyze_worker: AnalyzeWorker | None = None
        self._rematch_worker: RematchWorker | None = None
        self._analyzing = False

        self.recognition_box = RecognitionBox(
            settings,
            capture_processor=analyzer.capture_processor,
            capture_fn=analyzer.capture_region,
        )
        self.recognition_box.confirmed.connect(self._on_region_confirmed)
        self.recognition_box.cancelled.connect(self._on_region_cancelled)

        self._setup_window()
        self._apply_window_icon()
        self._setup_ui()
        self._apply_settings()

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(self.BUTTON_SIZE)

        bx, by = self.settings.button_position
        self.move(bx, by)

    def _apply_window_icon(self) -> None:
        icon_path = self.icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.result_panel = ResultPanel()
        self.result_panel.coordinate_clicked.connect(self._show_map_detail)
        self.result_panel.candidate_clicked.connect(self._show_candidate_detail)
        self.result_panel.candidate_confirmed.connect(self._confirm_candidate)
        self.result_panel.rematch_requested.connect(self._on_rematch_requested)
        self.result_panel.delete_requested.connect(self._clear_result)
        self.result_panel.learn_reset_requested.connect(self._reset_learned_data)
        root.addWidget(self.result_panel)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self.btn = QLabel()
        self.btn.setFixedSize(self.BUTTON_SIZE, self.BUTTON_SIZE)
        self.btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn.setPixmap(self._load_button_icon())
        self.btn.setStyleSheet("background: transparent;")
        self.btn.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        row.addWidget(self.btn, alignment=Qt.AlignmentFlag.AlignLeft)
        row.addStretch()
        root.addLayout(row)

        self.adjustSize()
        self._update_widget_width()

    def _apply_settings(self) -> None:
        self.setWindowOpacity(1.0)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, False)
        self.result_panel.set_background_opacity(
            self.settings.result_panel_bg_opacity
        )
        self.show()

    def _update_widget_width(self) -> None:
        if self.result_panel.isVisible():
            panel_w = self.result_panel.preferred_width()
            total_w = max(self.BUTTON_SIZE + panel_w + 12, self.sizeHint().width())
            self.setMinimumWidth(total_w)
            self.setMaximumWidth(16777215)
            self.adjustSize()
            self.setFixedWidth(total_w)
        else:
            self.setMinimumWidth(self.BUTTON_SIZE)
            self.setMaximumWidth(self.BUTTON_SIZE)
            self.setFixedWidth(self.BUTTON_SIZE)

    def _show_candidate_detail(self, candidate: RefCandidate) -> None:
        if self._analyzing or self._current_result is None:
            return
        zone = self.coordinate_service.get_effective_zone(
            self._current_result.zone_id
        )
        if zone is None:
            zone = {
                "id": self._current_result.zone_id,
                "name_ko": self._current_result.zone_name,
            }
        result = self.coordinate_service.build_result(zone, candidate.x, candidate.y)
        result.match_score = candidate.score
        result.ref_candidates = self._current_result.ref_candidates
        result.auto_candidate_rank = self._current_result.auto_candidate_rank
        result.confirmed_ref_name = self._current_result.confirmed_ref_name
        result.learn_hits = self._current_result.learn_hits
        already_confirmed = (
            self._current_result.confirmed_ref_name == candidate.ref_name
        )
        self._open_detail_dialog(
            result,
            None if already_confirmed else candidate,
        )

    def _open_detail_dialog(
        self,
        result: RecognitionResult,
        confirm_candidate: RefCandidate | None,
    ) -> None:
        try:
            if self._detail_dialog is not None and self._detail_dialog.isVisible():
                self._detail_dialog.close()

            self._detail_dialog = MapDetailDialog(
                result,
                self.coordinate_service,
                self.settings,
                None,
                show_confirm=confirm_candidate is not None,
            )
            if confirm_candidate is not None:
                self._detail_dialog.confirmed.connect(
                    lambda: self._confirm_candidate(confirm_candidate)
                )
            self._detail_dialog.finished.connect(self._on_detail_closed)
            self._detail_dialog.show()
            self._detail_dialog.raise_()
            self._detail_dialog.activateWindow()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "오류",
                f"상세 지도를 열 수 없습니다.\n{exc}",
            )

    def _confirm_candidate(self, candidate: RefCandidate) -> None:
        if self._analyzing or self._current_result is None:
            return
        try:
            result, hits = self.analyzer.confirm_user_ref(candidate)
            result.ref_candidates = self._current_result.ref_candidates
            result.excluded_ref_names = self._current_result.excluded_ref_names
            result.can_rematch = self._current_result.can_rematch
            result.confirmed_ref_name = candidate.ref_name
            result.learn_hits = hits
            self._show_result(result)
            if self._detail_dialog is not None and self._detail_dialog.isVisible():
                self._detail_dialog.close()
        except ValueError as exc:
            QMessageBox.warning(self, "확정 실패", str(exc))
        except Exception as exc:
            QMessageBox.critical(
                self,
                "오류",
                f"확정 저장 중 오류가 발생했습니다.\n{exc}",
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._analyzing:
            event.accept()
            return

        if self.result_panel.isVisible() and self._current_result is None:
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton and self._is_on_button(
            event.pos()
        ):
            self._btn_press_pos = event.globalPosition().toPoint()
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            self.btn.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_on_button(event.pos()):
                self._start_recognition()
            else:
                self._drag_pos = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                self.btn.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is None:
            return
        new_pos = event.globalPosition().toPoint() - self._drag_pos
        self.move(new_pos)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.RightButton
            and self._btn_press_pos is not None
        ):
            moved = (
                event.globalPosition().toPoint() - self._btn_press_pos
            ).manhattanLength()
            if moved < 8:
                self._show_context_menu(event.globalPosition().toPoint())
            self._btn_press_pos = None

        if self._drag_pos is not None:
            self._drag_pos = None
            self.btn.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            self.settings.button_position = (self.x(), self.y())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _is_on_button(self, pos: QPoint) -> bool:
        return self.btn.geometry().contains(pos)

    @classmethod
    def _load_button_icon(cls) -> QPixmap:
        icon_path = cls.icon_path()
        if icon_path is None:
            return QPixmap()
        pixmap = QPixmap(str(icon_path))
        if pixmap.isNull():
            return QPixmap()

        dpr = 1.0
        app = QApplication.instance()
        if app is not None:
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                dpr = max(1.0, screen.devicePixelRatio())

        target = max(1, int(cls.BUTTON_SIZE * dpr))
        scaled = pixmap.scaled(
            target,
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        return scaled

    def _start_recognition(self) -> None:
        if self._analyzing:
            return
        self.recognition_box.start_selection()

    def _on_region_confirmed(self, rect: tuple[int, int, int, int]) -> None:
        if self._analyzing:
            return

        self.settings.last_capture_rect = rect
        self._set_analyzing(True)
        self.result_panel.show_analyzing()
        self._update_widget_width()

        # 오버레이 hide() 직후 캡처하면 안내 문구가 섞일 수 있어 잠시 대기
        QTimer.singleShot(120, lambda: self._start_analyze_worker(rect))

    def _start_analyze_worker(self, rect: tuple[int, int, int, int]) -> None:
        if self._analyzing is False:
            return
        worker = AnalyzeWorker(self.analyzer, rect)
        worker.finished_ok.connect(self._on_analyze_ok)
        worker.finished_error.connect(self._on_analyze_error)
        worker.finished.connect(self._on_analyze_finished)
        self._analyze_worker = worker
        worker.start()

    def _set_analyzing(self, active: bool) -> None:
        self._analyzing = active
        self.setCursor(
            QCursor(Qt.CursorShape.WaitCursor)
            if active
            else QCursor(Qt.CursorShape.OpenHandCursor)
        )

    def _on_analyze_ok(self, result: RecognitionResult) -> None:
        self._show_result(result)

    def _on_analyze_error(self, message: str, is_value_error: bool) -> None:
        self.result_panel.clear_result()
        self._update_widget_width()

        if is_value_error:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("인식 실패")
            msg.setText(message)
            msg.setInformativeText(
                "네모를 보물지도 양피지 창에만 맞추세요 (소지품 창 제외).\n"
                "상단 지역명·빨간 X·하단 1/8 아이콘이 보이게 한 뒤 ✓ 버튼을 누르세요."
            )
            retry_btn = msg.addButton(
                "다시 캡처",
                QMessageBox.ButtonRole.AcceptRole,
            )
            msg.addButton(QMessageBox.StandardButton.Close)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == retry_btn:
                self._start_recognition()
        else:
            QMessageBox.critical(
                self,
                "오류",
                f"지도 분석 중 오류가 발생했습니다.\n{message}",
            )

    def _on_analyze_finished(self) -> None:
        self._set_analyzing(False)
        self._analyze_worker = None

    def _on_rematch_requested(self) -> None:
        if self._analyzing or self._current_result is None:
            return
        if not self._current_result.can_rematch:
            return

        self._set_analyzing(True)
        self.result_panel.set_rematch_enabled(False)
        self.result_panel.coord_label.setText("⏳ 다른 후보 검색 중...")

        worker = RematchWorker(
            self.analyzer,
            list(self._current_result.excluded_ref_names),
        )
        worker.finished_ok.connect(self._on_analyze_ok)
        worker.finished_error.connect(self._on_rematch_error)
        worker.finished.connect(self._on_rematch_finished)
        self._rematch_worker = worker
        worker.start()

    def _on_rematch_error(self, message: str, is_value_error: bool) -> None:
        if self._current_result is not None:
            self.result_panel.show_result(self._current_result)
        if is_value_error:
            QMessageBox.information(self, "재검색", message)
        else:
            QMessageBox.critical(
                self,
                "오류",
                f"재검색 중 오류가 발생했습니다.\n{message}",
            )

    def _on_rematch_finished(self) -> None:
        self._set_analyzing(False)
        self._rematch_worker = None

    def _manual_input_fallback(self) -> None:
        dialog = ManualInputDialog(self.coordinate_service, self)
        values = dialog.get_values()
        if values is None:
            return
        zone, x, y = values
        result = self.coordinate_service.build_result(zone, x, y)
        self._show_result(result)

    def _on_region_cancelled(self) -> None:
        pass

    def _show_result(self, result: RecognitionResult) -> None:
        self.analyzer.match_stats.record(result.match_source, result.zone_id)
        stats_hint = self.analyzer.match_stats.last_source_hint(
            result.match_source
        )
        self._current_result = result
        self.result_panel.show_result(result, stats_hint=stats_hint)
        self._update_widget_width()

    def _clear_result(self) -> None:
        self._current_result = None
        self.result_panel.clear_result()
        self._update_widget_width()

    def _show_map_detail(self, result: RecognitionResult) -> None:
        if self._analyzing or result is None:
            return
        self._open_detail_dialog(result, confirm_candidate=None)

    def _on_detail_closed(self) -> None:
        self._detail_dialog = None

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #2b2b2b;
                color: #eee;
                border: 1px solid #555;
            }
            QMenu::item:selected { background-color: #2196F3; }
            """
        )

        bg_menu = menu.addMenu("🔆 지도 위치 배경")
        for label, value in [
            ("0%", 0.0),
            ("25%", 0.25),
            ("50%", 0.5),
            ("75%", 0.75),
            ("100%", 1.0),
        ]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(
                abs(self.settings.result_panel_bg_opacity - value) < 0.01
            )
            action.triggered.connect(
                lambda _=False, v=value: self._set_result_panel_bg_opacity(v)
            )
            bg_menu.addAction(action)

        detail_bg_menu = menu.addMenu("🔆 상세 지도 배경")
        for label, value in [
            ("0%", 0.0),
            ("25%", 0.25),
            ("50%", 0.5),
            ("75%", 0.75),
            ("100%", 1.0),
        ]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(
                abs(self.settings.detail_map_bg_opacity - value) < 0.01
            )
            action.triggered.connect(
                lambda _=False, v=value: self._set_detail_map_bg_opacity(v)
            )
            detail_bg_menu.addAction(action)

        menu.addSeparator()
        stats_action = QAction("📊 매칭 통계", self)
        stats_action.triggered.connect(self._show_match_stats)
        menu.addAction(stats_action)
        exit_action = QAction("❌ 종료", self)
        exit_action.triggered.connect(self.exit_requested.emit)
        menu.addAction(exit_action)

        menu.exec(global_pos)

    def _set_result_panel_bg_opacity(self, opacity: float) -> None:
        self.settings.result_panel_bg_opacity = opacity
        self.result_panel.set_background_opacity(opacity)

    def _set_detail_map_bg_opacity(self, opacity: float) -> None:
        self.settings.detail_map_bg_opacity = opacity
        if self._detail_dialog is not None and self._detail_dialog.isVisible():
            self._detail_dialog.set_background_opacity(opacity)

    def _show_match_stats(self) -> None:
        dialog = MatchStatsDialog(self.analyzer.match_stats, self)
        dialog.exec()

    def _reset_learned_data(self) -> None:
        reply = QMessageBox.question(
            self,
            "학습 데이터 초기화",
            "확정(✓)으로 저장한 지도 학습을 모두 지웁니다.\n"
            "zip을 새로 풀어도 이 데이터는 PC에 남아 있었을 수 있습니다.\n\n"
            "초기화할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.analyzer.user_learn.clear_all()
        self.analyzer._party_size_by_zone.clear()
        self._clear_result()
        QMessageBox.information(
            self,
            "완료",
            "학습 데이터를 초기화했습니다.\n"
            "다시 확정(✓)하면 1회부터 학습이 시작됩니다.",
        )


def run_overlay() -> int:
    app = QApplication(sys.argv)
    configure_qt_app(app)
    app.setQuitOnLastWindowClosed(True)
    icon_path = get_app_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    configure_tesseract()

    map_pack = MapPackService()
    if not map_pack.is_ready():
        setup = MapPackSetupDialog(map_pack)
        if setup.exec() != MapPackSetupDialog.DialogCode.Accepted:
            return 1
        if not map_pack.is_ready():
            return 1

    settings = SettingsService()
    coordinate_service = CoordinateService(maps_dir=map_pack.maps_dir)
    analyzer = MapAnalyzer(coordinate_service, map_pack=map_pack)

    overlay = FloatingButton(settings, analyzer, coordinate_service)
    overlay.show()

    overlay.exit_requested.connect(app.quit)

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_overlay())
