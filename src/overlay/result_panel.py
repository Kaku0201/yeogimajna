from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QFont, QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.models.recognition_result import RecognitionResult
from src.models.ref_candidate import RefCandidate
from src.app_config import APP_NAME
from src.services.aetheryte_icon_service import AetheryteIconService

# ref 보물지도 썸네일 — 지형 비교용 (224px의 80%)
CANDIDATE_THUMB_SIZE = 179


def _party_label_from_ref(ref_name: str) -> str:
    """ref_name 접두어(solo/·party8/)로 인원수 배지."""
    if ref_name.startswith("solo/"):
        return "1인"
    if ref_name.startswith("party8/"):
        return "8인"
    return ""


def _party_badge_text(party_size: int | None, uncertain: bool = False) -> str:
    """결과 패널용 인원 배지."""
    if party_size == 1:
        return "1인" + ("?" if uncertain else "")
    if party_size == 8:
        return "8인" + ("?" if uncertain else "")
    return ""


class _CandidateTile(QFrame):
    """ref DB 후보 1칸 — 클릭 시 상세 지도, 확정 시 학습"""

    clicked = pyqtSignal(object)
    confirmed = pyqtSignal(object)

    THUMB_SIZE = CANDIDATE_THUMB_SIZE

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._candidate: RefCandidate | None = None
        self.setObjectName("candidateTile")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.rank_label = QLabel("-")
        self.rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rank_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setScaledContents(False)

        self.coord_label = QLabel("-")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.coord_label.setFont(QFont("Segoe UI", 10))
        self.coord_label.setWordWrap(True)

        self.confirm_btn = QPushButton("✓ 확정")
        self.confirm_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.confirm_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.confirm_btn.clicked.connect(self._on_confirm_clicked)

        layout.addWidget(self.rank_label)
        layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.coord_label)
        layout.addWidget(self.confirm_btn)

        self.hide()

    def _on_confirm_clicked(self) -> None:
        if self._candidate is not None:
            self.confirmed.emit(self._candidate)

    def set_candidate(
        self,
        candidate: RefCandidate,
        *,
        is_auto: bool,
        is_confirmed: bool = False,
        learn_hits: int | None = None,
    ) -> None:
        self._candidate = candidate
        party = _party_label_from_ref(candidate.ref_name)
        self.rank_label.setText(
            f"#{candidate.rank} · {party}" if party else f"#{candidate.rank}"
        )
        self.coord_label.setText(f"{candidate.x:.1f}, {candidate.y:.1f}")

        pixmap = QPixmap(candidate.ref_image_path)
        if pixmap.isNull():
            pixmap = QPixmap(self.THUMB_SIZE, self.THUMB_SIZE)
            pixmap.fill(Qt.GlobalColor.darkGray)
        else:
            pixmap = pixmap.scaled(
                self.THUMB_SIZE,
                self.THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.thumb_label.setPixmap(pixmap)

        if is_confirmed:
            border = "#66bb6a"
            bg = "rgba(102, 187, 106, 55)"
            if learn_hits and learn_hits > 0:
                self.confirm_btn.setText(f"✓ 확정됨 ({learn_hits}회)")
            else:
                self.confirm_btn.setText("✓ 확정됨")
            self.confirm_btn.setEnabled(False)
        else:
            border = "#4fc3f7" if is_auto else "rgba(100, 100, 120, 120)"
            bg = "rgba(79, 195, 247, 40)" if is_auto else "rgba(0, 0, 0, 30)"
            self.confirm_btn.setText("✓ 확정")
            self.confirm_btn.setEnabled(True)

        self.confirm_btn.setStyleSheet(
            """
            QPushButton {
                color: #ffffff;
                background-color: rgba(56, 142, 60, 200);
                border: 1px solid rgba(102, 187, 106, 220);
                border-radius: 4px;
                padding: 3px 8px;
            }
            QPushButton:hover:enabled {
                background-color: rgba(76, 175, 80, 230);
            }
            QPushButton:disabled {
                color: #a5d6a7;
                background-color: rgba(40, 80, 45, 180);
            }
            """
        )
        self.setStyleSheet(
            f"""
            #candidateTile {{
                background-color: {bg};
                border: 2px solid {border};
                border-radius: 6px;
            }}
            QLabel {{
                color: #eeeeee;
                background: transparent;
            }}
            """
        )
        self.show()

    def clear_tile(self) -> None:
        self._candidate = None
        self.thumb_label.clear()
        self.confirm_btn.setEnabled(True)
        self.confirm_btn.setText("✓ 확정")
        self.hide()

    def mousePressEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._candidate is not None
        ):
            self.clicked.emit(self._candidate)
            event.accept()
            return
        super().mousePressEvent(event)


class ResultPanel(QFrame):
    """지도 위치 표시 패널"""

    coordinate_clicked = pyqtSignal(object)
    candidate_clicked = pyqtSignal(object)
    candidate_confirmed = pyqtSignal(object)
    rematch_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    learn_reset_requested = pyqtSignal()

    ICON_SIZE = 32
    FONT_TITLE = 11
    FONT_BODY = 11
    FONT_HINT = 9
    CENTER = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: RecognitionResult | None = None
        self._bg_opacity = 0.5
        self._icon_service = AetheryteIconService()
        self.setObjectName("resultPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._setup_ui()
        self.hide()

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        title = QLabel(f"🗺️ {APP_NAME}")
        title.setFont(QFont("Segoe UI", self.FONT_TITLE, QFont.Weight.Bold))
        title.setAlignment(self.CENTER)
        layout.addWidget(title)

        self.zone_label = QLabel("-")
        self.coord_label = QLabel("-")
        self.coord_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        for label in (self.zone_label, self.coord_label):
            label.setFont(QFont("Segoe UI", self.FONT_BODY))
            label.setWordWrap(True)
            label.setAlignment(self.CENTER)
            layout.addWidget(label)

        ae_row = QHBoxLayout()
        ae_row.setSpacing(6)
        ae_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aetheryte_icon = QLabel()
        self.aetheryte_icon.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
        self.aetheryte_icon.setScaledContents(True)
        self.aetheryte_label = QLabel("-")
        self.aetheryte_label.setFont(QFont("Segoe UI", self.FONT_BODY))
        ae_row.addWidget(self.aetheryte_icon)
        ae_row.addWidget(self.aetheryte_label)
        layout.addLayout(ae_row)

        self.candidates_title = QLabel("후보 (클릭 → 상세 지도)")
        self.candidates_title.setFont(QFont("Segoe UI", self.FONT_HINT))
        self.candidates_title.setAlignment(self.CENTER)
        layout.addWidget(self.candidates_title)

        candidates_row = QHBoxLayout()
        candidates_row.setSpacing(8)
        candidates_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._candidate_tiles = [
            _CandidateTile(self),
            _CandidateTile(self),
            _CandidateTile(self),
        ]
        for tile in self._candidate_tiles:
            tile.clicked.connect(self.candidate_clicked.emit)
            tile.confirmed.connect(self.candidate_confirmed.emit)
            candidates_row.addWidget(tile)
        layout.addLayout(candidates_row)
        self.candidates_title.hide()

        self.rematch_btn = QPushButton("다른 후보 찾기")
        self.rematch_btn.setFont(QFont("Segoe UI", self.FONT_HINT))
        self.rematch_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.rematch_btn.clicked.connect(self.rematch_requested.emit)
        self.rematch_btn.hide()
        layout.addWidget(self.rematch_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.hint_label = QLabel("좌표 클릭 → 상세 지도 | 우클릭 → 삭제")
        self.hint_label.setFont(QFont("Segoe UI", self.FONT_HINT))
        self.hint_label.setAlignment(self.CENTER)
        layout.addWidget(self.hint_label)

        self.stats_hint_label = QLabel("")
        self.stats_hint_label.setFont(QFont("Segoe UI", 8))
        self.stats_hint_label.setAlignment(self.CENTER)
        self.stats_hint_label.setWordWrap(True)
        self.stats_hint_label.hide()
        layout.addWidget(self.stats_hint_label)

        self.learn_reset_btn = QPushButton("학습 데이터 초기화")
        self.learn_reset_btn.setFont(QFont("Segoe UI", self.FONT_HINT))
        self.learn_reset_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.learn_reset_btn.clicked.connect(self.learn_reset_requested.emit)
        self.learn_reset_btn.hide()
        layout.addWidget(self.learn_reset_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.coord_label.installEventFilter(self)
        self.set_background_opacity(self._bg_opacity)

    @classmethod
    def preferred_width(cls) -> int:
        """후보 3열 썸네일이 잘리지 않는 최소 패널 너비"""
        return 3 * CANDIDATE_THUMB_SIZE + 80

    @staticmethod
    def _visible_bg_alpha(opacity: float) -> int:
        """0%일 때도 클릭이 통과되지 않도록 최소 알파 1 유지"""
        return max(1, int(opacity * 255))

    def set_background_opacity(self, opacity: float) -> None:
        """배경·테두리만 투명도 조절 — 글자는 항상 불투명"""
        self._bg_opacity = max(0.0, min(1.0, opacity))
        bg_alpha = self._visible_bg_alpha(self._bg_opacity)
        border_alpha = max(0, int(self._bg_opacity * 180))

        self.setStyleSheet(
            f"""
            #resultPanel {{
                background-color: rgba(30, 30, 35, {bg_alpha});
                border: 1px solid rgba(100, 100, 120, {border_alpha});
                border-radius: 8px;
            }}
            QLabel {{
                color: #eeeeee;
                background: transparent;
            }}
            """
        )
        self.coord_label.setStyleSheet(
            "color: #4fc3f7; text-decoration: underline; background: transparent;"
        )
        self.hint_label.setStyleSheet("color: #888888; background: transparent;")
        self.stats_hint_label.setStyleSheet("color: #777777; background: transparent;")
        self.candidates_title.setStyleSheet(
            "color: #aaaaaa; background: transparent;"
        )
        self.rematch_btn.setStyleSheet(
            """
            QPushButton {
                color: #eeeeee;
                background-color: rgba(60, 60, 70, 200);
                border: 1px solid rgba(120, 120, 140, 180);
                border-radius: 6px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: rgba(79, 195, 247, 80);
                border-color: #4fc3f7;
            }
            QPushButton:disabled {
                color: #666666;
                border-color: rgba(80, 80, 90, 120);
            }
            """
        )
        self.learn_reset_btn.setStyleSheet(
            """
            QPushButton {
                color: #999999;
                background: transparent;
                border: none;
                padding: 2px 6px;
                text-decoration: underline;
            }
            QPushButton:hover {
                color: #ef9a9a;
            }
            """
        )

    def _clear_candidates(self) -> None:
        for tile in self._candidate_tiles:
            tile.clear_tile()
        self.candidates_title.hide()
        self.rematch_btn.hide()

    def set_rematch_enabled(self, enabled: bool) -> None:
        self.rematch_btn.setEnabled(enabled)
        self.rematch_btn.setVisible(enabled or self.rematch_btn.isEnabled())

    def show_analyzing(self) -> None:
        """백그라운드 분석 중 상태 표시"""
        self._result = None
        self.zone_label.setText("⏳ 분석 중...")
        self.coord_label.setText("잠시만 기다려 주세요")
        self.aetheryte_label.setText("")
        self.aetheryte_icon.clear()
        self.aetheryte_icon.hide()
        self._clear_candidates()
        self.hint_label.setText("")
        self.stats_hint_label.hide()
        self.learn_reset_btn.hide()
        self.setMinimumWidth(200)
        self.show()
        self.update()

    def show_result(
        self, result: RecognitionResult, stats_hint: str = ""
    ) -> None:
        self._result = result
        party_badge = _party_badge_text(
            result.party_size, result.party_size_uncertain
        )
        party_suffix = f" · {party_badge}" if party_badge else ""
        if result.confirmed_ref_name and result.learn_hits:
            self.zone_label.setText(
                f"📍 {result.zone_name}{party_suffix}  ✓ 확정 ({result.learn_hits}회 학습)"
            )
        else:
            self.zone_label.setText(f"📍 {result.zone_name}{party_suffix}")

        if result.match_source == "learned":
            hits = result.learn_hits or 1
            self.coord_label.setText(
                f"⚡ 학습 매칭: {result.coordinate_text}\n"
                f"({hits}회 학습·앱 꺼도 유지)"
            )
        elif result.match_source == "ref_tentative":
            self.coord_label.setText(
                f"참고: {result.coordinate_text}\n(후보에서 확인 권장)"
            )
        elif result.match_source == "detail":
            self.coord_label.setText(
                f"좌표: {result.coordinate_text}\n(ref 없음·상세지도 추정)"
            )
        else:
            self.coord_label.setText(f"좌표: {result.coordinate_text}")

        icon_path = self._icon_service.resolve_icon_path(
            result.nearest_aetheryte,
            result.nearest_aetheryte_icon,
        )
        if icon_path:
            pixmap = AetheryteIconService.load_qpixmap(icon_path, self.ICON_SIZE)
            self.aetheryte_icon.setPixmap(pixmap)
            self.aetheryte_icon.show()
        else:
            self.aetheryte_icon.clear()
            self.aetheryte_icon.hide()

        self.aetheryte_label.setText(result.nearest_aetheryte)
        self._show_candidates(result)
        if result.match_source == "learned":
            self.candidates_title.setText("확정 지도 (학습 매칭)")
            self.rematch_btn.hide()
        elif result.ref_candidates:
            self.rematch_btn.setVisible(True)
            self.rematch_btn.setEnabled(result.can_rematch)
        else:
            self.rematch_btn.hide()
        self.hint_label.setText(
            "후보 클릭 → 상세 지도 | 다른 후보 찾기 | 우클릭 → 삭제"
        )
        if stats_hint:
            self.stats_hint_label.setText(stats_hint)
            self.stats_hint_label.show()
        else:
            self.stats_hint_label.hide()
        self.learn_reset_btn.show()
        self.setMinimumWidth(self.preferred_width())
        self.show()
        self.update()

    def _show_candidates(self, result: RecognitionResult) -> None:
        if not result.ref_candidates:
            for tile in self._candidate_tiles:
                tile.clear_tile()
            self.candidates_title.hide()
            self.rematch_btn.hide()
            return

        self.candidates_title.show()
        for index, tile in enumerate(self._candidate_tiles):
            if index >= len(result.ref_candidates):
                tile.clear_tile()
                continue
            candidate = result.ref_candidates[index]
            if not Path(candidate.ref_image_path).is_file():
                tile.clear_tile()
                continue
            is_confirmed = candidate.ref_name == result.confirmed_ref_name
            learn_hits = result.learn_hits if is_confirmed else None
            tile.set_candidate(
                candidate,
                is_auto=candidate.rank == result.auto_candidate_rank,
                is_confirmed=is_confirmed,
                learn_hits=learn_hits,
            )

    def clear_result(self) -> None:
        self._result = None
        self.aetheryte_icon.clear()
        self._clear_candidates()
        self.learn_reset_btn.hide()
        self.stats_hint_label.hide()
        self.hide()

    def eventFilter(self, obj, event) -> bool:
        if (
            obj is self.coord_label
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and self._result is not None
        ):
            self.coordinate_clicked.emit(self._result)
            return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:
        if self._result is None:
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.delete_requested.emit()
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._result is not None
            and self.coord_label.geometry().contains(event.pos())
        ):
            self.coordinate_clicked.emit(self._result)
            event.accept()
            return

        super().mousePressEvent(event)
