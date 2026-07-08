"""매칭 경로 통계 다이얼로그"""

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout

from src.services.match_stats import MatchStatsService


class MatchStatsDialog(QDialog):
    def __init__(self, stats: MatchStatsService, parent=None) -> None:
        super().__init__(parent)
        self.stats = stats
        self.setWindowTitle("매칭 통계")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        title = QLabel("ref DB vs 상세지도 — 인식 경로 누적")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setStyleSheet("color: #dddddd; line-height: 1.4;")
        layout.addWidget(self.body)

        reset_btn = QPushButton("통계 초기화")
        reset_btn.clicked.connect(self._reset)
        layout.addWidget(reset_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._refresh()

    def _refresh(self) -> None:
        self.body.setText("\n".join(self.stats.summary_lines()))

    def _reset(self) -> None:
        self.stats.clear()
        self._refresh()
