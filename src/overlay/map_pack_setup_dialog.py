"""첫 실행 시 지도 데이터 팩 확인·다운로드"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.app_config import APP_NAME
from src.services.app_paths import get_app_icon_path
from src.services.map_pack_service import MapPackService


class _DownloadWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, map_pack: MapPackService) -> None:
        super().__init__()
        self.map_pack = map_pack

    def run(self) -> None:
        try:
            self.map_pack.download_and_install(
                progress=lambda cur, total, msg: self.progress.emit(cur, total, msg)
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MapPackSetupDialog(QDialog):
    """지도 팩이 없을 때 표시 — 사용자는 PNG를 직접 넣지 않음"""

    def __init__(self, map_pack: MapPackService, parent=None) -> None:
        super().__init__(parent)
        self.map_pack = map_pack
        self._worker: _DownloadWorker | None = None
        self._setup_ui()
        self._refresh_status()

    def _setup_ui(self) -> None:
        self.setWindowTitle(f"{APP_NAME} — 지도 데이터")
        self.setMinimumWidth(420)
        icon_path = get_app_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        layout = QVBoxLayout(self)

        title = QLabel("🗺️ 지도 데이터 확인")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.progress_msg = QLabel()
        self.progress_msg.setVisible(False)
        layout.addWidget(self.progress_msg)

        self.download_btn = QPushButton("지도 데이터 다운로드")
        self.download_btn.clicked.connect(self._start_download)
        layout.addWidget(self.download_btn)

        buttons = QDialogButtonBox()
        self.retry_btn = buttons.addButton(
            "다시 확인", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.start_btn = buttons.addButton(
            "시작", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.quit_btn = buttons.addButton(
            "종료", QDialogButtonBox.ButtonRole.RejectRole
        )
        self.retry_btn.clicked.connect(self._refresh_status)
        self.start_btn.clicked.connect(self.accept)
        self.quit_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_status(self) -> None:
        status = self.map_pack.get_status()
        count_line = f"• 지역 지도: {status.detail_count}장"
        if status.match_count:
            count_line += f" | match: {status.match_count}장"
        lines = [
            status.message,
            "",
            f"• 버전: {status.version}",
            f"• 출처: {status.source}",
            count_line,
        ]

        if not status.ready:
            lines.extend(
                [
                    "",
                    "일반 사용자는 지도 PNG를 직접 추가할 필요가 없습니다.",
                    "배포용 설치 파일에 지도 팩이 포함되어 있어야 합니다.",
                ]
            )
            if self.map_pack.download_url:
                lines.append("또는 아래 버튼으로 자동 다운로드할 수 있습니다.")
            else:
                lines.append("개발자에게 전체 설치 파일을 받으세요.")

        self.status_label.setText("\n".join(lines))
        self.start_btn.setEnabled(status.ready)
        self.download_btn.setEnabled(
            not status.ready and bool(self.map_pack.download_url)
        )
        self.download_btn.setVisible(bool(self.map_pack.download_url))

    def _start_download(self) -> None:
        if not self.map_pack.download_url:
            return

        self.download_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress_msg.setVisible(True)
        self.progress.setValue(0)

        self._worker = _DownloadWorker(self.map_pack)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_download_ok)
        self._worker.failed.connect(self._on_download_failed)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.setValue(current)
        self.progress_msg.setText(message)

    def _on_download_ok(self) -> None:
        self.progress_msg.setText("설치 완료")
        self.retry_btn.setEnabled(True)
        self._refresh_status()
        if self.map_pack.is_ready():
            QMessageBox.information(
                self,
                "준비 완료",
                "지도 데이터 설치가 완료되었습니다.\n「시작」을 눌러 사용하세요.",
            )

    def _on_download_failed(self, message: str) -> None:
        self.progress_msg.setText("")
        self.download_btn.setEnabled(True)
        self.retry_btn.setEnabled(True)
        QMessageBox.warning(
            self,
            "다운로드 실패",
            f"지도 데이터를 받지 못했습니다.\n\n{message}",
        )
        self._refresh_status()
