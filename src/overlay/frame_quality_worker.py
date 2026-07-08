import os
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from PIL import Image

from src.services.treasure_capture import CaptureReadiness, TreasureCaptureProcessor


class FrameQualityWorker(QThread):
    """드래그 프레임 실시간 품질 검사 — UI 스레드 블로킹 방지"""

    finished_check = pyqtSignal(object, tuple)

    def __init__(
        self,
        processor: TreasureCaptureProcessor,
        rect: tuple[int, int, int, int],
        *,
        capture_fn=None,
        shot: Image.Image | None = None,
        bare_capture: bool = False,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.capture_fn = capture_fn
        self.rect = rect
        self.shot = shot
        self.bare_capture = bare_capture

    def run(self) -> None:
        try:
            if self.shot is not None:
                shot = self.shot
            elif self.capture_fn is not None:
                shot = self.capture_fn(self.rect)
            else:
                raise ValueError("capture_fn 또는 shot이 필요합니다")

            debug_dir = os.environ.get("TR_QC_DEBUG_DIR")
            if debug_dir:
                out = Path(debug_dir)
                out.mkdir(parents=True, exist_ok=True)
                shot.save(out / f"qc_{self.rect[0]}_{self.rect[1]}.png")

            readiness = self.processor.assess_capture_readiness(
                shot, bare_capture=self.bare_capture
            )
            self.finished_check.emit(readiness, self.rect)
        except Exception:
            readiness = CaptureReadiness(
                score=0,
                max_score=2,
                ready=False,
                hint="화면 캡처에 실패했습니다",
                has_marker=False,
                has_parchment=False,
                has_banner=False,
                has_banner_aligned=False,
                has_party_icon=False,
            )
            self.finished_check.emit(readiness, self.rect)
