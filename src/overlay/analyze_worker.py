from PyQt6.QtCore import QThread, pyqtSignal

from src.services.map_analyzer import MapAnalyzer


class AnalyzeWorker(QThread):
    """보물지도 분석 — UI 스레드 블로킹 방지"""

    finished_ok = pyqtSignal(object)
    finished_error = pyqtSignal(str, bool)

    def __init__(
        self,
        analyzer: MapAnalyzer,
        rect: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.rect = rect

    def run(self) -> None:
        try:
            result = self.analyzer.analyze(self.rect)
            self.finished_ok.emit(result)
        except ValueError as exc:
            self.finished_error.emit(str(exc), True)
        except Exception as exc:
            self.finished_error.emit(str(exc), False)


class RematchWorker(QThread):
    """동일 캡처로 ref 후보 재검색"""

    finished_ok = pyqtSignal(object)
    finished_error = pyqtSignal(str, bool)

    def __init__(
        self,
        analyzer: MapAnalyzer,
        excluded_ref_names: list[str],
    ) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.excluded_ref_names = excluded_ref_names

    def run(self) -> None:
        try:
            result = self.analyzer.rematch_refs(self.excluded_ref_names)
            self.finished_ok.emit(result)
        except ValueError as exc:
            self.finished_error.emit(str(exc), True)
        except Exception as exc:
            self.finished_error.emit(str(exc), False)
