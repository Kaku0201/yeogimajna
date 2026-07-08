import json
from pathlib import Path
from typing import Any

from src.services.app_paths import get_user_data_dir


class SettingsService:
    """오버레이 설정 저장/로드"""

    DEFAULTS: dict[str, Any] = {
        "button_x": 100,
        "button_y": 100,
        "opacity": 0.92,
        "result_panel_bg_opacity": 0.5,
        "detail_map_bg_opacity": 0.5,
        "last_capture_rect": None,
    }

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = get_user_data_dir() / "settings.json"
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return dict(self.DEFAULTS)
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(self.DEFAULTS)
            merged.update(data)
            return merged
        except (json.JSONDecodeError, OSError):
            return dict(self.DEFAULTS)

    def save(self) -> None:
        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self._settings, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise RuntimeError(f"설정 저장 실패: {exc}") from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._settings[key] = value
        self.save()

    @property
    def button_position(self) -> tuple[int, int]:
        return int(self.get("button_x", 100)), int(self.get("button_y", 100))

    @button_position.setter
    def button_position(self, pos: tuple[int, int]) -> None:
        self.set("button_x", pos[0])
        self.set("button_y", pos[1])

    @property
    def opacity(self) -> float:
        return float(self.get("opacity", 0.92))

    @opacity.setter
    def opacity(self, value: float) -> None:
        self.set("opacity", max(0.3, min(1.0, value)))

    @property
    def result_panel_bg_opacity(self) -> float:
        return float(self.get("result_panel_bg_opacity", 0.5))

    @result_panel_bg_opacity.setter
    def result_panel_bg_opacity(self, value: float) -> None:
        self.set("result_panel_bg_opacity", max(0.0, min(1.0, value)))

    @property
    def detail_map_bg_opacity(self) -> float:
        return float(self.get("detail_map_bg_opacity", 0.5))

    @detail_map_bg_opacity.setter
    def detail_map_bg_opacity(self, value: float) -> None:
        self.set("detail_map_bg_opacity", max(0.0, min(1.0, value)))

    @property
    def last_capture_rect(self) -> tuple[int, int, int, int] | None:
        rect = self.get("last_capture_rect")
        if rect and len(rect) == 4:
            return tuple(rect)
        return None

    @last_capture_rect.setter
    def last_capture_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        self.set("last_capture_rect", list(rect) if rect else None)
