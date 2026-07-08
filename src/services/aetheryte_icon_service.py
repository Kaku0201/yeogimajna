from pathlib import Path

from PIL import Image

from src.services.app_paths import get_app_root


class AetheryteIconService:
    """에테라이트 아이콘 PNG 경로 탐색"""

    ICON_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")

    def __init__(self, assets_dir: Path | None = None) -> None:
        if assets_dir is None:
            assets_dir = get_app_root() / "assets"
        self.assets_dir = assets_dir
        self._search_dirs = (
            assets_dir / "aetherytes",
            assets_dir / "Aetheryte",
            assets_dir,
        )
        self._default_icons = (
            assets_dir / "aetherytes" / "Aetheryte.png",
            assets_dir / "Aetheryte.png",
        )

    def resolve_icon_path(
        self, name_ko: str, icon_hint: str | None = None
    ) -> Path | None:
        """에테라이트 이름 → 아이콘 파일 경로"""
        candidates: list[str] = []
        if icon_hint:
            candidates.append(icon_hint)
        candidates.append(name_ko)

        for name in candidates:
            if not name:
                continue
            path = self._find_named_icon(name)
            if path is not None:
                return path

        if self._default_icon.exists():
            return self._default_icon
        return None

    @property
    def _default_icon(self) -> Path:
        for path in self._default_icons:
            if path.is_file():
                return path
        return self._default_icons[-1]

    def _find_named_icon(self, name: str) -> Path | None:
        stem = Path(name).stem
        for folder in self._search_dirs:
            if not folder.exists():
                continue
            for ext in self.ICON_EXTENSIONS:
                path = folder / f"{stem}{ext}"
                if path.is_file():
                    return path
        return None

    @staticmethod
    def load_qpixmap(path: Path, size: int):
        """PyQt QPixmap 로드 (지연 import)"""
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPixmap

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return QPixmap()
        return pixmap.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    @staticmethod
    def tint_for_overlay(image: Image.Image, size: int) -> Image.Image:
        """상세 지도 오버레이용 리사이즈"""
        rgba = image.convert("RGBA")
        return rgba.resize((size, size), Image.Resampling.LANCZOS)
