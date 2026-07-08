"""사용자 설정·다운로드 지도 팩 저장 위치"""

import sys
from pathlib import Path

from src.app_config import APP_SLUG


def get_app_root() -> Path:
    """번들 리소스 루트 (개발: 프로젝트 루트, PyInstaller: _MEIPASS)"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def get_user_data_dir() -> Path:
    """사용자 데이터 (~/.yeogimajna). 구버전 ~/.tr_overlay 호환."""
    preferred = Path.home() / f".{APP_SLUG}"
    legacy = Path.home() / ".tr_overlay"
    if not preferred.exists() and legacy.exists():
        return legacy
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred


def get_app_icon_png_path() -> Path | None:
    """UI 렌더링용 고해상도 PNG"""
    path = get_app_root() / "assets" / "waymarks.png"
    return path if path.is_file() else None


def get_app_icon_path() -> Path | None:
    """창/exe 아이콘 — Windows .ico 우선"""
    assets = get_app_root() / "assets"
    for name in ("waymarks.ico", "waymarks.png"):
        path = assets / name
        if path.is_file():
            return path
    return None
