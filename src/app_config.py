"""앱 공통 설정 — 이름, 로깅, 디버그 플래그"""

from __future__ import annotations

import logging
import os
import sys

APP_NAME = "여기맞나?"
APP_SLUG = "yeogimajna"


def is_debug() -> bool:
    """TR_DEBUG=1 이면 개발용 상세 로그·디버그 이미지 저장"""
    return os.environ.get("TR_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _resolve_log_level() -> int:
    """로그 레벨: TR_DEBUG 명시 > 소스 실행(터미널) > 배포 exe 기본 WARNING"""
    raw = os.environ.get("TR_DEBUG", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return logging.DEBUG
    if raw in ("0", "false", "no"):
        return logging.WARNING
    # PyInstaller exe(console=False)는 터미널 없음 → 조용히
    if getattr(sys, "frozen", False):
        return logging.WARNING
    # python main.py 등 소스 실행 시 timing 등 debug 로그를 터미널에 출력
    return logging.DEBUG


def is_debug_party() -> bool:
    """TR_DEBUG_PARTY=1 이면 party OCR 입력 이미지를 debug_party/에 저장"""
    return os.environ.get("TR_DEBUG_PARTY", "").strip().lower() in ("1", "true", "yes")


def is_debug_query() -> bool:
    """TR_DEBUG_QUERY=1 이면 match()마다 debug_query.png 저장"""
    if os.environ.get("TR_DEBUG_QUERY", "").strip() == "1":
        return True
    return is_debug()


def setup_logging() -> None:
    """배포 exe: WARNING. 소스 실행·TR_DEBUG=1: DEBUG."""
    level = _resolve_log_level()
    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s %(message)s",
        force=True,
    )
    noisy = (
        "PIL",
        "PIL.PngImagePlugin",
        "pytesseract",
    )
    cap = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    for name in noisy:
        logging.getLogger(name).setLevel(cap)


def configure_qt_app(app) -> None:
    """QApplication 메타데이터"""
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                f"FFXIV.{APP_SLUG}"
            )
        except (AttributeError, OSError):
            pass
