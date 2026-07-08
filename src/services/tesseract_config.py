"""Tesseract OCR 경로 자동 설정 (Windows 등)"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.services.app_paths import get_app_root


def get_bundled_tesseract_dir() -> Path | None:
    """앱에 포함된 Tesseract (vendor/tesseract)"""
    root = get_app_root() / "vendor" / "tesseract"
    if (root / "tesseract.exe").is_file():
        return root
    return None


def get_bundled_tesseract_dll_dir() -> Path | None:
    """tesserocr 등 DLL 로딩용 — libtesseract/leptonica 포함 폴더"""
    bundled = get_bundled_tesseract_dir()
    if bundled is None:
        return None
    for name in ("libtesseract-5.dll", "libtesseract-4.dll", "tesseract50.dll"):
        if (bundled / name).is_file():
            return bundled
    # UB Mannheim 배포는 tesseract.exe만 있고 DLL 이름이 다를 수 있음
    if any(bundled.glob("*.dll")):
        return bundled
    return None


def prepare_tesseract_dll_path() -> None:
    """Windows: tesserocr·libtesseract DLL 검색 경로 등록"""
    if sys.platform != "win32":
        return

    dirs: list[Path] = []
    bundled = get_bundled_tesseract_dll_dir()
    if bundled is not None:
        dirs.append(bundled)

    for path in (
        Path(r"C:\Program Files\Tesseract-OCR"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR"),
    ):
        if path.is_dir():
            dirs.append(path)

    seen: set[str] = set()
    for directory in dirs:
        key = str(directory.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            os.add_dll_directory(key)
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")


def get_bundled_tessdata_dir() -> Path | None:
    """번들 tessdata 폴더"""
    bundled = get_bundled_tesseract_dir()
    if bundled is not None:
        tessdata = bundled / "tessdata"
        if (tessdata / "kor.traineddata").is_file():
            return tessdata

    legacy = get_app_root() / "data" / "tessdata"
    if (legacy / "kor.traineddata").is_file():
        return legacy
    return None


def configure_tesseract() -> bool:
    """
    pytesseract fallback용 — tesseract 실행 파일·tessdata 경로 설정.
    번들(vendor/tesseract) → 환경변수 → 시스템 설치 순.
    """
    prepare_tesseract_dll_path()
    try:
        import pytesseract
    except ImportError:
        return False

    if getattr(configure_tesseract, "_configured", False):
        return _probe_tesseract()

    candidates: list[Path] = []

    bundled = get_bundled_tesseract_dir()
    if bundled is not None:
        candidates.append(bundled / "tesseract.exe")

    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd:
        candidates.append(Path(env_cmd))

    if sys.platform == "win32":
        candidates.extend(
            [
                Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            ]
        )

    for path in candidates:
        if path.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            break

    tessdata = get_bundled_tessdata_dir()
    if tessdata is not None:
        os.environ["TESSDATA_PREFIX"] = str(tessdata)
    elif os.environ.get("TESSDATA_PREFIX"):
        prefix = Path(os.environ["TESSDATA_PREFIX"])
        if not (prefix / "kor.traineddata").is_file() and not (
            prefix / "tessdata" / "kor.traineddata"
        ).is_file():
            os.environ.pop("TESSDATA_PREFIX", None)

    configure_tesseract._configured = True
    return _probe_tesseract()


def _probe_tesseract() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def get_tessdata_dir() -> Path | None:
    """tessdata 폴더 경로"""
    bundled = get_bundled_tessdata_dir()
    if bundled is not None:
        return bundled

    try:
        import pytesseract

        cmd = Path(pytesseract.pytesseract.tesseract_cmd)
        if cmd.is_file():
            return cmd.parent / "tessdata"
    except Exception:
        pass

    for path in (
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
    ):
        if path.is_dir():
            return path
    return None


def has_korean_language() -> bool:
    prefix = os.environ.get("TESSDATA_PREFIX", "")
    if prefix:
        tessdata = Path(prefix)
        if (tessdata / "kor.traineddata").is_file():
            return True
        nested = tessdata / "tessdata" / "kor.traineddata"
        if nested.is_file():
            return True

    tessdata = get_tessdata_dir()
    if tessdata is None:
        return False
    return (tessdata / "kor.traineddata").is_file() or (
        tessdata / "kor_vert.traineddata"
    ).is_file()
