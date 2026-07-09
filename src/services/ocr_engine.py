"""Tesseract OCR — tesserocr(인프로세스) 우선, pytesseract(서브프로세스) fallback"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from PIL import Image

from src.services.tesseract_config import (
    configure_tesseract,
    get_tessdata_dir,
    prepare_tesseract_dll_path,
)

logger = logging.getLogger(__name__)

OcrLang = Literal["kor", "eng"]

# --psm N → tesserocr PSM enum 이름
_PSM_NAMES: dict[int, str] = {
    6: "SINGLE_BLOCK",
    7: "SINGLE_LINE",
    8: "SINGLE_WORD",
    10: "SINGLE_CHAR",
    11: "SPARSE_TEXT",
    13: "RAW_LINE",
}


class _OcrBackend(ABC):
    name: str

    @abstractmethod
    def read_text(
        self,
        image: Image.Image,
        *,
        lang: OcrLang,
        psm: int,
        whitelist: str | None = None,
    ) -> str:
        raise NotImplementedError


class _TesserocrBackend(_OcrBackend):
    name = "tesserocr"

    def __init__(self, tessdata: Path) -> None:
        prepare_tesseract_dll_path()
        from tesserocr import PSM, PyTessBaseAPI

        self._psm_enum = PSM
        self._apis: dict[OcrLang, object] = {}
        for lang in ("kor", "eng"):
            trained = tessdata / f"{lang}.traineddata"
            if not trained.is_file():
                if lang == "kor":
                    raise FileNotFoundError(f"kor traineddata 없음: {trained}")
                continue
            self._apis[lang] = PyTessBaseAPI(path=str(tessdata), lang=lang)
        if "eng" not in self._apis:
            raise FileNotFoundError("eng traineddata 없음")

    def _api_for(self, lang: OcrLang):
        return self._apis.get(lang) or self._apis["eng"]

    def read_text(
        self,
        image: Image.Image,
        *,
        lang: OcrLang,
        psm: int,
        whitelist: str | None = None,
    ) -> str:
        api = self._api_for(lang)
        psm_name = _PSM_NAMES.get(psm)
        if psm_name is None:
            raise ValueError(f"지원하지 않는 psm: {psm}")
        try:
            api.SetPageSegMode(getattr(self._psm_enum, psm_name))
            # 이전 호출의 whitelist가 남지 않도록 항상 초기화
            api.SetVariable("tessedit_char_whitelist", whitelist or "")
            api.SetImage(image)
            text = api.GetUTF8Text() or ""
            return text.strip()
        finally:
            api.Clear()
            api.SetVariable("tessedit_char_whitelist", "")


class _PytesseractBackend(_OcrBackend):
    name = "pytesseract"

    def __init__(self) -> None:
        if not configure_tesseract():
            raise RuntimeError("pytesseract 설정 실패")

    def read_text(
        self,
        image: Image.Image,
        *,
        lang: OcrLang,
        psm: int,
        whitelist: str | None = None,
    ) -> str:
        import pytesseract

        parts = [f"--psm {psm}"]
        if whitelist:
            parts.append(f"-c tessedit_char_whitelist={whitelist}")
        config = " ".join(parts)
        return pytesseract.image_to_string(
            image,
            lang=lang,
            config=config,
        ).strip()


class OcrEngine:
    """스레드 안전 OCR — kor/eng 엔진 재사용(tesserocr) 또는 subprocess fallback"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: _OcrBackend | None = None

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend is not None else "none"

    @property
    def available(self) -> bool:
        return self._backend is not None

    def initialize(self) -> bool:
        if self._backend is not None:
            return True

        tessdata = get_tessdata_dir()
        if tessdata is None:
            logger.warning("tessdata 경로를 찾지 못했습니다")
            return False

        backend = self._try_tesserocr(tessdata)
        if backend is None:
            backend = self._try_pytesseract()
        if backend is None:
            return False

        self._backend = backend
        logger.info("OCR 백엔드: %s (tessdata=%s)", backend.name, tessdata)
        return True

    def _try_tesserocr(self, tessdata: Path) -> _OcrBackend | None:
        try:
            return _TesserocrBackend(tessdata)
        except ImportError:
            logger.debug("tesserocr 미설치 — pytesseract fallback")
        except Exception as exc:
            logger.warning("tesserocr 초기화 실패: %s", exc)
        return None

    def _try_pytesseract(self) -> _OcrBackend | None:
        try:
            return _PytesseractBackend()
        except Exception as exc:
            logger.warning("pytesseract 초기화 실패: %s", exc)
        return None

    def read_text(
        self,
        image: Image.Image,
        *,
        lang: OcrLang = "eng",
        psm: int = 7,
        whitelist: str | None = None,
    ) -> str:
        if self._backend is None:
            return ""
        with self._lock:
            try:
                return self._backend.read_text(
                    image,
                    lang=lang,
                    psm=psm,
                    whitelist=whitelist,
                )
            except Exception as exc:
                logger.debug("OCR 실패 lang=%s psm=%s: %s", lang, psm, exc)
                return ""
