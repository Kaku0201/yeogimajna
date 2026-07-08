"""지도 데이터 팩 — 설치형 배포용 (사용자는 PNG를 직접 추가하지 않음)"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

from src.services.app_paths import get_app_root, get_user_data_dir

EXPANSION_FOLDERS = ("신생", "창천", "홍련", "칠흑", "효월", "황금")


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class MapPackStatus:
    """지도 팩 준비 상태"""

    ready: bool
    source: str
    maps_dir: Path
    detail_count: int
    match_count: int
    version: str
    message: str


class MapPackService:
    """
    지도 팩 우선순위:
    1. 사용자 폴더 (~/.yeogimajna/map_pack) — 다운로드/업데이트본
    2. 앱 번들 (assets/maps) — 설치 파일에 포함된 기본 팩
    """

    MAP_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

    def __init__(
        self,
        data_dir: Path | None = None,
        bundled_maps_dir: Path | None = None,
        user_maps_dir: Path | None = None,
    ) -> None:
        app_root = get_app_root()
        self.data_dir = data_dir or app_root / "data"
        self.bundled_maps_dir = bundled_maps_dir or app_root / "assets" / "maps"
        self.user_maps_dir = user_maps_dir or get_user_data_dir() / "map_pack"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        path = self.data_dir / "map_pack_manifest.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def maps_dir(self) -> Path:
        """현재 사용할 지도 루트"""
        if self._has_maps(self.user_maps_dir):
            return self.user_maps_dir
        return self.bundled_maps_dir

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "0.0.0"))

    @property
    def download_url(self) -> str:
        return str(self.manifest.get("download_url", "")).strip()

    def get_status(self) -> MapPackStatus:
        maps_dir = self.maps_dir
        detail_count = self._count_detail_maps(maps_dir)
        match_count = self._count_match_maps(maps_dir)
        min_detail = int(self.manifest.get("min_detail_maps", 1))
        min_match = int(self.manifest.get("min_match_templates", 0))

        ready = detail_count >= min_detail and match_count >= min_match
        if self._has_maps(self.user_maps_dir):
            source = "다운로드/업데이트"
        elif self._has_maps(self.bundled_maps_dir):
            source = "설치 파일 포함"
        else:
            source = "없음"

        if ready:
            message = f"지도 데이터 준비됨 — 지역 지도 {detail_count}장"
            if match_count:
                message += f", match {match_count}장"
        else:
            message = (
                "지도 데이터가 없습니다.\n"
                "배포용 설치 파일에는 지도 팩이 포함되어야 합니다."
            )

        return MapPackStatus(
            ready=ready,
            source=source,
            maps_dir=maps_dir,
            detail_count=detail_count,
            match_count=match_count,
            version=self.version,
            message=message,
        )

    def is_ready(self) -> bool:
        return self.get_status().ready

    def download_and_install(
        self,
        progress: ProgressCallback | None = None,
    ) -> None:
        """manifest의 download_url에서 zip 받아 사용자 map_pack 폴더에 설치"""
        url = self.download_url
        if not url:
            raise ValueError("지도 팩 다운로드 URL이 설정되지 않았습니다.")

        self.user_maps_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.user_maps_dir.parent / "map_pack_download.zip"

        try:
            self._download_file(url, zip_path, progress)
            self._extract_zip(zip_path, self.user_maps_dir, progress)
        finally:
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)

    def _download_file(
        self,
        url: str,
        dest: Path,
        progress: ProgressCallback | None,
    ) -> None:
        if progress:
            progress(0, 100, "지도 데이터 다운로드 중…")

        try:
            with urlopen(url, timeout=120) as response:
                total = int(response.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                chunk_size = 1024 * 256

                with dest.open("wb") as out:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if progress and total > 0:
                            pct = min(99, int(downloaded * 100 / total))
                            progress(pct, 100, "지도 데이터 다운로드 중…")
        except URLError as exc:
            raise ValueError(f"다운로드 실패: {exc}") from exc

        if progress:
            progress(100, 100, "다운로드 완료")

    def _extract_zip(
        self,
        zip_path: Path,
        dest: Path,
        progress: ProgressCallback | None,
    ) -> None:
        if progress:
            progress(0, 100, "지도 데이터 설치 중…")

        staging = dest.parent / "map_pack_staging"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(staging)

            extracted_root = self._find_maps_root(staging)
            if extracted_root is None:
                raise ValueError(
                    "zip 안에 확장팩 폴더(신생·창천 등) 또는 maps/를 찾지 못했습니다."
                )

            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)

            for item in extracted_root.iterdir():
                target = dest / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        if progress:
            progress(100, 100, "설치 완료")

    def _find_maps_root(self, root: Path) -> Path | None:
        """압축 해제 루트 — maps/ 또는 확장팩 폴더 직접"""
        maps_dir = root / "maps"
        if maps_dir.is_dir() and self._has_expansion_folders(maps_dir):
            return maps_dir

        if self._has_expansion_folders(root):
            return root

        # 구 match/detail/ 구조 호환
        if (root / "detail").is_dir() or (root / "match").is_dir():
            return root

        for path in root.rglob("detail"):
            if path.is_dir():
                return path.parent

        for name in EXPANSION_FOLDERS:
            if (root / name).is_dir():
                return root

        return None

    def _has_expansion_folders(self, maps_dir: Path) -> bool:
        return any((maps_dir / name).is_dir() for name in EXPANSION_FOLDERS)

    def _has_maps(self, maps_dir: Path) -> bool:
        if not maps_dir.exists():
            return False
        return self._count_detail_maps(maps_dir) > 0

    def _iter_map_files(self, maps_dir: Path):
        """확장팩 폴더 + 구 match/detail 하위 PNG"""
        for name in EXPANSION_FOLDERS:
            folder = maps_dir / name
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in self.MAP_IMAGE_EXTENSIONS:
                    yield path

        for path in maps_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.MAP_IMAGE_EXTENSIONS:
                continue
            parts = set(path.parts)
            if "match" in parts or "detail" in parts:
                yield path

    def _count_detail_maps(self, maps_dir: Path) -> int:
        count = 0
        for path in self._iter_map_files(maps_dir):
            if "match" in path.parts:
                continue
            if "detail" in path.parts:
                count += 1
                continue
            if re.search(r"_\d+$", path.stem):
                continue
            count += 1
        return count

    def _count_match_maps(self, maps_dir: Path) -> int:
        count = 0
        for path in self._iter_map_files(maps_dir):
            if "match" in path.parts:
                count += 1
            elif "detail" not in path.parts and re.search(r"_\d+$", path.stem):
                count += 1
        return count
