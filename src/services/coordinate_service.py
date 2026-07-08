import json
import math
import re
from pathlib import Path
from typing import Optional

from src.models.recognition_result import RecognitionResult
from src.services.app_paths import get_app_root


class CoordinateService:
    """지역/에테라이트 데이터 및 좌표 계산"""

    def __init__(
        self,
        data_dir: Path | None = None,
        maps_dir: Path | None = None,
    ) -> None:
        if data_dir is None:
            data_dir = get_app_root() / "data"
        self.data_dir = data_dir
        if maps_dir is None:
            maps_dir = data_dir.parent / "assets" / "maps"
        self.maps_dir = maps_dir
        self.zones = self._load_zones("zones.json")
        self.aetherytes = self._load_json("aetherytes.json")

    def _load_json(self, filename: str) -> list[dict]:
        path = self.data_dir / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_zones(self, filename: str) -> list[dict]:
        """확장팩별 객체 또는 flat 배열 zones.json 로드"""
        path = self.data_dir / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        zones: list[dict] = []
        for expansion, items in data.items():
            for zone in items:
                merged = dict(zone)
                merged.setdefault("expansion", expansion)
                zones.append(merged)
        return zones

    def find_zone_by_name(self, text: str) -> Optional[dict]:
        """OCR 텍스트에서 지역명 매칭"""
        normalized = text.lower().replace(" ", "").replace("'", "")
        for zone in self.zones:
            for key in ("name_ko", "name_en", "id"):
                candidate = str(zone.get(key, "")).lower().replace(" ", "").replace("'", "")
                if candidate and candidate in normalized:
                    return zone
                if normalized and normalized in candidate:
                    return zone
        return None

    def resolve_detail_zone_id(self, zone_id: str) -> str:
        """상세 지도 공유 ID (central_shroud_01 → central_shroud)"""
        zone = self.get_zone(zone_id)
        if zone and zone.get("detail_zone_id"):
            return str(zone["detail_zone_id"])

        base_id = re.sub(r"_\d+$", "", zone_id)
        return base_id if base_id != zone_id else zone_id

    def get_effective_zone(self, zone_id: str) -> Optional[dict]:
        """match 지역 + detail 공유 설정 병합"""
        zone = self.get_zone(zone_id)
        if zone is None:
            return None

        detail_id = self.resolve_detail_zone_id(zone_id)
        if detail_id == zone_id:
            return zone

        detail = self.get_zone(detail_id)
        if detail is None:
            return zone

        merged = dict(detail)
        merged.update(zone)
        merged["id"] = zone_id
        merged["detail_zone_id"] = detail_id
        return merged

    def _collect_aetherytes(self, zone_id: str) -> list[dict]:
        """zones.json aetherytes[] + spots[].aetheryte + aetherytes.json 수집"""
        collected: list[dict] = []
        seen: set[tuple] = set()
        detail_id = self.resolve_detail_zone_id(zone_id)

        for zid in (zone_id, detail_id):
            zone = self.get_zone(zid)
            if zone is None:
                continue

            sources: list[dict] = list(zone.get("aetherytes", []))
            for spot in zone.get("spots", []):
                ae = spot.get("aetheryte")
                if isinstance(ae, dict):
                    sources.append(ae)

            for ae in sources:
                name = str(ae.get("name_ko", ""))
                ax = ae.get("x")
                ay = ae.get("y")
                if ax is None or ay is None:
                    continue
                key = (name, float(ax), float(ay))
                if key in seen:
                    continue
                seen.add(key)
                collected.append(ae)

        for ae in self.aetherytes:
            if ae.get("zone_id") not in (zone_id, detail_id):
                continue
            name = str(ae.get("name_ko", ""))
            ax = ae.get("x")
            ay = ae.get("y")
            if ax is None or ay is None:
                continue
            key = (name, float(ax), float(ay))
            if key in seen:
                continue
            seen.add(key)
            collected.append(ae)

        return collected

    def find_nearest_aetheryte(
        self, zone_id: str, treasure_x: float, treasure_y: float
    ) -> tuple[str, float, dict | None]:
        """보물 좌표에서 가장 가까운 에테라이트 (이름, 거리, 항목)"""
        detail_id = self.resolve_detail_zone_id(zone_id)
        zone = self.get_zone(detail_id)

        # spots에 등록된 보물·에테 쌍이 있으면 우선 참고 (±0.6 이내)
        if zone is not None:
            for spot in zone.get("spots", []):
                treasure = spot.get("treasure", {})
                ae = spot.get("aetheryte")
                tx = treasure.get("x")
                ty = treasure.get("y")
                if tx is None or ty is None or not isinstance(ae, dict):
                    continue
                if abs(float(tx) - treasure_x) > 0.6 or abs(float(ty) - treasure_y) > 0.6:
                    continue
                ax = ae.get("x")
                ay = ae.get("y")
                if ax is not None and ay is not None:
                    dist = math.hypot(float(ax) - treasure_x, float(ay) - treasure_y)
                else:
                    dist = 0.0
                return str(ae.get("name_ko", "에테라이트")), dist, ae

        candidates = self._collect_aetherytes(zone_id)
        if not candidates:
            return "에테라이트 정보 없음 (zones.json에 추가)", 0.0, None

        def distance(ae: dict) -> float:
            return math.hypot(
                float(ae["x"]) - treasure_x,
                float(ae["y"]) - treasure_y,
            )

        best = min(candidates, key=distance)
        best_name = str(best.get("name_ko", "에테라이트"))
        return best_name, distance(best), best

    def build_result(
        self, zone: dict, x: float, y: float, map_index: int = 1
    ) -> RecognitionResult:
        zone_id = zone["id"]
        detail_zone_id = self.resolve_detail_zone_id(zone_id)
        ae_name, ae_dist, ae_entry = self.find_nearest_aetheryte(
            detail_zone_id, x, y
        )
        ae_x = float(ae_entry["x"]) if ae_entry and ae_entry.get("x") is not None else None
        ae_y = float(ae_entry["y"]) if ae_entry and ae_entry.get("y") is not None else None
        ae_icon = str(ae_entry.get("icon", "")) if ae_entry else None
        if ae_icon == "":
            ae_icon = None
        return RecognitionResult(
            zone_id=zone_id,
            zone_name=zone.get("name_ko", zone_id),
            x=x,
            y=y,
            nearest_aetheryte=ae_name,
            aetheryte_distance=ae_dist,
            nearest_aetheryte_x=ae_x,
            nearest_aetheryte_y=ae_y,
            nearest_aetheryte_icon=ae_icon,
            map_index=map_index,
        )

    def get_zone(self, zone_id: str) -> Optional[dict]:
        for zone in self.zones:
            if zone["id"] == zone_id:
                return zone
        return None

    MAP_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
    EXPANSION_FOLDERS = ("신생", "창천", "홍련", "칠흑", "효월", "황금")

    MAP_KIND_MATCH = "match"
    MAP_KIND_DETAIL = "detail"

    def _expansion_dir_for_zone(self, zone_id: str) -> Path | None:
        """zones.json expansion → maps/{확장팩}/ 경로"""
        zone = self.get_zone(zone_id)
        if zone and zone.get("expansion"):
            folder = self.maps_dir / str(zone["expansion"])
            if folder.is_dir():
                return folder

        for name in self.EXPANSION_FOLDERS:
            folder = self.maps_dir / name
            if not folder.is_dir():
                continue
            for ext in self.MAP_IMAGE_EXTENSIONS:
                if (folder / f"{zone_id}{ext}").exists():
                    return folder
        return None

    def get_match_map_path(self, zone_id: str, map_index: int = 1) -> Path:
        """보물지도 모양 매칭용 이미지"""
        return self.get_map_image_path(zone_id, map_index, kind=self.MAP_KIND_MATCH)

    def get_detail_map_path(self, zone_id: str, map_index: int = 1) -> Path:
        """좌표 표시용 상세 지도 (지역당 1장 공유)"""
        detail_id = self.resolve_detail_zone_id(zone_id)
        path = self.get_map_image_path(detail_id, map_index, kind=self.MAP_KIND_DETAIL)
        if path.exists():
            return path
        return self.get_map_image_path(zone_id, map_index, kind=self.MAP_KIND_DETAIL)

    # FFXIV 지도: 좌표 1~41, Y는 남쪽(아래)으로 증가, 1024 텍스처에 여백 있음
    FFXIV_MAP_UNIT = 40.96  # 4096 / 100

    def _resolve_detail_transform(
        self,
        zone: dict,
        display_width: int,
        display_height: int,
    ) -> tuple[float, float, float, float, float]:
        """게임 좌표 변환에 쓸 (unit, offset_x, offset_y, inner_w, inner_h)"""
        unit = float(zone.get("map_unit", self.FFXIV_MAP_UNIT))

        if zone.get("detail_offset_x") is not None:
            offset_x = float(zone["detail_offset_x"])
            offset_y = float(zone.get("detail_offset_y", 0.0))
            inner_w = float(
                zone.get("detail_inner_width", display_width)
            )
            inner_h = float(
                zone.get("detail_inner_height", display_height)
            )
            return unit, offset_x, offset_y, inner_w, inner_h

        for pos in zone.get("positions", []):
            px_ref = pos.get("pixel_x")
            py_ref = pos.get("pixel_y")
            if px_ref is None or py_ref is None:
                continue
            gx = float(pos["game_x"])
            gy = float(pos["game_y"])
            offset_x = float(px_ref) - (gx - 1.0) / unit * display_width
            offset_y = float(py_ref) - (gy - 1.0) / unit * display_height
            return unit, offset_x, offset_y, float(display_width), float(display_height)

        inset = float(zone.get("detail_inset", 0.0))
        if inset > 0:
            offset_x = display_width * inset
            offset_y = display_height * inset
            inner_w = display_width * (1.0 - inset * 2)
            inner_h = display_height * (1.0 - inset * 2)
        else:
            offset_x = 0.0
            offset_y = 0.0
            inner_w = float(display_width)
            inner_h = float(display_height)

        return unit, offset_x, offset_y, inner_w, inner_h

    def game_to_pixel(
        self,
        zone: dict,
        game_x: float,
        game_y: float,
        display_width: int,
        display_height: int,
    ) -> tuple[int, int]:
        """게임 좌표 → 상세 지도 픽셀 (FFXIV 1-based 좌표계)"""
        unit, offset_x, offset_y, inner_w, inner_h = self._resolve_detail_transform(
            zone, display_width, display_height
        )
        px = int(offset_x + ((game_x - 1.0) / unit) * inner_w)
        py = int(offset_y + ((game_y - 1.0) / unit) * inner_h)
        return px, py

    def get_detail_image_size(
        self, zone_id: str, map_index: int = 1
    ) -> tuple[int, int]:
        """detail PNG 실제 해상도 (2048/1024 혼재 대응)"""
        detail_id = self.resolve_detail_zone_id(zone_id)
        path = self.get_detail_map_path(detail_id, map_index)
        if path.exists():
            from PIL import Image

            with Image.open(path) as img:
                return img.size

        zone = self.get_zone(detail_id) or self.get_zone(zone_id)
        if zone is not None:
            return (
                int(zone.get("detail_width", 1024)),
                int(zone.get("detail_height", 1024)),
            )
        return 1024, 1024

    def game_to_detail_pixel(
        self,
        zone: dict,
        game_x: float,
        game_y: float,
        map_index: int = 1,
    ) -> tuple[int, int]:
        """상세 지도 원본 해상도 기준 픽셀 좌표"""
        zone_id = str(zone.get("detail_zone_id") or zone.get("id", ""))
        width, height = self.get_detail_image_size(zone_id, map_index)
        return self.game_to_pixel(zone, game_x, game_y, width, height)

    def refine_treasure_coords(
        self,
        zone_id: str,
        x: float,
        y: float,
        max_dist: float = 3.5,
    ) -> tuple[float, float]:
        """인식 좌표를 zones.json spots 중 가장 가까운 보물 위치로 보정"""
        detail_id = self.resolve_detail_zone_id(zone_id)
        zone = self.get_zone(detail_id)
        if zone is None:
            return x, y

        best_x, best_y = x, y
        best_dist = max_dist
        for spot in zone.get("spots", []):
            treasure = spot.get("treasure", {})
            tx = treasure.get("x")
            ty = treasure.get("y")
            if tx is None or ty is None:
                continue
            dist = math.hypot(float(tx) - x, float(ty) - y)
            if dist < best_dist:
                best_dist = dist
                best_x, best_y = float(tx), float(ty)
        return best_x, best_y

    def validate_treasure_coords(
        self,
        zone_id: str,
        x: float,
        y: float,
        *,
        max_spot_dist: float = 8.0,
    ) -> bool:
        """보물 좌표가 지역 spots 근처·유효 범위인지 검증 (OCR 오인식 차단)"""
        if not (5.0 <= x <= 38.0 and 5.0 <= y <= 38.0):
            return False

        detail_id = self.resolve_detail_zone_id(zone_id)
        zone = self.get_zone(detail_id)
        if zone is None:
            return True

        best_dist = float("inf")
        for spot in zone.get("spots", []):
            treasure = spot.get("treasure", {})
            tx = treasure.get("x")
            ty = treasure.get("y")
            if tx is None or ty is None:
                continue
            best_dist = min(best_dist, math.hypot(float(tx) - x, float(ty) - y))

        return best_dist <= max_spot_dist

    def pixel_to_game(
        self,
        zone: dict,
        pixel_x: float,
        pixel_y: float,
        display_width: int | None = None,
        display_height: int | None = None,
    ) -> tuple[float, float]:
        """상세 지도 픽셀 → 게임 좌표"""
        width = int(display_width or zone.get("detail_width", 1024))
        height = int(display_height or zone.get("detail_height", 1024))
        unit, offset_x, offset_y, inner_w, inner_h = self._resolve_detail_transform(
            zone, width, height
        )
        game_x = (pixel_x - offset_x) / inner_w * unit + 1.0
        game_y = (pixel_y - offset_y) / inner_h * unit + 1.0
        return round(game_x, 1), round(game_y, 1)

    def get_map_image_path(
        self, zone_id: str, map_index: int = 1, kind: str = MAP_KIND_DETAIL
    ) -> Path:
        """zone_id에 해당하는 지도 이미지 — maps/{확장팩}/{zone_id}.png"""
        expansion_dir = self._expansion_dir_for_zone(zone_id)
        default = (
            (expansion_dir or self.maps_dir / "...") / f"{zone_id}.png"
        )

        if expansion_dir is not None:
            flat = self._collect_flat_map_paths(expansion_dir, zone_id, kind)
            if flat:
                if map_index > 1:
                    for path in flat:
                        if self._extract_map_index(path.stem, zone_id) == map_index:
                            return path
                flat.sort(key=lambda p: self._map_sort_key(p.stem, zone_id))
                return flat[0]

        if not self.maps_dir.exists():
            return default

        legacy = self._collect_legacy_map_paths(self.maps_dir, zone_id, kind)
        if legacy:
            if map_index > 1:
                for path in legacy:
                    if self._extract_map_index(path.stem, zone_id) == map_index:
                        return path
            legacy.sort(key=lambda p: self._map_sort_key(p.stem, zone_id))
            return legacy[0]

        return default

    def _collect_flat_map_paths(
        self, expansion_dir: Path, zone_id: str, kind: str
    ) -> list[Path]:
        """maps/{확장팩}/ 직접 배치 (detail: id.png, match: id_01.png)"""
        found: list[Path] = []
        for path in expansion_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.MAP_IMAGE_EXTENSIONS:
                continue
            stem = path.stem
            if kind == self.MAP_KIND_DETAIL:
                if stem == zone_id:
                    found.append(path)
            elif re.fullmatch(rf"{re.escape(zone_id)}_\d+", stem):
                found.append(path)
        return found

    def _collect_legacy_map_paths(
        self, maps_dir: Path, zone_id: str, kind: str
    ) -> list[Path]:
        """구 match/detail/ 하위 폴더 호환"""
        typed: list[Path] = []
        legacy: list[Path] = []

        for path in maps_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.MAP_IMAGE_EXTENSIONS:
                continue
            if not self._map_filename_matches_zone(path.stem, zone_id):
                continue

            parts = set(path.parts)
            if kind in parts:
                typed.append(path)
            elif self.MAP_KIND_MATCH not in parts and self.MAP_KIND_DETAIL not in parts:
                legacy.append(path)

        return typed or legacy

    @staticmethod
    def _extract_map_index(stem: str, zone_id: str) -> Optional[int]:
        match = re.fullmatch(rf"{re.escape(zone_id)}_(\d+)", stem)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _map_sort_key(stem: str, zone_id: str) -> tuple[int, int]:
        """기본 지도 우선순위: {id}.png → {id}_1 → {id}_2 → … → 01_{id}"""
        if stem == zone_id:
            return (0, 0)
        match = re.fullmatch(rf"{re.escape(zone_id)}_(\d+)", stem)
        if match:
            return (1, int(match.group(1)))
        match = re.fullmatch(rf"(\d+)_{re.escape(zone_id)}", stem)
        if match:
            return (2, int(match.group(1)))
        return (3, 0)

    @staticmethod
    def _map_filename_matches_zone(stem: str, zone_id: str) -> bool:
        """파일명이 zone_id와 일치하는지 확인"""
        if stem == zone_id:
            return True
        if re.fullmatch(rf"{re.escape(zone_id)}_\d+", stem):
            return True
        if re.fullmatch(rf"\d+_{re.escape(zone_id)}", stem):
            return True
        return False
