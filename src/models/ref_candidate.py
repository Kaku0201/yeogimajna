from dataclasses import dataclass


@dataclass(frozen=True)
class RefCandidate:
    """ref DB 매칭 후보 — 사용자가 직접 선택해 상세 지도 확인"""

    rank: int
    x: float
    y: float
    score: float
    terrain_score: float
    marker_dist: float
    ref_name: str
    ref_image_path: str

    @property
    def coordinate_text(self) -> str:
        return f"X: {self.x:.1f}  Y: {self.y:.1f}"
