from dataclasses import dataclass, field
from typing import Optional

from src.models.ref_candidate import RefCandidate


@dataclass
class RecognitionResult:
    """지도 인식 결과 모델"""

    zone_id: str
    zone_name: str
    x: float
    y: float
    nearest_aetheryte: str
    aetheryte_distance: float
    map_index: int = 1
    match_score: float | None = None
    nearest_aetheryte_x: float | None = None
    nearest_aetheryte_y: float | None = None
    nearest_aetheryte_icon: str | None = None
    ref_candidates: list[RefCandidate] = field(default_factory=list)
    auto_candidate_rank: int | None = None
    match_source: str = "ref"
    excluded_ref_names: list[str] = field(default_factory=list)
    can_rematch: bool = False
    confirmed_ref_name: str | None = None
    learn_hits: int | None = None
    party_size: int | None = None
    party_size_uncertain: bool = False

    @property
    def coordinate_text(self) -> str:
        return f"X: {self.x:.1f}  Y: {self.y:.1f}"

    @property
    def distance_text(self) -> str:
        return f"{self.aetheryte_distance:.1f} 유닛"
