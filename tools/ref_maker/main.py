"""ref 이미지 제작 도구 — 상세지도 확대·크롭·X 마커·좌표 표시.

실행 (프로젝트 루트):
    python tools/ref_maker/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ref_maker_app import main

if __name__ == "__main__":
    main()
