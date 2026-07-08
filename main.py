"""여기맞나? — FFXIV 보물지도 좌표 인식 오버레이"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app_config import setup_logging
from src.overlay.floating_button import run_overlay


def main() -> int:
    # 소스 실행 시 매칭 query 이미지 저장 (배포 exe 제외)
    if not getattr(sys, "frozen", False):
        os.environ.setdefault("TR_DEBUG_QUERY", "1")
    setup_logging()
    return run_overlay()


if __name__ == "__main__":
    raise SystemExit(main())
