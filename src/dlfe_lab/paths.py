"""경로/부트스트랩 유틸.

이 저장소는 연구 스크립트(`flows/flow*/*.py`)와 루트 `config.py`를 모듈로
직접 import 한다. `bootstrap()`은 ROOT, 모든 flow 디렉터리, `src`를
sys.path에 넣어 스크립트가 루트에 있든 flows/로 이동했든 동일하게
import 되도록 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def _flow_dirs() -> list[Path]:
    flows = ROOT / "flows"
    if not flows.exists():
        return []
    return sorted(p for p in flows.glob("flow*") if p.is_dir())


FLOW_DIRS = _flow_dirs()


def bootstrap() -> None:
    """idempotent: ROOT + flows/flow* + src 를 sys.path 앞에 삽입."""
    global FLOW_DIRS
    FLOW_DIRS = _flow_dirs()
    for p in (ROOT, *FLOW_DIRS, ROOT / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
