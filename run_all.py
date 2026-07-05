"""
End-to-end reproduction driver.

Runs every stage in order:
    s1 data -> s2 word2vec -> s3 events -> s4 NTN -> s5 features
    -> s7 train/eval (+ s8 simulation) -> s9 report

Usage (from the alphamale venv):
    python run_all.py
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, *sorted((ROOT / "flows").glob("flow*"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STAGES = ["s1_data", "s2_word2vec", "s3_events", "s4_ntn",
          "t1_embed", "s5_features", "s7_train_eval", "s9_report"]

def main():
    for name in STAGES:
        print("\n" + "#" * 70 + f"\n# {name}\n" + "#" * 70, flush=True)
        t0 = time.time()
        mod = importlib.import_module(name)
        mod.main()
        print(f"# {name} done in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
