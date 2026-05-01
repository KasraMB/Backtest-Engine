"""
Delete all run_ml_collect outputs for a clean re-run.
"""
from pathlib import Path

ROOT = Path(__file__).parent

TO_DELETE = [
    ROOT / "cache" / "sens_runs_cache.json",
    ROOT / "cache" / "sensitivity_cache.json",
    ROOT / "cache" / "trades_cache.json",
    ROOT / "cache" / "val_trades_cache.json",
    ROOT / "cache" / "regime_map_cache.pkl",
    ROOT / "cache" / "atr_rank_map_cache.pkl",
    ROOT / "data"  / "ml_dataset.parquet",
    ROOT / "data"  / "validated_configs.json",
]

for p in TO_DELETE:
    if p.exists():
        p.unlink()
        print(f"deleted  {p.relative_to(ROOT)}")
    else:
        print(f"missing  {p.relative_to(ROOT)}")

print("\nDone. Run: python run_ml_collect.py")
