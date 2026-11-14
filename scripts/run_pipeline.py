"""Run the full pipeline:  python scripts/run_pipeline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from salesintel import pipeline

if __name__ == "__main__":
    out = pipeline.run()
    m = out["metrics"]
    print(f"Revenue ${m['revenue']:,.0f} | Profit ${m['profit']:,.0f} | Best model: {m['forecast']['best_model']}")
