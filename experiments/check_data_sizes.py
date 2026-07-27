"""Quick dataset size check."""
import pandas as pd
from pathlib import Path

for d in [Path("data/raw"), Path("data/processed")]:
    if d.exists():
        for f in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                print(f"{f.name}: {df.shape[0]} rows x {df.shape[1]} cols")
            except Exception as e:
                print(f"{f.name}: {e}")
