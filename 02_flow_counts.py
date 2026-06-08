import pandas as pd
from pathlib import Path

ROOT = Path("NFSTREAM_CSV")

for folder in ROOT.iterdir():

    if not folder.is_dir():
        continue

    total_rows = 0

    for csv_file in folder.glob("*.csv"):

        rows = sum(
            1 for _ in open(csv_file)
        ) - 1

        total_rows += rows

    print(folder.name, ":", total_rows)