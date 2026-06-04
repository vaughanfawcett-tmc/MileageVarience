"""Generate a sample Milcap-style report to test the classifier."""

from pathlib import Path
import pandas as pd

ROWS = [
    ("D001", "2026-05-01", "Birmingham", "Coventry", 22.0, 21.5, "Customer visit"),
    ("D002", "2026-05-01", "Leeds", "Manchester", 65.0, 42.0, "Roadworks on M62 diversion"),
    ("D003", "2026-05-02", "London", "Reading", 50.0, 40.0, "Multiple stops at client sites"),
    ("D004", "2026-05-02", "Bristol", "Cardiff", 60.0, 45.0, "Took scenic route"),
    ("D005", "2026-05-03", "Liverpool", "Chester", 30.0, 18.0, "Personal errand on way back"),
    ("D006", "2026-05-03", "Newcastle", "Durham", 25.0, 16.0, "School run"),
    ("D007", "2026-05-04", "Glasgow", "Edinburgh", 55.0, 47.0, "Detour due to GPS"),
    ("D008", "2026-05-04", "Sheffield", "Leeds", 40.0, 35.0, "Meeting"),
    ("D009", "2026-05-05", "Nottingham", "Derby", 20.0, 16.0, ""),
    ("D010", "2026-05-05", "Oxford", "Reading", 30.0, 27.0, "n/a"),
    ("D011", "2026-05-06", "Cambridge", "Norwich", 70.0, 62.0, "Accident on A11"),
    ("D012", "2026-05-06", "Brighton", "Portsmouth", 55.0, 49.0, "Heavy traffic on A27"),
    ("D013", "2026-05-07", "Plymouth", "Exeter", 50.0, 44.0, "Vehicle breakdown - had to reroute"),
    ("D014", "2026-05-07", "Aberdeen", "Inverness", 110.0, 105.0, "Visit"),
    ("D015", "2026-05-08", "Cardiff", "Swansea", 48.0, 41.0, "Stopped at gym briefly"),
]

COLUMNS = [
    "Driver ID",
    "Trip Date",
    "Origin",
    "Destination",
    "Claimed Miles",
    "Expected Miles",
    "Trip Reason",
]


def main() -> None:
    df = pd.DataFrame(ROWS, columns=COLUMNS)
    out = Path(__file__).parent / "sample_milcap_report.xlsx"
    df.to_excel(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
