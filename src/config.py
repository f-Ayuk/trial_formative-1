# ============================================================
# src/config.py — Central configuration
# ============================================================
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "milan_data"
CACHE_DIR = BASE_DIR / "cache"
FIG_DIR   = BASE_DIR / "figures"
RES_DIR   = BASE_DIR / "results"
MET_DIR   = RES_DIR / "metrics"
PRED_DIR  = RES_DIR / "predictions"

for d in (FIG_DIR, MET_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Milan column names (tab-separated, no header row)
MILAN_COLS = [
    "Square id", "Time Interval", "Country code",
    "SMS-in activity", "SMS-out activity",
    "Call-in activity", "Call-out activity",
    "Internet traffic activity",
]

OPT_DTYPES = {
    "Square id": "int32",
    "Time Interval": "int64",
    "Country code": "int16",
    "SMS-in activity": "float32",
    "SMS-out activity": "float32",
    "Call-in activity": "float32",
    "Call-out activity": "float32",
    "Internet traffic activity": "float32",
}

# Forecasting setup
TEST_START = "2013-12-16"
TEST_END   = "2013-12-22"
SEQ_LEN    = 288              # 2 days of 10-min intervals
FORECAST_HORIZON = 1          # one-step-ahead

TRAIN_END = "2013-12-09"
VAL_START = "2013-12-10"
VAL_END   = "2013-12-15"

RANDOM_STATE = 42