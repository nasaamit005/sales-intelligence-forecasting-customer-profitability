"""Central configuration: paths and analytical constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "superstore_global_2011_2014.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_FILE = PROCESSED_DIR / "sales_clean.csv"
DB_FILE = ROOT / "database" / "salesintel.db"
SCHEMA_FILE = ROOT / "database" / "schema.sql"
QUERIES_FILE = ROOT / "database" / "queries.sql"
RESULTS_DIR = ROOT / "results"
EXCEL_DIR = ROOT / "reports" / "excel"
HTML_DIR = ROOT / "reports" / "html"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
FORECAST_HORIZON_WEEKS = 13          # ~ 91 days (30/60/90-day views = 4/9/13 weeks)
SEASONAL_PERIOD = 52
VALIDATION_FOLDS = 3                 # rolling-origin folds used for model SELECTION
DISCOUNT_BINS = [-0.001, 0.0, 0.10, 0.20, 0.30, 0.50, 1.0]
DISCOUNT_LABELS = ["0%", "1-10%", "11-20%", "21-30%", "31-50%", ">50%"]
