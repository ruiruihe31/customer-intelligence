"""Shared constants and path configuration for the entire project."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "dataset"

# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------
CLUSTER_RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
COUNTRY_META: dict[str, dict[str, str]] = {
    "AE": {"name": "United Arab Emirates", "alpha3": "ARE", "region": "Middle East & Africa"},
    "AU": {"name": "Australia",             "alpha3": "AUS", "region": "Asia Pacific"},
    "BR": {"name": "Brazil",                "alpha3": "BRA", "region": "Latin America"},
    "CA": {"name": "Canada",                "alpha3": "CAN", "region": "North America"},
    "DE": {"name": "Germany",               "alpha3": "DEU", "region": "Europe"},
    "ES": {"name": "Spain",                 "alpha3": "ESP", "region": "Europe"},
    "FR": {"name": "France",                "alpha3": "FRA", "region": "Europe"},
    "GB": {"name": "United Kingdom",        "alpha3": "GBR", "region": "Europe"},
    "IN": {"name": "India",                 "alpha3": "IND", "region": "Asia Pacific"},
    "JP": {"name": "Japan",                 "alpha3": "JPN", "region": "Asia Pacific"},
    "MX": {"name": "Mexico",                "alpha3": "MEX", "region": "North America"},
    "NL": {"name": "Netherlands",           "alpha3": "NLD", "region": "Europe"},
    "PL": {"name": "Poland",                "alpha3": "POL", "region": "Europe"},
    "SE": {"name": "Sweden",                "alpha3": "SWE", "region": "Europe"},
    "SG": {"name": "Singapore",             "alpha3": "SGP", "region": "Asia Pacific"},
    "US": {"name": "United States",         "alpha3": "USA", "region": "North America"},
    "ZA": {"name": "South Africa",          "alpha3": "ZAF", "region": "Middle East & Africa"},
}

# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
AGE_BINS: list[int] = [17, 24, 34, 44, 54, 64, 75]
AGE_LABELS: list[str] = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-75"]
TITLE_GENDER: dict[str, str] = {
    "mr.": "Male",
    "mrs.": "Female",
    "ms.": "Female",
    "miss": "Female",
}

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
COLOR_SEQUENCE: list[str] = [
    "#6C8CFF",
    "#00C2A8",
    "#FFB84D",
    "#FF6B6B",
    "#B388FF",
    "#56CCF2",
]
