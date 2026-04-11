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
    "AE": {"name": "United Arab Emirates", "region": "Middle East & Africa"},
    "AU": {"name": "Australia", "region": "Asia Pacific"},
    "BR": {"name": "Brazil", "region": "Latin America"},
    "CA": {"name": "Canada", "region": "North America"},
    "DE": {"name": "Germany", "region": "Europe"},
    "ES": {"name": "Spain", "region": "Europe"},
    "FR": {"name": "France", "region": "Europe"},
    "GB": {"name": "United Kingdom", "region": "Europe"},
    "IN": {"name": "India", "region": "Asia Pacific"},
    "JP": {"name": "Japan", "region": "Asia Pacific"},
    "MX": {"name": "Mexico", "region": "North America"},
    "NL": {"name": "Netherlands", "region": "Europe"},
    "PL": {"name": "Poland", "region": "Europe"},
    "SE": {"name": "Sweden", "region": "Europe"},
    "SG": {"name": "Singapore", "region": "Asia Pacific"},
    "US": {"name": "United States", "region": "North America"},
    "ZA": {"name": "South Africa", "region": "Middle East & Africa"},
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
