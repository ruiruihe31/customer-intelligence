"""Formatting helpers, name/gender inference, and small data utilities."""
from __future__ import annotations

import re

import pandas as pd
from faker.providers.person.en_US import Provider

from .config import COUNTRY_META, TITLE_GENDER

# ---------------------------------------------------------------------------
# Name patterns (compiled once)
# ---------------------------------------------------------------------------
FIRST_NAME_RE = re.compile(r"^(?:Mr\.|Mrs\.|Ms\.|Miss|Dr\.)?\s*([A-Za-z][A-Za-z'\-]+)")

# Lazy-loaded name sets
_MALE_NAMES: set[str] | None = None
_FEMALE_NAMES: set[str] | None = None


def get_name_sets() -> tuple[set[str], set[str]]:
    global _MALE_NAMES, _FEMALE_NAMES
    if _MALE_NAMES is None:
        _MALE_NAMES = {item.lower() for item in Provider.first_names_male}
        _FEMALE_NAMES = {item.lower() for item in Provider.first_names_female}
    return _MALE_NAMES, _FEMALE_NAMES


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_number(value: float) -> str:
    if pd.isna(value):
        return "-"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "-"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


# ---------------------------------------------------------------------------
# Name & gender utilities
# ---------------------------------------------------------------------------

def first_name_from_full_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    match = FIRST_NAME_RE.match(name.strip())
    return match.group(1) if match else ""


def infer_gender(name: str, male_names: set[str], female_names: set[str]) -> str:
    lowered = str(name).strip().lower()
    for title, label in TITLE_GENDER.items():
        if lowered.startswith(f"{title} "):
            return label
    first_name = first_name_from_full_name(name).lower()
    if first_name in male_names and first_name not in female_names:
        return "Male"
    if first_name in female_names and first_name not in male_names:
        return "Female"
    if first_name in male_names and first_name in female_names:
        return "Ambiguous"
    return "Unknown"


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

def safe_country_name(code: str) -> str:
    return COUNTRY_META.get(str(code), {}).get("name", str(code))


def safe_country_alpha3(code: str) -> str:
    return COUNTRY_META.get(str(code), {}).get("alpha3", "")


def safe_region(code: str) -> str:
    return COUNTRY_META.get(str(code), {}).get("region", "Other")


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def get_mode_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    output_name: str,
) -> pd.DataFrame:
    modes = (
        df.groupby([group_col, value_col], observed=False)
        .size()
        .reset_index(name="count")
        .sort_values([group_col, "count", value_col], ascending=[True, False, True])
    )
    return (
        modes.drop_duplicates(group_col)[[group_col, value_col]]
        .rename(columns={value_col: output_name})
    )
