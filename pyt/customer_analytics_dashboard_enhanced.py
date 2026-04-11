
from __future__ import annotations

import base64
import logging
import os
import re
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from faker.providers.person.en_US import Provider
from plotly.io import to_html
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"plotly\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"_plotly_utils\..*")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dataset"
OUTPUT_HTML = BASE_DIR / "customer_analytics_dashboard_enhanced.html"
CLUSTER_RANDOM_STATE = 42

COUNTRY_META = {
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

AGE_BINS = [17, 24, 34, 44, 54, 64, 75]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-75"]
TITLE_GENDER = {"mr.": "Male", "mrs.": "Female", "ms.": "Female", "miss": "Female"}
COLOR_SEQUENCE = ["#6C8CFF", "#00C2A8", "#FFB84D", "#FF6B6B", "#B388FF", "#56CCF2"]
FIRST_NAME_RE = re.compile(r"^(?:Mr\.|Mrs\.|Ms\.|Miss|Dr\.)?\s*([A-Za-z][A-Za-z'\-]+)")


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


def safe_country_name(code: str) -> str:
    return COUNTRY_META.get(str(code), {}).get("name", str(code))


def safe_region(code: str) -> str:
    return COUNTRY_META.get(str(code), {}).get("region", "Other")


def get_mode_by_group(df: pd.DataFrame, group_col: str, value_col: str, output_name: str) -> pd.DataFrame:
    modes = (
        df.groupby([group_col, value_col], observed=False)
        .size()
        .reset_index(name="count")
        .sort_values([group_col, "count", value_col], ascending=[True, False, True])
    )
    return modes.drop_duplicates(group_col)[[group_col, value_col]].rename(columns={value_col: output_name})


def load_data() -> dict[str, pd.DataFrame]:
    required_files = [
        "customers.csv",
        "sessions.csv",
        "events.csv",
        "orders.csv",
        "order_items.csv",
        "products.csv",
        "reviews.csv",
    ]
    missing = [name for name in required_files if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files in {DATA_DIR}: {', '.join(missing)}")

    customers = pd.read_csv(
        DATA_DIR / "customers.csv",
        parse_dates=["signup_date"],
        dtype={"customer_id": "int32", "country": "category", "age": "float32"},
    )
    sessions = pd.read_csv(
        DATA_DIR / "sessions.csv",
        parse_dates=["start_time"],
        dtype={
            "session_id": "int32",
            "customer_id": "int32",
            "device": "category",
            "source": "category",
            "country": "category",
        },
    )
    events = pd.read_csv(
        DATA_DIR / "events.csv",
        parse_dates=["timestamp"],
        dtype={
            "event_id": "int32",
            "session_id": "int32",
            "event_type": "category",
            "product_id": "float32",
            "qty": "float32",
            "cart_size": "float32",
            "payment": "category",
            "discount_pct": "float32",
            "amount_usd": "float32",
        },
    )
    orders = pd.read_csv(
        DATA_DIR / "orders.csv",
        parse_dates=["order_time"],
        dtype={
            "order_id": "int32",
            "customer_id": "int32",
            "payment_method": "category",
            "country": "category",
            "device": "category",
            "source": "category",
            "discount_pct": "float32",
            "subtotal_usd": "float32",
            "total_usd": "float32",
        },
    )
    order_items = pd.read_csv(
        DATA_DIR / "order_items.csv",
        dtype={
            "order_id": "int32",
            "product_id": "int32",
            "unit_price_usd": "float32",
            "quantity": "int16",
            "line_total_usd": "float32",
        },
    )
    products = pd.read_csv(
        DATA_DIR / "products.csv",
        dtype={
            "product_id": "int32",
            "category": "category",
            "price_usd": "float32",
            "cost_usd": "float32",
            "margin_usd": "float32",
        },
    )
    reviews = pd.read_csv(
        DATA_DIR / "reviews.csv",
        parse_dates=["review_time"],
        dtype={"review_id": "int32", "order_id": "int32", "product_id": "int32", "rating": "int16"},
    )
    return {
        "customers": customers,
        "sessions": sessions,
        "events": events,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "reviews": reviews,
    }


def build_customer_profile(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["customers"].copy()
    sessions = data["sessions"]
    events = data["events"]
    orders = data["orders"]
    reviews = data["reviews"]

    male_names = {item.lower() for item in Provider.first_names_male}
    female_names = {item.lower() for item in Provider.first_names_female}

    customers["country_name"] = customers["country"].astype(str).map(safe_country_name)
    customers["region"] = customers["country"].astype(str).map(safe_region)
    customers["age_band"] = pd.cut(customers["age"], AGE_BINS, labels=AGE_LABELS)
    customers["age_band"] = customers["age_band"].cat.add_categories(["Unknown"]).fillna("Unknown")
    customers["gender_inferred"] = customers["name"].map(lambda x: infer_gender(x, male_names, female_names))

    session_base = sessions.groupby("customer_id", observed=False).agg(
        session_count=("session_id", "count"),
        first_session=("start_time", "min"),
        last_session=("start_time", "max"),
        source_diversity=("source", "nunique"),
        device_diversity=("device", "nunique"),
    )
    dominant_source = get_mode_by_group(sessions, "customer_id", "source", "dominant_source")
    dominant_device = get_mode_by_group(sessions, "customer_id", "device", "dominant_device")

    event_user = events.merge(sessions[["session_id", "customer_id"]], on="session_id", how="left")
    event_summary = (
        event_user.pivot_table(
            index="customer_id",
            columns="event_type",
            values="event_id",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .rename(
            columns={
                "page_view": "page_views",
                "add_to_cart": "add_to_cart_events",
                "checkout": "checkout_events",
                "purchase": "purchase_events",
            }
        )
        .reset_index()
    )
    required_event_cols = ["customer_id", "page_views", "add_to_cart_events", "checkout_events", "purchase_events"]
    event_summary = event_summary.reindex(columns=required_event_cols, fill_value=0)

    qty_by_customer = event_user.groupby("customer_id").agg(
        total_events=("event_id", "count"),
        total_qty=("qty", "sum"),
        mean_cart_size=("cart_size", "mean"),
    )
    event_summary = event_summary.merge(qty_by_customer, on="customer_id", how="left")

    order_summary = orders.groupby("customer_id").agg(
        order_count=("order_id", "count"),
        revenue=("total_usd", "sum"),
        subtotal=("subtotal_usd", "sum"),
        avg_order_value=("total_usd", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
        first_order=("order_time", "min"),
        last_order=("order_time", "max"),
    )

    reviews_with_customer = reviews.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
    review_summary = reviews_with_customer.groupby("customer_id").agg(
        review_count=("review_id", "count"),
        avg_rating=("rating", "mean"),
    )

    reference_time = max(
        data["events"]["timestamp"].max(),
        data["orders"]["order_time"].max(),
        data["sessions"]["start_time"].max(),
    )

    profile = customers.merge(session_base, on="customer_id", how="left")
    profile = profile.merge(dominant_source, on="customer_id", how="left")
    profile = profile.merge(dominant_device, on="customer_id", how="left")
    profile = profile.merge(event_summary, on="customer_id", how="left")
    profile = profile.merge(order_summary, on="customer_id", how="left")
    profile = profile.merge(review_summary, on="customer_id", how="left")

    fill_zero_cols = [
        "session_count",
        "source_diversity",
        "device_diversity",
        "page_views",
        "add_to_cart_events",
        "checkout_events",
        "purchase_events",
        "total_events",
        "total_qty",
        "order_count",
        "revenue",
        "subtotal",
        "avg_order_value",
        "avg_discount_pct",
        "review_count",
        "avg_rating",
        "mean_cart_size",
    ]
    for column in fill_zero_cols:
        profile[column] = profile[column].fillna(0)

    profile["dominant_source"] = profile["dominant_source"].astype("object").fillna("No Session")
    profile["dominant_device"] = profile["dominant_device"].astype("object").fillna("No Session")
    profile["active_days"] = (
        profile["last_session"].fillna(profile["signup_date"]) - profile["first_session"].fillna(profile["signup_date"])
    ).dt.days.add(1).clip(lower=0)
    profile["recency_days"] = (reference_time - profile["last_session"].fillna(profile["signup_date"])).dt.days.clip(lower=0)
    profile["session_to_order_rate"] = np.where(profile["session_count"] > 0, profile["order_count"] / profile["session_count"], 0)
    profile["page_to_cart_rate"] = np.where(profile["page_views"] > 0, profile["add_to_cart_events"] / profile["page_views"], 0)
    profile["cart_to_checkout_rate"] = np.where(
        profile["add_to_cart_events"] > 0, profile["checkout_events"] / profile["add_to_cart_events"], 0
    )
    profile["checkout_to_purchase_rate"] = np.where(
        profile["checkout_events"] > 0, profile["purchase_events"] / profile["checkout_events"], 0
    )
    profile["review_rate"] = np.where(profile["order_count"] > 0, profile["review_count"] / profile["order_count"], 0)
    profile["revenue_per_session"] = np.where(profile["session_count"] > 0, profile["revenue"] / profile["session_count"], 0)
    profile["marketing_opt_in"] = profile["marketing_opt_in"].astype(int)
    return profile


def score_series(series: pd.Series, reverse: bool = False, bins: int = 5) -> pd.Series:
    filled = pd.Series(series).fillna(series.median() if not pd.isna(series.median()) else 0)
    ranked = filled.rank(method="first", ascending=not reverse)
    try:
        scored = pd.qcut(ranked, q=min(bins, ranked.nunique()), labels=False, duplicates="drop") + 1
    except ValueError:
        scored = pd.Series(np.ones(len(series), dtype=int), index=series.index)
    scored = pd.Series(scored, index=series.index).astype(int)
    if reverse:
        return scored
    return scored


def add_rfm_segments(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = profile.copy()
    profile["r_score"] = score_series(profile["recency_days"], reverse=True)
    profile["f_score"] = score_series(profile["order_count"])
    profile["m_score"] = score_series(profile["revenue"])
    profile["rfm_score"] = profile["r_score"].astype(str) + profile["f_score"].astype(str) + profile["m_score"].astype(str)
    total = profile["r_score"] + profile["f_score"] + profile["m_score"]

    conditions = [
        (profile["r_score"] >= 4) & (profile["f_score"] >= 4) & (profile["m_score"] >= 4),
        (profile["r_score"] >= 4) & (profile["f_score"] >= 3),
        (profile["r_score"] <= 2) & (profile["f_score"] >= 4),
        (profile["r_score"] <= 2) & (profile["f_score"] <= 2) & (profile["m_score"] <= 2),
        (profile["f_score"] <= 2) & (profile["m_score"] <= 2),
    ]
    choices = ["Champions", "Loyal Customers", "At-Risk High Value", "Lost Low Value", "Price / Casual Shoppers"]
    profile["rfm_segment"] = np.select(conditions, choices, default="Potential Loyalists")

    rfm_summary = (
        profile.groupby("rfm_segment", observed=False)
        .agg(
            customers=("customer_id", "count"),
            revenue=("revenue", "sum"),
            avg_orders=("order_count", "mean"),
            avg_recency=("recency_days", "mean"),
        )
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    return profile, rfm_summary


def add_lifecycle_stage(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = profile.copy()
    conditions = [
        (profile["order_count"] == 0) & (profile["recency_days"] <= 30),
        (profile["order_count"] == 0) & (profile["session_count"] > 0),
        (profile["order_count"] == 1) & (profile["recency_days"] <= 45),
        (profile["order_count"] >= 2) & (profile["recency_days"] <= 60),
        (profile["order_count"] >= 1) & (profile["recency_days"] > 60) & (profile["recency_days"] <= 120),
        (profile["order_count"] >= 1) & (profile["recency_days"] > 120),
    ]
    choices = [
        "New Visitor",
        "Browsing Prospect",
        "New Buyer",
        "Active Repeat Buyer",
        "Cooling Down",
        "Dormant Customer",
    ]
    profile["lifecycle_stage"] = np.select(conditions, choices, default="Unclassified")
    lifecycle_summary = (
        profile.groupby("lifecycle_stage", observed=False)
        .agg(customers=("customer_id", "count"), revenue=("revenue", "sum"))
        .sort_values("customers", ascending=False)
        .reset_index()
    )
    return profile, lifecycle_summary


def select_clusters(cluster_frame: pd.DataFrame) -> tuple[np.ndarray, int, list[tuple[int, float]], np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_frame)

    sample_size = min(5000, len(cluster_frame))
    rng = np.random.default_rng(CLUSTER_RANDOM_STATE)
    sample_index = (
        rng.choice(len(cluster_frame), size=sample_size, replace=False)
        if sample_size < len(cluster_frame)
        else np.arange(len(cluster_frame))
    )
    sample_scaled = scaled[sample_index]

    scores: list[tuple[int, float]] = []
    best_score = -1.0
    best_k = 0

    for k in range(3, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=CLUSTER_RANDOM_STATE)
        sample_labels = model.fit_predict(sample_scaled)
        if len(np.unique(sample_labels)) <= 1:
            score = -1.0
        else:
            score = silhouette_score(sample_scaled, sample_labels)
        scores.append((k, score))
        if score > best_score:
            best_score = score
            best_k = k

    final_model_cls = MiniBatchKMeans if len(cluster_frame) > 20000 else KMeans
    final_model = final_model_cls(n_clusters=best_k, n_init=20, random_state=CLUSTER_RANDOM_STATE)
    best_labels = final_model.fit_predict(scaled)
    return best_labels, best_k, scores, scaled


def assign_cluster_personas(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    rating_non_zero = profile.loc[profile["avg_rating"] > 0, "avg_rating"]
    rating_mean = 0 if rating_non_zero.empty else rating_non_zero.mean()

    cluster_features = pd.DataFrame(
        {
            "session_count_log": np.log1p(profile["session_count"]),
            "events_log": np.log1p(profile["total_events"]),
            "page_to_cart_rate": profile["page_to_cart_rate"],
            "cart_to_checkout_rate": profile["cart_to_checkout_rate"],
            "checkout_to_purchase_rate": profile["checkout_to_purchase_rate"],
            "orders_log": np.log1p(profile["order_count"]),
            "revenue_log": np.log1p(profile["revenue"]),
            "avg_order_value": profile["avg_order_value"],
            "avg_discount_pct": profile["avg_discount_pct"],
            "avg_rating": profile["avg_rating"].replace(0, np.nan).fillna(rating_mean),
            "active_days": profile["active_days"],
            "recency_days": profile["recency_days"],
            "source_diversity": profile["source_diversity"],
            "device_diversity": profile["device_diversity"],
            "review_rate": profile["review_rate"],
            "age": profile["age"].fillna(profile["age"].median()),
            "marketing_opt_in": profile["marketing_opt_in"],
        }
    ).fillna(0)

    labels, best_k, scores, scaled_features = select_clusters(cluster_features)
    profile = profile.copy()
    profile["cluster_id"] = labels
    components = PCA(n_components=2, random_state=CLUSTER_RANDOM_STATE).fit_transform(scaled_features)
    profile["cluster_x"] = components[:, 0]
    profile["cluster_y"] = components[:, 1]

    cluster_summary = profile.groupby("cluster_id").agg(
        users=("customer_id", "count"),
        avg_age=("age", "mean"),
        sessions=("session_count", "mean"),
        orders=("order_count", "mean"),
        revenue=("revenue", "mean"),
        aov=("avg_order_value", "mean"),
        conversion=("session_to_order_rate", "mean"),
        avg_discount=("avg_discount_pct", "mean"),
        avg_rating=("avg_rating", "mean"),
        recency=("recency_days", "mean"),
    )
    overall = cluster_summary.mean()

    revenue_threshold = cluster_summary["revenue"].quantile(0.75)
    order_threshold = cluster_summary["orders"].quantile(0.75)
    session_threshold = cluster_summary["sessions"].quantile(0.75)
    recency_threshold = cluster_summary["recency"].quantile(0.75)
    discount_threshold = cluster_summary["avg_discount"].quantile(0.75)
    conversion_threshold = cluster_summary["conversion"].quantile(0.75)

    persona_names: dict[int, str] = {}
    used = set()

    for cluster_id, row in cluster_summary.sort_values(["revenue", "orders"], ascending=False).iterrows():
        if row["revenue"] >= revenue_threshold and row["orders"] >= order_threshold:
            label = "High-Value Repeat Buyers"
        elif row["sessions"] >= session_threshold and row["conversion"] < overall["conversion"]:
            label = "Heavy Browsers, Low Conversion"
        elif row["avg_discount"] >= discount_threshold:
            label = "Promotion Sensitive"
        elif row["recency"] >= recency_threshold:
            label = "Dormant / Churn Risk"
        elif row["conversion"] >= conversion_threshold:
            label = "High Intent, High Conversion"
        else:
            label = "Steady Mainstream"
        if label in used:
            label = f"{label} {cluster_id + 1}"
        used.add(label)
        persona_names[cluster_id] = label

    profile["cluster_name"] = profile["cluster_id"].map(persona_names)
    cluster_summary["cluster_name"] = cluster_summary.index.map(persona_names)
    cluster_summary["silhouette_score"] = dict(scores)[best_k]

    region_mix = (
        profile.groupby(["cluster_name", "region"])
        .size()
        .reset_index(name="users")
        .sort_values(["cluster_name", "users"], ascending=[True, False])
        .drop_duplicates("cluster_name")
    )
    source_mix = (
        profile.groupby(["cluster_name", "dominant_source"])
        .size()
        .reset_index(name="users")
        .sort_values(["cluster_name", "users"], ascending=[True, False])
        .drop_duplicates("cluster_name")
    )

    cluster_display = profile.groupby("cluster_name").agg(
        users=("customer_id", "count"),
        revenue=("revenue", "mean"),
        orders=("order_count", "mean"),
        sessions=("session_count", "mean"),
        conversion=("session_to_order_rate", "mean"),
        avg_age=("age", "mean"),
        avg_rating=("avg_rating", "mean"),
    )

    persona_cards = []
    for cluster_name, row in cluster_display.sort_values("revenue", ascending=False).iterrows():
        top_region = region_mix.loc[region_mix["cluster_name"] == cluster_name, "region"].iloc[0]
        top_source = source_mix.loc[source_mix["cluster_name"] == cluster_name, "dominant_source"].iloc[0]
        persona_cards.append(
            {
                "cluster_name": cluster_name,
                "users": format_number(row["users"]),
                "revenue": format_currency(row["revenue"]),
                "orders": f"{row['orders']:.2f}",
                "sessions": f"{row['sessions']:.2f}",
                "conversion": format_pct(row["conversion"]),
                "avg_age": f"{row['avg_age']:.1f}" if not pd.isna(row["avg_age"]) else "-",
                "avg_rating": f"{row['avg_rating']:.2f}",
                "top_region": top_region,
                "top_source": top_source,
            }
        )

    return profile, cluster_summary.reset_index(), persona_cards


def monthly_overview(data: dict[str, pd.DataFrame], reviews_with_orders: pd.DataFrame) -> pd.DataFrame:
    sessions = data["sessions"].copy()
    orders = data["orders"].copy()
    reviews_with_orders = reviews_with_orders.copy()

    sessions["month"] = sessions["start_time"].dt.to_period("M").dt.to_timestamp()
    orders["month"] = orders["order_time"].dt.to_period("M").dt.to_timestamp()
    reviews_with_orders["month"] = reviews_with_orders["review_time"].dt.to_period("M").dt.to_timestamp()

    sessions_monthly = sessions.groupby("month").agg(
        sessions=("session_id", "count"),
        active_users=("customer_id", "nunique"),
    )
    orders_monthly = orders.groupby("month").agg(
        orders=("order_id", "count"),
        revenue=("total_usd", "sum"),
        aov=("total_usd", "mean"),
        buyers=("customer_id", "nunique"),
    )
    ratings_monthly = reviews_with_orders.groupby("month").agg(
        avg_rating=("rating", "mean"),
        review_count=("review_id", "count"),
    )
    monthly = sessions_monthly.join(orders_monthly, how="outer").join(ratings_monthly, how="left").fillna(0).reset_index()
    monthly["session_to_order"] = np.where(monthly["sessions"] > 0, monthly["orders"] / monthly["sessions"], 0)
    monthly["revenue_rolling_3m"] = monthly["revenue"].rolling(3, min_periods=1).mean()
    return monthly


def funnel_metrics(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = data["sessions"][["session_id", "device", "source"]].copy()
    stage_flags = (
        data["events"]
        .assign(flag=1)
        .pivot_table(index="session_id", columns="event_type", values="flag", aggfunc="max", fill_value=0, observed=False)
        .reset_index()
    )
    funnel_base = sessions.merge(stage_flags, on="session_id", how="left")
    stage_order = ["page_view", "add_to_cart", "checkout", "purchase"]
    for stage in stage_order:
        if stage not in funnel_base.columns:
            funnel_base[stage] = 0
    funnel_base[stage_order] = funnel_base[stage_order].fillna(0)

    overall = pd.DataFrame({"stage": stage_order, "sessions": [int(funnel_base[stage].sum()) for stage in stage_order]})
    device_rates = funnel_base.groupby("device", observed=False)[stage_order].mean().reindex(["desktop", "mobile", "tablet"])
    source_rates = funnel_base.groupby("source", observed=False)[stage_order].mean().reindex(
        ["organic", "direct", "paid", "social", "email", "referral"]
    )
    return overall, device_rates, source_rates


def geography_metrics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sessions_country = data["sessions"].groupby("country", observed=False).agg(
        sessions=("session_id", "count"),
        active_users=("customer_id", "nunique"),
    )
    orders_country = data["orders"].groupby("country", observed=False).agg(
        orders=("order_id", "count"),
        revenue=("total_usd", "sum"),
        aov=("total_usd", "mean"),
    )
    geo = sessions_country.join(orders_country, how="outer").fillna(0).reset_index()
    geo["country"] = geo["country"].astype(str)
    geo["country_name"] = geo["country"].map(safe_country_name)
    geo["region"] = geo["country"].map(safe_region)
    geo["conversion"] = np.where(geo["sessions"] > 0, geo["orders"] / geo["sessions"], 0)
    return geo


def demographic_frames(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sunburst_frame = (
        profile.groupby(["region", "gender_inferred", "age_band"], observed=True)
        .size()
        .reset_index(name="users")
    )
    sunburst_frame = sunburst_frame.loc[sunburst_frame["users"] > 0].copy()
    sunburst_frame["region"] = sunburst_frame["region"].fillna("Unknown")
    sunburst_frame["gender_inferred"] = sunburst_frame["gender_inferred"].fillna("Unknown")
    sunburst_frame["age_band"] = sunburst_frame["age_band"].astype(str).replace({"nan": "Unknown"})

    age_gender = profile.groupby(["age_band", "gender_inferred"], observed=True).size().reset_index(name="users")
    age_gender = age_gender.loc[age_gender["users"] > 0].copy()
    age_gender["plot_users"] = np.where(age_gender["gender_inferred"] == "Male", -age_gender["users"], age_gender["users"])
    age_gender["age_band"] = pd.Categorical(age_gender["age_band"], categories=AGE_LABELS + ["Unknown"], ordered=True)
    age_gender = age_gender.sort_values("age_band")
    return sunburst_frame, age_gender


def product_metrics(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order_items = data["order_items"]
    products = data["products"]
    orders = data["orders"][["order_id", "order_time"]]
    reviews = data["reviews"]

    order_line = (
        order_items.merge(products[["product_id", "category", "margin_usd"]], on="product_id", how="left")
        .merge(orders, on="order_id", how="left")
    )
    order_line["month"] = order_line["order_time"].dt.to_period("M").dt.to_timestamp()
    order_line["gross_margin"] = order_line["margin_usd"] * order_line["quantity"]

    category_monthly = (
        order_line.groupby(["month", "category"], observed=False)
        .agg(revenue=("line_total_usd", "sum"), units=("quantity", "sum"), gross_margin=("gross_margin", "sum"))
        .reset_index()
    )

    product_review = reviews.groupby("product_id").agg(avg_rating=("rating", "mean"), review_count=("review_id", "count")).reset_index()
    category_perf = (
        order_line.groupby("category", observed=False)
        .agg(revenue=("line_total_usd", "sum"), units=("quantity", "sum"), gross_margin=("gross_margin", "sum"), products=("product_id", "nunique"))
        .reset_index()
        .merge(
            products[["product_id", "category"]]
            .merge(product_review, on="product_id", how="left")
            .groupby("category", observed=False)
            .agg(avg_rating=("avg_rating", "mean"), review_count=("review_count", "sum"))
            .reset_index(),
            on="category",
            how="left",
        )
    )
    numeric_cols = category_perf.select_dtypes(include=[np.number]).columns
    category_perf[numeric_cols] = category_perf[numeric_cols].fillna(0)

    top_products = (
        data["order_items"]
        .merge(data["products"][["product_id", "name", "category"]], on="product_id", how="left")
        .groupby(["product_id", "name", "category"], observed=False)
        .agg(revenue=("line_total_usd", "sum"), units=("quantity", "sum"))
        .reset_index()
        .merge(product_review, on="product_id", how="left")
        .fillna({"avg_rating": 0, "review_count": 0})
        .sort_values(["revenue", "units"], ascending=False)
        .head(12)
    )
    return category_monthly, category_perf, top_products


def sentiment_product_metrics(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (product_master, sentiment_pct_by_category) using VADER sentiment."""
    products = data["products"]
    order_items = data["order_items"]
    reviews = data["reviews"].copy()

    # VADER sentiment scoring
    _sia = SentimentIntensityAnalyzer()

    def _vader_label(text: str) -> str:
        compound = _sia.polarity_scores(str(text))["compound"]
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    reviews["sentiment"] = reviews["review_text"].apply(_vader_label)

    # Product master: sales + ratings
    product_sales = (
        order_items.groupby("product_id")
        .agg(total_qty=("quantity", "sum"), total_revenue=("line_total_usd", "sum"))
        .reset_index()
    )
    product_ratings = (
        reviews.groupby("product_id")
        .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))
        .reset_index()
    )
    product_master = (
        products.merge(product_sales, on="product_id", how="left")
        .merge(product_ratings, on="product_id", how="left")
        .fillna(0)
    )

    # Sentiment pivot by category
    cat_sentiment = reviews.merge(products[["product_id", "category"]], on="product_id")
    sentiment_pivot = cat_sentiment.groupby(["category", "sentiment"]).size().unstack(fill_value=0)
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in sentiment_pivot.columns:
            sentiment_pivot[col] = 0
    sentiment_pct = sentiment_pivot.div(sentiment_pivot.sum(axis=1), axis=0) * 100

    return product_master, sentiment_pct, reviews


def retention_matrix(data: dict[str, pd.DataFrame], activity_source: str = "sessions") -> pd.DataFrame:
    customers = data["customers"][["customer_id", "signup_date"]].copy()

    if activity_source == "sessions":
        activity_df = data["sessions"][["customer_id", "start_time"]].rename(columns={"start_time": "activity_time"}).copy()
    else:
        activity_df = data["orders"][["customer_id", "order_time"]].rename(columns={"order_time": "activity_time"}).copy()

    customers["cohort_month"] = customers["signup_date"].dt.to_period("M").dt.to_timestamp()
    activity_df["activity_month"] = activity_df["activity_time"].dt.to_period("M").dt.to_timestamp()

    activity = customers.merge(activity_df[["customer_id", "activity_month"]], on="customer_id", how="left").dropna(
        subset=["activity_month"]
    )
    activity["month_number"] = (
        (activity["activity_month"].dt.year - activity["cohort_month"].dt.year) * 12
        + (activity["activity_month"].dt.month - activity["cohort_month"].dt.month)
    )
    activity = activity[(activity["month_number"] >= 0) & (activity["month_number"] <= 12)].drop_duplicates(
        ["customer_id", "activity_month"]
    )

    cohort_size = customers.groupby("cohort_month").agg(cohort_size=("customer_id", "nunique"))
    cohort_activity = activity.groupby(["cohort_month", "month_number"]).agg(active_users=("customer_id", "nunique")).reset_index()
    retention = cohort_activity.merge(cohort_size, on="cohort_month", how="left")
    retention["retention"] = retention["active_users"] / retention["cohort_size"]
    return retention.pivot(index="cohort_month", columns="month_number", values="retention").sort_index()


def build_theme() -> dict:
    return dict(
        layout=dict(
            font=dict(
                family="Segoe UI, Microsoft YaHei, sans-serif",
                color="#F8FAFC"   # global font color
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=COLOR_SEQUENCE,
            margin=dict(l=40, r=30, t=70, b=45),
            hovermode="x unified",

            # unified hover label style
            hoverlabel=dict(
                bgcolor="rgba(7, 17, 26, 0.96)",  # dark background
                bordercolor="#94A3B8",
                font=dict(
                    color="#FFFFFF",             # white text
                    size=15
                ),
                align="left"
            ),

            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.12)",
                zeroline=False,
                linecolor="rgba(148,163,184,0.18)",
                tickfont=dict(color="#EAF2FF", size=13),
                title_font=dict(color="#FFFFFF", size=16)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.12)",
                zeroline=False,
                linecolor="rgba(148,163,184,0.18)",
                tickfont=dict(color="#EAF2FF", size=13),
                title_font=dict(color="#FFFFFF", size=16)
            ),
        )
    )


def apply_theme(fig: go.Figure, title: str, height: int = 480) -> go.Figure:
    theme = build_theme()
    fig.update_layout(**theme["layout"])
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        height=height,
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    return fig


def make_overview_figure(monthly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["sessions"],
            mode="lines",
            name="Sessions",
            line=dict(width=2.5, color="#6C8CFF"),
            fill="tozeroy",
            fillcolor="rgba(108,140,255,0.12)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["orders"], mode="lines", name="Orders", line=dict(width=2.5, color="#00C2A8")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["revenue"], mode="lines", name="Revenue", line=dict(width=3, color="#FFB84D")),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["revenue_rolling_3m"],
            mode="lines",
            name="Revenue (3M Avg)",
            line=dict(width=2, dash="dash", color="#B388FF"),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Sessions / Orders", secondary_y=False)
    fig.update_yaxes(title_text="Revenue (USD)", secondary_y=True)
    return apply_theme(fig, "Traffic, Orders, and Revenue Overview", 470)


def make_geo_figure(geo: pd.DataFrame) -> go.Figure:
    fig = px.choropleth(
        geo,
        locations="country_name",
        locationmode="country names",
        color="revenue",
        hover_name="country_name",
        custom_data=["sessions", "orders", "conversion", "region"],
        color_continuous_scale=["#0B1F33", "#103B4C", "#00C2A8", "#FFB84D"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>Region=%{customdata[3]}"
            "<br>Sessions=%{customdata[0]:,.0f}<br>Orders=%{customdata[1]:,.0f}"
            "<br>Conversion=%{customdata[2]:.1%}<br>Revenue=$%{z:,.0f}<extra></extra>"
        )
    )
    fig.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False, projection_type="natural earth", landcolor="#0C1C2D", lakecolor="#0C1C2D"),
        coloraxis_colorbar=dict(title="Revenue"),
    )
    return apply_theme(fig, "Geographic Revenue Heatmap", 500)


def make_sunburst_figure(sunburst_frame: pd.DataFrame) -> go.Figure:
    sunburst_frame = sunburst_frame.copy()
    sunburst_frame = sunburst_frame.loc[sunburst_frame["users"] > 0].copy()

    if sunburst_frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No demographic data available",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return apply_theme(fig, "Audience Profile: Region × Gender × Age Band", 500)

    fig = px.sunburst(
        sunburst_frame,
        path=["region", "gender_inferred", "age_band"],
        values="users",
        color="users",
        color_continuous_scale=["#103B4C", "#6C8CFF", "#FF6B6B", "#FFB84D"],
    )
    fig.update_traces(
        insidetextorientation="radial",
        hovertemplate="%{label}<br>Users=%{value:,.0f}<extra></extra>"
    )
    return apply_theme(fig, "Audience Profile: Region × Gender × Age Band", 500)

def make_age_pyramid(age_gender: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    male = age_gender[age_gender["gender_inferred"] == "Male"]
    female = age_gender[age_gender["gender_inferred"] == "Female"]
    other = age_gender[~age_gender["gender_inferred"].isin(["Male", "Female"])]

    fig.add_trace(go.Bar(y=male["age_band"], x=male["plot_users"], orientation="h", name="Male", marker_color="#6C8CFF"))
    fig.add_trace(go.Bar(y=female["age_band"], x=female["plot_users"], orientation="h", name="Female", marker_color="#FF6B6B"))
    if not other.empty:
        fig.add_trace(go.Bar(y=other["age_band"], x=other["plot_users"], orientation="h", name="Unknown / Ambiguous", marker_color="#8EA6C9"))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(tickvals=[-4000, -2000, 0, 2000, 4000], ticktext=["4k", "2k", "0", "2k", "4k"])
    return apply_theme(fig, "Age Pyramid by Inferred Gender", 420)


def make_cluster_scatter(profile: pd.DataFrame) -> go.Figure:
    sample = profile.sample(n=min(12000, len(profile)), random_state=CLUSTER_RANDOM_STATE)
    fig = px.scatter(
        sample,
        x="cluster_x",
        y="cluster_y",
        color="cluster_name",
        size=np.clip(sample["revenue"] + 20, 20, 600),
        size_max=22,
        render_mode="webgl",
        hover_data={
            "region": True,
            "age": True,
            "session_count": ":,.0f",
            "order_count": ":,.0f",
            "revenue": ":.2f",
            "cluster_x": False,
            "cluster_y": False,
        },
    )
    fig.update_traces(marker=dict(opacity=0.72, line=dict(width=0)))
    fig.update_layout(legend_title_text="Persona")
    return apply_theme(fig, "Customer Clusters in PCA Space", 500)


def make_cluster_heatmap(cluster_summary: pd.DataFrame) -> go.Figure:
    display = cluster_summary[["cluster_name", "sessions", "orders", "revenue", "aov", "conversion", "avg_discount", "avg_rating", "recency", "avg_age"]].set_index("cluster_name")
    normalized = (display - display.mean()) / display.std(ddof=0).replace(0, 1)
    fig = go.Figure(
        data=go.Heatmap(
            z=normalized.values,
            x=["Sessions", "Orders", "Revenue / User", "AOV", "Conversion", "Discount", "Rating", "Recency", "Age"],
            y=normalized.index.tolist(),
            colorscale=[[0.0, "#112438"], [0.5, "#1C4258"], [0.75, "#00C2A8"], [1.0, "#FFB84D"]],
            hovertemplate="Persona=%{y}<br>Metric=%{x}<br>Relative Strength=%{z:.2f}<extra></extra>",
        )
    )
    return apply_theme(fig, "Cluster Feature Heatmap", 420)


def make_funnel_figure(overall_funnel: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Funnel(
            y=["Page View", "Add to Cart", "Checkout", "Purchase"],
            x=overall_funnel["sessions"],
            textinfo="value+percent previous",
            marker=dict(color=["#6C8CFF", "#56CCF2", "#00C2A8", "#FFB84D"]),
        )
    )
    return apply_theme(fig, "Session Funnel with Explicit Add-to-Cart Stage", 420)


def make_segment_heatmap(device_rates: pd.DataFrame, source_rates: pd.DataFrame) -> go.Figure:
    zmax = max(device_rates.max().max(), source_rates.max().max())
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Device Funnel Rates", "Channel Funnel Rates"))
    device_text = (device_rates[["page_view", "add_to_cart", "checkout", "purchase"]] * 100).round(1).astype(str) + "%"
    source_text = (source_rates[["page_view", "add_to_cart", "checkout", "purchase"]] * 100).round(1).astype(str) + "%"

    fig.add_trace(
        go.Heatmap(
            z=device_rates[["page_view", "add_to_cart", "checkout", "purchase"]].values,
            x=["Page View", "Add to Cart", "Checkout", "Purchase"],
            y=device_rates.index.tolist(),
            colorscale=[[0.0, "#112438"], [0.5, "#1C4258"], [0.75, "#00C2A8"], [1.0, "#FFB84D"]],
            zmin=0,
            zmax=zmax,
            text=device_text,
            texttemplate="%{text}",
            hovertemplate="Group=%{y}<br>Stage=%{x}<br>Rate=%{z:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=source_rates[["page_view", "add_to_cart", "checkout", "purchase"]].values,
            x=["Page View", "Add to Cart", "Checkout", "Purchase"],
            y=source_rates.index.tolist(),
            colorscale=[[0.0, "#112438"], [0.5, "#1C4258"], [0.75, "#00C2A8"], [1.0, "#FFB84D"]],
            zmin=0,
            zmax=zmax,
            text=source_text,
            texttemplate="%{text}",
            hovertemplate="Group=%{y}<br>Stage=%{x}<br>Rate=%{z:.1%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    return apply_theme(fig, "Funnel Performance by Device and Channel", 440)


def make_category_bubble(category_perf: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        category_perf,
        x="revenue",
        y="gross_margin",
        size="units",
        color="avg_rating",
        text="category",
        size_max=65,
        color_continuous_scale=["#6C8CFF", "#00C2A8", "#FFB84D"],
        hover_data={"review_count": ":,.0f", "products": ":,.0f"},
    )
    fig.update_traces(textposition="top center")
    return apply_theme(fig, "Category Matrix: Revenue, Margin, Units, Rating", 470)


def make_category_area(category_monthly: pd.DataFrame) -> go.Figure:
    fig = px.area(category_monthly, x="month", y="revenue", color="category", groupnorm=None, line_group="category")
    return apply_theme(fig, "Category Revenue Trend Over Time", 430)


def make_retention_heatmap(retention: pd.DataFrame, title: str) -> go.Figure:
    retention_text = retention.apply(lambda column: column.map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%")).values
    fig = go.Figure(
        data=go.Heatmap(
            z=retention.values,
            x=[f"M{int(col)}" for col in retention.columns],
            y=[pd.Timestamp(idx).strftime("%Y-%m") for idx in retention.index],
            colorscale=[[0.0, "#112438"], [0.35, "#1C4258"], [0.7, "#00C2A8"], [1.0, "#FFB84D"]],
            text=retention_text,
            texttemplate="%{text}",
            hovertemplate="Cohort=%{y}<br>Month=%{x}<br>Retention=%{z:.1%}<extra></extra>",
        )
    )
    return apply_theme(fig, title, 500)


def make_rfm_bar(rfm_summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        rfm_summary.sort_values("revenue", ascending=False),
        x="rfm_segment",
        y="revenue",
        color="customers",
        text="customers",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Revenue")
    return apply_theme(fig, "RFM Segments by Revenue Contribution", 430)


def make_lifecycle_donut(lifecycle_summary: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        lifecycle_summary,
        names="lifecycle_stage",
        values="customers",
        hole=0.55,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_theme(fig, "Customer Lifecycle Mix", 430)


def make_top_products_table(top_products: pd.DataFrame) -> go.Figure:
    display = top_products.copy()
    display["revenue"] = display["revenue"].map(lambda x: f"${x:,.0f}")
    display["avg_rating"] = display["avg_rating"].map(lambda x: f"{x:.2f}")
    display["review_count"] = display["review_count"].astype(int)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Product", "Category", "Revenue", "Units", "Rating", "Reviews"],
                    fill_color="#102638",
                    line_color="rgba(148,163,184,0.16)",
                    align="left",
                    font=dict(color="#E8EDF8", size=12),
                ),
                cells=dict(
                    values=[
                        display["name"],
                        display["category"],
                        display["revenue"],
                        display["units"],
                        display["avg_rating"],
                        display["review_count"],
                    ],
                    fill_color="rgba(8,21,34,0.82)",
                    line_color="rgba(148,163,184,0.12)",
                    align="left",
                    font=dict(color="#E8EDF8", size=11),
                    height=28,
                ),
            )
        ]
    )
    return apply_theme(fig, "Top Products Snapshot", 430)


def make_product_sales_rank(product_master: pd.DataFrame) -> go.Figure:
    top = product_master.sort_values("total_qty", ascending=False).head(10)
    fig = px.bar(
        top,
        x="total_qty",
        y="name",
        orientation="h",
        color="category",
        text="total_qty",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="Units Sold")
    return apply_theme(fig, "Top 10 Products by Sales Volume", 460)


def make_product_rating_rank(product_master: pd.DataFrame) -> go.Figure:
    top_rated = (
        product_master[product_master["review_count"] >= 5]
        .sort_values("avg_rating", ascending=False)
        .head(10)
    )
    fig = px.bar(
        top_rated,
        x="avg_rating",
        y="name",
        orientation="h",
        color="category",
        text="avg_rating",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="Average Rating", range=[0, 5.5])
    return apply_theme(fig, "Top 10 Rated Products (Min. 5 Reviews)", 460)


def make_sentiment_stacked_bar(sentiment_pct: pd.DataFrame) -> go.Figure:
    cats = sentiment_pct.index.tolist()
    fig = go.Figure()
    color_map = {"Positive": "#00C2A8", "Neutral": "#8FA2BE", "Negative": "#FF6B6B"}
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in sentiment_pct.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment,
                    y=cats,
                    x=sentiment_pct[sentiment].values,
                    orientation="h",
                    marker_color=color_map[sentiment],
                    text=sentiment_pct[sentiment].map(lambda v: f"{v:.1f}%"),
                    textposition="inside",
                    hovertemplate=f"{sentiment}: %{{x:.1f}}%<extra></extra>",
                )
            )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Share (%)", range=[0, 100])
    fig.update_yaxes(title=None)
    return apply_theme(fig, "Sentiment Distribution by Category (VADER)", 460)


def generate_wordcloud_section_html(reviews: pd.DataFrame, products: pd.DataFrame) -> str:
    """Generate one word cloud per category, return an HTML block of base64-embedded images."""
    merged = reviews.merge(products[["product_id", "category"]], on="product_id", how="left")
    categories = sorted(merged["category"].dropna().unique())
    blocks = []
    for cat in categories:
        texts = merged.loc[merged["category"] == cat, "review_text"].dropna()
        if texts.empty:
            continue
        combined = " ".join(texts.astype(str))
        wc = WordCloud(
            width=700,
            height=340,
            background_color=None,
            mode="RGBA",
            colormap="cool",
            max_words=120,
        ).generate(combined)
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        blocks.append(
            f"""<div class="wc-card reveal">
  <div class="wc-label">{cat}</div>
  <img src="data:image/png;base64,{b64}" alt="{cat} keyword cloud" style="width:100%;border-radius:14px;" />
</div>"""
        )
    return "\n".join(blocks)


def kpi_cards_html(kpis: list[dict[str, str]]) -> str:
    return "".join(
        f"""
        <div class="metric-block reveal">
            <div class="metric-label">{kpi['label']}</div>
            <div class="metric-value">{kpi['value']}</div>
            <div class="metric-detail">{kpi['detail']}</div>
        </div>
        """
        for kpi in kpis
    )


def persona_cards_html(cards: list[dict[str, str]]) -> str:
    return "".join(
        f"""
        <article class="persona-card reveal">
            <div class="persona-kicker">Persona</div>
            <h3>{item['cluster_name']}</h3>
            <p>Top region: {item['top_region']} / Main source: {item['top_source']}</p>
            <div class="persona-grid">
                <div><span>Users</span><strong>{item['users']}</strong></div>
                <div><span>Revenue / User</span><strong>{item['revenue']}</strong></div>
                <div><span>Orders / User</span><strong>{item['orders']}</strong></div>
                <div><span>Sessions / User</span><strong>{item['sessions']}</strong></div>
                <div><span>Conversion</span><strong>{item['conversion']}</strong></div>
                <div><span>Avg Age</span><strong>{item['avg_age']}</strong></div>
                <div><span>Avg Rating</span><strong>{item['avg_rating']}</strong></div>
            </div>
        </article>
        """
        for item in cards
    )


def figure_html(fig: go.Figure, include_js: bool = False) -> str:
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"displayModeBar": False, "responsive": True},
    )


def compose_dashboard(
    kpis: list[dict[str, str]],
    figures: dict[str, go.Figure],
    persona_cards: list[dict[str, str]],
    data_notes: list[str],
    wordcloud_html: str = "",
) -> str:
    ordered_keys = [
        "overview",
        "geo",
        "sunburst",
        "age_pyramid",
        "cluster_scatter",
        "cluster_heatmap",
        "rfm_bar",
        "lifecycle_donut",
        "funnel",
        "segment_heatmap",
        "product_sales_rank",
        "product_rating_rank",
        "sentiment_bar",
        "category_bubble",
        "category_area",
        "top_products",
        "retention_session",
        "retention_purchase",
    ]
    figure_blocks = {key: figure_html(figures[key], include_js=(index == 0)) for index, key in enumerate(ordered_keys)}
    note_items = "".join(f"<li>{item}</li>" for item in data_notes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Customer Analytics Dashboard</title>
<style>
:root {{--bg:#07111a;--panel:rgba(8,21,34,.82);--panel-strong:rgba(10,26,42,.96);--line:rgba(148,163,184,.16);--text:#eef3ff;--muted:#8fa2be;--accent:#00c2a8;--shadow:0 24px 70px rgba(0,0,0,.32);}}
* {{box-sizing:border-box;}}
html {{scroll-behavior:smooth;}}
body {{margin:0;font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:var(--text);background:radial-gradient(circle at top left,rgba(108,140,255,.22),transparent 28%),radial-gradient(circle at 78% 18%,rgba(0,194,168,.18),transparent 26%),linear-gradient(180deg,#061018 0%,#07111a 52%,#081520 100%);}}
a {{color:inherit;text-decoration:none;}}
.shell {{width:min(1480px,calc(100vw - 40px));margin:0 auto;padding:28px 0 56px;}}
.topbar {{position:sticky;top:10px;z-index:40;display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border:1px solid var(--line);border-radius:18px;background:rgba(7,17,26,.72);backdrop-filter:blur(18px);box-shadow:var(--shadow);}}
.brand {{display:flex;gap:14px;align-items:baseline;}}
.brand strong {{font-size:1.15rem;letter-spacing:.06em;text-transform:uppercase;}}
.brand span,.nav,.meta-title,.metric-label,.persona-grid span {{color:var(--muted);}}
.nav {{display:flex;gap:16px;flex-wrap:wrap;font-size:.92rem;}}
.hero {{min-height:calc(100svh - 92px);display:grid;grid-template-columns:1.2fr .8fr;gap:24px;align-items:end;padding:36px 0 22px;}}
.hero-copy {{padding:28px 8px 8px 0;}}
.eyebrow,.persona-kicker {{display:inline-flex;align-items:center;gap:8px;color:var(--accent);text-transform:uppercase;letter-spacing:.12em;font-size:.8rem;}}
.hero h1 {{margin:0;font-size:clamp(2.9rem,6vw,6rem);line-height:.92;letter-spacing:-.04em;max-width:900px;}}
.hero p,.meta-list,.section-head p,.persona-card p,.footer-note ul {{color:#c9d6ea;line-height:1.75;}}
.hero-meta {{display:grid;gap:16px;align-self:stretch;}}
.meta-panel,.section,.persona-card,.footer-note {{border:1px solid var(--line);background:var(--panel);border-radius:28px;box-shadow:var(--shadow);}}
.meta-panel {{padding:24px;display:grid;gap:16px;align-content:start;}}
.meta-list {{list-style:none;padding:0;margin:0;display:grid;gap:10px;}}
.kpi-strip {{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px;margin:0 0 24px;}}
.metric-block {{padding:18px 20px;border:1px solid var(--line);background:rgba(7,18,29,.88);border-radius:22px;min-height:124px;}}
.metric-value {{margin-top:12px;font-size:clamp(1.7rem,3vw,2.8rem);letter-spacing:-.04em;font-weight:700;}}
.metric-detail {{margin-top:8px;color:#c2d0e4;font-size:.92rem;}}
.section {{padding:24px 24px 14px;margin-bottom:24px;}}
.section-head {{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:end;margin-bottom:12px;}}
.section-head h2,.footer-note h3,.persona-card h3 {{margin:0;}}
.section-head h2 {{font-size:clamp(1.4rem,2vw,2.1rem);letter-spacing:-.03em;}}
.grid-2,.persona-row {{display:grid;gap:18px;}}
.grid-2 {{grid-template-columns:repeat(2,minmax(0,1fr));}}
.persona-row {{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:18px;}}
.figure-shell {{padding:4px 2px 0;border-radius:24px;overflow:hidden;background:linear-gradient(180deg,rgba(18,35,52,.48),rgba(10,24,38,.28));}}
.plotly-graph-div {{width:100% !important;}}
.persona-card {{padding:18px;background:var(--panel-strong);}}
.persona-card h3 {{margin:10px 0 8px;font-size:1.22rem;}}
.persona-card p {{margin:0 0 16px;}}
.persona-grid {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}}
.persona-grid strong {{font-size:1rem;letter-spacing:-.02em;}}
.footer-note {{padding:26px 28px 34px;}}
.reveal {{opacity:0;transform:translateY(24px);transition:opacity .7s ease,transform .7s ease;}}
.reveal.visible {{opacity:1;transform:translateY(0);}}
.wc-grid {{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:18px;}}
.wc-card {{border:1px solid var(--line);background:rgba(7,18,29,.88);border-radius:22px;padding:16px;}}
.wc-label {{font-size:.82rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);margin-bottom:10px;}}
@media (max-width:1180px) {{.hero,.grid-2,.persona-row,.wc-grid {{grid-template-columns:1fr;}}.kpi-strip {{grid-template-columns:repeat(3,minmax(0,1fr));}}}}
@media (max-width:780px) {{.shell {{width:min(100vw - 22px,1480px);padding-top:18px;}}.topbar {{align-items:start;gap:12px;flex-direction:column;}}.hero {{min-height:auto;padding-top:24px;}}.hero h1 {{font-size:clamp(2.2rem,13vw,4rem);}}.kpi-strip {{grid-template-columns:repeat(2,minmax(0,1fr));}}.metric-block {{min-height:110px;}}}}
@media (max-width:520px) {{.kpi-strip {{grid-template-columns:1fr;}}}}
</style>
</head>
<body>
<div class="shell">
<div class="topbar">
<div class="brand"><strong>Customer Intelligence</strong><span>7-table integrated customer analytics workspace</span></div>
<div class="nav"><a href="#overview">Overview</a><a href="#profile">Profile</a><a href="#cluster">Clusters</a><a href="#segments">Segments</a><a href="#funnel">Funnel</a><a href="#product">Products</a><a href="#wordclouds">Word Clouds</a><a href="#retention">Retention</a></div>
</div>
<header class="hero">
<div class="hero-copy reveal visible">
<div class="eyebrow">Customer Module</div>
<h1>Audience profile, segmentation, funnel, product, and retention in one dashboard.</h1>
<p>This dashboard is generated directly from Python using real analysis on <code>customers / sessions / events / orders / order_items / products / reviews</code>. It focuses on customer personas, profile structure, conversion path, product performance, and retention behavior.</p>
</div>
<div class="hero-meta">
<div class="meta-panel reveal"><div class="meta-title">Data scope</div><ul class="meta-list">{note_items}</ul></div>
<div class="meta-panel reveal"><div class="meta-title">Method notes</div><ul class="meta-list"><li>Gender is inferred from English names and titles; it is not a ground-truth demographic field.</li><li>The funnel is measured at the session level and explicitly includes add-to-cart.</li><li>Customer personas are generated from behavioral, transactional, and engagement features.</li></ul></div>
</div>
</header>
<section class="kpi-strip" id="overview">{kpi_cards_html(kpis)}</section>

<section class="section reveal">
    <div class="section-head"><div><h2>Business overview</h2></div></div>
    <div class="figure-shell">{figure_blocks["overview"]}</div>
</section>

<section class="section reveal" id="profile">
    <div class="section-head"><div><h2>Customer profile</h2></div></div>
    <div class="grid-2"><div class="figure-shell">{figure_blocks["sunburst"]}</div><div class="figure-shell">{figure_blocks["geo"]}</div></div>
    <div class="grid-2" style="margin-top:18px;"><div class="figure-shell">{figure_blocks["age_pyramid"]}</div><div class="figure-shell">{figure_blocks["lifecycle_donut"]}</div></div>
</section>

<section class="section reveal" id="cluster">
    <div class="section-head"><div><h2>Customer clustering</h2></div></div>
    <div class="grid-2"><div class="figure-shell">{figure_blocks["cluster_scatter"]}</div><div class="figure-shell">{figure_blocks["cluster_heatmap"]}</div></div>
    <div class="persona-row">{persona_cards_html(persona_cards)}</div>
</section>

<section class="section reveal" id="segments">
    <div class="section-head"><div><h2>Business segmentation</h2></div></div>
    <div class="grid-2"><div class="figure-shell">{figure_blocks["rfm_bar"]}</div><div class="figure-shell">{figure_blocks["funnel"]}</div></div>
</section>

<section class="section reveal" id="funnel">
    <div class="section-head"><div><h2>Funnel breakdown by device &amp; channel</h2></div></div>
    <div class="figure-shell">{figure_blocks["segment_heatmap"]}</div>
</section>

<section class="section reveal" id="product">
    <div class="section-head"><div><h2>Products &amp; Categories</h2></div></div>
    <div class="grid-2">
        <div class="figure-shell">{figure_blocks["product_sales_rank"]}</div>
        <div class="figure-shell">{figure_blocks["product_rating_rank"]}</div>
    </div>
    <div class="grid-2" style="margin-top:18px;">
        <div class="figure-shell">{figure_blocks["sentiment_bar"]}</div>
        <div class="figure-shell">{figure_blocks["category_bubble"]}</div>
    </div>
    <div class="grid-2" style="margin-top:18px;">
        <div class="figure-shell">{figure_blocks["category_area"]}</div>
        <div class="figure-shell">{figure_blocks["top_products"]}</div>
    </div>
</section>

<section class="section reveal" id="wordclouds">
    <div class="section-head"><div><h2>Review keyword clouds by category</h2></div></div>
    <div class="wc-grid">{wordcloud_html}</div>
</section>

<section class="section reveal" id="retention">
    <div class="section-head"><div><h2>Retention</h2></div></div>
    <div class="grid-2"><div class="figure-shell">{figure_blocks["retention_session"]}</div><div class="figure-shell">{figure_blocks["retention_purchase"]}</div></div>
</section>

<section class="footer-note reveal">
    <h3>Additional notes</h3>
    <ul>
        <li>All visuals are generated by a local Python script and exported into a standalone HTML file.</li>
        <li>Country codes are mapped to both country names and regions for high-level and drill-down reading.</li>
        <li>Retention uses signup-month cohorts and tracks up to 12 months of post-signup activity.</li>
    </ul>
</section>
</div>
<script>
const observer=new IntersectionObserver((entries)=>{{entries.forEach((entry)=>{{if(entry.isIntersecting){{entry.target.classList.add('visible');}}}});}},{{threshold:.16}});
document.querySelectorAll('.reveal').forEach((node)=>{{if(!node.classList.contains('visible')){{observer.observe(node);}}}});
</script>
</body>
</html>"""


def build_dashboard() -> None:
    data = load_data()

    profile = build_customer_profile(data)
    profile, rfm_summary = add_rfm_segments(profile)
    profile, lifecycle_summary = add_lifecycle_stage(profile)
    profile, cluster_summary, persona_cards = assign_cluster_personas(profile)

    reviews_with_orders = data["reviews"].merge(data["orders"][["order_id", "order_time"]], on="order_id", how="left")

    monthly = monthly_overview(data, reviews_with_orders)
    overall_funnel, device_rates, source_rates = funnel_metrics(data)
    geo = geography_metrics(data)
    sunburst_frame, age_gender = demographic_frames(profile)
    category_monthly, category_perf, top_products = product_metrics(data)
    product_master, sentiment_pct, reviews_with_sentiment = sentiment_product_metrics(data)
    retention_session = retention_matrix(data, activity_source="sessions")
    retention_purchase = retention_matrix(data, activity_source="orders")

    wc_html = generate_wordcloud_section_html(reviews_with_sentiment, data["products"])

    figures = {
        "overview": make_overview_figure(monthly),
        "geo": make_geo_figure(geo),
        "sunburst": make_sunburst_figure(sunburst_frame),
        "age_pyramid": make_age_pyramid(age_gender),
        "cluster_scatter": make_cluster_scatter(profile),
        "cluster_heatmap": make_cluster_heatmap(cluster_summary),
        "funnel": make_funnel_figure(overall_funnel),
        "segment_heatmap": make_segment_heatmap(device_rates, source_rates),
        "product_sales_rank": make_product_sales_rank(product_master),
        "product_rating_rank": make_product_rating_rank(product_master),
        "sentiment_bar": make_sentiment_stacked_bar(sentiment_pct),
        "category_bubble": make_category_bubble(category_perf),
        "category_area": make_category_area(category_monthly),
        "retention_session": make_retention_heatmap(retention_session, "Session Retention Cohort Heatmap"),
        "retention_purchase": make_retention_heatmap(retention_purchase, "Purchase Retention Cohort Heatmap"),
        "rfm_bar": make_rfm_bar(rfm_summary),
        "lifecycle_donut": make_lifecycle_donut(lifecycle_summary),
        "top_products": make_top_products_table(top_products),
    }

    max_time = max(
        data["events"]["timestamp"].max(),
        data["orders"]["order_time"].max(),
        data["reviews"]["review_time"].max(),
    )

    purchase_sessions = overall_funnel.loc[overall_funnel["stage"] == "purchase", "sessions"].iloc[0]
    page_sessions = overall_funnel.loc[overall_funnel["stage"] == "page_view", "sessions"].iloc[0]
    add_to_cart_sessions = overall_funnel.loc[overall_funnel["stage"] == "add_to_cart", "sessions"].iloc[0]

    kpis = [
        {
            "label": "Customers",
            "value": format_number(len(data["customers"])),
            "detail": f"{data['customers']['country'].nunique()} countries",
        },
        {
            "label": "Sessions",
            "value": format_number(len(data["sessions"])),
            "detail": f"{format_number(data['sessions']['customer_id'].nunique())} active users",
        },
        {
            "label": "Orders",
            "value": format_number(len(data["orders"])),
            "detail": f"{format_number(purchase_sessions)} purchase sessions",
        },
        {
            "label": "Revenue",
            "value": format_currency(data["orders"]["total_usd"].sum()),
            "detail": f"AOV {format_currency(data['orders']['total_usd'].mean())}",
        },
        {
            "label": "Session Conversion",
            "value": format_pct(purchase_sessions / page_sessions if page_sessions > 0 else 0),
            "detail": f"Add-to-cart rate {format_pct(add_to_cart_sessions / page_sessions if page_sessions > 0 else 0)}",
        },
        {
            "label": "Average Rating",
            "value": f"{data['reviews']['rating'].mean():.2f}",
            "detail": f"{format_number(len(data['reviews']))} reviews",
        },
    ]

    data_notes = [
        "Dataset: 7 CSV tables loaded at full granularity.",
        f"Date range: {data['customers']['signup_date'].min().date()} to {max_time.date()}.",
        f"Country coverage: {', '.join(sorted(COUNTRY_META.keys()))}.",
        f"Clusters selected automatically: {cluster_summary['cluster_id'].nunique()} (silhouette={cluster_summary['silhouette_score'].iloc[0]:.3f}).",
    ]

    OUTPUT_HTML.write_text(compose_dashboard(kpis, figures, persona_cards, data_notes, wc_html), encoding="utf-8")

    logger.info("dashboard_written=%s", OUTPUT_HTML)
    logger.info("customers=%s", len(data["customers"]))
    logger.info("sessions=%s", len(data["sessions"]))
    logger.info("orders=%s", len(data["orders"]))
    logger.info("revenue=%.2f", data["orders"]["total_usd"].sum())
    logger.info("clusters=%s", cluster_summary["cluster_id"].nunique())
    logger.info("silhouette=%.4f", cluster_summary["silhouette_score"].iloc[0])


if __name__ == "__main__":
    build_dashboard()
