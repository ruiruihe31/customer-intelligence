"""Pure analytics functions: customer profiling, RFM, clustering, funnels, retention."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import AGE_BINS, AGE_LABELS, CLUSTER_RANDOM_STATE
from .utils import (
    format_currency,
    format_number,
    format_pct,
    get_mode_by_group,
    get_name_sets,
    infer_gender,
    safe_country_name,
    safe_region,
)


# ---------------------------------------------------------------------------
# Customer profile
# ---------------------------------------------------------------------------

def build_customer_profile(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["customers"].copy()
    sessions = data["sessions"]
    events = data["events"]
    orders = data["orders"]
    reviews = data["reviews"]

    male_names, female_names = get_name_sets()

    customers["country_name"] = customers["country"].astype(str).map(safe_country_name)
    customers["region"] = customers["country"].astype(str).map(safe_region)
    customers["age_band"] = pd.cut(customers["age"], AGE_BINS, labels=AGE_LABELS)
    customers["age_band"] = customers["age_band"].cat.add_categories(["Unknown"]).fillna("Unknown")
    customers["gender_inferred"] = customers["name"].map(
        lambda x: infer_gender(x, male_names, female_names)
    )

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
    required_event_cols = [
        "customer_id",
        "page_views",
        "add_to_cart_events",
        "checkout_events",
        "purchase_events",
    ]
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

    reviews_with_customer = reviews.merge(
        orders[["order_id", "customer_id"]], on="order_id", how="left"
    )
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
        "session_count", "source_diversity", "device_diversity",
        "page_views", "add_to_cart_events", "checkout_events", "purchase_events",
        "total_events", "total_qty", "order_count", "revenue", "subtotal",
        "avg_order_value", "avg_discount_pct", "review_count", "avg_rating",
        "mean_cart_size",
    ]
    for column in fill_zero_cols:
        profile[column] = profile[column].fillna(0)

    profile["dominant_source"] = profile["dominant_source"].astype("object").fillna("No Session")
    profile["dominant_device"] = profile["dominant_device"].astype("object").fillna("No Session")
    profile["active_days"] = (
        profile["last_session"].fillna(profile["signup_date"])
        - profile["first_session"].fillna(profile["signup_date"])
    ).dt.days.add(1).clip(lower=0)
    profile["recency_days"] = (
        reference_time - profile["last_session"].fillna(profile["signup_date"])
    ).dt.days.clip(lower=0)
    profile["session_to_order_rate"] = np.where(
        profile["session_count"] > 0, profile["order_count"] / profile["session_count"], 0
    )
    profile["page_to_cart_rate"] = np.where(
        profile["page_views"] > 0, profile["add_to_cart_events"] / profile["page_views"], 0
    )
    profile["cart_to_checkout_rate"] = np.where(
        profile["add_to_cart_events"] > 0,
        profile["checkout_events"] / profile["add_to_cart_events"],
        0,
    )
    profile["checkout_to_purchase_rate"] = np.where(
        profile["checkout_events"] > 0,
        profile["purchase_events"] / profile["checkout_events"],
        0,
    )
    profile["review_rate"] = np.where(
        profile["order_count"] > 0, profile["review_count"] / profile["order_count"], 0
    )
    profile["revenue_per_session"] = np.where(
        profile["session_count"] > 0, profile["revenue"] / profile["session_count"], 0
    )
    profile["marketing_opt_in"] = profile["marketing_opt_in"].astype(int)
    return profile


# ---------------------------------------------------------------------------
# RFM segmentation
# ---------------------------------------------------------------------------

def score_series(series: pd.Series, reverse: bool = False, bins: int = 5) -> pd.Series:
    filled = pd.Series(series).fillna(
        series.median() if not pd.isna(series.median()) else 0
    )
    ranked = filled.rank(method="first", ascending=not reverse)
    try:
        scored = pd.qcut(ranked, q=min(bins, ranked.nunique()), labels=False, duplicates="drop") + 1
    except ValueError:
        scored = pd.Series(np.ones(len(series), dtype=int), index=series.index)
    scored = pd.Series(scored, index=series.index).astype(int)
    return scored


def add_rfm_segments(
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = profile.copy()
    profile["r_score"] = score_series(profile["recency_days"], reverse=True)
    profile["f_score"] = score_series(profile["order_count"])
    profile["m_score"] = score_series(profile["revenue"])
    profile["rfm_score"] = (
        profile["r_score"].astype(str)
        + profile["f_score"].astype(str)
        + profile["m_score"].astype(str)
    )

    conditions = [
        (profile["r_score"] >= 4) & (profile["f_score"] >= 4) & (profile["m_score"] >= 4),
        (profile["r_score"] >= 4) & (profile["f_score"] >= 3),
        (profile["r_score"] <= 2) & (profile["f_score"] >= 4),
        (profile["r_score"] <= 2) & (profile["f_score"] <= 2) & (profile["m_score"] <= 2),
        (profile["f_score"] <= 2) & (profile["m_score"] <= 2),
    ]
    choices = [
        "Champions",
        "Loyal Customers",
        "At-Risk High Value",
        "Lost Low Value",
        "Price / Casual Shoppers",
    ]
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


# ---------------------------------------------------------------------------
# Lifecycle stages
# ---------------------------------------------------------------------------

def add_lifecycle_stage(
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


# ---------------------------------------------------------------------------
# Clustering / personas
# ---------------------------------------------------------------------------

def select_clusters(
    cluster_frame: pd.DataFrame,
) -> tuple[np.ndarray, int, list[tuple[int, float]], np.ndarray]:
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
        score = (
            silhouette_score(sample_scaled, sample_labels)
            if len(np.unique(sample_labels)) > 1
            else -1.0
        )
        scores.append((k, score))
        if score > best_score:
            best_score = score
            best_k = k

    final_cls = MiniBatchKMeans if len(cluster_frame) > 20_000 else KMeans
    final_model = final_cls(n_clusters=best_k, n_init=20, random_state=CLUSTER_RANDOM_STATE)
    best_labels = final_model.fit_predict(scaled)
    return best_labels, best_k, scores, scaled


def assign_cluster_personas(
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    rating_non_zero = profile.loc[profile["avg_rating"] > 0, "avg_rating"]
    rating_mean = 0.0 if rating_non_zero.empty else rating_non_zero.mean()

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
    used: set[str] = set()

    for cluster_id, row in cluster_summary.sort_values(["revenue", "orders"], ascending=False).iterrows():
        if row["revenue"] <= 0 and row["orders"] <= 0:
            label = "Never Purchased"
        elif row["revenue"] >= revenue_threshold and row["orders"] >= order_threshold:
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

    region_counts = profile.groupby(["cluster_name", "region"]).size().reset_index(name="users")
    region_totals = profile.groupby("cluster_name").size().rename("total").reset_index()
    region_mix = region_counts.merge(region_totals, on="cluster_name")
    region_mix["share"] = region_mix["users"] / region_mix["total"]
    region_mix = (
        region_mix.sort_values(["cluster_name", "users"], ascending=[True, False])
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

    persona_cards: list[dict[str, str]] = []
    for cluster_name, row in cluster_display.sort_values("revenue", ascending=False).iterrows():
        region_row = region_mix.loc[region_mix["cluster_name"] == cluster_name].iloc[0]
        top_region = region_row["region"]
        top_region_share = float(region_row["share"])
        top_source = source_mix.loc[source_mix["cluster_name"] == cluster_name, "dominant_source"].iloc[0]
        persona_cards.append(
            {
                "cluster_name": str(cluster_name),
                "users": format_number(row["users"]),
                "revenue": format_currency(row["revenue"]),
                "orders": f"{row['orders']:.2f}",
                "sessions": f"{row['sessions']:.2f}",
                "conversion": format_pct(row["conversion"]),
                "avg_age": f"{row['avg_age']:.1f}" if not pd.isna(row["avg_age"]) else "-",
                "avg_rating": f"{row['avg_rating']:.2f}",
                "top_region": top_region,
                "top_region_pct": f"{top_region_share * 100:.0f}%",
                "top_source": top_source,
            }
        )

    return profile, cluster_summary.reset_index(), persona_cards


# ---------------------------------------------------------------------------
# Time-series / aggregates
# ---------------------------------------------------------------------------

def monthly_overview(
    data: dict[str, pd.DataFrame],
    reviews_with_orders: pd.DataFrame,
) -> pd.DataFrame:
    sessions = data["sessions"].copy()
    orders = data["orders"].copy()
    rwo = reviews_with_orders.copy()

    sessions["month"] = sessions["start_time"].dt.to_period("M").dt.to_timestamp()
    orders["month"] = orders["order_time"].dt.to_period("M").dt.to_timestamp()
    rwo["month"] = rwo["review_time"].dt.to_period("M").dt.to_timestamp()

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
    ratings_monthly = rwo.groupby("month").agg(
        avg_rating=("rating", "mean"),
        review_count=("review_id", "count"),
    )
    monthly = (
        sessions_monthly.join(orders_monthly, how="outer")
        .join(ratings_monthly, how="left")
        .fillna(0)
        .reset_index()
    )
    monthly["session_to_order"] = np.where(
        monthly["sessions"] > 0, monthly["orders"] / monthly["sessions"], 0
    )
    monthly["revenue_rolling_3m"] = monthly["revenue"].rolling(3, min_periods=1).mean()
    return monthly


def funnel_metrics(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = data["sessions"][["session_id", "device", "source"]].copy()
    stage_flags = (
        data["events"]
        .assign(flag=1)
        .pivot_table(
            index="session_id",
            columns="event_type",
            values="flag",
            aggfunc="max",
            fill_value=0,
            observed=False,
        )
        .reset_index()
    )
    funnel_base = sessions.merge(stage_flags, on="session_id", how="left")
    stage_order = ["page_view", "add_to_cart", "checkout", "purchase"]
    for stage in stage_order:
        if stage not in funnel_base.columns:
            funnel_base[stage] = 0
    funnel_base[stage_order] = funnel_base[stage_order].fillna(0)

    overall = pd.DataFrame(
        {"stage": stage_order, "sessions": [int(funnel_base[stage].sum()) for stage in stage_order]}
    )
    device_rates = funnel_base.groupby("device", observed=False)[stage_order].mean().reindex(
        ["desktop", "mobile", "tablet"]
    )
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


def demographic_frames(
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    age_gender["plot_users"] = np.where(
        age_gender["gender_inferred"] == "Male", -age_gender["users"], age_gender["users"]
    )
    age_gender["age_band"] = pd.Categorical(
        age_gender["age_band"], categories=AGE_LABELS + ["Unknown"], ordered=True
    )
    age_gender = age_gender.sort_values("age_band")
    return sunburst_frame, age_gender


def product_metrics(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order_items = data["order_items"]
    products = data["products"]
    orders = data["orders"][["order_id", "order_time"]]
    reviews = data["reviews"]

    order_line = order_items.merge(
        products[["product_id", "category", "margin_usd"]], on="product_id", how="left"
    ).merge(orders, on="order_id", how="left")
    order_line["month"] = order_line["order_time"].dt.to_period("M").dt.to_timestamp()
    order_line["gross_margin"] = order_line["margin_usd"] * order_line["quantity"]

    category_monthly = (
        order_line.groupby(["month", "category"], observed=False)
        .agg(
            revenue=("line_total_usd", "sum"),
            units=("quantity", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    product_review = (
        reviews.groupby("product_id")
        .agg(avg_rating=("rating", "mean"), review_count=("review_id", "count"))
        .reset_index()
    )
    category_perf = (
        order_line.groupby("category", observed=False)
        .agg(
            revenue=("line_total_usd", "sum"),
            units=("quantity", "sum"),
            gross_margin=("gross_margin", "sum"),
            products=("product_id", "nunique"),
        )
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


def sentiment_product_metrics(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (product_master, sentiment_pct_by_category, reviews_with_sentiment)."""
    products = data["products"]
    order_items = data["order_items"]
    reviews = data["reviews"].copy()

    _sia = SentimentIntensityAnalyzer()

    def _vader_label(text: str) -> str:
        compound = _sia.polarity_scores(str(text))["compound"]
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    reviews["sentiment"] = reviews["review_text"].apply(_vader_label)

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

    cat_sentiment = reviews.merge(products[["product_id", "category"]], on="product_id")
    sentiment_pivot = cat_sentiment.groupby(["category", "sentiment"]).size().unstack(fill_value=0)
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in sentiment_pivot.columns:
            sentiment_pivot[col] = 0
    sentiment_pct = sentiment_pivot.div(sentiment_pivot.sum(axis=1), axis=0) * 100

    return product_master, sentiment_pct, reviews


def retention_matrix(
    data: dict[str, pd.DataFrame],
    activity_source: str = "sessions",
) -> pd.DataFrame:
    customers = data["customers"][["customer_id", "signup_date"]].copy()

    if activity_source == "sessions":
        activity_df = (
            data["sessions"][["customer_id", "start_time"]]
            .rename(columns={"start_time": "activity_time"})
            .copy()
        )
    else:
        activity_df = (
            data["orders"][["customer_id", "order_time"]]
            .rename(columns={"order_time": "activity_time"})
            .copy()
        )

    customers["cohort_month"] = customers["signup_date"].dt.to_period("M").dt.to_timestamp()
    activity_df["activity_month"] = activity_df["activity_time"].dt.to_period("M").dt.to_timestamp()

    activity = (
        customers.merge(activity_df[["customer_id", "activity_month"]], on="customer_id", how="left")
        .dropna(subset=["activity_month"])
    )
    activity["month_number"] = (
        (activity["activity_month"].dt.year - activity["cohort_month"].dt.year) * 12
        + (activity["activity_month"].dt.month - activity["cohort_month"].dt.month)
    )
    activity = activity[
        (activity["month_number"] >= 0) & (activity["month_number"] <= 12)
    ].drop_duplicates(["customer_id", "activity_month"])

    cohort_size = customers.groupby("cohort_month").agg(cohort_size=("customer_id", "nunique"))
    cohort_activity = (
        activity.groupby(["cohort_month", "month_number"])
        .agg(active_users=("customer_id", "nunique"))
        .reset_index()
    )
    retention = cohort_activity.merge(cohort_size, on="cohort_month", how="left")
    retention["retention"] = retention["active_users"] / retention["cohort_size"]
    return retention.pivot(
        index="cohort_month", columns="month_number", values="retention"
    ).sort_index()
