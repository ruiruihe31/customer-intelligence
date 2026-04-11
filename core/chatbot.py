"""Rule-based chatbot engine: customer lookup, CRM recommendations, LTV tier scoring.

Phase 2 will replace answer_question() with a Claude API call.
"""
from __future__ import annotations

import pandas as pd

from .utils import format_currency, format_number, format_pct

# ---------------------------------------------------------------------------
# CRM recommendation table
# ---------------------------------------------------------------------------

_CRM_RULES: dict[str, str] = {
    # Lifecycle-based
    "New Visitor": "Send a welcome email with a first-order discount code (10–15%).",
    "Browsing Prospect": "Push a limited-time offer on browsed products; highlight top-rated items.",
    "New Buyer": "Send a product guide + cross-sell recommendations 3 days after purchase.",
    "Active Repeat Buyer": "Invite to loyalty programme / points rewards; offer early access to new arrivals.",
    "Cooling Down": "Send a win-back email ('We miss you') with a personalised exclusive discount.",
    "Dormant Customer": "High-value: direct outreach with premium incentive. Low-value: low-cost EDM re-engagement.",
    # RFM-based overrides
    "Champions": "Invite to VIP events; offer first access to new collections and limited editions.",
    "Loyal Customers": "Recognise loyalty with milestone rewards; promote complementary product categories.",
    "At-Risk High Value": "Urgent win-back: targeted discount + proactive customer-service contact.",
    "Lost Low Value": "Low-cost EDM only; A/B test reactivation creative.",
    "Price / Casual Shoppers": "Flash sales and discount events; avoid full-price messaging.",
    "Potential Loyalists": "Nurture with content marketing and personalised product recommendations.",
}


def get_crm_recommendation(lifecycle_stage: str, rfm_segment: str) -> str:
    """Return the CRM action string for a given lifecycle + RFM state."""
    # RFM segment overrides take priority for high-stakes groups
    priority_rfm = {"Champions", "At-Risk High Value", "Lost Low Value"}
    if rfm_segment in priority_rfm:
        return _CRM_RULES.get(rfm_segment, "Monitor and re-evaluate in 30 days.")
    return _CRM_RULES.get(lifecycle_stage, "Monitor and re-evaluate in 30 days.")


# ---------------------------------------------------------------------------
# LTV tier scoring (rule-based, no ML required)
# ---------------------------------------------------------------------------

def get_ltv_tier(revenue: float, order_count: int, recency_days: float) -> tuple[str, int]:
    """
    Return (tier_label, score_0_to_100).

    Tiers:  Platinum ≥ 80 | Gold 60–79 | Silver 40–59 | Bronze < 40
    """
    score = 0
    # Revenue contribution (max 40 pts)
    if revenue >= 500:
        score += 40
    elif revenue >= 200:
        score += 28
    elif revenue >= 80:
        score += 16
    elif revenue > 0:
        score += 6

    # Order frequency (max 30 pts)
    if order_count >= 8:
        score += 30
    elif order_count >= 4:
        score += 20
    elif order_count >= 2:
        score += 10
    elif order_count == 1:
        score += 4

    # Recency (max 30 pts — lower recency = more recent = better)
    if recency_days <= 30:
        score += 30
    elif recency_days <= 60:
        score += 22
    elif recency_days <= 120:
        score += 12
    elif recency_days <= 180:
        score += 4

    score = min(score, 100)
    if score >= 80:
        tier = "Platinum"
    elif score >= 60:
        tier = "Gold"
    elif score >= 40:
        tier = "Silver"
    else:
        tier = "Bronze"
    return tier, score


# ---------------------------------------------------------------------------
# Customer lookup
# ---------------------------------------------------------------------------

def lookup_customer(
    customer_id: int,
    profile: pd.DataFrame,
) -> dict | None:
    """Return a rich dict summary for a single customer, or None if not found."""
    row = profile[profile["customer_id"] == customer_id]
    if row.empty:
        return None
    r = row.iloc[0]

    tier, score = get_ltv_tier(
        float(r.get("revenue", 0)),
        int(r.get("order_count", 0)),
        float(r.get("recency_days", 999)),
    )
    crm = get_crm_recommendation(
        str(r.get("lifecycle_stage", "")),
        str(r.get("rfm_segment", "")),
    )

    return {
        "customer_id": customer_id,
        "name": str(r.get("name", "-")),
        "country": str(r.get("country_name", str(r.get("country", "-")))),
        "region": str(r.get("region", "-")),
        "age": r.get("age", None),
        "age_band": str(r.get("age_band", "-")),
        "gender_inferred": str(r.get("gender_inferred", "-")),
        "marketing_opt_in": bool(r.get("marketing_opt_in", 0)),
        # Behaviour
        "session_count": int(r.get("session_count", 0)),
        "order_count": int(r.get("order_count", 0)),
        "revenue": float(r.get("revenue", 0)),
        "avg_order_value": float(r.get("avg_order_value", 0)),
        "avg_discount_pct": float(r.get("avg_discount_pct", 0)),
        "recency_days": float(r.get("recency_days", 0)),
        "active_days": float(r.get("active_days", 0)),
        "page_to_cart_rate": float(r.get("page_to_cart_rate", 0)),
        "avg_rating": float(r.get("avg_rating", 0)),
        # Segments
        "rfm_segment": str(r.get("rfm_segment", "-")),
        "lifecycle_stage": str(r.get("lifecycle_stage", "-")),
        "cluster_name": str(r.get("cluster_name", "-")),
        # LTV
        "ltv_tier": tier,
        "ltv_score": score,
        # CRM
        "crm_recommendation": crm,
    }


# ---------------------------------------------------------------------------
# Simple keyword Q&A (Phase 1 — no Claude API)
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    data: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
) -> str:
    """
    Keyword-based Q&A.  Phase 2 will replace this with a Claude API call.
    Returns a markdown-formatted answer string.
    """
    q = question.lower()

    # --- Revenue questions ---
    if any(w in q for w in ["revenue", "sales", "income", "earn"]):
        total = data["orders"]["total_usd"].sum()
        monthly = data["orders"].copy()
        monthly["month"] = monthly["order_time"].dt.to_period("M").dt.to_timestamp()
        top_month = monthly.groupby("month")["total_usd"].sum().idxmax()
        return (
            f"**Total revenue:** {format_currency(total)}  \n"
            f"**Peak month:** {top_month.strftime('%B %Y')}"
        )

    # --- Order questions ---
    if any(w in q for w in ["order", "purchase", "buy", "bought"]):
        total = len(data["orders"])
        aov = data["orders"]["total_usd"].mean()
        return (
            f"**Total orders:** {format_number(total)}  \n"
            f"**Average order value:** {format_currency(aov)}"
        )

    # --- Customer / user questions ---
    if any(w in q for w in ["customer", "user", "buyer", "shopper"]):
        n = len(data["customers"])
        countries = data["customers"]["country"].nunique()
        return (
            f"**Total customers:** {format_number(n)}  \n"
            f"**Countries:** {countries}"
        )

    # --- Retention / churn questions ---
    if any(w in q for w in ["retain", "churn", "dormant", "cooling", "lost"]):
        if "lifecycle_stage" in profile.columns:
            counts = profile["lifecycle_stage"].value_counts()
            dormant = counts.get("Dormant Customer", 0)
            cooling = counts.get("Cooling Down", 0)
            return (
                f"**Dormant customers:** {format_number(dormant)}  \n"
                f"**Cooling down:** {format_number(cooling)}"
            )

    # --- Segment questions ---
    if any(w in q for w in ["champion", "rfm", "segment", "loyal"]):
        if "rfm_segment" in profile.columns:
            seg = profile["rfm_segment"].value_counts().reset_index()
            seg.columns = ["Segment", "Customers"]
            lines = ["**RFM Segments:**"]
            for _, row in seg.iterrows():
                lines.append(f"- {row['Segment']}: {format_number(row['Customers'])}")
            return "\n".join(lines)

    # --- Rating / review questions ---
    if any(w in q for w in ["rating", "review", "sentiment", "feedback"]):
        avg = data["reviews"]["rating"].mean()
        n = len(data["reviews"])
        return (
            f"**Average rating:** {avg:.2f} / 5  \n"
            f"**Total reviews:** {format_number(n)}"
        )

    # --- Conversion questions ---
    if any(w in q for w in ["conversion", "funnel", "cart", "checkout"]):
        orders = len(data["orders"])
        sessions = len(data["sessions"])
        rate = orders / sessions if sessions > 0 else 0
        return (
            f"**Session-to-order conversion:** {format_pct(rate)}  \n"
            f"**Sessions:** {format_number(sessions)} | **Orders:** {format_number(orders)}"
        )

    # --- Top products ---
    if any(w in q for w in ["product", "item", "top", "best"]):
        top = (
            data["order_items"]
            .merge(data["products"][["product_id", "name"]], on="product_id", how="left")
            .groupby("name")["quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        lines = ["**Top 5 products by units sold:**"]
        for name, qty in top.items():
            lines.append(f"- {name}: {format_number(qty)} units")
        return "\n".join(lines)

    return (
        "I couldn't find a specific answer for that question.  \n"
        "Try asking about: **revenue**, **orders**, **customers**, **retention**, "
        "**RFM segments**, **ratings**, **conversion**, or **top products**.  \n\n"
        "_AI-powered natural language Q&A is coming in Phase 2 (Claude API)._"
    )
