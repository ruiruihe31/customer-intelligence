"""Hybrid chatbot: rule-based customer lookup/CRM/LTV + Claude-powered natural-language Q&A."""
from __future__ import annotations

import logging
import os

import pandas as pd
from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    anthropic = None

from .utils import format_currency, format_number, format_pct

load_dotenv(override=True)

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 600

# ---------------------------------------------------------------------------
# CRM recommendation table
# ---------------------------------------------------------------------------

_CRM_RULES: dict[str, str] = {
    "New Visitor": "Send a welcome email with a first-order discount code (10–15%).",
    "Browsing Prospect": "Push a limited-time offer on browsed products; highlight top-rated items.",
    "New Buyer": "Send a product guide + cross-sell recommendations 3 days after purchase.",
    "Active Repeat Buyer": "Invite to loyalty programme / points rewards; offer early access to new arrivals.",
    "Cooling Down": "Send a win-back email ('We miss you') with a personalised exclusive discount.",
    "Dormant Customer": "High-value: direct outreach with premium incentive. Low-value: low-cost EDM re-engagement.",
    "Champions": "Invite to VIP events; offer first access to new collections and limited editions.",
    "Loyal Customers": "Recognise loyalty with milestone rewards; promote complementary product categories.",
    "At-Risk High Value": "Urgent win-back: targeted discount + proactive customer-service contact.",
    "Lost Low Value": "Low-cost EDM only; A/B test reactivation creative.",
    "Price / Casual Shoppers": "Flash sales and discount events; avoid full-price messaging.",
    "Potential Loyalists": "Nurture with content marketing and personalised product recommendations.",
}


def get_crm_recommendation(lifecycle_stage: str, rfm_segment: str) -> str:
    priority_rfm = {"Champions", "At-Risk High Value", "Lost Low Value"}
    if rfm_segment in priority_rfm:
        return _CRM_RULES.get(rfm_segment, "Monitor and re-evaluate in 30 days.")
    return _CRM_RULES.get(lifecycle_stage, "Monitor and re-evaluate in 30 days.")


# ---------------------------------------------------------------------------
# LTV tier scoring
# ---------------------------------------------------------------------------

def get_ltv_tier(revenue: float, order_count: int, recency_days: float) -> tuple[str, int]:
    score = 0
    if revenue >= 500:
        score += 40
    elif revenue >= 200:
        score += 28
    elif revenue >= 80:
        score += 16
    elif revenue > 0:
        score += 6

    if order_count >= 8:
        score += 30
    elif order_count >= 4:
        score += 20
    elif order_count >= 2:
        score += 10
    elif order_count == 1:
        score += 4

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
        "session_count": int(r.get("session_count", 0)),
        "order_count": int(r.get("order_count", 0)),
        "revenue": float(r.get("revenue", 0)),
        "avg_order_value": float(r.get("avg_order_value", 0)),
        "avg_discount_pct": float(r.get("avg_discount_pct", 0)),
        "recency_days": float(r.get("recency_days", 0)),
        "active_days": float(r.get("active_days", 0)),
        "page_to_cart_rate": float(r.get("page_to_cart_rate", 0)),
        "avg_rating": float(r.get("avg_rating", 0)),
        "rfm_segment": str(r.get("rfm_segment", "-")),
        "lifecycle_stage": str(r.get("lifecycle_stage", "-")),
        "cluster_name": str(r.get("cluster_name", "-")),
        "ltv_tier": tier,
        "ltv_score": score,
        "crm_recommendation": crm,
    }


# ---------------------------------------------------------------------------
# Business context for Claude (stable per dataset load — cached via id())
# ---------------------------------------------------------------------------

_context_cache: dict[tuple, str] = {}


def _build_business_context(
    data: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
) -> str:
    orders = data["orders"]
    sessions = data["sessions"]
    reviews = data["reviews"]
    customers = data["customers"]

    total_revenue = orders["total_usd"].sum()
    total_orders = len(orders)
    total_customers = len(customers)
    total_sessions = len(sessions)
    aov = orders["total_usd"].mean() if total_orders else 0
    conversion = total_orders / total_sessions if total_sessions else 0
    avg_rating = reviews["rating"].mean() if len(reviews) else 0

    monthly = orders.copy()
    monthly["month"] = monthly["order_time"].dt.to_period("M").dt.to_timestamp()
    monthly_rev = monthly.groupby("month")["total_usd"].sum().sort_values(ascending=False)
    top_month = monthly_rev.index[0] if len(monthly_rev) else None
    top_month_rev = monthly_rev.iloc[0] if len(monthly_rev) else 0

    top_products = (
        data["order_items"]
        .merge(data["products"][["product_id", "name", "category"]], on="product_id", how="left")
        .groupby(["name", "category"])["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    category_rev = (
        data["order_items"]
        .merge(data["products"][["product_id", "category"]], on="product_id", how="left")
        .merge(orders[["order_id", "total_usd"]], on="order_id", how="left")
        .groupby("category")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    lifecycle_counts = profile["lifecycle_stage"].value_counts() if "lifecycle_stage" in profile.columns else pd.Series(dtype=int)
    rfm_counts = profile["rfm_segment"].value_counts() if "rfm_segment" in profile.columns else pd.Series(dtype=int)
    cluster_counts = profile["cluster_name"].value_counts() if "cluster_name" in profile.columns else pd.Series(dtype=int)
    country_counts = customers["country"].value_counts().head(10) if "country" in customers.columns else pd.Series(dtype=int)
    region_counts = profile["region"].value_counts() if "region" in profile.columns else pd.Series(dtype=int)

    lines = [
        "=== BUSINESS SNAPSHOT (InsightFlow retail dataset) ===",
        f"Customers: {format_number(total_customers)} across {customers['country'].nunique()} countries",
        f"Sessions: {format_number(total_sessions)} | Orders: {format_number(total_orders)}",
        f"Total revenue: {format_currency(total_revenue)}",
        f"Average order value: {format_currency(aov)}",
        f"Session→order conversion: {format_pct(conversion)}",
        f"Average review rating: {avg_rating:.2f}/5 across {format_number(len(reviews))} reviews",
    ]
    if top_month is not None:
        lines.append(f"Peak revenue month: {top_month.strftime('%B %Y')} ({format_currency(top_month_rev)})")

    if len(lifecycle_counts):
        lines.append("")
        lines.append("=== LIFECYCLE STAGES ===")
        for stage, n in lifecycle_counts.items():
            lines.append(f"- {stage}: {format_number(n)}")

    if len(rfm_counts):
        lines.append("")
        lines.append("=== RFM SEGMENTS ===")
        for seg, n in rfm_counts.items():
            lines.append(f"- {seg}: {format_number(n)}")

    if len(cluster_counts):
        lines.append("")
        lines.append("=== PERSONA CLUSTERS ===")
        for name, n in cluster_counts.items():
            lines.append(f"- {name}: {format_number(n)}")

    if len(region_counts):
        lines.append("")
        lines.append("=== REGIONS (customer count) ===")
        for reg, n in region_counts.items():
            lines.append(f"- {reg}: {format_number(n)}")

    if len(country_counts):
        lines.append("")
        lines.append("=== TOP COUNTRIES ===")
        for c, n in country_counts.items():
            lines.append(f"- {c}: {format_number(n)}")

    if len(top_products):
        lines.append("")
        lines.append("=== TOP 10 PRODUCTS (by units sold) ===")
        for (name, cat), qty in top_products.items():
            lines.append(f"- {name} [{cat}]: {format_number(qty)} units")

    if len(category_rev):
        lines.append("")
        lines.append("=== CATEGORIES (by units) ===")
        for cat, qty in category_rev.items():
            lines.append(f"- {cat}: {format_number(qty)} units")

    return "\n".join(lines)


def _get_cached_context(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> str:
    key = (id(data), id(profile))
    ctx = _context_cache.get(key)
    if ctx is None:
        ctx = _build_business_context(data, profile)
        _context_cache[key] = ctx
    return ctx


# ---------------------------------------------------------------------------
# Claude-powered Q&A
# ---------------------------------------------------------------------------

_CLAUDE_SYSTEM_TEMPLATE = (
    "You are the analytics assistant for InsightFlow — a retail customer intelligence platform. "
    "Answer questions using ONLY the business snapshot below. "
    "Be concise: 1–4 sentences plus optional bullets, in markdown. "
    "Use USD for currency. If the snapshot doesn't contain the answer, say so honestly and suggest "
    "which dashboard tab (Overview / Products / Customers / Retention) the user should check.\n\n"
    "{context}"
)


def _claude_answer(
    question: str,
    data: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic SDK is not installed; run `pip install anthropic`.")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")

    context = _get_cached_context(data, profile)
    system_text = _CLAUDE_SYSTEM_TEMPLATE.format(context=context)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Keyword fallback (used when Claude fails or API key missing)
# ---------------------------------------------------------------------------

def _answer_question_fallback(
    question: str,
    data: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
) -> str:
    q = question.lower()

    if any(w in q for w in ["revenue", "sales", "income", "earn"]):
        total = data["orders"]["total_usd"].sum()
        monthly = data["orders"].copy()
        monthly["month"] = monthly["order_time"].dt.to_period("M").dt.to_timestamp()
        top_month = monthly.groupby("month")["total_usd"].sum().idxmax()
        return (
            f"**Total revenue:** {format_currency(total)}  \n"
            f"**Peak month:** {top_month.strftime('%B %Y')}"
        )

    if any(w in q for w in ["order", "purchase", "buy", "bought"]):
        total = len(data["orders"])
        aov = data["orders"]["total_usd"].mean()
        return (
            f"**Total orders:** {format_number(total)}  \n"
            f"**Average order value:** {format_currency(aov)}"
        )

    if any(w in q for w in ["customer", "user", "buyer", "shopper"]):
        n = len(data["customers"])
        countries = data["customers"]["country"].nunique()
        return (
            f"**Total customers:** {format_number(n)}  \n"
            f"**Countries:** {countries}"
        )

    if any(w in q for w in ["retain", "churn", "dormant", "cooling", "lost"]):
        if "lifecycle_stage" in profile.columns:
            counts = profile["lifecycle_stage"].value_counts()
            dormant = counts.get("Dormant Customer", 0)
            cooling = counts.get("Cooling Down", 0)
            return (
                f"**Dormant customers:** {format_number(dormant)}  \n"
                f"**Cooling down:** {format_number(cooling)}"
            )

    if any(w in q for w in ["champion", "rfm", "segment", "loyal"]):
        if "rfm_segment" in profile.columns:
            seg = profile["rfm_segment"].value_counts().reset_index()
            seg.columns = ["Segment", "Customers"]
            lines = ["**RFM Segments:**"]
            for _, row in seg.iterrows():
                lines.append(f"- {row['Segment']}: {format_number(row['Customers'])}")
            return "\n".join(lines)

    if any(w in q for w in ["rating", "review", "sentiment", "feedback"]):
        avg = data["reviews"]["rating"].mean()
        n = len(data["reviews"])
        return (
            f"**Average rating:** {avg:.2f} / 5  \n"
            f"**Total reviews:** {format_number(n)}"
        )

    if any(w in q for w in ["conversion", "funnel", "cart", "checkout"]):
        orders = len(data["orders"])
        sessions = len(data["sessions"])
        rate = orders / sessions if sessions > 0 else 0
        return (
            f"**Session-to-order conversion:** {format_pct(rate)}  \n"
            f"**Sessions:** {format_number(sessions)} | **Orders:** {format_number(orders)}"
        )

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
        "**RFM segments**, **ratings**, **conversion**, or **top products**."
    )


def answer_question(
    question: str,
    data: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
) -> str:
    """Prefer Claude when ANTHROPIC_API_KEY is set; fall back to keyword matching on error."""
    if ANTHROPIC_API_KEY and anthropic is not None:
        try:
            return _claude_answer(question, data, profile)
        except Exception as exc:
            logger.warning("Claude call failed, falling back to keyword engine: %s", exc)

    return _answer_question_fallback(question, data, profile)
