"""FastAPI backend for Customer Intelligence Platform.

Run from the Customer/ project root:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

import base64
import math
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from core.analytics import (
    add_lifecycle_stage,
    add_rfm_segments,
    assign_cluster_personas,
    build_customer_profile,
    demographic_frames,
    funnel_metrics,
    geography_metrics,
    monthly_overview,
    product_metrics,
    retention_matrix,
    sentiment_product_metrics,
)
from core.charts import generate_wordcloud_images
from core.chatbot import answer_question, lookup_customer
from core.data_loader import load_data

FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache: dict[str, Any] = {}


def _to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe records (NaN/inf → None)."""
    records = df.to_dict(orient="records")
    out = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = None if (math.isnan(float(v)) or math.isinf(float(v))) else float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    data = load_data()

    profile = build_customer_profile(data)
    profile, rfm_summary = add_rfm_segments(profile)
    profile, lifecycle_summary = add_lifecycle_stage(profile)
    profile, cluster_summary, persona_cards = assign_cluster_personas(profile)

    reviews_with_orders = data["reviews"].merge(
        data["orders"][["order_id", "order_time"]], on="order_id", how="left"
    )
    monthly = monthly_overview(data, reviews_with_orders)
    overall_funnel, device_rates, source_rates = funnel_metrics(data)
    geo = geography_metrics(data)
    sunburst_frame, age_gender = demographic_frames(profile)
    category_monthly, category_perf, top_products = product_metrics(data)
    product_master, sentiment_pct, reviews_with_sentiment = sentiment_product_metrics(data)
    retention_session = retention_matrix(data, activity_source="sessions")
    retention_purchase = retention_matrix(data, activity_source="orders")

    # Pre-compute word cloud base64 PNGs for the frontend
    wc_arrays = generate_wordcloud_images(data["reviews"], data["products"])
    wc_b64: dict[str, str] = {}
    for cat, arr in wc_arrays.items():
        try:
            img = Image.fromarray(arr)
            buf = BytesIO()
            img.save(buf, format="PNG")
            wc_b64[cat] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            continue

    _cache.update(
        data=data,
        profile=profile,
        rfm_summary=rfm_summary,
        lifecycle_summary=lifecycle_summary,
        persona_cards=persona_cards,
        monthly=monthly,
        overall_funnel=overall_funnel,
        device_rates=device_rates,
        source_rates=source_rates,
        geo=geo,
        sunburst_frame=sunburst_frame,
        age_gender=age_gender,
        category_monthly=category_monthly,
        category_perf=category_perf,
        top_products=top_products,
        product_master=product_master,
        sentiment_pct=sentiment_pct,
        reviews_with_sentiment=reviews_with_sentiment,
        retention_session=retention_session,
        retention_purchase=retention_purchase,
        wordcloud_b64=wc_b64,
    )
    yield


app = FastAPI(title="Customer Intelligence Platform API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
@app.get("/api/kpis")
def api_kpis():
    data = _cache["data"]
    overall_funnel = _cache["overall_funnel"]
    purchase_sessions = int(
        overall_funnel.loc[overall_funnel["stage"] == "purchase", "sessions"].iloc[0]
    )
    page_sessions = int(
        overall_funnel.loc[overall_funnel["stage"] == "page_view", "sessions"].iloc[0]
    )
    return {
        "customers": len(data["customers"]),
        "sessions": len(data["sessions"]),
        "orders": len(data["orders"]),
        "revenue": float(data["orders"]["total_usd"].sum()),
        "aov": float(data["orders"]["total_usd"].mean()),
        "conversion_rate": purchase_sessions / page_sessions if page_sessions else 0.0,
        "avg_rating": float(data["reviews"]["rating"].mean()),
    }


# ---------------------------------------------------------------------------
# Monthly overview  (analytics returns "revenue_rolling_3m"; frontend wants "rolling_revenue_3m")
# ---------------------------------------------------------------------------
@app.get("/api/monthly")
def api_monthly():
    monthly = _cache["monthly"].copy()
    monthly["month"] = monthly["month"].astype(str)
    monthly = monthly.rename(columns={"revenue_rolling_3m": "rolling_revenue_3m"})
    return _to_records(monthly)


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
@app.get("/api/geography")
def api_geography():
    return _to_records(_cache["geo"])


# ---------------------------------------------------------------------------
# Customers — RFM
# ---------------------------------------------------------------------------
@app.get("/api/customers/rfm")
def api_rfm():
    return _to_records(_cache["rfm_summary"])


# ---------------------------------------------------------------------------
# Customers — Lifecycle
# ---------------------------------------------------------------------------
@app.get("/api/customers/lifecycle")
def api_lifecycle():
    return _to_records(_cache["lifecycle_summary"])


# ---------------------------------------------------------------------------
# Customers — Personas
# ---------------------------------------------------------------------------
@app.get("/api/customers/personas")
def api_personas():
    return _cache["persona_cards"]


# ---------------------------------------------------------------------------
# Customers — Cluster scatter (sampled to keep payload small)
# ---------------------------------------------------------------------------
@app.get("/api/customers/cluster")
def api_cluster():
    profile = _cache["profile"]
    scatter = profile[["cluster_x", "cluster_y", "cluster_name"]].copy()
    if len(scatter) > 3000:
        scatter = scatter.sample(3000, random_state=42)
    return {"scatter": _to_records(scatter)}


# ---------------------------------------------------------------------------
# Customers — Demographics  (must be before /{customer_id} to avoid route capture)
# ---------------------------------------------------------------------------
@app.get("/api/customers/demographics")
def api_demographics():
    sunburst = _cache["sunburst_frame"].copy()
    sunburst["age_band"] = sunburst["age_band"].astype(str)

    age_gender = _cache["age_gender"][["age_band", "gender_inferred", "users"]].copy()
    age_gender["age_band"] = age_gender["age_band"].astype(str)

    return {
        "sunburst": _to_records(sunburst),
        "age_gender": _to_records(age_gender),
    }


# ---------------------------------------------------------------------------
# Customers — Individual lookup
# ---------------------------------------------------------------------------
def _clean_customer(info: dict) -> dict:
    """Convert numpy scalars to plain Python types for JSON serialization."""
    out = {}
    for k, v in info.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = None if (math.isnan(float(v)) or math.isinf(float(v))) else float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


@app.get("/api/customers/{customer_id}")
def api_customer(customer_id: int):
    info = lookup_customer(customer_id, _cache["profile"])
    if info is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return _clean_customer(info)


# ---------------------------------------------------------------------------
# Products — Word clouds (base64 PNG per category)
# ---------------------------------------------------------------------------
@app.get("/api/products/wordcloud")
def api_wordcloud():
    return _cache.get("wordcloud_b64", {})


# ---------------------------------------------------------------------------
# Products — Sentiment
# ---------------------------------------------------------------------------
@app.get("/api/products/sentiment")
def api_sentiment():
    sentiment = _cache["sentiment_pct"].reset_index()
    return _to_records(sentiment)


# ---------------------------------------------------------------------------
# Products — Categories + top products
# ---------------------------------------------------------------------------
@app.get("/api/products/categories")
def api_categories():
    cat_monthly = _cache["category_monthly"].copy()
    cat_monthly["month"] = cat_monthly["month"].astype(str)

    cat_perf = _cache["category_perf"].copy()

    top_prod = _cache["top_products"].copy()

    return {
        "monthly": _to_records(cat_monthly),
        "performance": _to_records(cat_perf),
        "top_products": _to_records(top_prod),
    }


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------
def _retention_to_dict(matrix: pd.DataFrame) -> dict:
    cohorts = [str(c.date()) for c in matrix.index]
    months = [int(m) for m in matrix.columns]
    values = [
        [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in row]
        for row in matrix.values.tolist()
    ]
    return {"cohorts": cohorts, "months": months, "values": values}


@app.get("/api/retention")
def api_retention():
    return {
        "session": _retention_to_dict(_cache["retention_session"]),
        "purchase": _retention_to_dict(_cache["retention_purchase"]),
    }


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------
@app.get("/api/funnel")
def api_funnel():
    overall = _to_records(_cache["overall_funnel"])

    device_rates = _cache["device_rates"].reset_index()
    source_rates = _cache["source_rates"].reset_index()

    return {
        "overall": overall,
        "device": _to_records(device_rates),
        "source": _to_records(source_rates),
    }


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


@app.post("/api/chatbot/query")
def api_chatbot(req: ChatRequest):
    answer = answer_question(req.question, _cache["data"], _cache["profile"])
    return {"answer": answer}
