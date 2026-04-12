"""Customer Intelligence Platform — Streamlit frontend.

Run:  streamlit run frontend/app.py
      (from the Customer/ project root)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore", category=FutureWarning)

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
from core.chatbot import answer_question, lookup_customer
from core.charts import (
    generate_wordcloud_images,
    make_age_pyramid,
    make_category_area,
    make_category_bubble,
    make_cluster_heatmap,
    make_cluster_scatter,
    make_funnel_figure,
    make_geo_figure,
    make_lifecycle_donut,
    make_overview_figure,
    make_product_rating_rank,
    make_product_sales_rank,
    make_retention_heatmap,
    make_rfm_bar,
    make_segment_heatmap,
    make_sentiment_stacked_bar,
    make_sunburst_figure,
    make_top_products_table,
)
from core.data_loader import load_data
from core.utils import format_currency, format_number, format_pct

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark theme overlay
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Prevent any Plotly figure from causing horizontal page overflow */
    .main .block-container {
        overflow-x: hidden;
        max-width: 100%;
    }
    .kpi-card {
        background: rgba(8, 21, 34, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 18px 20px;
        text-align: center;
    }
    .kpi-label { color: #8fa2be; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #eef3ff; letter-spacing: -0.03em; }
    .kpi-detail { color: #c9d6ea; font-size: 0.82rem; margin-top: 4px; }
    .persona-box {
        background: rgba(10, 26, 42, 0.96);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 16px;
    }
    .persona-title { color: #00c2a8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .persona-name { font-size: 1.05rem; font-weight: 700; color: #eef3ff; margin: 6px 0 10px; }
    .crm-box {
        background: rgba(0, 194, 168, 0.08);
        border: 1px solid rgba(0, 194, 168, 0.3);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 12px;
    }
    .tier-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading datasets...")
def _load_data() -> dict[str, pd.DataFrame]:
    return load_data()


@st.cache_data(show_spinner="Running analytics pipeline...")
def _compute_all(_data: dict[str, pd.DataFrame]) -> dict:
    profile = build_customer_profile(_data)
    profile, rfm_summary = add_rfm_segments(profile)
    profile, lifecycle_summary = add_lifecycle_stage(profile)
    profile, cluster_summary, persona_cards = assign_cluster_personas(profile)

    reviews_with_orders = _data["reviews"].merge(
        _data["orders"][["order_id", "order_time"]], on="order_id", how="left"
    )
    monthly = monthly_overview(_data, reviews_with_orders)
    overall_funnel, device_rates, source_rates = funnel_metrics(_data)
    geo = geography_metrics(_data)
    sunburst_frame, age_gender = demographic_frames(profile)
    category_monthly, category_perf, top_products = product_metrics(_data)
    product_master, sentiment_pct, reviews_with_sentiment = sentiment_product_metrics(_data)
    retention_session = retention_matrix(_data, activity_source="sessions")
    retention_purchase = retention_matrix(_data, activity_source="orders")

    return dict(
        profile=profile,
        rfm_summary=rfm_summary,
        lifecycle_summary=lifecycle_summary,
        cluster_summary=cluster_summary,
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
    )


@st.cache_data(show_spinner="Generating word clouds...")
def _generate_wordclouds(
    reviews: pd.DataFrame, products: pd.DataFrame
) -> dict[str, np.ndarray]:
    return generate_wordcloud_images(reviews, products)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
data = _load_data()
result = _compute_all(data)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
overall_funnel = result["overall_funnel"]
purchase_sessions = int(overall_funnel.loc[overall_funnel["stage"] == "purchase", "sessions"].iloc[0])
page_sessions = int(overall_funnel.loc[overall_funnel["stage"] == "page_view", "sessions"].iloc[0])
add_to_cart_sessions = int(overall_funnel.loc[overall_funnel["stage"] == "add_to_cart", "sessions"].iloc[0])

st.markdown("## Customer Intelligence Platform")

kpi_cols = st.columns(6)
kpis = [
    ("Customers", format_number(len(data["customers"])), f"{data['customers']['country'].nunique()} countries"),
    ("Sessions", format_number(len(data["sessions"])), f"{format_number(data['sessions']['customer_id'].nunique())} active users"),
    ("Orders", format_number(len(data["orders"])), f"{format_number(purchase_sessions)} purchase sessions"),
    ("Revenue", format_currency(data["orders"]["total_usd"].sum()), f"AOV {format_currency(data['orders']['total_usd'].mean())}"),
    ("Conversion", format_pct(purchase_sessions / page_sessions if page_sessions else 0), f"Cart rate {format_pct(add_to_cart_sessions / page_sessions if page_sessions else 0)}"),
    ("Avg Rating", f"{data['reviews']['rating'].mean():.2f}", f"{format_number(len(data['reviews']))} reviews"),
]
for col, (label, value, detail) in zip(kpi_cols, kpis):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-detail">{detail}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_products, tab_customers, tab_retention, tab_chatbot = st.tabs(
    ["Overview", "Products", "Customers", "Retention", "Chatbot"]
)

# ── Tab 1: Overview ─────────────────────────────────────────────────────────
with tab_overview:
    st.plotly_chart(make_overview_figure(result["monthly"]), use_container_width=True)
    st.plotly_chart(make_geo_figure(result["geo"]), use_container_width=True)

# ── Tab 2: Products ─────────────────────────────────────────────────────────
with tab_products:
    # Bar charts are full-width: long Y-axis product names overflow in half-width columns
    st.plotly_chart(make_product_sales_rank(result["product_master"]), use_container_width=True)
    st.plotly_chart(make_product_rating_rank(result["product_master"]), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(make_sentiment_stacked_bar(result["sentiment_pct"]), use_container_width=True)
    with col4:
        st.plotly_chart(make_category_bubble(result["category_perf"]), use_container_width=True)

    st.plotly_chart(make_category_area(result["category_monthly"]), use_container_width=True)
    st.plotly_chart(make_top_products_table(result["top_products"]), use_container_width=True)

    st.subheader("Review Keyword Clouds by Category")
    wc_images = _generate_wordclouds(
        result["reviews_with_sentiment"].reset_index(drop=True),
        data["products"].reset_index(drop=True),
    )
    if wc_images:
        items = list(wc_images.items())
        n_cols = 3
        for row_start in range(0, len(items), n_cols):
            row_items = items[row_start : row_start + n_cols]
            img_cols = st.columns(n_cols)
            for img_col, (cat, img) in zip(img_cols, row_items):
                img_col.image(img, caption=cat, use_container_width=True)

# ── Tab 3: Customers ────────────────────────────────────────────────────────
with tab_customers:
    # Persona cards
    with st.expander("Customer Personas", expanded=True):
        persona_cards = result["persona_cards"]
        n_cols = min(len(persona_cards), 4)
        p_cols = st.columns(n_cols)
        for p_col, card in zip(p_cols, persona_cards):
            p_col.markdown(
                f'<div class="persona-box">'
                f'<div class="persona-title">Persona</div>'
                f'<div class="persona-name">{card["cluster_name"]}</div>'
                f'<small style="color:#8fa2be">{card["top_region"]} / {card["top_source"]}</small><br><br>'
                f'<b>Users:</b> {card["users"]}<br>'
                f'<b>Revenue/User:</b> {card["revenue"]}<br>'
                f'<b>Orders/User:</b> {card["orders"]}<br>'
                f'<b>Conversion:</b> {card["conversion"]}<br>'
                f'<b>Avg Age:</b> {card["avg_age"]}'
                f"</div>",
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_cluster_scatter(result["profile"]), use_container_width=True)
    with col2:
        st.plotly_chart(make_cluster_heatmap(result["cluster_summary"]), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(make_rfm_bar(result["rfm_summary"]), use_container_width=True)
    with col4:
        st.plotly_chart(make_lifecycle_donut(result["lifecycle_summary"]), use_container_width=True)

    st.plotly_chart(make_funnel_figure(result["overall_funnel"]), use_container_width=True)
    st.plotly_chart(
        make_segment_heatmap(result["device_rates"], result["source_rates"]),
        use_container_width=True,
    )

    st.markdown("#### Demographics Breakdown")
    dem_col1, dem_col2 = st.columns([1, 1])
    with dem_col1:
        st.plotly_chart(make_sunburst_figure(result["sunburst_frame"]), use_container_width=True)
    with dem_col2:
        st.plotly_chart(make_age_pyramid(result["age_gender"]), use_container_width=True)

# ── Tab 4: Retention ────────────────────────────────────────────────────────
with tab_retention:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            make_retention_heatmap(result["retention_session"], "Session Retention Cohort"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            make_retention_heatmap(result["retention_purchase"], "Purchase Retention Cohort"),
            use_container_width=True,
        )

# ── Tab 5: Chatbot ───────────────────────────────────────────────────────────
with tab_chatbot:
    st.subheader("Customer Lookup")
    st.caption(
        "Enter a customer ID to view their profile, segment, LTV tier, and CRM recommendation."
    )

    cid_input = st.text_input("Customer ID", placeholder="e.g. 1042")
    if cid_input:
        try:
            cid = int(cid_input)
            info = lookup_customer(cid, result["profile"])
            if info is None:
                st.error(f"Customer ID {cid} not found in the dataset.")
            else:
                # LTV tier badge colours
                tier_colours = {
                    "Platinum": ("#b388ff", "#1a0a2e"),
                    "Gold": ("#FFB84D", "#2a1a00"),
                    "Silver": ("#8fa2be", "#0c1828"),
                    "Bronze": ("#FF6B6B", "#2a0a0a"),
                }
                tc, tbg = tier_colours.get(info["ltv_tier"], ("#00c2a8", "#001a14"))

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**Profile**")
                    st.write(f"**Name:** {info['name']}")
                    st.write(f"**Country:** {info['country']} ({info['region']})")
                    st.write(f"**Age:** {info.get('age', '-')} ({info['age_band']})")
                    st.write(f"**Gender:** {info['gender_inferred']}")
                    st.write(f"**Marketing opt-in:** {'Yes' if info['marketing_opt_in'] else 'No'}")

                with col_b:
                    st.markdown("**Behaviour**")
                    st.write(f"**Sessions:** {info['session_count']}")
                    st.write(f"**Orders:** {info['order_count']}")
                    st.write(f"**Revenue:** {format_currency(info['revenue'])}")
                    st.write(f"**AOV:** {format_currency(info['avg_order_value'])}")
                    st.write(f"**Avg Discount:** {format_pct(info['avg_discount_pct'])}")
                    st.write(f"**Recency:** {int(info['recency_days'])} days ago")
                    st.write(f"**Page→Cart Rate:** {format_pct(info['page_to_cart_rate'])}")

                with col_c:
                    st.markdown("**Segments & LTV**")
                    st.write(f"**RFM Segment:** {info['rfm_segment']}")
                    st.write(f"**Lifecycle:** {info['lifecycle_stage']}")
                    st.write(f"**Persona:** {info['cluster_name']}")
                    st.markdown(
                        f'<span class="tier-badge" style="background:{tbg};color:{tc};border:1px solid {tc}">'
                        f"LTV Tier: {info['ltv_tier']} ({info['ltv_score']}/100)</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="crm-box"><b>CRM Recommendation</b><br>{info["crm_recommendation"]}</div>',
                    unsafe_allow_html=True,
                )

        except ValueError:
            st.error("Please enter a valid integer customer ID.")

    st.divider()

    # -------------------------------------------------------------------
    # Q&A
    # -------------------------------------------------------------------
    st.subheader("Data Q&A")
    st.caption(
        "Ask a business question in plain English. "
        "AI-powered answers (Claude API) will be available in Phase 2."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_q = st.chat_input("Ask a question about the data...")
    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        answer = answer_question(user_q, data, result["profile"])
        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # -------------------------------------------------------------------
    # CRM reference table
    # -------------------------------------------------------------------
    with st.expander("CRM Recommendations Reference Table"):
        crm_data = {
            "Stage / Segment": [
                "New Visitor", "Browsing Prospect", "New Buyer",
                "Active Repeat Buyer", "Cooling Down", "Dormant Customer",
                "Champions", "Loyal Customers", "At-Risk High Value",
                "Lost Low Value", "Price / Casual Shoppers", "Potential Loyalists",
            ],
            "Recommended Action": [
                "Welcome email + first-order discount code (10–15%)",
                "Limited-time offer on browsed products; highlight top-rated items",
                "Product guide + cross-sell recommendations 3 days post-purchase",
                "Loyalty programme invitation; early access to new arrivals",
                "Win-back email ('We miss you') with personalised exclusive discount",
                "High-value: direct outreach + premium incentive; Low-value: low-cost EDM",
                "VIP events; first access to new collections and limited editions",
                "Milestone rewards; promote complementary categories",
                "Urgent win-back: targeted discount + proactive CS contact",
                "Low-cost EDM; A/B test reactivation creative",
                "Flash sales and discount events; avoid full-price messaging",
                "Content marketing and personalised product recommendations",
            ],
        }
        st.dataframe(pd.DataFrame(crm_data), use_container_width=True, hide_index=True)
