"""Plotly chart builders and word-cloud generators (no file I/O)."""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from .config import AGE_LABELS, CLUSTER_RANDOM_STATE, COLOR_SEQUENCE
from .utils import format_currency, format_number, format_pct


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def build_theme() -> dict:
    return dict(
        layout=dict(
            font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", color="#F8FAFC"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=COLOR_SEQUENCE,
            margin=dict(l=40, r=30, t=70, b=45),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(7, 17, 26, 0.96)",
                bordercolor="#94A3B8",
                font=dict(color="#FFFFFF", size=15),
                align="left",
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.12)",
                zeroline=False,
                linecolor="rgba(148,163,184,0.18)",
                tickfont=dict(color="#EAF2FF", size=13),
                title_font=dict(color="#FFFFFF", size=16),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.12)",
                zeroline=False,
                linecolor="rgba(148,163,184,0.18)",
                tickfont=dict(color="#EAF2FF", size=13),
                title_font=dict(color="#FFFFFF", size=16),
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


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

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
        go.Scatter(
            x=monthly["month"],
            y=monthly["orders"],
            mode="lines",
            name="Orders",
            line=dict(width=2.5, color="#00C2A8"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["revenue"],
            mode="lines",
            name="Revenue",
            line=dict(width=3, color="#FFB84D"),
        ),
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


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

def make_geo_figure(geo: pd.DataFrame) -> go.Figure:
    fig = px.choropleth(
        geo,
        locations="iso_alpha3",
        locationmode="ISO-3",
        color="revenue",
        hover_name="country_name",
        custom_data=["sessions", "orders", "conversion", "region"],
        # Light-to-dark scale — readable on both light and dark backgrounds
        color_continuous_scale=["#cff0ec", "#56CCF2", "#00C2A8", "#006a68"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>Region=%{customdata[3]}"
            "<br>Sessions=%{customdata[0]:,.0f}<br>Orders=%{customdata[1]:,.0f}"
            "<br>Conversion=%{customdata[2]:.1%}<br>Revenue=$%{z:,.0f}<extra></extra>"
        )
    )
    fig.update_layout(
        geo=dict(
            bgcolor="rgba(255,255,255,0.04)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(148,163,184,0.3)",
            showland=True,
            landcolor="#e8eff4",
            showocean=True,
            oceancolor="#f0f6fa",
            showlakes=False,
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title="Revenue",
            tickprefix="$",
            thickness=14,
            len=0.7,
            x=1.01,
        ),
    )
    fig = apply_theme(fig, "Global Revenue Heatmap", 520)
    # Override margin AFTER apply_theme so the colorbar is not clipped
    fig.update_layout(margin=dict(l=0, r=130, t=70, b=10))
    return fig


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def make_sunburst_figure(sunburst_frame: pd.DataFrame) -> go.Figure:
    sunburst_frame = sunburst_frame.loc[sunburst_frame["users"] > 0].copy()
    if sunburst_frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No demographic data available",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16),
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
        hovertemplate="%{label}<br>Users=%{value:,.0f}<extra></extra>",
    )
    return apply_theme(fig, "Audience Profile: Region × Gender × Age Band", 500)


def make_age_pyramid(age_gender: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    male = age_gender[age_gender["gender_inferred"] == "Male"]
    female = age_gender[age_gender["gender_inferred"] == "Female"]
    other = age_gender[~age_gender["gender_inferred"].isin(["Male", "Female"])]

    fig.add_trace(
        go.Bar(y=male["age_band"], x=male["plot_users"], orientation="h", name="Male", marker_color="#6C8CFF")
    )
    fig.add_trace(
        go.Bar(y=female["age_band"], x=female["plot_users"], orientation="h", name="Female", marker_color="#FF6B6B")
    )
    if not other.empty:
        fig.add_trace(
            go.Bar(
                y=other["age_band"],
                x=other["plot_users"],
                orientation="h",
                name="Unknown / Ambiguous",
                marker_color="#8EA6C9",
            )
        )
    fig.update_layout(barmode="relative")
    fig.update_xaxes(tickvals=[-4000, -2000, 0, 2000, 4000], ticktext=["4k", "2k", "0", "2k", "4k"])
    return apply_theme(fig, "Age Pyramid by Inferred Gender", 420)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

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
    display = cluster_summary[
        ["cluster_name", "sessions", "orders", "revenue", "aov", "conversion", "avg_discount", "avg_rating", "recency", "avg_age"]
    ].set_index("cluster_name")
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


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------

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

    stage_cols = ["page_view", "add_to_cart", "checkout", "purchase"]
    stage_labels = ["Page View", "Add to Cart", "Checkout", "Purchase"]

    device_text = (device_rates[stage_cols] * 100).round(1).astype(str) + "%"
    source_text = (source_rates[stage_cols] * 100).round(1).astype(str) + "%"

    fig.add_trace(
        go.Heatmap(
            z=device_rates[stage_cols].values,
            x=stage_labels,
            y=device_rates.index.tolist(),
            colorscale=[[0.0, "#112438"], [0.5, "#1C4258"], [0.75, "#00C2A8"], [1.0, "#FFB84D"]],
            zmin=0, zmax=zmax,
            text=device_text, texttemplate="%{text}",
            hovertemplate="Group=%{y}<br>Stage=%{x}<br>Rate=%{z:.1%}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=source_rates[stage_cols].values,
            x=stage_labels,
            y=source_rates.index.tolist(),
            colorscale=[[0.0, "#112438"], [0.5, "#1C4258"], [0.75, "#00C2A8"], [1.0, "#FFB84D"]],
            zmin=0, zmax=zmax,
            text=source_text, texttemplate="%{text}",
            hovertemplate="Group=%{y}<br>Stage=%{x}<br>Rate=%{z:.1%}<extra></extra>",
        ),
        row=1, col=2,
    )
    return apply_theme(fig, "Funnel Performance by Device and Channel", 440)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

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
    fig = px.area(
        category_monthly, x="month", y="revenue", color="category",
        groupnorm=None, line_group="category",
    )
    return apply_theme(fig, "Category Revenue Trend Over Time", 430)


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
                        display["name"], display["category"],
                        display["revenue"], display["units"],
                        display["avg_rating"], display["review_count"],
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
        top, x="total_qty", y="name", orientation="h",
        color="category", text="total_qty",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_yaxes(autorange="reversed", title=None, tickfont=dict(size=11))
    fig.update_xaxes(title="Units Sold")
    fig = apply_theme(fig, "Top 10 Products by Sales Volume", 460)
    fig.update_layout(margin=dict(l=210, r=80, t=70, b=45))
    return fig


def make_product_rating_rank(product_master: pd.DataFrame) -> go.Figure:
    top_rated = (
        product_master[product_master["review_count"] >= 5]
        .sort_values("avg_rating", ascending=False)
        .head(10)
    )
    fig = px.bar(
        top_rated, x="avg_rating", y="name", orientation="h",
        color="category", text="avg_rating",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(autorange="reversed", title=None, tickfont=dict(size=11))
    fig.update_xaxes(title="Avg Rating", range=[0, 5.5])
    fig = apply_theme(fig, "Top 10 Rated Products (Min. 5 Reviews)", 460)
    fig.update_layout(margin=dict(l=210, r=80, t=70, b=45))
    return fig


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


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

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
    fig = px.pie(lifecycle_summary, names="lifecycle_stage", values="customers", hole=0.55)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_theme(fig, "Customer Lifecycle Mix", 430)


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def make_retention_heatmap(retention: pd.DataFrame, title: str) -> go.Figure:
    retention_text = retention.apply(
        lambda col: col.map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%")
    ).values
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


# ---------------------------------------------------------------------------
# Word clouds
# ---------------------------------------------------------------------------

def generate_wordcloud_images(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return {category: numpy RGBA array} for each product category.

    Numpy arrays are used instead of PIL Images so Streamlit's @st.cache_data
    can serialize/deserialize them reliably across all supported versions.
    """
    merged = reviews.merge(products[["product_id", "category"]], on="product_id", how="left")
    categories = sorted(merged["category"].dropna().unique())
    images: dict[str, np.ndarray] = {}
    for cat in categories:
        texts = merged.loc[merged["category"] == cat, "review_text"].dropna()
        if texts.empty:
            continue
        combined = " ".join(texts.astype(str))
        try:
            wc = WordCloud(
                width=700, height=340,
                background_color=None, mode="RGBA",
                colormap="cool", max_words=120,
            ).generate(combined)
            images[cat] = np.array(wc.to_image())
        except Exception:
            continue
    return images


def generate_wordcloud_section_html(reviews: pd.DataFrame, products: pd.DataFrame) -> str:
    """Return HTML block of base64-embedded word cloud images (for standalone HTML export)."""
    merged = reviews.merge(products[["product_id", "category"]], on="product_id", how="left")
    categories = sorted(merged["category"].dropna().unique())
    blocks: list[str] = []
    for cat in categories:
        texts = merged.loc[merged["category"] == cat, "review_text"].dropna()
        if texts.empty:
            continue
        combined = " ".join(texts.astype(str))
        wc = WordCloud(
            width=700, height=340,
            background_color=None, mode="RGBA",
            colormap="cool", max_words=120,
        ).generate(combined)
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        blocks.append(
            f'<div class="wc-card reveal">'
            f'<div class="wc-label">{cat}</div>'
            f'<img src="data:image/png;base64,{b64}" alt="{cat} keyword cloud" '
            f'style="width:100%;border-radius:14px;" /></div>'
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Serialisation helper (for HTML export)
# ---------------------------------------------------------------------------

def figure_html(fig: go.Figure, include_js: bool = False) -> str:
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"displayModeBar": False, "responsive": True},
    )
