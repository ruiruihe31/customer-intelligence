"""Generate the standalone HTML dashboard report.

Run:  python scripts/export_html.py [--output path/to/file.html]
      (from the Customer/ project root)
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
from core.charts import (
    figure_html,
    generate_wordcloud_section_html,
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
from core.config import COUNTRY_META, PROJECT_ROOT
from core.data_loader import load_data
from core.utils import format_currency, format_number, format_pct

DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "dashboard.html"


def _kpi_cards_html(kpis: list[dict[str, str]]) -> str:
    return "".join(
        f'<div class="metric-block reveal">'
        f'<div class="metric-label">{k["label"]}</div>'
        f'<div class="metric-value">{k["value"]}</div>'
        f'<div class="metric-detail">{k["detail"]}</div>'
        f"</div>"
        for k in kpis
    )


def _persona_cards_html(cards: list[dict[str, str]]) -> str:
    return "".join(
        f'<article class="persona-card reveal">'
        f'<div class="persona-kicker">Persona</div>'
        f"<h3>{c['cluster_name']}</h3>"
        f"<p>Top region: {c['top_region']} / Main source: {c['top_source']}</p>"
        f'<div class="persona-grid">'
        f"<div><span>Users</span><strong>{c['users']}</strong></div>"
        f"<div><span>Revenue / User</span><strong>{c['revenue']}</strong></div>"
        f"<div><span>Orders / User</span><strong>{c['orders']}</strong></div>"
        f"<div><span>Sessions / User</span><strong>{c['sessions']}</strong></div>"
        f"<div><span>Conversion</span><strong>{c['conversion']}</strong></div>"
        f"<div><span>Avg Age</span><strong>{c['avg_age']}</strong></div>"
        f"</div></article>"
        for c in cards
    )


def build_html(output_path: Path = DEFAULT_OUTPUT) -> None:
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
    product_master, sentiment_pct, reviews_ws = sentiment_product_metrics(data)
    retention_session = retention_matrix(data, activity_source="sessions")
    retention_purchase = retention_matrix(data, activity_source="orders")
    wc_html = generate_wordcloud_section_html(reviews_ws, data["products"])

    figs = {
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

    ordered_keys = [
        "overview", "geo", "sunburst", "age_pyramid", "cluster_scatter", "cluster_heatmap",
        "rfm_bar", "lifecycle_donut", "funnel", "segment_heatmap",
        "product_sales_rank", "product_rating_rank", "sentiment_bar",
        "category_bubble", "category_area", "top_products",
        "retention_session", "retention_purchase",
    ]
    fig_blocks = {k: figure_html(figs[k], include_js=(i == 0)) for i, k in enumerate(ordered_keys)}

    purchase_sessions = int(overall_funnel.loc[overall_funnel["stage"] == "purchase", "sessions"].iloc[0])
    page_sessions = int(overall_funnel.loc[overall_funnel["stage"] == "page_view", "sessions"].iloc[0])
    add_to_cart_sessions = int(overall_funnel.loc[overall_funnel["stage"] == "add_to_cart", "sessions"].iloc[0])

    kpis = [
        {"label": "Customers", "value": format_number(len(data["customers"])), "detail": f"{data['customers']['country'].nunique()} countries"},
        {"label": "Sessions", "value": format_number(len(data["sessions"])), "detail": f"{format_number(data['sessions']['customer_id'].nunique())} active users"},
        {"label": "Orders", "value": format_number(len(data["orders"])), "detail": f"{format_number(purchase_sessions)} purchase sessions"},
        {"label": "Revenue", "value": format_currency(data["orders"]["total_usd"].sum()), "detail": f"AOV {format_currency(data['orders']['total_usd'].mean())}"},
        {"label": "Session Conversion", "value": format_pct(purchase_sessions / page_sessions if page_sessions else 0), "detail": f"Add-to-cart rate {format_pct(add_to_cart_sessions / page_sessions if page_sessions else 0)}"},
        {"label": "Average Rating", "value": f"{data['reviews']['rating'].mean():.2f}", "detail": f"{format_number(len(data['reviews']))} reviews"},
    ]

    max_time = max(
        data["events"]["timestamp"].max(),
        data["orders"]["order_time"].max(),
        data["reviews"]["review_time"].max(),
    )
    data_notes = [
        "Dataset: 7 CSV tables loaded at full granularity.",
        f"Date range: {data['customers']['signup_date'].min().date()} to {max_time.date()}.",
        f"Country coverage: {', '.join(sorted(COUNTRY_META.keys()))}.",
        f"Clusters selected automatically: {cluster_summary['cluster_id'].nunique()} (silhouette={cluster_summary['silhouette_score'].iloc[0]:.3f}).",
    ]
    note_items = "".join(f"<li>{item}</li>" for item in data_notes)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Customer Analytics Dashboard</title>
<style>
:root{{--bg:#07111a;--panel:rgba(8,21,34,.82);--panel-strong:rgba(10,26,42,.96);--line:rgba(148,163,184,.16);--text:#eef3ff;--muted:#8fa2be;--accent:#00c2a8;--shadow:0 24px 70px rgba(0,0,0,.32);}}
*{{box-sizing:border-box;}}html{{scroll-behavior:smooth;}}
body{{margin:0;font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:var(--text);background:radial-gradient(circle at top left,rgba(108,140,255,.22),transparent 28%),radial-gradient(circle at 78% 18%,rgba(0,194,168,.18),transparent 26%),linear-gradient(180deg,#061018 0%,#07111a 52%,#081520 100%);}}
.shell{{width:min(1480px,calc(100vw - 40px));margin:0 auto;padding:28px 0 56px;}}
.topbar{{position:sticky;top:10px;z-index:40;display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border:1px solid var(--line);border-radius:18px;background:rgba(7,17,26,.72);backdrop-filter:blur(18px);}}
.brand strong{{font-size:1.15rem;letter-spacing:.06em;text-transform:uppercase;}}
.brand span,.nav,.meta-title,.metric-label,.persona-grid span{{color:var(--muted);}}
.nav{{display:flex;gap:16px;font-size:.92rem;}}
.kpi-strip{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px;margin:24px 0;}}
.metric-block{{padding:18px 20px;border:1px solid var(--line);background:rgba(7,18,29,.88);border-radius:22px;}}
.metric-label{{font-size:.82rem;text-transform:uppercase;letter-spacing:.1em;}}
.metric-value{{margin-top:12px;font-size:clamp(1.7rem,3vw,2.8rem);letter-spacing:-.04em;font-weight:700;}}
.metric-detail{{margin-top:8px;color:#c2d0e4;font-size:.92rem;}}
.section{{padding:24px;margin-bottom:24px;border:1px solid var(--line);background:var(--panel);border-radius:28px;}}
.section-head h2{{margin:0 0 12px;font-size:clamp(1.4rem,2vw,2.1rem);}}
.grid-2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;}}
.persona-row{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin-top:18px;}}
.figure-shell{{padding:4px 2px 0;border-radius:24px;overflow:hidden;}}
.persona-card{{padding:18px;background:var(--panel-strong);border:1px solid var(--line);border-radius:28px;}}
.persona-kicker{{color:var(--accent);text-transform:uppercase;font-size:.78rem;letter-spacing:.12em;}}
.persona-card h3{{margin:10px 0 8px;}}
.persona-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}}
.reveal{{opacity:0;transform:translateY(24px);transition:opacity .7s ease,transform .7s ease;}}
.reveal.visible{{opacity:1;transform:translateY(0);}}
.wc-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:18px;}}
.wc-card{{border:1px solid var(--line);background:rgba(7,18,29,.88);border-radius:22px;padding:16px;}}
.wc-label{{font-size:.82rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);margin-bottom:10px;}}
.meta-panel{{padding:24px;border:1px solid var(--line);background:var(--panel);border-radius:28px;}}
.meta-list{{list-style:none;padding:0;margin:0;display:grid;gap:10px;color:#c9d6ea;}}
.footer-note{{padding:26px 28px 34px;border:1px solid var(--line);background:var(--panel);border-radius:28px;}}
@media(max-width:1180px){{.grid-2,.persona-row,.wc-grid{{grid-template-columns:1fr;}}.kpi-strip{{grid-template-columns:repeat(3,minmax(0,1fr));}}}}
</style></head><body>
<div class="shell">
<div class="topbar">
<div class="brand"><strong>Customer Intelligence</strong> <span>7-table integrated analytics dashboard</span></div>
<div class="nav"><a href="#overview">Overview</a><a href="#cluster">Clusters</a><a href="#segments">Segments</a><a href="#funnel">Funnel</a><a href="#product">Products</a><a href="#retention">Retention</a></div>
</div>
<div class="meta-panel reveal" style="margin:24px 0"><div class="meta-title">Data scope</div><ul class="meta-list">{note_items}</ul></div>
<section class="kpi-strip" id="overview">{_kpi_cards_html(kpis)}</section>
<section class="section reveal"><div class="section-head"><h2>Business Overview</h2></div><div class="figure-shell">{fig_blocks["overview"]}</div></section>
<section class="section reveal" id="profile"><div class="section-head"><h2>Customer Profile</h2></div>
<div class="grid-2"><div class="figure-shell">{fig_blocks["sunburst"]}</div><div class="figure-shell">{fig_blocks["geo"]}</div></div>
<div class="grid-2" style="margin-top:18px"><div class="figure-shell">{fig_blocks["age_pyramid"]}</div><div class="figure-shell">{fig_blocks["lifecycle_donut"]}</div></div></section>
<section class="section reveal" id="cluster"><div class="section-head"><h2>Customer Clustering</h2></div>
<div class="grid-2"><div class="figure-shell">{fig_blocks["cluster_scatter"]}</div><div class="figure-shell">{fig_blocks["cluster_heatmap"]}</div></div>
<div class="persona-row">{_persona_cards_html(persona_cards)}</div></section>
<section class="section reveal" id="segments"><div class="section-head"><h2>Segmentation</h2></div>
<div class="grid-2"><div class="figure-shell">{fig_blocks["rfm_bar"]}</div><div class="figure-shell">{fig_blocks["funnel"]}</div></div></section>
<section class="section reveal" id="funnel"><div class="section-head"><h2>Funnel by Device &amp; Channel</h2></div>
<div class="figure-shell">{fig_blocks["segment_heatmap"]}</div></section>
<section class="section reveal" id="product"><div class="section-head"><h2>Products &amp; Categories</h2></div>
<div class="grid-2"><div class="figure-shell">{fig_blocks["product_sales_rank"]}</div><div class="figure-shell">{fig_blocks["product_rating_rank"]}</div></div>
<div class="grid-2" style="margin-top:18px"><div class="figure-shell">{fig_blocks["sentiment_bar"]}</div><div class="figure-shell">{fig_blocks["category_bubble"]}</div></div>
<div class="grid-2" style="margin-top:18px"><div class="figure-shell">{fig_blocks["category_area"]}</div><div class="figure-shell">{fig_blocks["top_products"]}</div></div></section>
<section class="section reveal" id="wordclouds"><div class="section-head"><h2>Review Keyword Clouds</h2></div>
<div class="wc-grid">{wc_html}</div></section>
<section class="section reveal" id="retention"><div class="section-head"><h2>Retention</h2></div>
<div class="grid-2"><div class="figure-shell">{fig_blocks["retention_session"]}</div><div class="figure-shell">{fig_blocks["retention_purchase"]}</div></div></section>
<div class="footer-note reveal"><ul class="meta-list">
<li>All charts are interactive Plotly figures embedded directly in this HTML file.</li>
<li>Retention uses signup-month cohorts tracked over 12 months of post-signup activity.</li>
</ul></div>
</div>
<script>const obs=new IntersectionObserver(entries=>entries.forEach(e=>{{if(e.isIntersecting)e.target.classList.add('visible');}}),{{threshold:.16}});document.querySelectorAll('.reveal').forEach(n=>{{if(!n.classList.contains('visible'))obs.observe(n);}});</script>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML dashboard written to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export standalone HTML dashboard.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML path")
    args = parser.parse_args()
    build_html(args.output)
