"""CSV data loading with typed dtypes and parse dates."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR


def load_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all 7 CSV tables and return as a dict keyed by table name."""
    required_files = [
        "customers.csv",
        "sessions.csv",
        "events.csv",
        "orders.csv",
        "order_items.csv",
        "products.csv",
        "reviews.csv",
    ]
    missing = [name for name in required_files if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {data_dir}: {', '.join(missing)}"
        )

    customers = pd.read_csv(
        data_dir / "customers.csv",
        parse_dates=["signup_date"],
        dtype={"customer_id": "int32", "country": "category", "age": "float32"},
    )
    sessions = pd.read_csv(
        data_dir / "sessions.csv",
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
        data_dir / "events.csv",
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
        data_dir / "orders.csv",
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
        data_dir / "order_items.csv",
        dtype={
            "order_id": "int32",
            "product_id": "int32",
            "unit_price_usd": "float32",
            "quantity": "int16",
            "line_total_usd": "float32",
        },
    )
    products = pd.read_csv(
        data_dir / "products.csv",
        dtype={
            "product_id": "int32",
            "category": "category",
            "price_usd": "float32",
            "cost_usd": "float32",
            "margin_usd": "float32",
        },
    )
    reviews = pd.read_csv(
        data_dir / "reviews.csv",
        parse_dates=["review_time"],
        dtype={
            "review_id": "int32",
            "order_id": "int32",
            "product_id": "int32",
            "rating": "int16",
        },
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
