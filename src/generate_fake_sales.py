import os
import re
import unicodedata
import hashlib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

DEFAULT_RAW_OUT = "data/raw/fake_sales_history_raw.csv"
DEFAULT_CLEAN_OUT = "data/clean_data_fake.csv"
REAL_RAW_PATH = "data/raw/sales_history_raw.csv"


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_product_name(name: str):
    if pd.isna(name):
        return np.nan

    s = str(name).lower().strip()
    s = strip_accents(s)

    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Unificar sinónimos (igual que tu fase1)
    s = s.replace("memoria ram", "ram")
    s = s.replace("memoria", "ram")
    s = s.replace("disco solido", "ssd")

    tokens = s.split()
    tokens = [t for t in tokens if t not in {"de", "para", "con", "y", "el", "la"}]
    tokens.sort()

    return " ".join(tokens)


def make_product_id(normalized_name: str) -> str:
    h = hashlib.sha1(normalized_name.encode("utf-8")).hexdigest()[:10]
    return f"PROD_{h}"


def generate_raw_fake(days: int = 365) -> pd.DataFrame:
    start_date = datetime(2025, 1, 1)

    # (producto, categoria, costo, demanda_base)
    products = [
        ("Ryzen 5 5600", "CPU_AMD", 700, 12),
        ("Ryzen 7 5800X", "CPU_AMD", 1000, 8),
        ("Intel i5 12400", "CPU_INTEL", 900, 10),
        ("Placa B450", "MB_AMD", 450, 9),
        ("Placa B550", "MB_AMD", 600, 7),
        ("RAM 8GB Kingston", "RAM", 80, 20),
        ("SSD 1TB Kingston", "SSD", 250, 14),
    ]

    rows = []

    for i in range(days):
        d = start_date + timedelta(days=i)
        dow = d.weekday()
        month = d.month
        dom = d.day

        # Estacionalidad semanal
        weekly_factor = 1.0
        if dow == 0:      # lunes
            weekly_factor = 1.15
        elif dow == 5:    # sábado
            weekly_factor = 1.25
        elif dow == 6:    # domingo
            weekly_factor = 0.90

        # Fin de mes
        end_month_factor = 1.30 if dom >= 25 else 1.00

        # Navidad
        christmas_factor = 1.50 if month == 12 else 1.00

        # Tendencia anual leve (+10%)
        trend = 1.00 + (i / (days - 1)) * 0.10

        # Feriado simple (domingo)
        is_holiday = 1 if dow == 6 else 0

        for product, category, cost, base_demand in products:
            # Promo corta: 3 días cada ~45
            promo = 1 if (i % 45 in [0, 1, 2]) else 0

            # Precio y elasticidad
            if promo == 1:
                price = cost * np.random.uniform(0.75, 0.85)
                promo_boost = 1.60
            else:
                price = cost * np.random.uniform(1.15, 1.30)
                promo_boost = 1.00

            # Elasticidad: más precio => menos demanda
            typical_price = cost * 1.20
            price_factor = np.clip(typical_price / max(price, 1e-6), 0.80, 1.25)

            # Demanda real con ruido
            demand = base_demand * weekly_factor * end_month_factor * christmas_factor * trend * promo_boost * price_factor
            demand = np.random.normal(demand, demand * 0.06)  # 6% ruido
            demand = max(0, int(round(demand)))

            # Stock inicial con stockouts
            if random.random() < 0.10:
                stock_inicial = random.randint(0, 6)
            else:
                stock_inicial = random.randint(10, 60)

            # Ventas observadas
            sales = min(demand, stock_inicial)
            stock_final = stock_inicial - sales

            rows.append([
                d.strftime("%Y-%m-%d"),
                product,
                category,
                sales,
                round(price, 2),
                cost,
                promo,
                stock_final,
                is_holiday,
                stock_inicial
            ])

    raw = pd.DataFrame(rows, columns=[
        "fecha", "producto", "categoria",
        "cantidad", "precio_venta", "costo",
        "promo", "stock_final", "is_holiday", "stock_inicial"
    ])
    return raw


def to_clean_like_fase1(raw: pd.DataFrame) -> pd.DataFrame:
    # Replica lo esencial que genera tu fase1_cleaning.py
    df = raw.drop_duplicates().copy()

    required_cols = ["fecha", "producto", "cantidad", "precio_venta", "costo", "categoria"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"El RAW fake no cumple columnas requeridas: {missing}")

    df_clean = df.dropna(subset=required_cols).copy()

    df_clean["producto_norm"] = df_clean["producto"].apply(normalize_product_name)
    df_clean = df_clean.dropna(subset=["producto_norm"])
    df_clean = df_clean[df_clean["producto_norm"].str.len() > 0]

    df_clean["product_id"] = df_clean["producto_norm"].apply(make_product_id)
    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Genera dataset FAKE alineado al pipeline (RAW + CLEAN).")
    parser.add_argument("--days", type=int, default=365, help="Cantidad de días a simular (default 365)")
    parser.add_argument("--raw_out", default=DEFAULT_RAW_OUT, help="Salida RAW fake")
    parser.add_argument("--clean_out", default=DEFAULT_CLEAN_OUT, help="Salida CLEAN fake (similar a fase1)")
    parser.add_argument("--overwrite-real", action="store_true",
                        help="⚠️ Si lo activas, sobrescribe data/raw/sales_history_raw.csv (NO recomendado)")
    args = parser.parse_args()

    ensure_dirs()

    raw = generate_raw_fake(days=args.days)
    raw.to_csv(args.raw_out, index=False)
    print(f"✅ RAW FAKE generado: {args.raw_out} (filas={len(raw)})")

    clean = to_clean_like_fase1(raw)
    clean.to_csv(args.clean_out, index=False)
    print(f"✅ CLEAN FAKE generado: {args.clean_out} (filas={len(clean)})")

    if args.overwrite_real:
        raw.to_csv(REAL_RAW_PATH, index=False)
        print(f"⚠️ También se sobrescribió el RAW principal: {REAL_RAW_PATH}")


if __name__ == "__main__":
    main()