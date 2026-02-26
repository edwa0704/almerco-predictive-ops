import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

RAW_PATH = "data/raw/sales_history_raw.csv"


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)


def generate_data():
    start_date = datetime(2025, 1, 1)
    days = 365

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

        # Estacionalidad semanal fuerte
        weekly_factor = 1.0
        if dow == 0:      # lunes
            weekly_factor = 1.15
        elif dow == 5:    # sábado
            weekly_factor = 1.25
        elif dow == 6:    # domingo
            weekly_factor = 0.90

        # Fin de mes fuerte
        end_month_factor = 1.30 if dom >= 25 else 1.00

        # Navidad fuerte
        christmas_factor = 1.50 if month == 12 else 1.00

        # Tendencia anual leve (+10%)
        trend = 1.00 + (i / (days - 1)) * 0.10

        # Feriado simple (domingo como “cerrado/parcial” en este prototipo)
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

            # Elasticidad fuerte: más precio => menos demanda
            typical_price = cost * 1.20
            price_factor = np.clip(typical_price / max(price, 1e-6), 0.80, 1.25)

            # Demanda “real” (no observada) con ruido controlado
            demand = base_demand * weekly_factor * end_month_factor * christmas_factor * trend * promo_boost * price_factor
            demand = np.random.normal(demand, demand * 0.06)  # 6% ruido (más aprendible)
            demand = max(0, int(round(demand)))

            # Stock inicial: a veces bajo o 0 (stockout real)
            # 10% días con shock fuerte
            if random.random() < 0.10:
                stock_inicial = random.randint(0, 6)
            else:
                stock_inicial = random.randint(10, 60)

            # Ventas observadas = min(demanda, stock)
            sales = min(demand, stock_inicial)

            # Stock final real
            stock_final = stock_inicial - sales

            rows.append([
                d.strftime("%Y-%m-%d"),
                product,
                category,
                sales,               # OJO: esto es venta observada
                round(price, 2),
                cost,
                promo,
                stock_final,
                is_holiday,
                stock_inicial,       # opcional extra (muy útil)
                demand               # opcional extra (si quieres medir “verdad” del generador)
            ])

    df = pd.DataFrame(rows, columns=[
        "fecha", "producto", "categoria",
        "cantidad", "precio_venta", "costo",
        "promo", "stock_final", "is_holiday",
        "stock_inicial", "demand_real"
    ])
    return df


if __name__ == "__main__":
    ensure_dirs()
    df = generate_data()
    df.to_csv(RAW_PATH, index=False)
    print(f"Dataset generado (con stockouts reales): {RAW_PATH} (rows={len(df)})")