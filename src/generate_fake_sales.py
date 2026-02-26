import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
OUT_PATH = "data/raw/sales_history_raw.csv"

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)

def seasonal_multiplier(d: datetime) -> float:
    # Estacionalidad simple
    if d.month == 12:
        return 1.30  # navidad
    if d.month == 3:
        return 1.20  # inicio clases
    return 1.00

def is_holiday(d: datetime) -> int:
    # Prototipo simple: marca algunos feriados “fijos” (puedes ajustar luego)
    # (solo para aprender el patrón; en real vendrá del calendario de la empresa)
    fixed = {(1, 1), (7, 28), (7, 29), (12, 25)}
    return 1 if (d.month, d.day) in fixed else 0

def generate_data():
    start_date = datetime(2025, 1, 1)
    days = 365

    # (producto, categoria, costo, base_demanda, stock_base)
    products = [
        ("Ryzen 5 5600", "CPU_AMD", 700, 9, 60),
        ("Ryzen 7 5800X", "CPU_AMD", 1000, 6, 40),
        ("Intel i5 12400", "CPU_INTEL", 900, 7, 50),
        ("Placa B450", "MB_AMD", 450, 6, 70),
        ("Placa B550", "MB_AMD", 600, 5, 60),
        ("RAM 8GB Kingston", "RAM", 80, 10, 200),
        ("SSD 1TB Kingston", "SSD", 250, 8, 120),
    ]

    rows = []

    for i in range(days):
        d = start_date + timedelta(days=i)
        season = seasonal_multiplier(d)
        fer = is_holiday(d)

        # Promos: campañas cortas (3 días) cada ~30 días
        promo_window = (i % 30) in {10, 11, 12}

        # Descuento placas cada 60 días por 15 días (como tu lógica previa)
        placa_discount = (i % 60) < 15

        for product, category, cost, base, stock_base in products:
            promo = 0

            # Precio
            if category == "MB_AMD" and placa_discount:
                price = cost * np.random.uniform(0.85, 0.95)
            else:
                price = cost * np.random.uniform(1.10, 1.25)

            # Promo explícita (afecta precio y demanda)
            if promo_window and category in {"CPU_AMD", "SSD", "RAM"}:
                promo = 1
                price *= np.random.uniform(0.88, 0.95)  # baja precio en promo

            # Efecto cruzado: placas baratas -> sube AMD
            cross_boost = 1.0
            if category == "CPU_AMD" and placa_discount:
                cross_boost = 1.20

            # Efecto feriado (puede subir o bajar; aquí lo hacemos subir moderado)
            holiday_boost = 1.10 if fer == 1 else 1.00

            # Efecto precio suave
            typical_price = cost * 1.18
            price_factor = np.clip(typical_price / max(price, 1e-6), 0.85, 1.15)

            # Demanda esperada (sin stock)
            expected = base * season * cross_boost * holiday_boost * price_factor
            expected *= (1.25 if promo == 1 else 1.0)

            # Ruido CONTROLADO (no extremo)
            noise = np.random.normal(0, 1.2)
            demand = int(max(0, round(expected + noise)))

            # Stock simulado (stock_final)
            # Stock final varía y puede quedar 0 (stockout)
            stock_today = int(max(0, round(stock_base * np.random.uniform(0.15, 1.0))))

            # Ventas reales = min(demanda, stock)
            sales = min(demand, stock_today)

            # Stock final después de vender
            stock_final = stock_today - sales

            rows.append([
                d.strftime("%Y-%m-%d"),
                product,
                category,
                sales,
                round(price, 2),
                cost,
                promo,
                stock_final,
                fer
            ])

    df = pd.DataFrame(rows, columns=[
        "fecha", "producto", "categoria",
        "cantidad", "precio_venta", "costo",
        "promo", "stock_final", "is_holiday"
    ])
    return df

if __name__ == "__main__":
    ensure_dirs()
    df = generate_data()
    df.to_csv(OUT_PATH, index=False)
    print(f"Dataset simulado generado: {OUT_PATH} (rows={len(df)})")