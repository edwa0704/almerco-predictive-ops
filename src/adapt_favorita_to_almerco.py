import pandas as pd
import numpy as np
import os

IN_TRAIN = "data/raw/favorita/train.csv"
IN_HOL = "data/raw/favorita/holidays_events.csv"
OUT_RAW = "data/raw/sales_history_raw.csv"

def main():
    if not os.path.exists(IN_TRAIN):
        raise FileNotFoundError(f"No existe {IN_TRAIN}. Descárgalo desde Hugging Face primero.")
    if not os.path.exists(IN_HOL):
        raise FileNotFoundError(f"No existe {IN_HOL}. Descárgalo desde Hugging Face primero.")

    # Favorita train columns: date, store_nbr, family, sales, onpromotion (según descripción)
    df = pd.read_csv(IN_TRAIN)
    df["date"] = pd.to_datetime(df["date"])

    # Holidays: usaremos fechas tipo Holiday/Event como indicador binario simple
    hol = pd.read_csv(IN_HOL)
    hol["date"] = pd.to_datetime(hol["date"])
    hol_dates = set(hol["date"].dt.date.unique())

    # Convertir a tu esquema:
    # fecha, producto, categoria, cantidad, precio_venta, costo, promo, stock_final, is_holiday
    out = pd.DataFrame()
    out["fecha"] = df["date"].dt.strftime("%Y-%m-%d")

    # producto: combinamos store + family para que sea único y “tipo SKU”
    out["producto"] = "STORE_" + df["store_nbr"].astype(str) + "_" + df["family"].astype(str)

    # categoria: family
    out["categoria"] = df["family"].astype(str)

    # cantidad: sales (Favorita puede ser decimal; tu pipeline trabaja bien con floats)
    out["cantidad"] = df["sales"].astype(float)

    # promo: onpromotion (en Favorita es cantidad de items promocionados; lo convertimos a 0/1)
    if "onpromotion" in df.columns:
        out["promo"] = (df["onpromotion"].fillna(0).astype(float) > 0).astype(int)
    else:
        out["promo"] = 0

    # is_holiday: si la fecha está en holidays_events.csv => 1
    out["is_holiday"] = df["date"].dt.date.apply(lambda d: 1 if d in hol_dates else 0)

    # precio_venta y costo NO existen en este dataset, así que los simulamos de forma estable:
    # precio_venta: base por categoria + ruido pequeño
    cat_base = (out["categoria"].astype("category").cat.codes % 50) + 10  # 10..59
    base_price = cat_base.astype(float)

    # promo reduce precio ligeramente (como comportamiento real)
    price = base_price * (1.0 - 0.08 * out["promo"])
    price = price * (1.0 + np.random.default_rng(42).normal(0, 0.02, size=len(price)))
    out["precio_venta"] = price.round(2)

    # costo: margen fijo (ej. 70% del precio)
    out["costo"] = (out["precio_venta"] * 0.70).round(2)

    # stock_final: no hay stock real; ponemos un proxy simple para no romper tu pipeline
    # (0 cuando ventas muy altas sugiere posible quiebre; si no, stock positivo)
    q95 = out["cantidad"].quantile(0.95)
    out["stock_final"] = np.where(out["cantidad"] >= q95, 0, 20).astype(int)

    # Guardar
    os.makedirs("data/raw", exist_ok=True)
    out.to_csv(OUT_RAW, index=False)
    print(f"✅ Adaptación completada. Nuevo raw listo: {OUT_RAW}")
    print("Columnas:", list(out.columns))

if __name__ == "__main__":
    main()