import os
import pandas as pd

IN_PATH = "data/raw/source/supermarket_sales.csv"
OUT_PATH = "data/raw/sales_history_raw.csv"

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"No existe: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    out = pd.DataFrame()

    out["fecha"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    out["producto"] = df["Product line"].astype(str)
    out["categoria"] = df["Product line"].astype(str)
    out["cantidad"] = df["Quantity"].astype(float)
    out["precio_venta"] = df["Unit price"].astype(float)

    # costo unitario = cogs total / cantidad
    out["costo"] = (df["cogs"] / df["Quantity"]).astype(float)

    out["promo"] = 0
    out["stock_final"] = 0
    out["stock_inicial"] = 0
    out["is_holiday"] = 0

    out = out.dropna()
    out = out[out["cantidad"] > 0]

    os.makedirs("data/raw", exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("✅ Adaptación completada")
    print("Archivo generado:", OUT_PATH)
    print("Filas:", len(out))

if __name__ == "__main__":
    main()