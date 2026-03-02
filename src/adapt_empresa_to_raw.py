import os
import argparse
import pandas as pd

# Salida estándar para todo tu proyecto
OUT_RAW = "data/raw/sales_history_raw.csv"

# Sinónimos típicos que llegan en CSV empresa
SYNONYMS = {
    "fecha": ["fecha", "date", "invoice_date", "order_date", "datetime", "day"],
    "producto": ["producto", "product", "item", "sku", "description", "product_name", "name"],
    "categoria": ["categoria", "category", "family", "line", "product_line", "department"],
    "cantidad": ["cantidad", "qty", "quantity", "units", "sales_qty", "items_sold", "sales"],
    "precio_venta": ["precio_venta", "unit_price", "price", "sale_price", "unitprice", "precio"],
    "costo": ["costo", "cost", "cogs", "unit_cost", "cost_price"],
    "promo": ["promo", "promotion", "onpromotion", "discount", "is_promo"],
}

def find_col(cols_lower, candidates):
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV real de la empresa (ruta)")
    parser.add_argument("--sep", default=None, help="Separador si sabes (ej: , o ;). Si no, autodetecta.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No existe: {args.input}")

    # Leer robusto (coma/semicolon)
    if args.sep:
        df = pd.read_csv(args.input, sep=args.sep)
    else:
        # intento simple: coma, si falla o queda 1 columna, prueba ;
        df = pd.read_csv(args.input)
        if df.shape[1] == 1:
            df = pd.read_csv(args.input, sep=";")

    # Mapeo automático
    cols_lower = {c.strip().lower(): c for c in df.columns}

    mapping = {}
    for std_col, cand in SYNONYMS.items():
        found = find_col(cols_lower, cand)
        if found:
            mapping[found] = std_col

    df = df.rename(columns=mapping)

    # Validación mínima
    required_min = ["fecha", "cantidad"]
    missing_min = [c for c in required_min if c not in df.columns]
    if missing_min:
        raise ValueError(
            f"CSV no tiene lo mínimo para demanda. Falta: {missing_min}\n"
            f"Columnas actuales: {list(df.columns)}"
        )

    # Si no viene producto/categoria, crea placeholders (para que el pipeline nunca rompa)
    if "producto" not in df.columns:
        df["producto"] = "PRODUCTO_UNICO"
    if "categoria" not in df.columns:
        df["categoria"] = df["producto"]

    # Si no hay precio/costo, crea NaN (fase1 puede limpiarlo o dejarlo)
    if "precio_venta" not in df.columns:
        df["precio_venta"] = pd.NA
    if "costo" not in df.columns:
        df["costo"] = pd.NA
    if "promo" not in df.columns:
        df["promo"] = 0

    # Normaliza fecha
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Filtra filas sin fecha o cantidad
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df = df.dropna(subset=["fecha", "cantidad"])
    df = df[df["cantidad"] >= 0]

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUT_RAW, index=False)
    print(f"✅ Raw estándar listo: {OUT_RAW}")
    print("Columnas estándar disponibles:", [c for c in ["fecha","producto","categoria","cantidad","precio_venta","costo","promo"] if c in df.columns])
    print("Filas:", len(df))

if __name__ == "__main__":
    main()
