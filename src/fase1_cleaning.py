import os
import re
import unicodedata
import hashlib
import numpy as np
import pandas as pd

# ✅ Detecta automáticamente cuál CSV raw usar
RAW_PATHS = [
    "data/raw/sales_history_raw.csv",           # generate_fake_sales.py  ← prioridad 1
    "data/raw/supermarket_sales_clean.csv",      # dataset supermarket     ← prioridad 2
]
OUT_PATH    = "data/clean_data.csv"
REPORT_PATH = "reports/null_report.txt"


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("reports",  exist_ok=True)
    os.makedirs("data",     exist_ok=True)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_product_name(name: str):
    if pd.isna(name):
        return np.nan
    s = strip_accents(str(name).lower().strip())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("memoria ram", "ram").replace("memoria", "ram").replace("disco solido", "ssd")
    tokens = [t for t in s.split() if t not in {"de", "para", "con", "y", "el", "la"}]
    tokens.sort()
    return " ".join(tokens)


def make_product_id(normalized_name: str) -> str:
    h = hashlib.sha1(normalized_name.encode("utf-8")).hexdigest()[:10]
    return f"PROD_{h}"


def main():
    ensure_dirs()

    # Detectar archivo raw disponible
    RAW_PATH = next((p for p in RAW_PATHS if os.path.exists(p)), None)
    if RAW_PATH is None:
        raise FileNotFoundError(
            "No se encontró ningún archivo raw. Opciones:\n"
            "  1) python src/generate_fake_sales.py --days 730 --overwrite-real\n"
            "  2) Coloca supermarket_sales_clean.csv en data/raw/"
        )

    print(f"📂 Usando: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    nulls_before = df.isna().sum().sort_values(ascending=False)
    df = df.drop_duplicates()

    required_cols = ["fecha", "categoria", "cantidad", "precio_venta", "costo"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}\nColumnas actuales: {list(df.columns)}")

    for col in ["promo", "is_holiday", "stock_final", "stock_inicial"]:
        if col not in df.columns:
            df[col] = 0

    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[df_clean["cantidad"] > 0].copy()

    # producto = categoria si no existe columna producto
    if "producto" not in df_clean.columns:
        df_clean["producto"] = df_clean["categoria"]

    df_clean["producto_norm"] = df_clean["producto"].apply(normalize_product_name)
    df_clean = df_clean.dropna(subset=["producto_norm"])
    df_clean = df_clean[df_clean["producto_norm"].str.len() > 0].copy()
    df_clean["product_id"] = df_clean["producto_norm"].apply(make_product_id)

    nulls_after = df_clean.isna().sum().sort_values(ascending=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("REPORTE DE VALORES NULOS - FASE 1\n\n")
        f.write(f"Fuente: {RAW_PATH}\n\n")
        f.write("NULOS ANTES:\n")
        f.write(nulls_before.to_string())
        f.write("\n\nNULOS DESPUÉS:\n")
        f.write(nulls_after.to_string())
        f.write(f"\n\nFilas originales : {len(df)}\n")
        f.write(f"Filas limpias    : {len(df_clean)}\n")
        f.write(f"Filas eliminadas : {len(df) - len(df_clean)}\n")

    df_clean.to_csv(OUT_PATH, index=False)

    print("✅ Fase 1 completada")
    print(f"Archivo limpio : {OUT_PATH}  ({len(df_clean)} filas)")
    print(f"Reporte        : {REPORT_PATH}")


if __name__ == "__main__":
    main()