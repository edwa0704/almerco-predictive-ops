import os
import re
import unicodedata
import hashlib

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/sales_history_raw.csv"
OUT_PATH = "data/processed/clean_data.csv"
REPORT_PATH = "reports/null_report.txt"


def ensure_dirs() -> None:
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
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

    # Unificar sinónimos
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


def main() -> None:
    ensure_dirs()

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo raw: {RAW_PATH}\n"
            "Crea data/raw/sales_history_raw.csv primero."
        )

    df = pd.read_csv(RAW_PATH)

    # Reporte de nulos ANTES
    nulls_before = df.isna().sum().sort_values(ascending=False)

    # Limpieza básica
    df = df.drop_duplicates()

    required_cols = ["fecha", "producto", "cantidad", "precio_venta", "costo", "categoria"]
    optional_cols = ["promo", "stock_final", "is_holiday", "stock_inicial"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    # Si opcionales no existen, crearlas en 0
    for c in optional_cols:
        if c not in df.columns:
            df[c] = 0

    # Mantener opcionales y evitar errores de SettingWithCopy
    df_clean = df.dropna(subset=required_cols).copy()

    # Asegurar opcionales en df_clean (copiando por índice)
    for c in optional_cols:
        df_clean[c] = df.loc[df_clean.index, c]

    # Normalización
    df_clean["producto_norm"] = df_clean["producto"].apply(normalize_product_name)
    df_clean = df_clean.dropna(subset=["producto_norm"])
    df_clean = df_clean[df_clean["producto_norm"].str.len() > 0]

    # product_id
    df_clean["product_id"] = df_clean["producto_norm"].apply(make_product_id)

    # Reporte de nulos DESPUÉS
    nulls_after = df_clean.isna().sum().sort_values(ascending=False)

    # Guardar reporte
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("REPORTE DE VALORES NULOS - FASE 1\n\n")
        f.write("NULOS ANTES:\n")
        f.write(nulls_before.to_string())
        f.write("\n\nNULOS DESPUÉS:\n")
        f.write(nulls_after.to_string())
        f.write("\n\nFilas originales: " + str(len(df)) + "\n")
        f.write("Filas limpias: " + str(len(df_clean)) + "\n")
        f.write("Filas eliminadas: " + str(len(df) - len(df_clean)) + "\n")

    # Guardar clean_data.csv (IMPORTANTE)
    df_clean.to_csv(OUT_PATH, index=False)

    print("✅ Fase 1 completada")
    print(f"Archivo limpio: {OUT_PATH}")
    print(f"Reporte: {REPORT_PATH}")


if __name__ == "__main__":
    main()