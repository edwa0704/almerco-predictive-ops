import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

CLEAN_PATH = "data/clean_data.csv"   # ✅ ruta corregida

def ensure_dirs():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

def main():
    ensure_dirs()

    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(
            f"No existe {CLEAN_PATH}\n"
            "Ejecuta primero: python src/fase1_cleaning.py"
        )

    df = pd.read_csv(CLEAN_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])

    # ==============================
    # 1️⃣ Desviación estándar por categoría
    # ==============================
    std_by_category = df.groupby("categoria")["cantidad"].std().sort_values(ascending=False)
    print("\nDesviación estándar por categoría:")
    print(std_by_category)
    std_by_category.to_csv("reports/std_by_category.csv")

    # ==============================
    # 2️⃣ Heatmap de correlaciones
    # ==============================
    pivot_df = df.pivot_table(
        index="fecha",
        columns="categoria",
        values="cantidad",
        aggfunc="sum"
    )

    correlation_matrix = pivot_df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlación entre categorías")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()
    print("\nHeatmap de correlación guardado: reports/figures/correlation_heatmap.png")

    # ==============================
    # 3️⃣ Test de hipótesis
    # ==============================
    cats = df["categoria"].unique()

    # Detectar categorías AMD automáticamente
    mb_cat  = next((c for c in cats if "MB_AMD"  in str(c).upper() or "MB AMD"  in str(c).upper()), None)
    cpu_cat = next((c for c in cats if "CPU_AMD" in str(c).upper() or "CPU AMD" in str(c).upper()), None)

    if mb_cat is None or cpu_cat is None:
        print(f"\n⚠️  Categorías MB_AMD / CPU_AMD no encontradas en el dataset.")
        print(f"   Categorías disponibles: {list(cats)}")
        print("   Saltando test de hipótesis.")
        return

    mb_prices = df[df["categoria"] == mb_cat].groupby("fecha")["precio_venta"].mean()
    threshold = mb_prices.mean()

    discount_days = mb_prices[mb_prices <  threshold].index
    normal_days   = mb_prices[mb_prices >= threshold].index

    amd_sales     = df[df["categoria"] == cpu_cat].groupby("fecha")["cantidad"].sum()
    sales_discount = amd_sales[amd_sales.index.isin(discount_days)]
    sales_normal   = amd_sales[amd_sales.index.isin(normal_days)]

    if len(sales_discount) < 2 or len(sales_normal) < 2:
        print("\n⚠️  Datos insuficientes para el test de hipótesis.")
        return

    t_stat, p_value = stats.ttest_ind(sales_discount, sales_normal, equal_var=False)

    print(f"\nTest de hipótesis ({cpu_cat} vs precio {mb_cat}):")
    print("t-statistic:", round(t_stat, 4))
    print("p-value:",     round(p_value, 6))

    with open("reports/hypothesis_test.txt", "w") as f:
        f.write(f"Test {cpu_cat} vs precio {mb_cat}\n")
        f.write(f"t-statistic: {t_stat}\n")
        f.write(f"p-value: {p_value}\n")

    if p_value < 0.05:
        print("Resultado: Existe evidencia estadística significativa.")
    else:
        print("Resultado: No hay evidencia suficiente.")

if __name__ == "__main__":
    main()