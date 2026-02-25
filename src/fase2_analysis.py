import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

CLEAN_PATH = "data/processed/clean_data.csv"

def ensure_dirs():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

def main():
    ensure_dirs()

    df = pd.read_csv(CLEAN_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])

    # ==============================
    # 1️⃣ Desviación estándar por categoría
    # ==============================

    std_by_category = df.groupby("categoria")["cantidad"].std().sort_values(ascending=False)

    print("\nDesviación estándar por categoría:")
    print(std_by_category)

    # Guardar reporte
    std_by_category.to_csv("reports/std_by_category.csv")

    # ==============================
    # 2️⃣ Dashboard de correlaciones
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

    print("\nHeatmap de correlación guardado.")

    # ==============================
    # 3️⃣ Test de hipótesis
    # ==============================

    # Identificar días con descuento en placas AMD
    mb_prices = df[df["categoria"] == "MB_AMD"].groupby("fecha")["precio_venta"].mean()

    threshold = mb_prices.mean()

    discount_days = mb_prices[mb_prices < threshold].index
    normal_days = mb_prices[mb_prices >= threshold].index

    amd_sales = df[df["categoria"] == "CPU_AMD"].groupby("fecha")["cantidad"].sum()

    sales_discount = amd_sales[amd_sales.index.isin(discount_days)]
    sales_normal = amd_sales[amd_sales.index.isin(normal_days)]

    t_stat, p_value = stats.ttest_ind(sales_discount, sales_normal, equal_var=False)

    print("\nTest de hipótesis (CPU AMD vs precio placas):")
    print("t-statistic:", round(t_stat, 4))
    print("p-value:", round(p_value, 6))

    with open("reports/hypothesis_test.txt", "w") as f:
        f.write("Test CPU AMD vs precio placas AMD\n")
        f.write(f"t-statistic: {t_stat}\n")
        f.write(f"p-value: {p_value}\n")

    if p_value < 0.05:
        print("Resultado: Existe evidencia estadística significativa.")
    else:
        print("Resultado: No hay evidencia suficiente.")

if __name__ == "__main__":
    main()