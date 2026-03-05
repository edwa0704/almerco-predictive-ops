
import matplotlib
matplotlib.use('Agg')

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ── Rutas ──────────────────────────────────────────────
CLEAN_PATH  = "data/clean_data.csv"
MODEL_PATH  = "models/demand_model.joblib"
REPORT_PATH = "reports/predict_report.txt"
FIG_PATH    = "reports/figures/predict_vs_real.png"

os.makedirs("models",          exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# ── Métricas ────────────────────────────────────────────
def wmape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 1e-9 else np.nan


# ── Features temporales ─────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["fecha"])
    df["month"]               = d.dt.month
    df["week"]                = d.dt.isocalendar().week.astype(int)
    df["day"]                 = d.dt.day
    df["dow"]                 = d.dt.dayofweek
    df["is_weekend"]          = (df["dow"] >= 5).astype(int)
    df["is_christmas_season"] = d.dt.month.isin([11, 12]).astype(int)
    df["is_back_to_school"]   = d.dt.month.isin([3, 4]).astype(int)
    df["is_end_month"]        = (d.dt.day >= 25).astype(int)
    return df


# ── Lags y rolling ──────────────────────────────────────
def add_lags_roll(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.sort_values([id_col, "fecha"]).copy()
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(id_col)["cantidad"].shift(lag)
    df["roll_mean_7"] = (
        df.groupby(id_col)["cantidad"]
        .shift(1).rolling(7).mean()
        .reset_index(level=0, drop=True)
    )
    df["roll_mean_28"] = (
        df.groupby(id_col)["cantidad"]
        .shift(1).rolling(28).mean()
        .reset_index(level=0, drop=True)
    )
    return df


# ════════════════════════════════════════════════════════
def main():
    # 1. Cargar datos
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(
            f"No existe {CLEAN_PATH}\n"
            "Ejecuta primero: python src/fase1_cleaning.py"
        )

    df = pd.read_csv(CLEAN_PATH)
    df["fecha"]    = pd.to_datetime(df["fecha"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df = df.dropna(subset=["fecha", "cantidad"]).copy()

    # 2. Columna ID de producto
    if "product_id" in df.columns:
        id_col = "product_id"
    elif "producto" in df.columns:
        id_col = "producto"
    else:
        df["producto"] = "PRODUCTO_UNICO"
        id_col = "producto"

    df[id_col] = df[id_col].astype(str)

    # 3. Agregar columnas opcionales si no existen
    for col in ["promo", "is_holiday", "stock_inicial"]:
        if col not in df.columns:
            df[col] = 0

    # 4. Agregación diaria por producto
    agg_dict = {"cantidad": "sum", "promo": "max", "is_holiday": "max", "stock_inicial": "max"}
    daily    = df.groupby([id_col, "fecha"], as_index=False).agg(agg_dict)
    daily    = daily.sort_values([id_col, "fecha"]).reset_index(drop=True)

    # 5. Encoding de categoría si existe
    if "categoria" in df.columns:
        cat_map      = df[[id_col, "categoria"]].drop_duplicates(subset=id_col)
        daily        = daily.merge(cat_map, on=id_col, how="left")
        le           = LabelEncoder()
        daily["cat_enc"] = le.fit_transform(daily["categoria"].fillna("unknown"))
    else:
        daily["cat_enc"] = 0
        le = None

    # 6. Features
    daily = add_calendar_features(daily)
    daily = add_lags_roll(daily, id_col=id_col)

    FEATURE_COLS = [
        "cat_enc",
        "month", "week", "day", "dow", "is_weekend",
        "is_christmas_season", "is_back_to_school", "is_end_month",
        "promo", "is_holiday", "stock_inicial",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_28",
    ]

    daily = daily.dropna(subset=FEATURE_COLS + ["cantidad"]).copy()

    if len(daily) < 100:
        print("⚠️  Pocas filas tras lags/rolling.")
        print("   Ejecuta: python src/generate_fake_sales.py --days 730")

    # 7. Split temporal por fecha (correcto — no por índice)
    all_dates  = sorted(daily["fecha"].unique())
    split_date = all_dates[int(len(all_dates) * 0.80)]

    train_df = daily[daily["fecha"] <  split_date].copy()
    test_df  = daily[daily["fecha"] >= split_date].copy()

    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df["cantidad"].astype(float).values
    X_test  = test_df[FEATURE_COLS].astype(float)
    y_test  = test_df["cantidad"].astype(float).values

    print(f"Train: {len(X_train)} filas  |  Test: {len(X_test)} filas")
    print(f"Split date: {pd.Timestamp(split_date).date()}")

    # 8. Modelo
    model = HistGradientBoostingRegressor(
        max_iter          = 800,
        max_depth         = 6,
        learning_rate     = 0.03,
        min_samples_leaf  = 5,
        l2_regularization = 0.1,
        random_state      = 42,
    )
    model.fit(X_train, y_train)

    # 9. Métricas
    y_pred    = np.maximum(model.predict(X_test), 0)
    mask      = y_test > 0
    mape_val  = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    wmape_val = wmape(y_test, y_pred) * 100

    print(f"\nMAPE(%):  {mape_val:.2f}")
    print(f"WMAPE(%): {wmape_val:.2f}")

    # 10. Gráfico por categoría
    test_df       = test_df.copy()
    test_df["pred"] = y_pred
    group_col     = "categoria" if "categoria" in test_df.columns else id_col
    cats          = test_df[group_col].unique()

    fig, axes = plt.subplots(len(cats), 1, figsize=(12, 3 * len(cats)), sharex=False)
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        sub   = test_df[test_df[group_col] == cat].sort_values("fecha")
        w_cat = wmape(sub["cantidad"], sub["pred"]) * 100
        ax.plot(sub["fecha"], sub["cantidad"], label="Real",     linewidth=2)
        ax.plot(sub["fecha"], sub["pred"],     label="Predicho", linewidth=2, linestyle="--")
        ax.set_title(f"{cat}  |  WMAPE: {w_cat:.1f}%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Predicción vs Real  —  WMAPE global: {wmape_val:.2f}%", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_PATH, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\n📊 Gráfico guardado: {FIG_PATH}")

    # 11. Guardar modelo
    joblib.dump({
        "model":           model,
        "feature_columns": FEATURE_COLS,
        "id_col":          id_col,
        "label_encoder":   le,
    }, MODEL_PATH)
    print(f"✅ Modelo guardado: {MODEL_PATH}")

    # 12. Reporte
    with open(REPORT_PATH, "w") as f:
        f.write(f"Split date : {pd.Timestamp(split_date).date()}\n")
        f.write(f"Train rows : {len(X_train)}\n")
        f.write(f"Test rows  : {len(X_test)}\n")
        f.write(f"MAPE (%)   : {mape_val:.2f}\n")
        f.write(f"WMAPE (%)  : {wmape_val:.2f}\n")
        f.write(f"Features   : {FEATURE_COLS}\n")

    print(f"📄 Reporte: {REPORT_PATH}")
    print()

    if wmape_val <= 15:
        print("🏆 ¡WMAPE <= 15%! Objetivo alcanzado.")
    elif wmape_val <= 30:
        print("✅ Buen resultado. Más histórico puede bajar el error.")
    else:
        print("⚠️  WMAPE > 30%. Ejecuta con más días:")
        print("   python src/generate_fake_sales.py --days 730")
        print("   python src/fase1_cleaning.py")
        print("   python src/predict_demand.py")


if __name__ == "__main__":
    main()