import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

CLEAN_PATH = "data/processed/clean_data.csv"
REPORT_PATH = "reports/phase3_metrics.txt"
PER_PRODUCT_PATH = "reports/phase3_per_product_metrics.csv"
PRED_SAMPLE_PATH = "reports/phase3_predictions_sample.csv"

GRANULARITY = "weekly"  # weekly recomendado para reducir ruido


def ensure_dirs():
    os.makedirs("reports", exist_ok=True)


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def wmape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["month"] = df["fecha"].dt.month
    df["day_of_year"] = df["fecha"].dt.dayofyear
    return df


def add_seasonality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_christmas_season"] = (df["month"] == 12).astype(int)
    df["is_back_to_school"] = (df["month"] == 3).astype(int)
    return df


def build_weekly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Semana empezando lunes
    df["week_start"] = df["fecha"].dt.to_period("W-MON").dt.start_time

    df_w = (
        df.groupby(["week_start", "product_id", "categoria"], as_index=False)
          .agg(
              cantidad=("cantidad", "sum"),
              precio_venta=("precio_venta", "mean"),
              promo=("promo", "max"),
              is_holiday=("is_holiday", "max"),
              stock_final=("stock_final", "min"),  # si alguna vez llegó a 0, es alerta
          )
          .rename(columns={"week_start": "fecha"})
    )

    # Flag de stockout semanal
    df_w["is_stockout"] = (df_w["stock_final"] <= 0).astype(int)

    df_w = add_time_features(df_w)
    df_w = add_seasonality_flags(df_w)

    # Orden por producto y fecha
    df_w = df_w.sort_values(["product_id", "fecha"])

    # Features de memoria por producto (semanales)
    df_w["lag_1w"] = df_w.groupby("product_id")["cantidad"].shift(1)
    df_w["lag_4w"] = df_w.groupby("product_id")["cantidad"].shift(4)

    df_w["roll4w_mean"] = (
        df_w.groupby("product_id")["cantidad"]
            .shift(1)
            .rolling(window=4, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
    )

    # Quitamos filas sin historia suficiente
    df_w = df_w.dropna(subset=["lag_1w", "lag_4w", "roll4w_mean"]).reset_index(drop=True)
    return df_w


def temporal_split_single_series(df_prod: pd.DataFrame, test_ratio=0.2):
    df_prod = df_prod.sort_values("fecha").reset_index(drop=True)
    if len(df_prod) < 15:
        return None, None, None

    cutoff_idx = int(len(df_prod) * (1 - test_ratio))
    cutoff_date = df_prod.loc[cutoff_idx, "fecha"]

    train = df_prod[df_prod["fecha"] < cutoff_date]
    test = df_prod[df_prod["fecha"] >= cutoff_date]

    if len(train) < 10 or len(test) < 3:
        return None, None, None

    return train, test, cutoff_date


def main():
    ensure_dirs()

    df = pd.read_csv(CLEAN_PATH)

    if GRANULARITY != "weekly":
        raise ValueError("Este script está configurado para weekly.")

    df_ml = build_weekly_dataset(df)

    # Features numéricas
    feature_cols = [
        "month", "day_of_year",
        "is_christmas_season", "is_back_to_school",
        "precio_venta",
        "promo", "is_holiday", "stock_final", "is_stockout",
        "lag_1w", "lag_4w", "roll4w_mean",
    ]

    per_product_rows = []
    sample_preds_rows = []

    # WMAPE global (todo)
    global_abs_err_sum = 0.0
    global_actual_sum = 0.0

    # WMAPE global in-stock (solo semanas sin stockout)
    global_abs_err_sum_in = 0.0
    global_actual_sum_in = 0.0

    for pid, g in df_ml.groupby("product_id"):
        train, test, cutoff_date = temporal_split_single_series(g, test_ratio=0.2)
        if train is None:
            continue

        # Entrenamiento SOLO con semanas con stock
        train_in = train[train["is_stockout"] == 0].copy()
        if len(train_in) < 10:
            continue

        X_train = train_in[feature_cols]
        y_train = train_in["cantidad"]

        X_test = test[feature_cols]
        y_test = test["cantidad"]

        test_in = test[test["is_stockout"] == 0].copy()

        model = HistGradientBoostingRegressor(
            random_state=42,
            max_depth=8,
            learning_rate=0.06,
            max_iter=600,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Métricas generales
        mae_val = mean_absolute_error(y_test, preds)
        mape_val = mape(y_test, preds)
        wmape_val = wmape(y_test, preds)

        global_abs_err_sum += float(np.sum(np.abs(y_test.values - preds)))
        global_actual_sum += float(np.sum(np.abs(y_test.values)))

        # Métricas in-stock
        if len(test_in) > 0:
            preds_in = model.predict(test_in[feature_cols])
            wmape_in = wmape(test_in["cantidad"].values, preds_in)
            global_abs_err_sum_in += float(np.sum(np.abs(test_in["cantidad"].values - preds_in)))
            global_actual_sum_in += float(np.sum(np.abs(test_in["cantidad"].values)))
        else:
            wmape_in = np.nan

        categoria = g["categoria"].iloc[0] if "categoria" in g.columns else "NA"

        per_product_rows.append({
            "product_id": pid,
            "categoria": categoria,
            "cutoff_date_test_from": pd.to_datetime(cutoff_date).date(),
            "n_train_instock": len(train_in),
            "n_test_total": len(test),
            "MAE": round(mae_val, 4),
            "MAPE_%": round(mape_val, 4),
            "WMAPE_%": round(wmape_val, 4),
            "WMAPE_instock_%": (round(wmape_in, 4) if not np.isnan(wmape_in) else ""),
        })

        # Muestra de predicciones
        tmp = test[["fecha"]].copy()
        tmp["product_id"] = pid
        tmp["categoria"] = categoria
        tmp["y_true"] = y_test.values
        tmp["y_pred"] = preds
        sample_preds_rows.append(tmp.head(10))

    if not per_product_rows:
        raise RuntimeError("No se pudo entrenar ningún producto (muy pocos datos por serie).")

    per_prod_df = pd.DataFrame(per_product_rows).sort_values("WMAPE_%")
    per_prod_df.to_csv(PER_PRODUCT_PATH, index=False)

    global_wmape = (global_abs_err_sum / global_actual_sum * 100.0) if global_actual_sum > 0 else 0.0
    global_wmape_in = (global_abs_err_sum_in / global_actual_sum_in * 100.0) if global_actual_sum_in > 0 else 0.0

    sample_df = pd.concat(sample_preds_rows, ignore_index=True)
    sample_df.head(200).to_csv(PRED_SAMPLE_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("PHASE 3 - DEMAND PREDICTION METRICS\n\n")
        f.write(f"Granularidad: {GRANULARITY}\n")
        f.write("Estrategia: 1 modelo por product_id (HistGradientBoostingRegressor)\n")
        f.write("Métrica principal (error medio): WMAPE global ponderado\n")
        f.write("WMAPE_global_instock excluye semanas con stockout.\n")
        f.write("MAPE se reporta como referencia.\n\n")
        f.write(f"WMAPE_global(%): {global_wmape:.4f}\n")
        f.write(f"WMAPE_global_instock(%): {global_wmape_in:.4f}\n")
        f.write(f"Productos evaluados: {len(per_prod_df)}\n\n")
        f.write("Archivos:\n")
        f.write(f"- {PER_PRODUCT_PATH}\n")
        f.write(f"- {PRED_SAMPLE_PATH}\n")

    print("✅ Fase 3 ejecutada (modelo por producto)")
    print(f"Granularidad: {GRANULARITY}")
    print(f"WMAPE_global(%): {global_wmape:.4f}")
    print(f"WMAPE_global_instock(%): {global_wmape_in:.4f}")
    print(f"Detalle por producto: {PER_PRODUCT_PATH}")
    print(f"Muestra de predicciones: {PRED_SAMPLE_PATH}")

    if global_wmape_in < 15.0:
        print("✅ Cumple: WMAPE_global_instock < 15%")
    else:
        print("❌ Aún no cumple: WMAPE_global_instock >= 15% (necesita ajuste)")


if __name__ == "__main__":
    main()