import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


CLEAN_PATH = "data/processed/clean_data.csv"
REPORT_PATH = "reports/phase3_metrics.txt"
PRED_PATH = "reports/predictions_sample.csv"


def ensure_dirs():
    os.makedirs("reports", exist_ok=True)


def mape(y_true, y_pred) -> float:
    """MAPE en % (referencia). Castiga fuerte cuando y_true es pequeño."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def wmape(y_true, y_pred) -> float:
    """WMAPE en % (principal). Más estable para demanda."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dow"] = df["fecha"].dt.dayofweek
    df["month"] = df["fecha"].dt.month
    df["day"] = df["fecha"].dt.day
    df["day_of_year"] = df["fecha"].dt.dayofyear
    return df


def add_seasonality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estacionalidad simple (ajustable según Perú/empresa):
    - Navidad: diciembre
    - Inicio de clases: marzo
    """
    df = df.copy()
    df["is_christmas_season"] = (df["month"] == 12).astype(int)
    df["is_back_to_school"] = (df["month"] == 3).astype(int)
    return df


def build_daily_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demanda diaria por producto.
    """
    df_daily = (
        df.groupby(["fecha", "product_id", "categoria"], as_index=False)
          .agg(
              cantidad=("cantidad", "sum"),
              precio_venta=("precio_venta", "mean"),
          )
    )
    df_daily = add_time_features(df_daily)
    df_daily = add_seasonality_flags(df_daily)

    # Orden temporal por producto para features de memoria
    df_daily = df_daily.sort_values(["product_id", "fecha"])

    # Lags por producto
    df_daily["lag_1"] = df_daily.groupby("product_id")["cantidad"].shift(1)
    df_daily["lag_7"] = df_daily.groupby("product_id")["cantidad"].shift(7)

    # Promedio móvil 7 días (sin mirar el día actual -> shift(1))
    df_daily["roll7_mean"] = (
        df_daily.groupby("product_id")["cantidad"]
        .shift(1)
        .rolling(window=7, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Quitamos filas donde aún no hay historia suficiente (lags NaN)
    df_daily = df_daily.dropna(subset=["lag_1", "lag_7", "roll7_mean"]).reset_index(drop=True)
    return df_daily


def temporal_train_test_split(df: pd.DataFrame, test_ratio=0.2):
    """
    Split temporal: train = fechas antiguas, test = fechas recientes.
    """
    df = df.sort_values("fecha").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_ratio))
    cutoff_date = df.loc[cutoff_idx, "fecha"]
    train = df[df["fecha"] < cutoff_date]
    test = df[df["fecha"] >= cutoff_date]
    return train, test, cutoff_date


def main():
    ensure_dirs()

    df = pd.read_csv(CLEAN_PATH)
    df_ml = build_daily_dataset(df)

    train_df, test_df, cutoff_date = temporal_train_test_split(df_ml, test_ratio=0.2)

    # Features
    num_features = [
        "dow", "month", "day", "day_of_year",
        "is_christmas_season", "is_back_to_school",
        "precio_venta",
        "lag_1", "lag_7", "roll7_mean",
    ]
    cat_features = ["categoria"]  # opcional: agregar "product_id" si quieres más precisión

    X_train = train_df[num_features + cat_features]
    y_train = train_df["cantidad"]
    X_test = test_df[num_features + cat_features]
    y_test = test_df["cantidad"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.08,
        max_iter=400,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, preds)
    mape_val = mape(y_test, preds)
    wmape_val = wmape(y_test, preds)

    # Reporte
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("PHASE 3 - DEMAND PREDICTION METRICS\n\n")
        f.write("Modelo: HistGradientBoostingRegressor (sklearn)\n")
        f.write(f"Split temporal (test desde): {cutoff_date.date()}\n\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MAPE(%): {mape_val:.4f}  (referencia)\n")
        f.write(f"WMAPE(%): {wmape_val:.4f} (principal / 'error medio')\n\n")
        f.write("Nota: WMAPE es más estable cuando hay días con demanda pequeña.\n")

    # Muestra de predicciones
    sample = X_test.copy()
    sample["y_true"] = y_test.values
    sample["y_pred"] = preds
    sample.head(80).to_csv(PRED_PATH, index=False)

    print("✅ Fase 3 ejecutada")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE(%): {mape_val:.4f}  (referencia)")
    print(f"WMAPE(%): {wmape_val:.4f} (principal)")
    print(f"Reporte: {REPORT_PATH}")
    print(f"Predicciones (muestra): {PRED_PATH}")

    # Criterio de aceptación (según interpretación)
    if wmape_val < 15.0:
        print("✅ Cumple: WMAPE < 15%")
    else:
        print("❌ Aún no cumple: WMAPE >= 15% (necesita ajuste)")


if __name__ == "__main__":
    main()