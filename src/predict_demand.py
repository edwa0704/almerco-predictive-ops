import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

CLEAN_PATH = "data/processed/clean_data.csv"
REPORT_PATH = "reports/phase3_metrics.txt"
PRED_PATH = "reports/predictions_sample.csv"


def ensure_dirs():
    os.makedirs("reports", exist_ok=True)


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (en %)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dow"] = df["fecha"].dt.dayofweek          # 0=Lunes
    df["month"] = df["fecha"].dt.month
    df["day"] = df["fecha"].dt.day
    df["day_of_year"] = df["fecha"].dt.dayofyear
    return df


def add_seasonality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estacionalidad simple:
    - Navidad: Diciembre
    - Inicio de clases: Marzo (ajustable)
    """
    df = df.copy()
    df["is_christmas_season"] = (df["month"] == 12).astype(int)
    df["is_back_to_school"] = (df["month"] == 3).astype(int)
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregamos demanda diaria por producto.
    """
    df_daily = (
        df.groupby(["fecha", "product_id", "categoria"], as_index=False)
          .agg(cantidad=("cantidad", "sum"),
               precio_venta=("precio_venta", "mean"))
    )
    df_daily = add_time_features(df_daily)
    df_daily = add_seasonality_flags(df_daily)
    return df_daily


def main():
    ensure_dirs()

    df = pd.read_csv(CLEAN_PATH)

    # Dataset listo para ML
    df_ml = build_dataset(df)

    # Features (numéricas)
    feature_cols = ["dow", "month", "day", "day_of_year", "is_christmas_season", "is_back_to_school", "precio_venta"]
    X = df_ml[feature_cols]
    y = df_ml["cantidad"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo principal (Gradient Boosting)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mape_val = mape(y_test, preds)

    # Guardar reporte
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("PHASE 3 - DEMAND PREDICTION METRICS\n")
        f.write(f"Model: GradientBoostingRegressor\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MAPE(%): {mape_val:.4f}\n")

    # Guardar muestra de predicciones
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["y_pred"] = preds
    out.head(50).to_csv(PRED_PATH, index=False)

    print("✅ Fase 3 ejecutada")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE(%): {mape_val:.4f}")
    print(f"Reporte: {REPORT_PATH}")
    print(f"Predicciones (muestra): {PRED_PATH}")

    # Si el MAPE no cumple, probamos un RandomForest rápido (fallback)
    if mape_val >= 15.0:
        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        mae_rf = mean_absolute_error(y_test, preds_rf)
        mape_rf = mape(y_test, preds_rf)

        with open(REPORT_PATH, "a", encoding="utf-8") as f:
            f.write("\n--- FALLBACK MODEL ---\n")
            f.write("Model: RandomForestRegressor\n")
            f.write(f"MAE: {mae_rf:.4f}\n")
            f.write(f"MAPE(%): {mape_rf:.4f}\n")

        print("\n⚠️ GradientBoosting no cumplió <15%. Probando RandomForest...")
        print(f"RF MAE: {mae_rf:.4f}")
        print(f"RF MAPE(%): {mape_rf:.4f}")


if __name__ == "__main__":
    main()