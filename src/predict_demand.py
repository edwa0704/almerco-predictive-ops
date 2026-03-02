import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib

DEFAULT_INPUT = "data/clean_data.csv"
MODEL_OUT = "models/demand_model.joblib"


def wmape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom <= 1e-9:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["fecha"], errors="coerce")
    df["month"] = d.dt.month
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Estacionalidad macro (ajustable)
    df["is_christmas_season"] = df["month"].isin([11, 12]).astype(int)
    df["is_back_to_school"] = df["month"].isin([3, 4]).astype(int)
    df["is_end_month"] = (df["day"] >= 25).astype(int)
    return df


def add_lags_roll(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.sort_values([id_col, "fecha"]).copy()

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(id_col)["cantidad"].shift(lag)

    # Rolling means (shift(1) para no usar futuro)
    df["roll_mean_7"] = (
        df.groupby(id_col)["cantidad"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    )
    df["roll_mean_28"] = (
        df.groupby(id_col)["cantidad"].shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    )
    return df


def main():
    input_path = DEFAULT_INPUT
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe {input_path}. Ejecuta Fase 1 primero.")

    df = pd.read_csv(input_path)

    if "fecha" not in df.columns or "cantidad" not in df.columns:
        raise ValueError("clean_data.csv debe tener al menos: fecha, cantidad")

    # ID de producto
    if "product_id" in df.columns:
        id_col = "product_id"
    elif "producto" in df.columns:
        id_col = "producto"
    else:
        df["producto"] = "PRODUCTO_UNICO"
        id_col = "producto"

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df = df.dropna(subset=["fecha", "cantidad"]).copy()

    df[id_col] = df[id_col].astype(str)

    # ✅ Agregación diaria por producto (clave)
    daily = df.groupby([id_col, "fecha"], as_index=False)["cantidad"].sum()
    daily = daily.sort_values([id_col, "fecha"]).reset_index(drop=True)

    # Features
    daily = add_calendar_features(daily)
    daily = add_lags_roll(daily, id_col=id_col)

    feature_cols = [
        "month", "week", "day", "dow", "is_weekend",
        "is_christmas_season", "is_back_to_school", "is_end_month",
        "lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28"
    ]

    daily = daily.dropna(subset=feature_cols + ["cantidad"]).copy()

    if len(daily) < 200:
        print("⚠️ Dataset pequeño luego de lags/rolling. Igual entreno, pero el error puede variar.")

    # One-hot producto (modelo global)
    X_prod = pd.get_dummies(daily[id_col], prefix="prod", drop_first=True)
    X_num = daily[feature_cols].astype(float).reset_index(drop=True)
    X = pd.concat([X_num, X_prod.reset_index(drop=True)], axis=1)
    y = daily["cantidad"].astype(float).values

    # Split temporal: últimas 28 fechas como test (si hay)
    all_dates = np.array(sorted(daily["fecha"].unique()))
    if len(all_dates) > 40:
        split_date = all_dates[-28]
    else:
        split_date = all_dates[int(len(all_dates) * 0.8)]

    train = daily[daily["fecha"] < split_date].copy()
    test = daily[daily["fecha"] >= split_date].copy()

    X_train_prod = pd.get_dummies(train[id_col], prefix="prod", drop_first=True)
    X_test_prod = pd.get_dummies(test[id_col], prefix="prod", drop_first=True)

    X_train_num = train[feature_cols].astype(float).reset_index(drop=True)
    X_test_num = test[feature_cols].astype(float).reset_index(drop=True)

    X_train = pd.concat([X_train_num, X_train_prod.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test_num, X_test_prod.reset_index(drop=True)], axis=1)

    # Alinear columnas
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    y_train = train["cantidad"].astype(float).values
    y_test = test["cantidad"].astype(float).values

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=150
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, np.maximum(pred, 1e-6)) * 100
    w = wmape(y_test, pred) * 100

    print("✅ Fase 3 ejecutada (MODELO GLOBAL + lags/rolling)")
    print(f"Split date: {split_date}")
    print(f"MAPE(%): {mape:.2f}")
    print(f"WMAPE(%): {w:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X_train.columns), "id_col": id_col}, MODEL_OUT)
    print(f"✅ Modelo guardado: {MODEL_OUT}")

    if w < 15:
        print("🎉 Cumple el requisito: WMAPE < 15%")
    else:
        print("⚠️ Aún no baja de 15%. Con más histórico (meses/años) normalmente baja. También puedo añadir lags extra y rolling por 56 días.")


if __name__ == "__main__":
    main()