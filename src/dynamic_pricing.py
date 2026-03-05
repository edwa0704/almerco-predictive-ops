"""
dynamic_pricing.py  —  Fase 4: Red Neuronal de Pricing Dinámico
================================================================
Objetivo : Recomendar el precio óptimo que MAXIMIZA el revenue
           (precio × demanda_esperada) respetando margen mínimo.

Arquitectura:
  - Red 1 (DemandNet)  : predice demanda dado un precio
  - Red 2 (PricingNet) : recomienda precio óptimo por contexto

Herramientas: TensorFlow / Keras
Dataset     : data/clean_data.csv  (generado por fase1_cleaning.py)
"""

import matplotlib
matplotlib.use('Agg')

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

tf.get_logger().setLevel('ERROR')

# ── Rutas ───────────────────────────────────────────────────────────────
CLEAN_PATH    = "data/clean_data.csv"
MODEL_DIR     = "models"
DEMAND_MODEL  = f"{MODEL_DIR}/demand_nn.keras"
PRICING_MODEL = f"{MODEL_DIR}/pricing_nn.keras"
SCALER_PATH   = f"{MODEL_DIR}/pricing_scalers.joblib"
REPORT_PATH   = "reports/pricing_report.txt"
FIG_PATH      = "reports/figures/pricing_elasticity.png"

os.makedirs(MODEL_DIR,         exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN
# ════════════════════════════════════════════════════════════════════════
def load_and_prepare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe {path}\n"
            "Ejecuta primero: python src/fase1_cleaning.py"
        )

    df = pd.read_csv(path)
    df["fecha"]    = pd.to_datetime(df["fecha"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df["precio_venta"] = pd.to_numeric(df["precio_venta"], errors="coerce")
    df["costo"]    = pd.to_numeric(df["costo"], errors="coerce")
    df = df.dropna(subset=["fecha", "cantidad", "precio_venta", "costo"])
    df = df[df["cantidad"] > 0].copy()

    # Features temporales
    df["mes"]        = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["dia_mes"]    = df["fecha"].dt.day
    df["is_weekend"] = (df["dia_semana"] >= 5).astype(int)
    df["is_xmas"]    = df["mes"].isin([11, 12]).astype(int)
    df["is_school"]  = df["mes"].isin([3, 4]).astype(int)
    df["is_end_month"] = (df["dia_mes"] >= 25).astype(int)

    # Margen
    df["margen_ratio"] = (df["precio_venta"] - df["costo"]) / df["precio_venta"].clip(lower=1)
    df["revenue"]      = df["precio_venta"] * df["cantidad"]

    # Columnas opcionales
    for col in ["promo", "is_holiday", "stock_inicial"]:
        if col not in df.columns:
            df[col] = 0

    return df


# ════════════════════════════════════════════════════════════════════════
# 2. RED 1 — DemandNet: predice cantidad dado precio y contexto
# ════════════════════════════════════════════════════════════════════════
def build_demand_net(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="context")
    x   = layers.Dense(128, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear", name="demand")(x)
    model = keras.Model(inp, out, name="DemandNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"]
    )
    return model


def train_demand_net(df: pd.DataFrame):
    print("\n── Entrenando DemandNet (elasticidad precio-demanda) ──")

    le = LabelEncoder()
    df = df.copy()
    df["cat_enc"] = le.fit_transform(df["categoria"].astype(str))

    FEATURES = [
        "precio_venta", "costo", "margen_ratio",
        "cat_enc", "mes", "dia_semana", "dia_mes",
        "is_weekend", "is_xmas", "is_school", "is_end_month",
        "promo", "is_holiday", "stock_inicial",
    ]

    X = df[FEATURES].astype(float).values
    y = df["cantidad"].astype(float).values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X_sc, y_sc, test_size=0.2, random_state=42
    )

    model = build_demand_net(len(FEATURES))

    cb = [
        callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=cb,
        verbose=0,
    )

    # Métricas
    y_pred_sc = model.predict(X_val, verbose=0).ravel()
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    y_real    = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_real    = np.maximum(y_real, 1e-6)

    mape  = mean_absolute_percentage_error(y_real, np.maximum(y_pred, 1e-6)) * 100
    wmape = np.sum(np.abs(y_real - y_pred)) / np.sum(y_real) * 100

    print(f"  DemandNet — MAPE: {mape:.2f}%  |  WMAPE: {wmape:.2f}%")
    print(f"  Epochs entrenados: {len(history.history['loss'])}")

    return model, scaler_X, scaler_y, le, FEATURES, mape, wmape


# ════════════════════════════════════════════════════════════════════════
# 3. RED 2 — PricingNet: recomienda precio óptimo
#    Input : contexto sin precio
#    Output: precio_optimo normalizado
#    Target: precio que históricamente generó mayor revenue en contexto similar
# ════════════════════════════════════════════════════════════════════════
def build_pricing_net(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="context")
    x   = layers.Dense(128, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.15)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="precio_opt")(x)
    model = keras.Model(inp, out, name="PricingNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_pricing_net(df: pd.DataFrame, le: LabelEncoder):
    print("\n── Entrenando PricingNet (precio óptimo por contexto) ──")

    df = df.copy()
    df["cat_enc"] = le.transform(df["categoria"].astype(str))

    # Target: precio del registro con mayor revenue en el mismo
    #         (categoria, mes, dia_semana) — precio de mejor desempeño histórico
    df["ctx_key"] = (
        df["cat_enc"].astype(str) + "_" +
        df["mes"].astype(str)     + "_" +
        df["dia_semana"].astype(str)
    )
    best_price = (
        df.groupby("ctx_key")
        .apply(lambda g: g.loc[g["revenue"].idxmax(), "precio_venta"])
        .rename("precio_optimo")
    )
    df = df.join(best_price, on="ctx_key")

    FEATURES_P = [
        "costo", "cat_enc", "mes", "dia_semana", "dia_mes",
        "is_weekend", "is_xmas", "is_school", "is_end_month",
        "promo", "is_holiday", "stock_inicial",
    ]

    X = df[FEATURES_P].astype(float).values
    y = df["precio_optimo"].astype(float).values

    scaler_Xp = MinMaxScaler()
    scaler_yp = MinMaxScaler()
    X_sc = scaler_Xp.fit_transform(X)
    y_sc = scaler_yp.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X_sc, y_sc, test_size=0.2, random_state=42
    )

    model = build_pricing_net(len(FEATURES_P))

    cb = [
        callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=cb,
        verbose=0,
    )

    y_pred_sc = model.predict(X_val, verbose=0).ravel()
    y_pred    = scaler_yp.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    y_real    = scaler_yp.inverse_transform(y_val.reshape(-1, 1)).ravel()

    mae_price = np.mean(np.abs(y_real - y_pred))
    print(f"  PricingNet — MAE precio: {mae_price:.2f}  |  Epochs: {len(history.history['loss'])}")

    return model, scaler_Xp, scaler_yp, FEATURES_P, mae_price


# ════════════════════════════════════════════════════════════════════════
# 4. GRÁFICO: curva elasticidad precio → demanda por categoría
# ════════════════════════════════════════════════════════════════════════
def plot_elasticity(df, demand_model, scaler_X, scaler_y, le, features):
    print("\n── Generando gráfico de elasticidad ──")

    cats  = df["categoria"].unique()
    ncols = 2
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.ravel()

    for i, cat in enumerate(cats):
        sub  = df[df["categoria"] == cat]
        row  = sub.iloc[0].copy()

        p_min = sub["costo"].mean() * 0.75
        p_max = sub["costo"].mean() * 1.50
        prices = np.linspace(p_min, p_max, 60)

        demands = []
        for p in prices:
            row["precio_venta"] = p
            row["margen_ratio"] = (p - row["costo"]) / max(p, 1)
            row["cat_enc"]      = le.transform([cat])[0]
            x_row = np.array([[row[f] for f in features]], dtype=float)
            x_sc  = scaler_X.transform(x_row)
            d_sc  = demand_model.predict(x_sc, verbose=0)[0, 0]
            d     = scaler_y.inverse_transform([[d_sc]])[0, 0]
            demands.append(max(d, 0))

        revenues = np.array(prices) * np.array(demands)
        opt_idx  = np.argmax(revenues)

        ax = axes[i]
        ax2 = ax.twinx()
        ax.plot(prices, demands,  color="#2196F3", linewidth=2, label="Demanda")
        ax2.plot(prices, revenues, color="#4CAF50", linewidth=2, linestyle="--", label="Revenue")
        ax.axvline(prices[opt_idx], color="#F44336", linestyle=":", linewidth=1.5,
                   label=f"Precio óptimo: {prices[opt_idx]:.1f}")
        ax.set_title(cat, fontsize=11)
        ax.set_xlabel("Precio")
        ax.set_ylabel("Demanda", color="#2196F3")
        ax2.set_ylabel("Revenue", color="#4CAF50")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    # Ocultar subplots vacíos
    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Curva Elasticidad Precio → Demanda / Revenue por Categoría",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_PATH, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Gráfico guardado: {FIG_PATH}")


# ════════════════════════════════════════════════════════════════════════
# 5. DEMO: precio recomendado para un contexto dado
# ════════════════════════════════════════════════════════════════════════
def demo_pricing(df, pricing_model, scaler_Xp, scaler_yp, le, features_p):
    print("\n── Demo: Precios recomendados por categoría ──")
    print(f"{'Categoría':<25} {'Costo':>8} {'P. Actual':>10} {'P. Óptimo':>10} {'Δ%':>7}")
    print("-" * 65)

    cats = df["categoria"].unique()
    results = []

    for cat in cats:
        sub  = df[df["categoria"] == cat]
        row  = sub.iloc[len(sub)//2].copy()   # registro representativo
        costo = row["costo"]
        p_actual = row["precio_venta"]

        row["cat_enc"] = le.transform([cat])[0]
        x_row = np.array([[row[f] for f in features_p]], dtype=float)
        x_sc  = scaler_Xp.transform(x_row)
        p_sc  = pricing_model.predict(x_sc, verbose=0)[0, 0]
        p_opt = scaler_yp.inverse_transform([[p_sc]])[0, 0]
        p_opt = max(p_opt, costo * 1.05)  # mínimo 5% de margen

        delta = ((p_opt - p_actual) / p_actual) * 100
        print(f"{cat:<25} {costo:>8.2f} {p_actual:>10.2f} {p_opt:>10.2f} {delta:>+7.1f}%")
        results.append({"categoria": cat, "costo": costo,
                        "precio_actual": p_actual, "precio_optimo": p_opt})

    return results


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  FASE 4 — Red Neuronal de Pricing Dinámico")
    print("=" * 60)

    # 1. Datos
    df = load_and_prepare(CLEAN_PATH)
    print(f"\n✅ Dataset cargado: {len(df)} filas | "
          f"{df['categoria'].nunique()} categorías")
    print(f"   Rango fechas: {df['fecha'].min().date()} → {df['fecha'].max().date()}")

    # 2. DemandNet
    demand_model, scaler_X, scaler_y, le, features, mape_d, wmape_d = \
        train_demand_net(df)

    # 3. PricingNet
    pricing_model, scaler_Xp, scaler_yp, features_p, mae_p = \
        train_pricing_net(df, le)

    # 4. Curva de elasticidad
    plot_elasticity(df, demand_model, scaler_X, scaler_y, le, features)

    # 5. Demo recomendaciones
    results = demo_pricing(df, pricing_model, scaler_Xp, scaler_yp, le, features_p)

    # 6. Guardar modelos y scalers
    demand_model.save(DEMAND_MODEL)
    pricing_model.save(PRICING_MODEL)
    joblib.dump({
        "scaler_X_demand":  scaler_X,
        "scaler_y_demand":  scaler_y,
        "scaler_X_pricing": scaler_Xp,
        "scaler_y_pricing": scaler_yp,
        "label_encoder":    le,
        "features_demand":  features,
        "features_pricing": features_p,
    }, SCALER_PATH)

    print(f"\n✅ Modelos guardados:")
    print(f"   {DEMAND_MODEL}")
    print(f"   {PRICING_MODEL}")
    print(f"   {SCALER_PATH}")

    # 7. Reporte
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("FASE 4 - RED NEURONAL DE PRICING DINAMICO\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset      : {CLEAN_PATH}\n")
        f.write(f"Filas        : {len(df)}\n")
        f.write(f"Categorias   : {df['categoria'].nunique()}\n\n")
        f.write("-- DemandNet --\n")
        f.write(f"  MAPE  : {mape_d:.2f}%\n")
        f.write(f"  WMAPE : {wmape_d:.2f}%\n\n")
        f.write("-- PricingNet --\n")
        f.write(f"  MAE precio : {mae_p:.2f}\n\n")
        f.write("-- Precios recomendados --\n")
        for r in results:
            f.write(f"  {r['categoria']}: {r['precio_actual']:.2f} -> {r['precio_optimo']:.2f}\n")

    print(f"\n📄 Reporte: {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("  RESUMEN FASE 4")
    print("=" * 60)
    print(f"  DemandNet  WMAPE : {wmape_d:.2f}%")
    print(f"  PricingNet MAE   : {mae_p:.2f}")
    if wmape_d <= 15:
        print("  🏆 DemandNet cumple requisito WMAPE < 15%")
    print("  📊 Curva elasticidad: reports/figures/pricing_elasticity.png")
    print("=" * 60)


if __name__ == "__main__":
    main()