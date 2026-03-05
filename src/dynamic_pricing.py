"""
dynamic_pricing.py  -  Fase 4: Red Neuronal de Pricing Dinamico
================================================================
Objetivo : Sugerir el precio de venta final que MAXIMIZA el revenue
           recibiendo: stock_actual, costo y demanda_proyectada.

Arquitectura:
  - DemandNet  : predice demanda dado precio y contexto
  - PricingNet : recibe (stock_actual, costo, demanda_proyectada)
                 y sugiere el precio de venta final

Funcion de perdida personalizada (StockoutAwareLoss):
  - Penaliza MAS quedarse sin stock (x3) que vender barato (x2)
  - Stockout = precio muy alto -> cliente no compra -> stock sin rotar
  - Cheap    = precio muy bajo -> perdida de margen

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


# ════════════════════════════════════════════════════════════════════════
# FUNCION DE PERDIDA PERSONALIZADA — StockoutAwareLoss
# ════════════════════════════════════════════════════════════════════════
#
#  Logica de negocio:
#  - "Vender barato"  -> perdemos margen (malo, penaliza x2)
#  - "Quedarse sin stock" -> perdemos ventas futuras + dano reputacion
#                           (peor, penaliza x3)
#
#  Como funciona:
#  - Error base = MSE entre precio_predicho y precio_optimo
#  - Si pred < optimo * 0.85  => modelo sugiere precio muy bajo  -> x2
#  - Si pred > optimo * 1.10  => modelo sugiere precio muy alto,
#    nadie compra, stock no rota -> riesgo stockout -> x3
#
#  STOCKOUT_PENALTY (3) > CHEAP_PENALTY (2) porque el stockout
#  tiene consecuencias en cascada que van mas alla de una sola venta.
#
# ════════════════════════════════════════════════════════════════════════

STOCKOUT_PENALTY = 3.0   # penalizacion stockout (precio muy alto)
CHEAP_PENALTY    = 2.0   # penalizacion vender barato (precio muy bajo)
CHEAP_THRESHOLD  = 0.85  # pred < 85% del optimo => barato
STOCKOUT_THRESHOLD = 1.10  # pred > 110% del optimo => riesgo stockout


def stockout_aware_loss(y_true, y_pred):
    """
    Funcion de perdida personalizada.
    Penaliza mas quedarse sin stock que vender barato.

    y_true : precio optimo normalizado [0, 1]
    y_pred : precio sugerido por la red [0, 1]
    """
    # Error base MSE
    base_loss = tf.square(y_true - y_pred)

    # Penalizacion 1: precio demasiado bajo (vender barato)
    is_cheap   = tf.cast(y_pred < (y_true * CHEAP_THRESHOLD), tf.float32)
    cheap_loss = base_loss * CHEAP_PENALTY * is_cheap

    # Penalizacion 2: precio demasiado alto (stock sin rotar = stockout)
    is_stockout   = tf.cast(y_pred > (y_true * STOCKOUT_THRESHOLD), tf.float32)
    stockout_loss = base_loss * STOCKOUT_PENALTY * is_stockout

    total = base_loss + cheap_loss + stockout_loss
    return tf.reduce_mean(total)


# ── Rutas ────────────────────────────────────────────────────────────────
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
# 1. CARGA Y PREPARACION
# ════════════════════════════════════════════════════════════════════════
def load_and_prepare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe {path}\n"
            "Ejecuta primero: python src/fase1_cleaning.py"
        )

    df = pd.read_csv(path)
    df["fecha"]       = pd.to_datetime(df["fecha"], errors="coerce")
    df["cantidad"]    = pd.to_numeric(df["cantidad"],    errors="coerce")
    df["precio_venta"]= pd.to_numeric(df["precio_venta"],errors="coerce")
    df["costo"]       = pd.to_numeric(df["costo"],       errors="coerce")
    df = df.dropna(subset=["fecha","cantidad","precio_venta","costo"])
    df = df[df["cantidad"] > 0].copy()

    df["mes"]         = df["fecha"].dt.month
    df["dia_semana"]  = df["fecha"].dt.dayofweek
    df["dia_mes"]     = df["fecha"].dt.day
    df["is_weekend"]  = (df["dia_semana"] >= 5).astype(int)
    df["is_xmas"]     = df["mes"].isin([11,12]).astype(int)
    df["is_school"]   = df["mes"].isin([3,4]).astype(int)
    df["is_end_month"]= (df["dia_mes"] >= 25).astype(int)
    df["margen_ratio"]= (df["precio_venta"] - df["costo"]) / df["precio_venta"].clip(lower=1)
    df["revenue"]     = df["precio_venta"] * df["cantidad"]

    for col in ["promo","is_holiday","stock_inicial"]:
        if col not in df.columns:
            df[col] = 0

    return df


# ════════════════════════════════════════════════════════════════════════
# 2. DEMANDNET — predice cantidad dado precio y contexto
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
    print("\n-- Entrenando DemandNet (elasticidad precio-demanda) --")

    le = LabelEncoder()
    df = df.copy()
    df["cat_enc"] = le.fit_transform(df["categoria"].astype(str))

    FEATURES = [
        "precio_venta","costo","margen_ratio",
        "cat_enc","mes","dia_semana","dia_mes",
        "is_weekend","is_xmas","is_school","is_end_month",
        "promo","is_holiday","stock_inicial",
    ]

    X = df[FEATURES].astype(float).values
    y = df["cantidad"].astype(float).values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

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
        epochs=200, batch_size=32,
        callbacks=cb, verbose=0,
    )

    y_pred_sc = model.predict(X_val, verbose=0).ravel()
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()
    y_real    = scaler_y.inverse_transform(y_val.reshape(-1,1)).ravel()
    y_real    = np.maximum(y_real, 1e-6)

    mape  = mean_absolute_percentage_error(y_real, np.maximum(y_pred,1e-6)) * 100
    wmape = np.sum(np.abs(y_real - y_pred)) / np.sum(y_real) * 100

    print(f"  DemandNet  MAPE: {mape:.2f}%  |  WMAPE: {wmape:.2f}%")
    print(f"  Epochs: {len(history.history['loss'])}")

    return model, scaler_X, scaler_y, le, FEATURES, mape, wmape


# ════════════════════════════════════════════════════════════════════════
# 3. PRICINGNET — sugiere precio de venta dado stock + costo + demanda
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
        loss=stockout_aware_loss,    # <- perdida personalizada
        metrics=["mae"]
    )
    return model


def train_pricing_net(df: pd.DataFrame, le: LabelEncoder,
                      demand_model, scaler_X_d, scaler_y_d, features_d):
    print("\n-- Entrenando PricingNet (stock + costo + demanda proyectada) --")
    print(f"   Loss personalizada: stockout x{STOCKOUT_PENALTY} | barato x{CHEAP_PENALTY}")

    df = df.copy()
    df["cat_enc"] = le.transform(df["categoria"].astype(str))

    # Demanda proyectada usando DemandNet ya entrenado
    X_d    = scaler_X_d.transform(df[features_d].astype(float).values)
    d_sc   = demand_model.predict(X_d, verbose=0).ravel()
    df["demanda_proyectada"] = scaler_y_d.inverse_transform(
        d_sc.reshape(-1,1)
    ).ravel().clip(min=0)

    # Target: precio que historicamente genero mayor revenue por contexto
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

    # Features clave segun especificacion:
    # stock_actual + costo + demanda_proyectada + contexto
    FEATURES_P = [
        "stock_inicial",         # stock actual disponible
        "costo",                 # costo del producto
        "demanda_proyectada",    # salida de DemandNet
        "cat_enc","mes","dia_semana","dia_mes",
        "is_weekend","is_xmas","is_school","is_end_month",
        "promo","is_holiday",
    ]

    X = df[FEATURES_P].astype(float).values
    y = df["precio_optimo"].astype(float).values

    scaler_Xp = MinMaxScaler()
    scaler_yp = MinMaxScaler()
    X_sc = scaler_Xp.fit_transform(X)
    y_sc = scaler_yp.fit_transform(y.reshape(-1,1)).ravel()

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
        epochs=200, batch_size=32,
        callbacks=cb, verbose=0,
    )

    y_pred_sc = model.predict(X_val, verbose=0).ravel()
    y_pred    = scaler_yp.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()
    y_real    = scaler_yp.inverse_transform(y_val.reshape(-1,1)).ravel()
    mae_p     = np.mean(np.abs(y_real - y_pred))

    print(f"  PricingNet  MAE precio: {mae_p:.2f}  |  Epochs: {len(history.history['loss'])}")

    return model, scaler_Xp, scaler_yp, FEATURES_P, mae_p


# ════════════════════════════════════════════════════════════════════════
# 4. GRAFICO: curva elasticidad precio -> demanda por categoria
# ════════════════════════════════════════════════════════════════════════
def plot_elasticity(df, demand_model, scaler_X, scaler_y, le, features):
    print("\n-- Generando grafico de elasticidad --")

    cats  = df["categoria"].unique()
    ncols = 2
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
    axes = axes.ravel()

    for i, cat in enumerate(cats):
        sub = df[df["categoria"] == cat]
        row = sub.iloc[0].copy()

        p_min  = sub["costo"].mean() * 0.75
        p_max  = sub["costo"].mean() * 1.50
        prices = np.linspace(p_min, p_max, 60)

        demands = []
        for p in prices:
            row["precio_venta"] = p
            row["margen_ratio"] = (p - row["costo"]) / max(p, 1)
            row["cat_enc"]      = le.transform([cat])[0]
            x_row = np.array([[row[f] for f in features]], dtype=float)
            x_sc  = scaler_X.transform(x_row)
            d_sc  = demand_model.predict(x_sc, verbose=0)[0,0]
            d     = scaler_y.inverse_transform([[d_sc]])[0,0]
            demands.append(max(d, 0))

        revenues = np.array(prices) * np.array(demands)
        opt_idx  = np.argmax(revenues)

        ax  = axes[i]
        ax2 = ax.twinx()
        ax.plot(prices, demands,  color="#2196F3", linewidth=2, label="Demanda")
        ax2.plot(prices, revenues, color="#4CAF50", linewidth=2,
                 linestyle="--", label="Revenue")
        ax.axvline(prices[opt_idx], color="#F44336", linestyle=":",
                   linewidth=1.5, label=f"Optimo: {prices[opt_idx]:.1f}")
        ax.set_title(cat, fontsize=11)
        ax.set_xlabel("Precio")
        ax.set_ylabel("Demanda",color="#2196F3")
        ax2.set_ylabel("Revenue",color="#4CAF50")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Elasticidad Precio -> Demanda / Revenue por Categoria",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_PATH, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Grafico guardado: {FIG_PATH}")


# ════════════════════════════════════════════════════════════════════════
# 5. DEMO: precio sugerido para cada categoria
# ════════════════════════════════════════════════════════════════════════
def demo_pricing(df, pricing_model, scaler_Xp, scaler_yp,
                 le, features_p, demand_model, scaler_X_d, scaler_y_d, features_d):
    print("\n-- Demo: Precios sugeridos por categoria --")
    print(f"{'Categoria':<25} {'Stock':>6} {'Costo':>8} {'Dem.Proy':>9} "
          f"{'P.Actual':>9} {'P.Optimo':>9} {'Delta%':>7}")
    print("-" * 78)

    results = []
    for cat in df["categoria"].unique():
        sub  = df[df["categoria"] == cat]
        row  = sub.iloc[len(sub)//2].copy()
        costo    = row["costo"]
        p_actual = row["precio_venta"]
        stock    = row["stock_inicial"]

        # Demanda proyectada
        row["cat_enc"] = le.transform([cat])[0]
        x_d  = np.array([[row[f] for f in features_d]], dtype=float)
        x_dsc= scaler_X_d.transform(x_d)
        d_sc = demand_model.predict(x_dsc, verbose=0)[0,0]
        dem_proy = max(scaler_y_d.inverse_transform([[d_sc]])[0,0], 0)

        # Precio sugerido
        row["demanda_proyectada"] = dem_proy
        x_p  = np.array([[row[f] for f in features_p]], dtype=float)
        x_psc= scaler_Xp.transform(x_p)
        p_sc = pricing_model.predict(x_psc, verbose=0)[0,0]
        p_opt= scaler_yp.inverse_transform([[p_sc]])[0,0]
        p_opt= max(p_opt, costo * 1.05)  # minimo 5% margen

        delta = ((p_opt - p_actual) / p_actual) * 100
        print(f"{cat:<25} {stock:>6.0f} {costo:>8.2f} {dem_proy:>9.1f} "
              f"{p_actual:>9.2f} {p_opt:>9.2f} {delta:>+7.1f}%")
        results.append({
            "categoria": cat, "stock": stock, "costo": costo,
            "demanda_proyectada": dem_proy,
            "precio_actual": p_actual, "precio_optimo": p_opt
        })

    return results


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  FASE 4 - Red Neuronal de Pricing Dinamico")
    print("=" * 60)

    df = load_and_prepare(CLEAN_PATH)
    print(f"\nDataset: {len(df)} filas | {df['categoria'].nunique()} categorias")
    print(f"Fechas : {df['fecha'].min().date()} -> {df['fecha'].max().date()}")

    # Entrenar DemandNet
    demand_model, scaler_X, scaler_y, le, features_d, mape_d, wmape_d = \
        train_demand_net(df)

    # Entrenar PricingNet (usa demanda proyectada de DemandNet)
    pricing_model, scaler_Xp, scaler_yp, features_p, mae_p = \
        train_pricing_net(df, le, demand_model, scaler_X, scaler_y, features_d)

    # Grafico elasticidad
    plot_elasticity(df, demand_model, scaler_X, scaler_y, le, features_d)

    # Demo precios sugeridos
    results = demo_pricing(
        df, pricing_model, scaler_Xp, scaler_yp, le, features_p,
        demand_model, scaler_X, scaler_y, features_d
    )

    # Guardar modelos
    demand_model.save(DEMAND_MODEL)
    pricing_model.save(PRICING_MODEL)
    joblib.dump({
        "scaler_X_demand":  scaler_X,
        "scaler_y_demand":  scaler_y,
        "scaler_X_pricing": scaler_Xp,
        "scaler_y_pricing": scaler_yp,
        "label_encoder":    le,
        "features_demand":  features_d,
        "features_pricing": features_p,
        "stockout_penalty": STOCKOUT_PENALTY,
        "cheap_penalty":    CHEAP_PENALTY,
    }, SCALER_PATH)

    print(f"\nModelos guardados:")
    print(f"  {DEMAND_MODEL}")
    print(f"  {PRICING_MODEL}")
    print(f"  {SCALER_PATH}")

    # Reporte
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("FASE 4 - RED NEURONAL DE PRICING DINAMICO\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset      : {CLEAN_PATH}\n")
        f.write(f"Filas        : {len(df)}\n")
        f.write(f"Categorias   : {df['categoria'].nunique()}\n\n")
        f.write("Loss personalizada (StockoutAwareLoss):\n")
        f.write(f"  Stockout penalty : x{STOCKOUT_PENALTY}\n")
        f.write(f"  Cheap penalty    : x{CHEAP_PENALTY}\n\n")
        f.write("DemandNet:\n")
        f.write(f"  MAPE  : {mape_d:.2f}%\n")
        f.write(f"  WMAPE : {wmape_d:.2f}%\n\n")
        f.write("PricingNet:\n")
        f.write(f"  MAE precio : {mae_p:.2f}\n\n")
        f.write("Precios sugeridos:\n")
        for r in results:
            f.write(f"  {r['categoria']}: stock={r['stock']:.0f} "
                    f"demanda={r['demanda_proyectada']:.1f} "
                    f"precio {r['precio_actual']:.2f} -> {r['precio_optimo']:.2f}\n")

    print(f"\nReporte: {REPORT_PATH}")
    print("\n" + "=" * 60)
    print("  RESUMEN FASE 4")
    print("=" * 60)
    print(f"  DemandNet  WMAPE       : {wmape_d:.2f}%")
    print(f"  PricingNet MAE precio  : {mae_p:.2f}")
    print(f"  Loss StockoutAware     : stockout x{STOCKOUT_PENALTY} > barato x{CHEAP_PENALTY}")
    if wmape_d <= 15:
        print("  DemandNet cumple WMAPE < 15%")
    print(f"  Grafico: {FIG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()