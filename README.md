Almerco Predictive Ops
Sistema modular de prediccion de demanda y pricing dinamico para operaciones comerciales.
Limpieza de datos -> Analisis estadistico -> Modelo ML -> Red Neuronal de Pricing.
Resultados:

Fase 3 WMAPE: 7.97% (objetivo < 15%)
Fase 4 DemandNet WMAPE: 10.68% (objetivo < 15%)


Estructura del proyecto
almerco-predictive-ops/
├── data/
│   ├── raw/                        # Archivos fuente CSV
│   └── clean_data.csv              # Generado por Fase 1
├── models/
│   ├── demand_model.joblib         # Modelo Fase 3
│   ├── demand_nn.keras             # DemandNet Fase 4
│   ├── pricing_nn.keras            # PricingNet Fase 4
│   └── pricing_scalers.joblib      # Scalers Fase 4
├── reports/
│   ├── figures/                    # Graficos generados
│   ├── null_report.txt
│   ├── std_by_category.csv
│   ├── hypothesis_test.txt
│   ├── predict_report.txt
│   └── pricing_report.txt
├── src/
│   ├── generate_fake_sales.py      # Genera dataset con patrones reales
│   ├── adapt_empresa_to_raw.py     # Adapta CSV empresarial al formato estandar
│   ├── fase1_cleaning.py           # Fase 1: Limpieza y normalizacion
│   ├── fase2_analysis.py           # Fase 2: Analisis estadistico
│   ├── predict_demand.py           # Fase 3: Prediccion de demanda
│   └── dynamic_pricing.py         # Fase 4: Red neuronal pricing dinamico
├── .gitignore
├── requirements.txt
└── README.md

Instalacion
1. Clonar el repositorio
bash
```bash
git clone https://github.com/TU_USUARIO/almerco-predictive-ops.git
cd almerco-predictive-ops
```

2. Crear entorno virtual
Windows - CMD
cmd
```bash
python -m venv .venv
.venv\Scripts\activate
```
Windows - PowerShell
powershell
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea scripts:
powershellSet-ExecutionPolicy RemoteSigned -Scope CurrentUser
Windows - Git Bash
bash
```bash
python -m venv .venv
source .venv/Scripts/activate
```

Linux / Mac
bash
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalar dependencias
bash
```bash
pip install -r requirements.txt
```

Ejecucion del pipeline completo
Ejecuta los pasos en este orden exacto:
Paso 1 - Generar dataset con patrones reales
Windows
cmd
```bash
python src/generate_fake_sales.py --days 730 --overwrite-real
```

Linux / Mac
bash
```bash
python3 src/generate_fake_sales.py --days 730 --overwrite-real
```

Genera 2 anos de historico con estacionalidad, promociones y tendencia.
Resultado: data/raw/sales_history_raw.csv (aprox 5110 filas)

Paso 2 - Fase 1: Limpieza de datos
bash
```bash
python src/fase1_cleaning.py        # Windows
python3 src/fase1_cleaning.py       # Linux / Mac
```

Resultado: data/clean_data.csv (aprox 5036 filas limpias)
Reporte: reports/null_report.txt

Paso 3 - Fase 2: Analisis estadistico
bash
```bash
python src/fase2_analysis.py        # Windows
python3 src/fase2_analysis.py       # Linux / Mac
```

Resultados:

reports/std_by_category.csv
reports/figures/correlation_heatmap.png
reports/hypothesis_test.txt


Paso 4 - Fase 3: Prediccion de demanda
bash
```bash
python src/predict_demand.py        # Windows
python3 src/predict_demand.py       # Linux / Mac
```

Resultado esperado:
Train: 3874 filas  |  Test: 966 filas
Split date: 2024-08-12
MAPE(%):   8.15
WMAPE(%):  7.97
Modelo guardado: models/demand_model.joblib
WMAPE <= 15% - Objetivo alcanzado.

Paso 5 - Fase 4: Red Neuronal de Pricing Dinamico
bash
```bash
python src/dynamic_pricing.py       # Windows
python3 src/dynamic_pricing.py      # Linux / Mac
```

Resultado esperado:
Dataset cargado: 5036 filas | 5 categorias
DemandNet  WMAPE : 10.68%
PricingNet MAE   : 63.13
Graficos: reports/figures/pricing_elasticity.png
Modelos guardados: models/demand_nn.keras + pricing_nn.keras
La Fase 4 entrena dos redes neuronales:

DemandNet: aprende la curva de elasticidad precio -> demanda por contexto
PricingNet: recomienda el precio optimo que maximiza el revenue por categoria


Usar tu propio CSV empresarial
Si tienes datos reales de tu empresa usa el adaptador:
Windows - CMD / PowerShell
cmd
```bash
python src\adapt_empresa_to_raw.py --input "C:\ruta\de\tu_archivo.csv"
```

Windows - Git Bash
bash
```bash
python src/adapt_empresa_to_raw.py --input "/c/Users/TU_USUARIO/Downloads/archivo.csv"
```

Linux / Mac
bash
```bash
python3 src/adapt_empresa_to_raw.py --input "/home/usuario/archivo.csv"
```

Luego ejecuta el pipeline desde el Paso 2.
Para WMAPE menor al 15% necesitas al menos 6 meses de historico diario.
Con 1 a 2 anos el modelo alcanza WMAPE menor al 10%.

Modelos utilizados
Fase 3 - HistGradientBoostingRegressor
ParametroValormax_iter800max_depth6learning_rate0.03min_samples_leaf5l2_regularization0.1
Fase 4 - Red Neuronal (TensorFlow / Keras)
DemandNet - predice demanda dado precio y contexto:
Input(14) -> Dense(128) -> BN -> Dropout(0.2)
          -> Dense(64)  -> BN -> Dropout(0.2)
          -> Dense(32)  -> Dense(1) lineal
PricingNet - recomienda precio optimo:
Input(12) -> Dense(128) -> BN -> Dropout(0.2)
          -> Dense(64)  -> BN -> Dropout(0.15)
          -> Dense(32)  -> Dense(1) sigmoid
Features incluidas: temporales, estacionalidad (Navidad, inicio de clases, fin de mes), promo, is_holiday, stock, lags, rolling mean.

Metrica de evaluacion
WMAPE (Weighted Mean Absolute Percentage Error):
WMAPE = suma(|real - predicho|) / suma(|real|) x 100
ResultadoInterpretacionmenor 10%Excelente10 a 15%Objetivo cumplido15 a 30%Aceptable, mejorablemayor 30%Insuficiente historico

Errores comunes
FileNotFoundError: data/raw/sales_history_raw.csv
bash
```bash
python src/generate_fake_sales.py --days 730 --overwrite-real
```

FileNotFoundError: data/clean_data.csv
bash
```bash
python src/fase1_cleaning.py
```

WMAPE mayor al 30%
El dataset tiene poco historico. Usa --days 730 o CSV empresarial con mas de 6 meses.
PowerShell bloquea scripts
powershellSet-ExecutionPolicy RemoteSigned -Scope CurrentUser
Graficos no abren / error Tcl/Tk
Ya resuelto. El proyecto usa backend Agg (guarda PNG, no abre ventana).
UnicodeEncodeError en Windows
Ya resuelto. Todos los archivos usan encoding="utf-8".

Autor
Frank Edwar Perez Bustillos
Ingenieria de Programacion, IA y Software