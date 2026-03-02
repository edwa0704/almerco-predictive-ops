# Almerco Predictive Ops

📊 Sistema de Predicción de Demanda con Machine Learning
📌 Descripción

Este proyecto implementa un pipeline completo de análisis y predicción de demanda utilizando:

🐼 Pandas (limpieza y transformación)

📈 Análisis estadístico

🤖 Machine Learning (HistGradientBoostingRegressor)

📉 Métrica de evaluación WMAPE

El sistema permite procesar un CSV empresarial, limpiarlo, analizarlo y entrenar un modelo de predicción considerando estacionalidad y comportamiento histórico.

🏗️ Estructura del Proyecto
almerco-predictive-ops/
│
├── data/
│   ├── raw/                    # Datos crudos
│   ├── clean_data.csv          # Datos limpios listos para modelado
│
├── models/                     # Modelo entrenado
│
├── reports/                    # Reportes generados
│
├── src/
│   ├── adapt_empresa_to_raw.py
│   ├── fase1_cleaning.py
│   ├── fase2_analysis.py
│   ├── predict_demand.py
│
├── requirements.txt
└── README.md

🛠️ Instalación

✅ CMD

✅ PowerShell

✅ Git Bash

✅ Linux / Mac

´´´bash
git clone https://github.com/TU_USUARIO/almerco-predictive-ops.git
cd almerco-predictive-ops
´´´

2️⃣ Crear entorno virtual

🟦 En Windows (CMD)

´´´cmd
python -m venv .venv
.venv\Scripts\activate
´´´

🟦 En Windows (PowerShell)
´´´powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
´´´

🟦 En Windows (Git Bash)
´´´bash
python -m venv .venv
source .venv/Scripts/activate
´´´

🟢 En Linux / Mac
´´´bash
python3 -m venv .venv
source .venv/bin/activate
´´´

3️⃣ Instalar dependencias

Una vez activado el entorno virtual:
´´´bash
pip install -r requirements.txt
´´´

🔍 Verificación opcional

Para confirmar que el entorno está activo:
´´´bash
python --version
pip list
´´´

▶️ Cómo Ejecutar el Proyecto

1️⃣ Adaptar CSV de empresa

Este paso convierte cualquier CSV empresarial al formato estándar del proyecto.

🟦 Windows (CMD)
´´´cmd
python src\adapt_empresa_to_raw.py --input "C:\ruta\del\archivo_empresa.csv"
´´´

🟦 Windows (PowerShell)
´´´powershell
python src\adapt_empresa_to_raw.py --input "C:\ruta\del\archivo_empresa.csv"
´´´

🟦 Windows (Git Bash)
´´´bash
python src/adapt_empresa_to_raw.py --input "/c/ruta/del/archivo_empresa.csv"
´´´
Nota: En Git Bash las rutas usan formato tipo /c/Users/...

🟢 Linux / Mac
´´´bash
python3 src/adapt_empresa_to_raw.py --input "/home/usuario/archivo_empresa.csv"
´´´
2️⃣ Fase 1 – Limpieza de datos

CMD / PowerShell
´´´bash
python src\fase1_cleaning.py
´´´
Git Bash / Linux / Mac
´´´bash
python src/fase1_cleaning.py
´´´
3️⃣ Fase 2 – Análisis estadístico
CMD / PowerShell
´´´bash
python src\fase2_analysis.py
´´´
Git Bash / Linux / Mac
´´´bash
python src/fase2_analysis.py
´´´
4️⃣ Fase 3 – Predicción de demanda
CMD / PowerShell
´´´powershell
python src\predict_demand.py
´´´
Git Bash / Linux / Mac
´´´bash
python src/predict_demand.py
´´´

⚠️ Errores Comunes y Soluciones
1️⃣ Error: No existe clean_data.csv

Causa: No se ejecutó la Fase 1 antes de la Fase 3.

Solución:

python src/fase1_cleaning.py
2️⃣ Error: FileNotFoundError al adaptar CSV

Causa: Ruta incorrecta del archivo.

Solución:

En Git Bash usar rutas tipo /c/Users/...

En CMD / PowerShell usar C:\Users\...

3️⃣ Error alto en MAPE o WMAPE

Causa posible:

Pocos días históricos

Muchas ventas en cero

Dataset muy pequeño

Recomendación:

Utilizar datasets con mayor histórico

Agregar más lags (14, 28 días)

Incrementar max_iter del modelo

4️⃣ Error en PowerShell: ejecución de scripts bloqueada

Solución:

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
5️⃣ Entorno virtual no se activa

Verifique que esté ejecutando el comando correcto según su terminal:

CMD → .venv\Scripts\activate

PowerShell → .venv\Scripts\Activate.ps1

Git Bash → source .venv/Scripts/activate

Linux/Mac → source .venv/bin/activate


📈 Modelo Utilizado

HistGradientBoostingRegressor

max_iter = 500

max_depth = 6

learning_rate = 0.05

Se aplican:

Lags (1 y 7 días)

Transformación logarítmica

División temporal (80/20)

📊 Métrica de Evaluación

Se utiliza WMAPE (Weighted Mean Absolute Percentage Error) debido a su estabilidad ante valores cero en demanda.

🧠 Conclusión

El proyecto implementa un flujo completo de ingeniería de datos y modelado predictivo listo para entornos empresariales.

La arquitectura permite:

Adaptarse a distintos CSV de empresas

Escalar a grandes volúmenes de datos

Reentrenar el modelo con nuevos históricos

Incorporar mejoras futuras como más lags o features de promoción

El sistema está diseñado para ser reproducible, modular y escalable.

📌 Autor

Frank Edwar Pérez Bustillos
Ingeniería de Programación, IA y Software