# Almerco Predictive Ops

📊 **Sistema de Predicción de Demanda con Machine Learning**

## 📌 Descripción
Este proyecto implementa un pipeline completo de análisis y predicción de demanda utilizando:
- 🐼 Pandas (limpieza y transformación)
- 📈 Análisis estadístico
- 🤖 Machine Learning (HistGradientBoostingRegressor)
- 📉 Métrica de evaluación WMAPE

El sistema permite procesar un CSV empresarial, limpiarlo, analizarlo y entrenar un modelo considerando estacionalidad y comportamiento histórico.

---

## 🏗️ Estructura del Proyecto


almerco-predictive-ops/
├── data/
│ ├── raw/ # Datos crudos
│ └── clean_data.csv # Datos limpios listos para modelado
├── models/ # Modelo entrenado
├── reports/ # Reportes generados
├── src/
│ ├── adapt_empresa_to_raw.py
│ ├── fase1_cleaning.py
│ ├── fase2_analysis.py
│ └── predict_demand.py
├── requirements.txt
└── README.md


---

## 🛠️ Instalación

### 1) Clonar repositorio
```bash
git clone https://github.com/TU_USUARIO/almerco-predictive-ops.git
cd almerco-predictive-ops
2) Crear entorno virtual

🟦 Windows (CMD)

python -m venv .venv
.venv\Scripts\activate

🟦 Windows (PowerShell)

python -m venv .venv
.venv\Scripts\Activate.ps1

Si PowerShell bloquea scripts:

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

🟦 Windows (Git Bash)

python -m venv .venv
source .venv/Scripts/activate

🟢 Linux / Mac

python3 -m venv .venv
source .venv/bin/activate
3) Instalar dependencias
pip install -r requirements.txt
🔍 Verificación opcional
python --version
pip list
▶️ Cómo Ejecutar el Proyecto
1️⃣ Adaptar CSV de empresa

Convierte cualquier CSV empresarial al formato estándar del proyecto.

🟦 Windows (CMD)

python src\adapt_empresa_to_raw.py --input "C:\ruta\del\archivo_empresa.csv"

🟦 Windows (PowerShell)

python src\adapt_empresa_to_raw.py --input "C:\ruta\del\archivo_empresa.csv"

🟦 Windows (Git Bash)

python src/adapt_empresa_to_raw.py --input "/c/Users/DELL/Downloads/archivo_empresa.csv"

🟢 Linux / Mac

python3 src/adapt_empresa_to_raw.py --input "/home/usuario/archivo_empresa.csv"
2️⃣ Fase 1 – Limpieza de datos

CMD / PowerShell

python src\fase1_cleaning.py

Git Bash / Linux / Mac

python src/fase1_cleaning.py
3️⃣ Fase 2 – Análisis estadístico

CMD / PowerShell

python src\fase2_analysis.py

Git Bash / Linux / Mac

python src/fase2_analysis.py
4️⃣ Fase 3 – Predicción de demanda

CMD / PowerShell

python src\predict_demand.py

Git Bash / Linux / Mac

python src/predict_demand.py
⚠️ Errores Comunes y Soluciones
1) Error: No existe clean_data.csv

Causa: No se ejecutó la Fase 1 antes de la Fase 3.
Solución:

python src/fase1_cleaning.py
2) Error: FileNotFoundError al adaptar CSV

Causa: Ruta incorrecta del archivo.
Solución:

Git Bash usa rutas tipo: /c/Users/...

CMD/PowerShell usa rutas tipo: C:\Users\...

3) Error alto en WMAPE / MAPE

Causas posibles:

Pocos días históricos

Muchas ventas en cero

Dataset pequeño

Recomendación:

Usar datasets con mayor histórico

Agregar más lags (14, 28 días)

Ajustar hiperparámetros (max_iter, max_depth)

4) PowerShell bloquea scripts
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
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

Se utiliza WMAPE por su estabilidad ante valores cero en demanda.

🧠 Conclusión

Proyecto modular y reproducible para:

Adaptarse a distintos CSV empresariales

Procesar, analizar y entrenar modelos de demanda

Reentrenar con nuevos históricos

Escalar mejoras futuras (más lags, promos reales, features adicionales)

📌 Autor

Frank Edwar Pérez Bustillos
Ingeniería de Programación, IA y Software