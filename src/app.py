# Your code here
# Paso 1: Carga del conjunto de datos
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import joblib

# Cargar el dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv'
df = pd.read_csv(url)
print(df.head())

# Convertir a formato de serie temporal
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Graficar
plt.figure(figsize=(12,6))
plt.plot(df['sales'])
plt.title('Ventas en el tiempo')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.grid()
plt.show()

# Paso 2: Análisis de la serie temporal
print("\n--- Análisis ---")
print(f"TENSOR (unidad mínima de tiempo): {pd.infer_freq(df.index)}")  # frecuencia
# Puede devolver 'MS' (Month Start), 'D' (Diario), etc.

# Tendencia
decomposition = sm.tsa.seasonal_decompose(df['sales'], model='additive', period=12)
decomposition.plot()
plt.show()

# Estacionariedad: Prueba de Dickey-Fuller aumentada
adf_test = adfuller(df['sales'])
print('ADF Statistic:', adf_test[0])
print('p-value:', adf_test[1])
if adf_test[1] < 0.05:
    print('✅ La serie ES estacionaria.')
else:
    print('⚠️ La serie NO es estacionaria.')

# Paso 3: Entrenar un ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Separar en train-test
train = df.iloc[:-12]  # Dejamos los últimos 12 meses para test
test = df.iloc[-12:]

# Parametrización simple (p,d,q) = (1,1,1) — después puedes ajustar mejor
model = ARIMA(train['sales'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Paso 4: Predicción y evaluación
forecast = model_fit.forecast(steps=12)
plt.figure(figsize=(10,5))
plt.plot(train.index, train['sales'], label='Train')
plt.plot(test.index, test['sales'], label='Test', color='green')
plt.plot(test.index, forecast, label='Predicción', color='red')
plt.legend()
plt.title('Predicción de ventas')
plt.grid()
plt.show()

# Métrica: Error cuadrático medio
mse = mean_squared_error(test['sales'], forecast)
print(f"Mean Squared Error: {mse:.2f}")

# Paso 5: Guardar el modelo
joblib.dump(model_fit, 'sales_forecast_arima.pkl')
print("✅ Modelo guardado en 'sales_forecast_arima.pkl'")
