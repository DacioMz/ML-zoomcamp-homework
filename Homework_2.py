"""
Homework 2
"""

import pandas as pd
import numpy as np
import os
import requests
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
filename = "car_fuel_efficiency.csv"

if not os.path.exists(filename):
    print("Descargando dataset...")
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

df = pd.read_csv(filename)
print(df.head())

#Question 1
# we keep only the col that we need 
df = df[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']]

df.isnull().sum()

#Question 2
df["horsepower"].median()

import numpy as np

# Fijar la semilla para reproducibilidad
np.random.seed(42)

# Barajar (shuffle) los índices del DataFrame
n = len(df)
indices = np.arange(n)
np.random.shuffle(indices)

# Reordenar el DataFrame con esos índices
df = df.iloc[indices].reset_index(drop=True)

# Dividir en 60% / 20% / 20%
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - n_test - n_val



df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train + n_val]
df_test = df.iloc[n_train + n_val:]

# Verificar tamaños
print(len(df_train), len(df_val), len(df_test))
df_test.isnull().sum()

#Question 3

#Model filling nan with 0
features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
target = 'fuel_efficiency_mpg'

# Copiar y rellenar nulos con 0
df_train_0 = df_train.fillna({'horsepower': 0})
df_val_0 = df_val.fillna({'horsepower': 0})

# Separar X e y
X_train_0 = df_train_0[features].values
y_train_0 = df_train_0[target].values

X_val_0 = df_val_0[features].values
y_val_0 = df_val_0[target].values

# Entrenar modelo
model_0 = LinearRegression()
model_0.fit(X_train_0, y_train_0)

# Predecir y evaluar
y_pred_val_0 = model_0.predict(X_val_0)
rmse_0 = np.sqrt(mean_squared_error(y_val_0, y_pred_val_0))

print("RMSE con horsepower=0:", rmse_0)

#Model with mean 

# Calcular media solo con train
mean_hp = df_train['horsepower'].mean()

# Reemplazar nulos usando esa media
df_train_mean = df_train.fillna({'horsepower': mean_hp})
df_val_mean = df_val.fillna({'horsepower': mean_hp})

# Separar X e y
X_train_mean = df_train_mean[features].values
y_train_mean = df_train_mean[target].values

X_val_mean = df_val_mean[features].values
y_val_mean = df_val_mean[target].values

# Entrenar modelo
model_mean = LinearRegression()
model_mean.fit(X_train_mean, y_train_mean)

# Predecir y evaluar
y_pred_val_mean = model_mean.predict(X_val_mean)
rmse_mean = np.sqrt(mean_squared_error(y_val_mean, y_pred_val_mean))

print("RMSE con horsepower=media:", rmse_mean)

#Entonces?
rmse_0 = round(rmse_0, 2)
rmse_mean = round(rmse_mean, 2)

print("RMSE (fill 0):", rmse_0)
print("RMSE (fill mean):", rmse_mean)

#Question 4

df_train_0 = df_train.fillna({'horsepower': 0})
df_val_0 = df_val.fillna({'horsepower': 0})

X_train = df_train_0[features].values
y_train = df_train_0[target].values

X_val = df_val_0[features].values
y_val = df_val_0[target].values

r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
scores = []

for r in r_values:
    model = Ridge(alpha=r)  # 'alpha' es el parámetro de regularización
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse = round(rmse, 2)
    
    scores.append((r, rmse))

scores

results = pd.DataFrame(scores, columns=["r", "RMSE"])
print(results)

best_r = results.loc[results["RMSE"].idxmin()]
print("Mejor valor de r:", best_r["r"])
print("Mejor RMSE:", best_r["RMSE"])
