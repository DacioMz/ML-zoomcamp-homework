"""
Homework 1 
"""
import pandas as pd
import numpy as np
import os
import requests

#Question 1
pd.__version__

# -------------------------------
# I load the data base to work with
# -------------------------------
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
filename = "car_fuel_efficiency.csv"

if not os.path.exists(filename):
    print("Descargando dataset...")
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

df = pd.read_csv(filename)
print(df.head())

# Question 2, how many records do i have?
print("Cantidad de registros:", df.shape[0])

# Question 3, how many fuel type do i have?
print("Tipos de combustible:", df["fuel_type"].unique())

# Question 4, how many columns have NULL?
print("Columnas con valores nulos:", df.columns[df.isnull().any()].tolist())

# Question 5, which is the Max fuel efficiency?
print("Máxima eficiencia de Asia:", df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max())

# Question 6, Median value of horsepower?
print("Media de horsepower:", df["horsepower"].mean())

# i use the fillna method to fill the missing values in the horsepower column with the mode.
moda = df["horsepower"].mode()[0]
df["horsepower"] = df["horsepower"].fillna(moda)
print("Media de horsepower (con moda rellenada):", df["horsepower"].mean())

# Question 7, Median value of horsepower?
asia_cars = df[df["origin"] == "Asia"]
asia_cars_subset = asia_cars[["vehicle_weight", "model_year"]]
X = asia_cars_subset.head(7).to_numpy()

print("Matriz X:")
print(X)

# Compute matrix-matrix multiplication
XTX = X.T @ X
print("XTX:\n", XTX)

# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
XTX_inv = np.linalg.inv(XTX)
print("XTX invertida:\n", XTX_inv)

#i create a new array called Y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y

print("Vector w:", w)
print("Suma de los elementos de w:", w.sum())
