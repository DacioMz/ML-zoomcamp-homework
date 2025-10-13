"""
Homework 3

"""

import pandas as pd
import numpy as np
import os
import requests
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np


# Definimos la URL y el nombre del archivo
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv"
filename = "course_lead_scoring.csv"

#Descargamos el archivo solo si no existe
if not os.path.exists(filename):
    print("Descargando dataset...")
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

# Cargamos el dataset
df = pd.read_csv(filename)
print(df.head())

# Verificar valores nulos
df.isnull().sum()

# Separar variables categóricas y numéricas
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['number']).columns

# Reemplazar nulos
df[cat_cols] = df[cat_cols].fillna('NA')
df[num_cols] = df[num_cols].fillna(0.0)

# Verificar que no queden nulos
print("Nulos restantes:", df.isnull().sum().sum())


#Question 1 
# Calcular la moda de la columna 'industry'
mode_industry = df['industry'].mode()[0]
print("Moda de industry:", mode_industry)

#Question 2

num_df = df.select_dtypes(include=['number'])
corr_matrix = num_df.corr()
corr_matrix

corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
# Eliminar autocorrelaciones (valor = 1)
corr_pairs = corr_pairs[corr_pairs < 1]
corr_pairs.head()


#Question 3

y = df['converted']
X = df.drop('converted', axis=1)

from sklearn.model_selection import train_test_split

# 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# dividir el 40% restante en 20% val y 20% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from sklearn.feature_selection import mutual_info_classif

# Filtramos solo las categóricas
cat_cols = X_train.select_dtypes(include=['object']).columns

# Convertimos las categorías a números usando factorize
X_train_enc = X_train[cat_cols].apply(lambda x: x.factorize()[0])

# Calculamos la información mutua
mi = mutual_info_classif(X_train_enc, y_train, discrete_features=True)

# Mostramos los resultados
mi_scores = pd.Series(mi, index=cat_cols).sort_values(ascending=False)
mi_scores.round(2)

#Question 4

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Identificar columnas
cat_cols = X_train.select_dtypes(include=['object']).columns
num_cols = X_train.select_dtypes(include=['number']).columns

# Crear el transformador de columnas
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

# Definir el modelo con los parámetros indicados
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

# Crear el pipeline
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(preprocessor, model)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Predicciones
y_pred = pipeline.predict(X_val)

# Calcular accuracy
acc = accuracy_score(y_val, y_pred)
print("Accuracy:", round(acc, 2))

#Question 5

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Repetimos configuración de la Q4
cat_cols = X_train.select_dtypes(include=['object']).columns
num_cols = X_train.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
pipeline = make_pipeline(preprocessor, model)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
base_acc = accuracy_score(y_val, y_pred)
print("Base accuracy:", base_acc)


acc_diffs = {}

for col in X_train.columns:
    # dataset sin esa columna
    X_train_drop = X_train.drop(columns=[col])
    X_val_drop = X_val.drop(columns=[col])
    
    # definir preprocesador nuevo
    cat_cols_drop = X_train_drop.select_dtypes(include=['object']).columns
    num_cols_drop = X_train_drop.select_dtypes(include=['number']).columns
    
    preprocessor_drop = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_drop),
        ('num', 'passthrough', num_cols_drop)
    ])
    
    pipeline_drop = make_pipeline(preprocessor_drop, LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42))
    pipeline_drop.fit(X_train_drop, y_train)
    y_pred_drop = pipeline_drop.predict(X_val_drop)
    acc_drop = accuracy_score(y_val, y_pred_drop)
    
    acc_diffs[col] = base_acc - acc_drop

# Mostrar diferencias ordenadas
pd.Series(acc_diffs).sort_values()

#Question 6
C_values = [0.01, 0.1, 1, 10, 100]

cat_cols = X_train.select_dtypes(include=['object']).columns
num_cols = X_train.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

results = {}

for C in C_values:
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    pipeline = make_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    results[C] = round(acc, 3)

results

