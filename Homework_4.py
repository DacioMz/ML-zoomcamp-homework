"""
Homework 4 

"""

import pandas as pd
import numpy as np
import os
import requests
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
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


# Primero: dividimos en train (60%) y temp (40%)
train, temp = train_test_split(df, test_size=0.4, random_state=1)

# Luego: dividimos temp en validation (20%) y test (20%)
# Como temp es el 40% del total, necesitamos que validation sea la mitad de temp (0.5 * 0.4 = 0.2)
val, test = train_test_split(temp, test_size=0.5, random_state=1)

# Verificamos tamaños
print(f"Train: {len(train)/len(df):.2%}")
print(f"Validation: {len(val)/len(df):.2%}")
print(f"Test: {len(test)/len(df):.2%}")

#Question 1
from sklearn.metrics import roc_auc_score

y_train = train['converted']

# Lista de variables numéricas
features = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']

# Diccionario para guardar los AUC
auc_scores = {}

for f in features:
    auc = roc_auc_score(y_train, train[f])
    # Si el AUC es menor que 0.5, invertimos el signo
    if auc < 0.5:
        auc = roc_auc_score(y_train, -train[f])
    auc_scores[f] = auc

# Mostrar resultados ordenados
sorted_auc = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
for feature, auc in sorted_auc:
    print(f"{feature}: {auc:.3f}")

#Question 2

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Definir variables
y_train = train['converted']
y_val = val['converted']

# Eliminamos la variable objetivo de los features
X_train = train.drop('converted', axis=1)
X_val = val.drop('converted', axis=1)

# One-hot encoding con DictVectorizer
dv = DictVectorizer(sparse=False)

# Convertir los dataframes en diccionarios de registros
train_dict = X_train.to_dict(orient='records')
val_dict = X_val.to_dict(orient='records')

# Fit + transform en train, solo transform en val
X_train_encoded = dv.fit_transform(train_dict)
X_val_encoded = dv.transform(val_dict)

# Entrenar la regresión logística
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_encoded, y_train)

# Predecir probabilidades y calcular AUC
y_pred_val = model.predict_proba(X_val_encoded)[:, 1]
auc = roc_auc_score(y_val, y_pred_val)

print(f"AUC en validación: {auc:.3f}")


#Question 3 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# Probabilidades predichas del modelo (ya las tenés del paso anterior)
y_pred_val = model.predict_proba(X_val_encoded)[:, 1]

thresholds = np.arange(0.0, 1.01, 0.01)

precisions = []
recalls = []

for t in thresholds:
    y_pred_bin = (y_pred_val >= t)
    precisions.append(precision_score(y_val, y_pred_bin))
    recalls.append(recall_score(y_val, y_pred_bin))

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs. Threshold')
plt.legend()
plt.grid(True)
plt.show()

# Encontrar el punto donde se cruzan (mínima diferencia entre ambos)
diff = np.abs(np.array(precisions) - np.array(recalls))
best_threshold = thresholds[np.argmin(diff)]
print(f"El umbral donde se cruzan Precision y Recall es aproximadamente: {best_threshold:.3f}")

#Question 4

import numpy as np
from sklearn.metrics import precision_score, recall_score

thresholds = np.arange(0.0, 1.01, 0.01)

f1_scores = []

for t in thresholds:
    y_pred_bin = (y_pred_val >= t)
    precision = precision_score(y_val, y_pred_bin)
    recall = recall_score(y_val, y_pred_bin)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    f1_scores.append(f1)

# Encontrar el threshold donde F1 es máximo
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = np.max(f1_scores)

print(f"F1 máximo: {best_f1:.3f} en threshold ≈ {best_threshold:.2f}")

#Question 5

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import numpy as np

# Variables
df_full_train = train.copy()  # si ya tenés df_full_train
y_full = df_full_train['converted']
X_full = df_full_train.drop('converted', axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=1)

auc_scores = []

for train_idx, val_idx in kf.split(X_full):
    X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
    
    # One-hot encoding
    dv = DictVectorizer(sparse=False)
    X_train_encoded = dv.fit_transform(X_train.to_dict(orient='records'))
    X_val_encoded = dv.transform(X_val.to_dict(orient='records'))
    
    # Entrenar modelo
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_encoded, y_train)
    
    # Predecir probabilidades y calcular AUC
    y_pred_val = model.predict_proba(X_val_encoded)[:, 1]
    auc = roc_auc_score(y_val, y_pred_val)
    auc_scores.append(auc)

# Calcular desviación estándar
std_auc = np.std(auc_scores)
print(f"AUC en cada fold: {auc_scores}")
print(f"Desviación estándar de los AUC: {std_auc:.3f}")

#Question 6

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import numpy as np

C_values = [0.000001, 0.001, 1]

kf = KFold(n_splits=5, shuffle=True, random_state=1)

results = []

for C in C_values:
    auc_scores = []
    for train_idx, val_idx in kf.split(X_full):
        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        # One-hot encoding
        dv = DictVectorizer(sparse=False)
        X_train_encoded = dv.fit_transform(X_train.to_dict(orient='records'))
        X_val_encoded = dv.transform(X_val.to_dict(orient='records'))
        
        # Entrenar modelo
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_encoded, y_train)
        
        # AUC
        y_pred_val = model.predict_proba(X_val_encoded)[:, 1]
        auc = roc_auc_score(y_val, y_pred_val)
        auc_scores.append(auc)
    
    mean_auc = np.round(np.mean(auc_scores), 3)
    std_auc = np.round(np.std(auc_scores), 3)
    results.append((C, mean_auc, std_auc))
    print(f"C={C}: mean AUC={mean_auc}, std={std_auc}")

# Elegir el mejor C
results_sorted = sorted(results, key=lambda x: (-x[1], x[2], x[0]))  # max mean, min std, min C
best_C = results_sorted[0][0]
print(f"\nMejor C: {best_C}")
