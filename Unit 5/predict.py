import pickle

# 1. Cargar el pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# 2. Crear el registro que queremos predecir
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# 3. El modelo espera una lista de diccionarios
X = [record]

# 4. Calcular la probabilidad de conversión
pred = model.predict_proba(X)[0, 1]

print("Probabilidad de conversión:", round(pred, 3))

