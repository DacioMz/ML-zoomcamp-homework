import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Cargar el pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# 2. Definir la estructura de los datos de entrada
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# 3. Crear la app FastAPI
app = FastAPI()

# 4. Endpoint para predecir
@app.post("/predict")
def predict(client: Lead):
    X = [client.dict()]
    pred = model.predict_proba(X)[0, 1]
    return {"conversion_probability": round(pred, 3)}


