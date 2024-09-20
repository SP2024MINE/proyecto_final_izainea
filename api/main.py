from fastapi import FastAPI
import uvicorn
import pandas as pd
import joblib

app = FastAPI(debug=True, title="API 50K", version="0.1",summary="API para ver si usted gana más de 50k al año")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(data: dict): # se recibe un diccionario
    model = joblib.load("../Datos/mi_primer_pipeline.pkl") # se carga el modelo
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    if prediction[0] == 0:
        return {"prediction": "Menos de 50K"}
    else:
        return {"prediction": "Más de 50K"}