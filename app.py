from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Heart Disease API",
    docs_url=None
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Heart Disease API",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
    )

model  = joblib.load("heart_nn_model.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Heart Disease API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features).reshape(1, -1)
        arr = scaler.transform(arr)
        prediction = int(model.predict(arr)[0])
        label = "Maladie cardiaque" if prediction == 1 else "Pas de maladie"
        return {"prediction": prediction, "label": label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

@app.get("/dashboard")
def dashboard():
    return FileResponse("index.html")
