from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import joblib, numpy as np, os, mlflow, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="HeartAI API", version="1.0.0")

model  = joblib.load("heart_nn_model.pkl")
scaler = joblib.load("scaler.pkl")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO  = os.getenv("GITHUB_REPO", "onsalouini/mlops-heartai")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "HeartAI API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
def predict(data: InputData):
    try:
        arr   = np.array(data.features).reshape(1, -1)
        arr   = scaler.transform(arr)
        pred  = int(model.predict(arr)[0])
        proba = float(model.predict_proba(arr)[0][pred])

        # Dans Heart Disease dataset :
        # target=0 → Maladie cardiaque
        # target=1 → Pas de maladie
        label = "Pas de maladie" if pred == 1 else "Maladie cardiaque"

        return {
            "prediction": pred,
            "label": label,
            "confidence": round(proba, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/metrics")
def get_metrics():
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("heart_disease_experiment")
        if not exp:
            return {"runs": [], "best_accuracy": 0}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=10
        )
        runs_data = []
        for r in runs:
            runs_data.append({
                "run_id":     r.info.run_id[:8],
                "run_name":   r.info.run_name,
                "status":     r.info.status,
                "accuracy":   r.data.metrics.get("accuracy", 0),
                "f1_score":   r.data.metrics.get("f1_score", 0),
                "auc_roc":    r.data.metrics.get("auc_roc", 0),
                "start_time": datetime.fromtimestamp(
                    r.info.start_time/1000).strftime("%d/%m %H:%M")
            })
        best = max([r["accuracy"] for r in runs_data], default=0)
        return {"runs": runs_data, "best_accuracy": best, "total": len(runs_data)}
    except Exception as e:
        return {"runs": [], "error": str(e)}

@app.get("/api/github")
def get_github():
    if not GITHUB_TOKEN:
        return {"status": "no_token", "runs": []}
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        url  = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=5"
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        runs = []
        for r in data.get("workflow_runs", []):
            runs.append({
                "name":       r["name"],
                "status":     r["status"],
                "conclusion": r.get("conclusion", ""),
                "branch":     r["head_branch"],
                "commit":     r["head_sha"][:7],
                "created_at": r["created_at"][:16].replace("T", " "),
                "url":        r["html_url"]
            })
        latest = runs[0] if runs else {}
        return {
            "status":   latest.get("conclusion", "unknown"),
            "runs":     runs,
            "repo":     GITHUB_REPO,
            "repo_url": f"https://github.com/{GITHUB_REPO}"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "runs": []}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return FileResponse("index.html")

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_PATH = "heart.csv"

@app.get("/api/dataset/info")
def dataset_info():
    df = pd.read_csv(DATA_PATH)
    return {
        "total_rows":    len(df),
        "sick":          int((df["target"]==0).sum()),
        "healthy":       int((df["target"]==1).sum()),
        "features":      list(df.columns),
        "preview":       df.head(5).to_dict(orient="records")
    }

@app.post("/api/dataset/add")
def add_row(data: InputData):
    try:
        df = pd.read_csv(DATA_PATH)
        if len(data.features) != 14:
            raise HTTPException(status_code=400, detail="14 valeurs requises (13 features + target)")
        cols = list(df.columns)
        new_row = dict(zip(cols, data.features))
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        return {"message": "Ligne ajoutée", "total_rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/dataset/delete/{index}")
def delete_row(index: int):
    try:
        df = pd.read_csv(DATA_PATH)
        if index < 0 or index >= len(df):
            raise HTTPException(status_code=400, detail="Index invalide")
        df = df.drop(index=index).reset_index(drop=True)
        df.to_csv(DATA_PATH, index=False)
        return {"message": f"Ligne {index} supprimée", "total_rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/dataset/retrain")
def retrain_after_edit():
    try:
        from model_pipeline import prepare_data, train_model, evaluate_model, save_model
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model_new = train_model(X_train, y_train)
        metrics   = evaluate_model(model_new, X_test, y_test)
        save_model(model_new)
        global model, scaler
        model  = joblib.load("heart_nn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return {"message": "Modèle réentraîné", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_PATH = "heart.csv"

@app.get("/api/dataset/info")
def dataset_info():
    df = pd.read_csv(DATA_PATH)
    return {
        "total_rows":    len(df),
        "sick":          int((df["target"]==0).sum()),
        "healthy":       int((df["target"]==1).sum()),
        "features":      list(df.columns),
        "preview":       df.head(5).to_dict(orient="records")
    }

@app.post("/api/dataset/add")
def add_row(data: InputData):
    try:
        df = pd.read_csv(DATA_PATH)
        if len(data.features) != 14:
            raise HTTPException(status_code=400, detail="14 valeurs requises (13 features + target)")
        cols = list(df.columns)
        new_row = dict(zip(cols, data.features))
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        return {"message": "Ligne ajoutée", "total_rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/dataset/delete/{index}")
def delete_row(index: int):
    try:
        df = pd.read_csv(DATA_PATH)
        if index < 0 or index >= len(df):
            raise HTTPException(status_code=400, detail="Index invalide")
        df = df.drop(index=index).reset_index(drop=True)
        df.to_csv(DATA_PATH, index=False)
        return {"message": f"Ligne {index} supprimée", "total_rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/dataset/retrain")
def retrain_after_edit():
    try:
        from model_pipeline import prepare_data, train_model, evaluate_model, save_model
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model_new = train_model(X_train, y_train)
        metrics   = evaluate_model(model_new, X_test, y_test)
        save_model(model_new)
        global model, scaler
        model  = joblib.load("heart_nn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return {"message": "Modèle réentraîné", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
