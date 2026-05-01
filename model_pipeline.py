import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib, mlflow, mlflow.sklearn
from datetime import datetime

DATA_PATH   = "heart.csv"
MODEL_PATH  = "heart_nn_model.pkl"
SCALER_PATH = "scaler.pkl"

mlflow.set_experiment("heart_disease_experiment")

# ── Elasticsearch ──────────────────────────────────────────
ES_OK = False
try:
    from elasticsearch import Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    ES_OK = es.ping()
    print("Elasticsearch connecte ✓" if ES_OK else "Elasticsearch non disponible")
except Exception as e:
    print(f"Elasticsearch : {e}")

def send_es(data):
    if not ES_OK: return
    try:
        es.index(index="mlflow-metrics", document=data)
        print(f"ES log envoye : {data['event']}")
    except Exception as e:
        print(f"ES erreur : {e}")

def prepare_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X  = df.drop("target", axis=1)
    y  = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Donnees pretes — Train: {len(X_train)}, Test: {len(X_test)}")
    send_es({"timestamp": datetime.now().isoformat(),
             "event": "prepare_data",
             "train_size": len(X_train),
             "test_size": len(X_test)})
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Modèle optimisé via GridSearchCV le 2026-05-01
    Scorer : 60% Recall + 40% AUC-ROC (priorité détection médicale)
    Meilleurs params : hidden=(64, 32), alpha=0.0001, lr=0.001
    """
    with mlflow.start_run():
        hidden, max_iter = (64, 32), 500
        alpha            = 0.0001
        lr_init          = 0.001

        mlflow.log_param("hidden_layer_sizes", str(hidden))
        mlflow.log_param("max_iter",           max_iter)
        mlflow.log_param("algorithm",          "MLPClassifier_Optimized")
        mlflow.log_param("alpha",              alpha)
        mlflow.log_param("learning_rate_init", lr_init)
        mlflow.log_param("early_stopping",     True)

        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=lr_init,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("Modèle optimisé entraîné avec succès.")
        mlflow.sklearn.log_model(model, "model")
        send_es({"timestamp": datetime.now().isoformat(),
                 "event": "train_model",
                 "hidden_layer_sizes": str(hidden),
                 "alpha": alpha,
                 "learning_rate_init": lr_init})
    return model

def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = round(accuracy_score(y_test, y_pred), 4)
    f1  = round(f1_score(y_test, y_pred),        4)
    auc = round(roc_auc_score(y_test, y_proba),  4)
    print(f"Accuracy : {acc}\nF1-Score : {f1}\nAUC-ROC  : {auc}")
    with mlflow.start_run(nested=True):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc",  auc)
    send_es({"timestamp": datetime.now().isoformat(),
             "event": "evaluate_model",
             "accuracy": acc, "f1_score": f1, "auc_roc": auc})
    return {"accuracy": acc, "f1_score": f1, "auc_roc": auc}

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"Modele sauvegarde -> {MODEL_PATH}")
    send_es({"timestamp": datetime.now().isoformat(),
             "event": "save_model", "model_path": MODEL_PATH})

def load_model():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modele charge.")
    return model, scaler
