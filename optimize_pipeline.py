
"""
optimize_pipeline.py — Optimisation complète du pipeline
1. GridSearchCV sur EarlyStopping_NN
2. Mise à jour automatique de model_pipeline.py avec les meilleurs params
3. Log MLflow + sauvegarde du meilleur modèle
Usage : python3 optimize_pipeline.py
"""

import pandas as pd
import numpy as np
import joblib, mlflow, mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    make_scorer
)

# ── Mêmes constantes que model_pipeline.py ─────────────────
DATA_PATH   = "heart.csv"
MODEL_PATH  = "heart_nn_model.pkl"
SCALER_PATH = "scaler.pkl"
mlflow.set_experiment("heart_disease_experiment")

# ── Elasticsearch ───────────────────────────────────────────
ES_OK = False
try:
    from elasticsearch import Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    ES_OK = es.ping()
    print("Elasticsearch connecté ✓" if ES_OK else "Elasticsearch non disponible")
except Exception as e:
    print(f"Elasticsearch : {e}")

def send_es(data):
    if not ES_OK: return
    try:
        es.index(index="mlflow-metrics", document=data)
        print(f"ES log envoyé : {data['event']}")
    except Exception as e:
        print(f"ES erreur : {e}")

# ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  OPTIMISATION PIPELINE — Heart Disease Detection")
print("="*60)

# ── 1. DONNÉES ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df.drop("target", axis=1)
y  = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

try:
    scaler     = joblib.load(SCALER_PATH)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"✓ Scaler chargé depuis {SCALER_PATH}")
except:
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Nouveau scaler sauvegardé → {SCALER_PATH}")

# ── 2. GRILLE DE PARAMÈTRES ─────────────────────────────────
# Optimisée pour un contexte médical : priorité au Recall
param_grid = {
    'hidden_layer_sizes': [
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128, 64),
        (128,),
        (256, 128),
    ],
    'alpha':              [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.005, 0.01],
}

base_model = MLPClassifier(
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.15,
    max_iter=500,
    random_state=42
)

# Scorer combiné : priorité Recall (critique médical) + AUC
# Recall pondéré à 60%, AUC à 40%
def medical_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_prob = estimator.predict_proba(X)[:, 1]
    rec = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    return 0.6 * rec + 0.4 * auc

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n🔍 GridSearchCV en cours...")
print(f"   Combinaisons à tester : "
      f"{len(param_grid['hidden_layer_sizes']) * len(param_grid['alpha']) * len(param_grid['learning_rate_init'])}")
print(f"   Scorer : 60% Recall + 40% AUC-ROC (priorité médicale)\n")

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring=make_scorer(medical_scorer, needs_proba=False),
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train_sc, y_train)

# ── 3. MEILLEURS PARAMÈTRES ─────────────────────────────────
best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_

print(f"\n{'='*60}")
print(f"✅ Meilleurs paramètres trouvés :")
for k, v in best_params.items():
    print(f"   {k:<25} : {v}")

# ── 4. ÉVALUATION FINALE ────────────────────────────────────
y_pred = best_model.predict(X_test_sc)
y_prob = best_model.predict_proba(X_test_sc)[:, 1]

acc  = round(accuracy_score(y_test, y_pred),  4)
f1   = round(f1_score(y_test, y_pred),         4)
auc  = round(roc_auc_score(y_test, y_prob),    4)
prec = round(precision_score(y_test, y_pred),  4)
rec  = round(recall_score(y_test, y_pred),     4)

print(f"\n📊 Performance du modèle optimisé :")
print(f"   Accuracy  : {acc}")
print(f"   AUC-ROC   : {auc}")
print(f"   F1-Score  : {f1}")
print(f"   Precision : {prec}")
print(f"   Recall    : {rec}  ← métrique clé (médical)")

print(f"\nClassification Report :\n")
print(classification_report(y_test, y_pred,
                             target_names=['Sans maladie', 'Maladie']))

# ── 5. LOG MLFLOW ────────────────────────────────────────────
with mlflow.start_run(run_name="optimized_EarlyStopping_NN"):
    mlflow.log_param("algorithm",          "MLPClassifier_Optimized")
    mlflow.log_param("hidden_layer_sizes", str(best_params['hidden_layer_sizes']))
    mlflow.log_param("alpha",              best_params['alpha'])
    mlflow.log_param("learning_rate_init", best_params['learning_rate_init'])
    mlflow.log_param("activation",         "relu")
    mlflow.log_param("early_stopping",     True)
    mlflow.log_param("scorer",             "0.6*Recall + 0.4*AUC")

    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("auc_roc",   auc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)

    mlflow.sklearn.log_model(best_model, "model")
    print(f"\n✅ Run MLflow loggé : optimized_EarlyStopping_NN")

# ── 6. SAUVEGARDE DU MODÈLE ────────────────────────────────
joblib.dump(best_model, MODEL_PATH)
print(f"✅ Modèle sauvegardé → {MODEL_PATH}")

# ── 7. MISE À JOUR AUTOMATIQUE model_pipeline.py ───────────
hidden_str = str(best_params['hidden_layer_sizes'])
alpha_val  = best_params['alpha']
lr_val     = best_params['learning_rate_init']

new_train_func = f'''def train_model(X_train, y_train):
    """Modèle optimisé via GridSearchCV le {datetime.now().strftime('%Y-%m-%d')}
    Scorer : 60% Recall + 40% AUC-ROC (priorité détection médicale)
    Meilleurs params : hidden={hidden_str}, alpha={alpha_val}, lr={lr_val}
    """
    with mlflow.start_run():
        hidden, max_iter = {hidden_str}, 500
        alpha            = {alpha_val}
        lr_init          = {lr_val}

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
        send_es({{"timestamp": datetime.now().isoformat(),
                 "event": "train_model",
                 "hidden_layer_sizes": str(hidden),
                 "alpha": alpha,
                 "learning_rate_init": lr_init}})
    return model
'''

# Lire le fichier existant
with open("model_pipeline.py", "r") as f:
    content = f.read()

# Remplacer la fonction train_model
import re
pattern = r'def train_model\(X_train, y_train\):.*?(?=\ndef |\Z)'
updated = re.sub(pattern, new_train_func, content, flags=re.DOTALL)

# Sauvegarde backup
with open("model_pipeline_backup.py", "w") as f:
    f.write(content)
print(f"✅ Backup sauvegardé → model_pipeline_backup.py")

# Écriture du fichier mis à jour
with open("model_pipeline.py", "w") as f:
    f.write(updated)
print(f"✅ model_pipeline.py mis à jour avec les meilleurs paramètres !")

# ── 8. ELASTICSEARCH ───────────────────────────────────────
send_es({
    "timestamp":         datetime.now().isoformat(),
    "event":             "optimize_pipeline",
    "best_hidden":       str(best_params['hidden_layer_sizes']),
    "best_alpha":        best_params['alpha'],
    "best_lr":           best_params['learning_rate_init'],
    "accuracy":          acc,
    "f1_score":          f1,
    "auc_roc":           auc,
    "recall":            rec,
})

print(f"\n{'='*60}")
print(f"🏆 PIPELINE OPTIMISÉ ET MIS À JOUR")
print(f"   AUC-ROC : {auc}  |  Recall : {rec}  |  F1 : {f1}")
print(f"{'='*60}")
print(f"\nPour visualiser dans MLflow UI :")
print(f"  mlflow ui --port 5000")
