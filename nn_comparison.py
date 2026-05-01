"""
nn_comparison.py — Comparaison d'architectures Neural Network
Intégré avec model_pipeline.py (même MLflow experiment, même scaler)
Usage : python3 nn_comparison.py
"""

import pandas as pd
import numpy as np
import joblib, mlflow, mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, roc_curve,
    classification_report
)

# ── Mêmes constantes que model_pipeline.py ─────────────────
DATA_PATH   = "heart.csv"
SCALER_PATH = "scaler.pkl"
mlflow.set_experiment("heart_disease_experiment")

# ── Elasticsearch (même logique que model_pipeline.py) ─────
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

# ── 1. DONNÉES (réutilise scaler.pkl existant si possible) ──
print("\n" + "="*60)
print("  COMPARAISON ARCHITECTURES NEURAL NETWORK — Heart Disease")
print("="*60)

df = pd.read_csv(DATA_PATH)
X  = df.drop("target", axis=1)
y  = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Réutilise le scaler existant s'il est déjà sauvegardé
try:
    scaler = joblib.load(SCALER_PATH)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"✓ Scaler chargé depuis {SCALER_PATH}")
except:
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Nouveau scaler sauvegardé → {SCALER_PATH}")

print(f"  Train: {len(X_train_sc)} | Test: {len(X_test_sc)} | Features: {X_train_sc.shape[1]}\n")

# ── 2. ARCHITECTURES À COMPARER ─────────────────────────────
ARCHITECTURES = {
    "Shallow_NN":   dict(hidden_layer_sizes=(64,),        activation='relu',  max_iter=500),
    "Deep_NN":      dict(hidden_layer_sizes=(128, 64, 32), activation='relu',  max_iter=500),
    "Wide_NN":      dict(hidden_layer_sizes=(256, 256),    activation='relu',  max_iter=500),
    "EarlyStopping_NN": dict(hidden_layer_sizes=(128, 64), activation='relu',
                             early_stopping=True, validation_fraction=0.15, max_iter=500),
    "Tanh_NN":      dict(hidden_layer_sizes=(100, 50),     activation='tanh',  max_iter=500),
}

COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#9B2226']
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# ── 3. ENTRAÎNEMENT + MLFLOW LOGGING ────────────────────────
print("Entraînement des modèles...\n")

for name, params in ARCHITECTURES.items():
    with mlflow.start_run(run_name=f"nn_comparison_{name}"):

        # Params MLflow
        mlflow.log_param("algorithm",          "MLPClassifier")
        mlflow.log_param("architecture_name",  name)
        mlflow.log_param("hidden_layer_sizes", str(params['hidden_layer_sizes']))
        mlflow.log_param("activation",         params.get('activation', 'relu'))
        mlflow.log_param("max_iter",           params.get('max_iter', 500))
        mlflow.log_param("early_stopping",     params.get('early_stopping', False))

        # Entraînement
        model = MLPClassifier(random_state=42, **params)
        model.fit(X_train_sc, y_train)

        # Prédictions
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]

        # Métriques
        acc  = round(accuracy_score(y_test, y_pred),              4)
        f1   = round(f1_score(y_test, y_pred),                    4)
        auc  = round(roc_auc_score(y_test, y_prob),               4)
        prec = round(precision_score(y_test, y_pred),             4)
        rec  = round(recall_score(y_test, y_pred),                4)
        cv_scores = cross_val_score(model, X_train_sc, y_train,
                                    cv=cv, scoring='roc_auc')

        # Log MLflow
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("auc_roc",   auc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("cv_auc_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_auc_std",  round(cv_scores.std(),  4))
        mlflow.sklearn.log_model(model, "model")

        # Stockage local
        results[name] = {
            'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
            'accuracy': acc, 'f1': f1, 'auc': auc,
            'precision': prec, 'recall': rec,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
            'cm': confusion_matrix(y_test, y_pred),
        }

        # Elasticsearch
        send_es({
            "timestamp": datetime.now().isoformat(),
            "event": "nn_comparison",
            "architecture": name,
            "accuracy": acc, "f1_score": f1, "auc_roc": auc,
        })

        print(f"  {name:<22} Acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}")

# ── 4. TABLEAU RÉCAPITULATIF ────────────────────────────────
print("\n" + "="*72)
print(f"{'Modèle':<22} {'Acc':>6} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Recall':>6} {'CV-AUC':>12}")
print("="*72)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>6.3f} {r['auc']:>6.3f} {r['f1']:>6.3f} "
          f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
          f"{r['cv_mean']:>6.3f}±{r['cv_std']:.3f}")
print("="*72)

best = max(results, key=lambda k: results[k]['auc'])
print(f"\n🏆 Meilleur modèle (AUC-ROC) : {best}")
print(f"\nClassification Report — {best}:\n")
print(classification_report(y_test, results[best]['y_pred'],
                             target_names=['Sans maladie', 'Maladie']))

# ── 5. VISUALISATION ────────────────────────────────────────
names  = list(results.keys())
colors = COLORS[:len(names)]

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0D1117')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# 5a. Métriques groupées
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#161B22')
metrics_keys   = ['accuracy', 'auc', 'f1', 'precision', 'recall']
metrics_labels = ['Accuracy', 'AUC-ROC', 'F1', 'Precision', 'Recall']
x    = np.arange(len(metrics_keys))
bw   = 0.13
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics_keys]
    ax1.bar(x + i*bw, vals, bw, label=name, color=colors[i],
            alpha=0.85, edgecolor='white', linewidth=0.4)
ax1.set_xticks(x + bw*2); ax1.set_xticklabels(metrics_labels, color='white', fontsize=11)
ax1.set_ylim(0.5, 1.05); ax1.set_ylabel('Score', color='white')
ax1.set_title('Toutes les métriques', color='white', fontsize=14, fontweight='bold')
ax1.legend(fontsize=8, facecolor='#21262D', labelcolor='white', edgecolor='none',
           loc='lower right')
ax1.tick_params(colors='white')
for sp in ax1.spines.values(): sp.set_color('#30363D')

# 5b. CV AUC ± std
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#161B22')
cv_m = [results[n]['cv_mean'] for n in names]
cv_s = [results[n]['cv_std']  for n in names]
short = [n.replace('_NN','').replace('_',' ') for n in names]
ax2.barh(short, cv_m, xerr=cv_s, color=colors, alpha=0.85,
         edgecolor='white', linewidth=0.4,
         error_kw=dict(ecolor='white', capsize=4))
ax2.set_xlim(0.5, 1.05); ax2.set_xlabel('AUC', color='white')
ax2.set_title('CV AUC-ROC (5-fold)', color='white', fontsize=12, fontweight='bold')
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_color('#30363D')

# 5c. Courbes ROC
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor('#161B22')
for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax3.plot(fpr, tpr, color=colors[i], lw=2,
             label=f"{short[i]} (AUC={r['auc']:.3f})")
ax3.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
ax3.set_xlabel('False Positive Rate', color='white')
ax3.set_ylabel('True Positive Rate', color='white')
ax3.set_title('Courbes ROC', color='white', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9, facecolor='#21262D', labelcolor='white', edgecolor='none')
ax3.tick_params(colors='white')
for sp in ax3.spines.values(): sp.set_color('#30363D')

# 5d. Matrices de confusion (top 2)
sorted_r = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
for idx, (name, r) in enumerate(sorted_r[:2]):
    ax = fig.add_subplot(gs[1, 2] if idx == 0 else gs[2, 2])
    ax.set_facecolor('#161B22')
    cm = r['cm']
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Sain', 'Malade'], color='white', fontsize=9)
    ax.set_yticklabels(['Sain', 'Malade'], color='white', fontsize=9)
    for ii in range(2):
        for jj in range(2):
            ax.text(jj, ii, str(cm[ii,jj]), ha='center', va='center',
                    color='white', fontsize=16, fontweight='bold')
    rank = "🥇" if idx == 0 else "🥈"
    ax.set_title(f"{rank} {name.replace('_NN','').replace('_',' ')}\nConfusion Matrix",
                 color='white', fontsize=10, fontweight='bold')
    for sp in ax.spines.values(): sp.set_color('#30363D')

# 5e. Radar
ax5 = fig.add_subplot(gs[2, :2], polar=True)
ax5.set_facecolor('#161B22')
cats   = ['Accuracy','AUC-ROC','F1','Precision','Recall']
N      = len(cats)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
ax5.set_theta_offset(np.pi / 2); ax5.set_theta_direction(-1)
ax5.set_xticks(angles[:-1]); ax5.set_xticklabels(cats, color='white', fontsize=10)
ax5.set_ylim(0.5, 1.0)
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics_keys] + [r['accuracy']]
    ax5.plot(angles, vals, color=colors[i], lw=2,
             label=name.replace('_NN','').replace('_',' '))
    ax5.fill(angles, vals, color=colors[i], alpha=0.07)
ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15),
           facecolor='#21262D', labelcolor='white', edgecolor='none', fontsize=9)
ax5.set_title('Radar — Vue globale', color='white', fontsize=13, fontweight='bold', pad=20)
ax5.spines['polar'].set_color('#30363D')
ax5.tick_params(colors='white')

fig.suptitle('🫀 Comparaison Neural Networks — Détection Maladie Cardiaque',
             color='white', fontsize=16, fontweight='bold', y=0.98)

OUT = "nn_comparison_results.png"
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='#0D1117')
print(f"\n✅ Graphique sauvegardé → {OUT}")
print("✅ Tous les runs loggés dans MLflow → experiment: heart_disease_experiment")
print("\nPour visualiser dans MLflow UI :")
print("  mlflow ui --port 5000")
