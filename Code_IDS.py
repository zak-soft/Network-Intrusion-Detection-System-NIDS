# ============================================================
# IDS CICIDS2017 - FULL PIPELINE (COMPLET)
# 1) Binaire (BENIGN vs ATTACK)
# 2) Simulation temps réel (Méthode A)
# 3) Multi-classes (type d’attaque)
# 4) Sauvegarde modèle + scaler + encoder
# 5) API Flask (option : génération app.py)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
import joblib


# ============================================================
# 0) CONFIG
# ============================================================
CSV_PATH   = r"C:\Users\meedj\Desktop\W_DATA\Wednesday-workingHours.pcap_ISCX.csv"
THRESHOLD  = 0.70   # si proba_attack >= seuil => ATTACK
DELAY      = 0.20   # délai entre 2 flows (simulation temps réel)
N_STREAM   = 200    # nb de flows streamés dans la simulation temps réel

WINDOW     = 50     # fenêtre glissante pour taux d'attaque
ALERT_RATE = 0.30   # si +30% d'attaques dans la fenêtre => ALERTE SOC

RANDOM_STATE = 42


# ============================================================
# 1) Charger dataset
# ============================================================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

if "Label" not in df.columns:
    raise ValueError("❌ Colonne 'Label' introuvable. Vérifie df.columns.")

print("✅ Dataset chargé:", df.shape)
print("\n✅ Top labels:")
print(df["Label"].astype(str).str.strip().value_counts().head(10))


# ============================================================
# 2) Nettoyage features (robuste)
# ============================================================
y_text = df["Label"].astype(str).str.strip()

X = df.drop(columns=["Label"])

# garder uniquement les colonnes numériques (important CICIDS)
X = X.select_dtypes(include=[np.number])

# remplacer inf -> nan
X = X.replace([np.inf, -np.inf], np.nan)

# remplir nan par médiane
X = X.fillna(X.median(numeric_only=True))

# clip extrêmes (sécurité)
X = X.clip(lower=-1e12, upper=1e12)

print("\n✅ Features:", X.shape)
print("NaN total:", int(X.isna().sum().sum()), "| Inf total:", int(np.isinf(X.values).sum()))


# ============================================================
# 3) IDS BINAIRE : BENIGN=0, ATTACK=1
# ============================================================
y_bin = (y_text != "BENIGN").astype(int)

print("\n✅ Répartition binaire:")
print(pd.Series(y_bin).value_counts().rename(index={0:"BENIGN", 1:"ATTACK"}))

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bin
)

# scaling
scaler_bin = StandardScaler()
X_train_s = scaler_bin.fit_transform(X_train)
X_test_s  = scaler_bin.transform(X_test)

print("\n✅ Split binaire OK")
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Test BENIGN:", int((y_test==0).sum()), "| Test ATTACK:", int((y_test==1).sum()))


# ============================================================
# 4) Entraîner modèles (LogReg + RandomForest)
# ============================================================
# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=2000, n_jobs=-1)
t0 = time.time()
lr.fit(X_train_s, y_train)
print(f"\n✅ LogisticRegression entraîné en {time.time()-t0:.2f}s")

# RandomForest (souvent meilleur)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
t0 = time.time()
rf.fit(X_train_s, y_train)
print(f"✅ RandomForest entraîné en {time.time()-t0:.2f}s")

# On choisit RF comme modèle IDS principal
model_bin = rf


# ============================================================
# 5) Évaluation binaire + visualisations
# ============================================================
y_pred  = model_bin.predict(X_test_s)
y_proba = model_bin.predict_proba(X_test_s)[:, 1]

print("\n===== REPORT (BINAIRE) =====")
print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"])
disp.plot(values_format="d")
plt.title("Confusion Matrix - IDS (BENIGN vs ATTACK)")
plt.show()

# ROC-AUC
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC (binaire): {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - IDS binaire")
plt.legend()
plt.show()

# Distribution des probabilités
plt.figure(figsize=(7,4))
plt.hist(y_proba[y_test==0], bins=30, alpha=0.7, label="BENIGN")
plt.hist(y_proba[y_test==1], bins=30, alpha=0.7, label="ATTACK")
plt.title("Distribution proba_attack (binaire)")
plt.xlabel("proba_attack")
plt.ylabel("Nb flows")
plt.legend()
plt.show()


# ============================================================
# 6) Simulation temps réel (Méthode A) - stream flows 1 par 1
# ============================================================
def stream_ids_realtime(X_test_scaled, y_test_series, model,
                        n=200, delay=0.2,
                        threshold=0.7, window=50, alert_rate=0.30, seed=42):
    """
    Stream de flows du TEST (simulation temps réel)
    - on mélange les indices
    - on affiche un flow à la fois
    - on calcule un taux d'attaque sur une fenêtre glissante
    """
    y = np.array(y_test_series)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    recent_preds = []
    alerts = 0

    print("\n==============================")
    print("🚨 IDS STREAM (simulation temps réel)")
    print("==============================")
    print(f"Seuil={threshold} | Fenêtre={window} | Alerte si taux>= {alert_rate:.0%}\n")

    for t, i in enumerate(idx[:n], 1):
        x = X_test_scaled[i].reshape(1, -1)
        proba = model.predict_proba(x)[0, 1]
        pred = int(proba >= threshold)

        true_label = "ATTACK" if y[i] == 1 else "BENIGN"
        pred_label = "🚨 ATTACK" if pred == 1 else "✅ BENIGN"

        recent_preds.append(pred)
        if len(recent_preds) > window:
            recent_preds.pop(0)

        attack_rate = sum(recent_preds) / len(recent_preds)

        print(f"[{t:03d}] idx={i:6d} | TRUE={true_label:<6} | PRED={pred_label:<9} | "
              f"proba_attack={proba:.3f} | last{len(recent_preds):02d}_attack_rate={attack_rate:.1%}")

        if len(recent_preds) == window and attack_rate >= alert_rate:
            alerts += 1
            print("🔥 SOC ALERT: Taux d'attaques élevé sur la fenêtre récente !")
            print("-"*70)

        time.sleep(delay)

    print("\n✅ Fin du stream. Alertes SOC:", alerts)


# Lancer le stream
stream_ids_realtime(
    X_test_scaled=X_test_s,
    y_test_series=y_test,
    model=model_bin,
    n=N_STREAM,
    delay=DELAY,
    threshold=THRESHOLD,
    window=WINDOW,
    alert_rate=ALERT_RATE,
    seed=RANDOM_STATE
)


# ============================================================
# 7) MULTI-CLASSES (type d’attaque)
# ============================================================
le = LabelEncoder()
y_multi = le.fit_transform(y_text)

print("\n==============================")
print("🧠 MULTI-CLASSES (TYPE D'ATTAQUE)")
print("==============================")
print("Nb classes:", len(le.classes_))
print("Classes:", list(le.classes_))

# Split multi
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X, y_multi, test_size=0.2, random_state=RANDOM_STATE, stratify=y_multi
)

# Scaling multi
scaler_multi = StandardScaler()
X_train_m_s = scaler_multi.fit_transform(X_train_m)
X_test_m_s  = scaler_multi.transform(X_test_m)

# Modèle multi-classes
rf_multi = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
rf_multi.fit(X_train_m_s, y_train_m)

y_pred_m = rf_multi.predict(X_test_m_s)

print("\n===== REPORT (MULTI-CLASSES) =====")
print(classification_report(y_test_m, y_pred_m, target_names=le.classes_))

# Confusion matrix multi-classes
cm_m = confusion_matrix(y_test_m, y_pred_m)
plt.figure(figsize=(10,10))
plt.imshow(cm_m, interpolation="nearest")
plt.title("Confusion Matrix - Multi-classes")
plt.colorbar()
plt.xticks(range(len(le.classes_)), le.classes_, rotation=90)
plt.yticks(range(len(le.classes_)), le.classes_)
plt.tight_layout()
plt.show()

# Exemple
i = 0
pred_class = rf_multi.predict(X_test_m_s[i].reshape(1,-1))[0]
print("\nExemple prédiction multi-classes:", le.inverse_transform([pred_class])[0])


# ============================================================
# 8) Sauvegarde modèles + scalers + encoder
# ============================================================
joblib.dump(model_bin, "ids_binary_model.pkl")
joblib.dump(scaler_bin, "ids_binary_scaler.pkl")

joblib.dump(rf_multi, "ids_multiclass_model.pkl")
joblib.dump(scaler_multi, "ids_multiclass_scaler.pkl")
joblib.dump(le, "ids_label_encoder.pkl")

print("\n✅ Modèles + scalers sauvegardés:")
print("- ids_binary_model.pkl / ids_binary_scaler.pkl")
print("- ids_multiclass_model.pkl / ids_multiclass_scaler.pkl / ids_label_encoder.pkl")


# ============================================================
# 9) (OPTION) Générer un fichier Flask app.py (local)
# ============================================================
APP_PY = r"""
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(_name_)

model = joblib.load("ids_binary_model.pkl")
scaler = joblib.load("ids_binary_scaler.pkl")

THRESHOLD = 0.70

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # attendu: {"features":[...]}
    feats = np.array(data["features"], dtype=float).reshape(1, -1)
    feats_s = scaler.transform(feats)

    proba = float(model.predict_proba(feats_s)[0,1])
    pred = int(proba >= THRESHOLD)

    return jsonify({
        "pred": pred,
        "label": "ATTACK" if pred==1 else "BENIGN",
        "proba_attack": proba
    })

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(APP_PY)

print("\n✅ Fichier app.py généré (API Flask locale).")
print("Pour lancer: python app.py")