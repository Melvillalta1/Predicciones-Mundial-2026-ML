"""
train_model.py
==============
Entrena XGBoost y Regresión Logística para predecir resultado de partidos.
División temporal — NO aleatoria (es una serie de tiempo).

  Train : hasta dic 2021  (32,751 partidos)
  Val   : 2022-2023       ( 2,023 partidos — incluye WC2022 para validación histórica)
  Test  : 2024-2025       ( 2,395 partidos — evaluación final honesta)

Métricas usadas: Log-Loss y Brier Score (requeridas por la rúbrica).
NO se usa Accuracy como métrica principal.

Salidas:
  modelo_xgb.pkl          — modelo XGBoost entrenado
  modelo_lr.pkl           — modelo Regresión Logística entrenado
  metricas_modelos.csv    — comparativa de métricas

Uso en Colab:
  !pip install pandas numpy scikit-learn xgboost joblib requests -q
  !python train_model.py
"""

import pandas as pd
import numpy as np
import requests, io, joblib, time
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics        import log_loss, brier_score_loss
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

start = time.time()

# ══════════════════════════════════════════════════════════════
# 1 — CARGAR DATOS
# ══════════════════════════════════════════════════════════════
print("📥 Cargando dataset...")
URL_DATASET = "https://raw.githubusercontent.com/Melvillalta1/Predicciones-Mundial-2026-ML/refs/heads/main/dataset_modelo_entrenamiento.csv"

try:
    r = requests.get(URL_DATASET, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=['date'])
    print(f"   Cargado desde GitHub: {len(df):,} filas")
except Exception as e:
    print(f"   GitHub no disponible ({e}), intentando local...")
    df = pd.read_csv("dataset_modelo_entrenamiento.csv", parse_dates=['date'])
    print(f"   Cargado local: {len(df):,} filas")

# ══════════════════════════════════════════════════════════════
# 2 — FEATURES Y TARGET
# ══════════════════════════════════════════════════════════════
FEATURES = [
    'elo_home', 'elo_away', 'elo_diff',
    'fifa_rank_home', 'fifa_rank_away', 'fifa_rank_diff',
    'fifa_pts_home',  'fifa_pts_away',  'fifa_pts_diff',
    'win_rate_home',    'goals_scored_home',    'goals_conceded_home',    'wc_win_rate_home',
    'win_rate_away',    'goals_scored_away',    'goals_conceded_away',    'wc_win_rate_away',
    'goals_rolling_home', 'goals_rolling_away',
    'pen_pct_home',       'pen_pct_away',
    'win_rate_diff', 'goals_diff',
    'is_neutral', 'is_wc', 'is_qualifier', 'is_knockout',
    'decay_weight',
]
TARGET = 'result'

# ══════════════════════════════════════════════════════════════
# 3 — DIVISIÓN TEMPORAL (NO aleatoria)
# ══════════════════════════════════════════════════════════════
train = df[df['date'] <  '2022-01-01']
val   = df[(df['date'] >= '2022-01-01') & (df['date'] < '2024-01-01')]
test  = df[df['date'] >= '2024-01-01']

X_train = train[FEATURES];  y_train = train[TARGET]
X_val   = val[FEATURES];    y_val   = val[TARGET]
X_test  = test[FEATURES];   y_test  = test[TARGET]

# Pesos de entrenamiento (decay_weight — partidos recientes pesan más)
w_train = train['decay_weight'].values

print(f"\n📊 División temporal:")
print(f"   Train : {len(X_train):,} partidos (hasta dic 2021)")
print(f"   Val   : {len(X_val):,}  partidos (2022-2023)")
print(f"   Test  : {len(X_test):,}  partidos (2024-2025)")
print(f"   WC2022 en val: {((val['is_wc']==1)).sum()} partidos")

print(f"\n   Distribución target (train):")
for v, lbl in [(0,'Pierde'),(1,'Empate'),(2,'Gana')]:
    n = (y_train==v).sum()
    print(f"     {lbl}: {n:,} ({n/len(y_train)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# 4 — MODELO 1: XGBoost
# ══════════════════════════════════════════════════════════════
print("\n⚡ Entrenando XGBoost...")

xgb = XGBClassifier(
    n_estimators     = 500,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,    # L1 regularización
    reg_lambda       = 1.0,    # L2 regularización
    objective        = 'multi:softprob',
    num_class        = 3,
    eval_metric      = 'mlogloss',
    early_stopping_rounds = 30,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

xgb.fit(
    X_train, y_train,
    sample_weight    = w_train,
    eval_set         = [(X_val, y_val)],
    verbose          = False,
)

# Probabilidades
prob_xgb_val  = xgb.predict_proba(X_val)
prob_xgb_test = xgb.predict_proba(X_test)

# Métricas XGBoost
ll_xgb_val  = log_loss(y_val,  prob_xgb_val)
ll_xgb_test = log_loss(y_test, prob_xgb_test)

# Brier Score multiclase (promedio de las 3 clases)
def brier_multi(y_true, y_prob):
    scores = []
    for c in range(3):
        scores.append(brier_score_loss((y_true==c).astype(int), y_prob[:,c]))
    return np.mean(scores)

bs_xgb_val  = brier_multi(y_val,  prob_xgb_val)
bs_xgb_test = brier_multi(y_test, prob_xgb_test)

print(f"   XGBoost — mejores árboles: {xgb.best_iteration}")
print(f"   Log-Loss  val: {ll_xgb_val:.4f}  |  test: {ll_xgb_test:.4f}")
print(f"   BrierScore val: {bs_xgb_val:.4f}  |  test: {bs_xgb_test:.4f}")

# ══════════════════════════════════════════════════════════════
# 5 — MODELO 2: Regresión Logística
# ══════════════════════════════════════════════════════════════
print("\n📐 Entrenando Regresión Logística...")

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),          # necesario para LR
    ('lr', LogisticRegression(
        multi_class = 'multinomial',
        solver      = 'lbfgs',
        max_iter    = 1000,
        C           = 0.5,                 # regularización moderada
        random_state= 42,
    ))
])

lr_pipe.fit(X_train, y_train, lr__sample_weight=w_train)

prob_lr_val  = lr_pipe.predict_proba(X_val)
prob_lr_test = lr_pipe.predict_proba(X_test)

ll_lr_val  = log_loss(y_val,  prob_lr_val)
ll_lr_test = log_loss(y_test, prob_lr_test)
bs_lr_val  = brier_multi(y_val,  prob_lr_val)
bs_lr_test = brier_multi(y_test, prob_lr_test)

print(f"   Log-Loss  val: {ll_lr_val:.4f}  |  test: {ll_lr_test:.4f}")
print(f"   BrierScore val: {bs_lr_val:.4f}  |  test: {bs_lr_test:.4f}")

# ══════════════════════════════════════════════════════════════
# 6 — COMPARATIVA Y FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
print("\n📊 Comparativa de modelos:")
print(f"{'Modelo':<22} {'LogLoss Val':>12} {'LogLoss Test':>13} {'Brier Val':>10} {'Brier Test':>11}")
print("-"*70)
print(f"{'XGBoost':<22} {ll_xgb_val:>12.4f} {ll_xgb_test:>13.4f} {bs_xgb_val:>10.4f} {bs_xgb_test:>11.4f}")
print(f"{'Logistic Regression':<22} {ll_lr_val:>12.4f} {ll_lr_test:>13.4f} {bs_lr_val:>10.4f} {bs_lr_test:>11.4f}")

# ══════════════════════════════════════════════════════════════
# 7 — VALIDACIÓN HISTÓRICA WC2022
# ══════════════════════════════════════════════════════════════
print("\n🏆 Validación histórica — Mundial 2022:")
wc22 = val[val['is_wc']==1]
if len(wc22) > 0:
    X_wc22  = wc22[FEATURES]
    y_wc22  = wc22[TARGET]
    p_wc22  = xgb.predict_proba(X_wc22)
    ll_wc22 = log_loss(y_wc22, p_wc22)
    bs_wc22 = brier_multi(y_wc22, p_wc22)
    acc_wc22 = (xgb.predict(X_wc22) == y_wc22).mean()
    print(f"   Partidos WC2022:  {len(wc22)}")
    print(f"   Log-Loss:         {ll_wc22:.4f}")
    print(f"   Brier Score:      {bs_wc22:.4f}")
    print(f"   Accuracy (ref):   {acc_wc22:.2%}  (no es la métrica principal)")

# ══════════════════════════════════════════════════════════════
# 8 — FEATURE IMPORTANCE (XGBoost)
# ══════════════════════════════════════════════════════════════
print("\n🔍 Top 10 features más importantes (XGBoost):")
fi = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.head(10).items():
    bar = '█' * int(imp * 200)
    print(f"   {feat:<30} {imp:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════
# 9 — GUARDAR MODELOS
# ══════════════════════════════════════════════════════════════
joblib.dump(xgb,     'modelo_xgb.pkl')
joblib.dump(lr_pipe, 'modelo_lr.pkl')

metricas = pd.DataFrame([
    {'modelo':'XGBoost',             'logloss_val':ll_xgb_val, 'logloss_test':ll_xgb_test,
     'brier_val':bs_xgb_val,         'brier_test':bs_xgb_test},
    {'modelo':'Logistic Regression', 'logloss_val':ll_lr_val,  'logloss_test':ll_lr_test,
     'brier_val':bs_lr_val,          'brier_test':bs_lr_test},
])
metricas.to_csv('metricas_modelos.csv', index=False)

elapsed = time.time() - start
print(f"""
{'='*55}
✅ Modelos entrenados en {elapsed:.1f} segundos

Archivos generados:
  modelo_xgb.pkl          — XGBoost (usar en Monte Carlo)
  modelo_lr.pkl           — Regresión Logística
  metricas_modelos.csv    — comparativa

⚠️  El modelo XGBoost será el usado en la simulación
   Monte Carlo por tener mejor Log-Loss generalmente.
{'='*55}
""")

print("📌 Interpretación de métricas:")
print("   Log-Loss  < 1.0  → modelo útil")
print("   Log-Loss  ~ 1.0  → similar a adivinar al azar")
print("   Brier     < 0.25 → buena calibración de probabilidades")
print("   (Un partido de fútbol es inherentemente impredecible,")
print("    Log-Loss ~0.95 ya es competitivo con modelos profesionales)")