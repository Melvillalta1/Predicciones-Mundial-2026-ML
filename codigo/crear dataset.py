"""
build_dataset_v3.py
===================
Pipeline completo — descarga TODO desde GitHub automáticamente.
No necesitas subir ningún archivo a Colab.

Fuentes:
  - martj42/international_results     → base de 49,000+ partidos
  - Melvillalta1/Predicciones-Mundial-2026-ML → tus archivos propios

Uso en Google Colab:
  !pip install pandas numpy requests -q
  !python build_dataset_v3.py

Salidas:
  dataset_mundial2026_v3.csv   — dataset enriquecido (37k+ filas, 39 columnas)
  elo_ratings_final_v3.csv     — ELO de cada selección al corte junio 2025
"""

import pandas as pd
import numpy as np
import requests
import io
import time

start = time.time()

# ══════════════════════════════════════════════════════════════
# URLs — todas las fuentes centralizadas aquí
# ══════════════════════════════════════════════════════════════
REPO = "https://raw.githubusercontent.com/Melvillalta1/Predicciones-Mundial-2026-ML/refs/heads/main"

URLS = {
    "results":      "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "goalscorers":  "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
    "former_names": f"{REPO}/former_names.csv",
    "wc50":         f"{REPO}/world_cup_last_50_years.csv",
    "fifa_rank_1":  f"{REPO}/fifa_ranking-2023-07-20.csv",
    "fifa_rank_2":  f"{REPO}/fifa_ranking-2024-04-04.csv",
    "fifa_rank_3":  f"{REPO}/fifa_ranking-2024-06-20.csv",
}

def descargar(nombre, url):
    """Descarga un CSV desde URL y retorna un DataFrame."""
    print(f"   Descargando {nombre}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# ══════════════════════════════════════════════════════════════
# PASO 1 — former_names (estandarización histórica de nombres)
# ══════════════════════════════════════════════════════════════
print("📋 PASO 1 — Nombres históricos")
former = descargar("former_names", URLS["former_names"])
former['start_date'] = pd.to_datetime(former['start_date'])
former['end_date']   = pd.to_datetime(former['end_date'])
print(f"   {len(former)} registros cargados")

def estandarizar_nombre(nombre, fecha, former_df):
    """
    Convierte nombre histórico al nombre actual del país.
    Ej: 'Zaïre' (1974) → 'DR Congo'
        'Netherlands Antilles' (2005) → 'Curaçao'
    """
    mask = (
        (former_df['former'] == nombre) &
        (former_df['start_date'] <= fecha) &
        (former_df['end_date']   >= fecha)
    )
    m = former_df[mask]
    return m.iloc[0]['current'] if len(m) > 0 else nombre

# Mapa estático para diferencias residuales entre martj42 y FIFA ranking
NAME_MAP = {
    "United States":          "USA",
    "South Korea":            "Korea Republic",
    "North Korea":            "Korea DPR",
    "Ivory Coast":            "Côte d'Ivoire",
    "Iran":                   "IR Iran",
    "Czech Republic":         "Czechia",
    "Macedonia":              "North Macedonia",
    "Swaziland":              "Eswatini",
    "China":                  "China PR",
    "Curacao":                "Curaçao",
    "Cape Verde":             "Cabo Verde",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Trinidad and Tobago":    "Trinidad and Tobago",
}

# Mapa adicional para hacer coincidir nombres con FIFA ranking
# (FIFA ranking usa nombres ligeramente distintos en algunos casos)
FIFA_NAME_MAP = {
    "Curaçao":              "Curacao",
    "Cabo Verde":           "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
}

# ══════════════════════════════════════════════════════════════
# PASO 2 — Dataset base de partidos (martj42)
# ══════════════════════════════════════════════════════════════
print("\n📥 PASO 2 — Dataset base de partidos (martj42)")
df = descargar("results", URLS["results"])
print(f"   {len(df):,} partidos descargados")

df['date'] = pd.to_datetime(df['date'])

print("   Estandarizando nombres históricos...")
df['home_team'] = df.apply(
    lambda r: estandarizar_nombre(r['home_team'], r['date'], former), axis=1
)
df['away_team'] = df.apply(
    lambda r: estandarizar_nombre(r['away_team'], r['date'], former), axis=1
)
df['home_team'] = df['home_team'].replace(NAME_MAP)
df['away_team'] = df['away_team'].replace(NAME_MAP)

df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
df = df.dropna(subset=['home_score', 'away_score'])
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

# Target: 2=local gana, 1=empate, 0=local pierde
df['result'] = np.where(
    df['home_score'] > df['away_score'], 2,
    np.where(df['home_score'] == df['away_score'], 1, 0)
)

df['is_wc'] = (
    df['tournament'].str.contains('FIFA World Cup', na=False) &
    ~df['tournament'].str.contains('qualif', case=False, na=False)
).astype(int)
df['is_qualifier'] = df['tournament'].str.contains('qualif', case=False, na=False).astype(int)
df['is_neutral']   = df['neutral'].astype(int)

df = df[df['date'] >= '1980-01-01'].sort_values('date').reset_index(drop=True)
print(f"   {len(df):,} partidos tras limpieza (desde 1980)")

# ══════════════════════════════════════════════════════════════
# PASO 3 — world_cup_last_50 (flag de stage / knockout)
# ══════════════════════════════════════════════════════════════
print("\n📥 PASO 3 — Stages de Mundiales")
wc50 = descargar("world_cup_last_50", URLS["wc50"])
wc50['date']      = pd.to_datetime(wc50['date'])
wc50['home_team'] = wc50['home_team'].replace(NAME_MAP)
wc50['away_team'] = wc50['away_team'].replace(NAME_MAP)
wc50['is_knockout'] = (
    ~wc50['stage'].str.contains('Group', case=False, na=False)
).astype(int)

df = df.merge(
    wc50[['date', 'home_team', 'away_team', 'stage', 'is_knockout']],
    on=['date', 'home_team', 'away_team'],
    how='left'
)
df['is_knockout'] = df['is_knockout'].fillna(0).astype(int)
df['stage']       = df['stage'].fillna('Regular')
print(f"   {(df['stage'] != 'Regular').sum():,} partidos con stage identificado")

# ══════════════════════════════════════════════════════════════
# PASO 4 — ELO rating (calculado secuencialmente)
# ══════════════════════════════════════════════════════════════
print("\n⚡ PASO 4 — Calculando ELO ratings...")

def k_factor(tournament, is_knockout=0):
    t = str(tournament).lower()
    if 'fifa world cup' in t and 'qualif' not in t:
        k = 60
    elif any(x in t for x in ['copa america', 'euro ', 'africa cup', 'asian cup']):
        k = 50
    elif 'qualif' in t:
        k = 40
    elif 'friendly' in t:
        k = 20
    else:
        k = 35
    return k * (1.2 if is_knockout else 1.0)

elo = {}
elo_home_list, elo_away_list = [], []

for row in df.itertuples():
    h, a  = row.home_team, row.away_team
    rh    = elo.get(h, 1500)
    ra    = elo.get(a, 1500)
    adj   = 100 if not row.is_neutral else 0
    exp_h = 1 / (1 + 10 ** ((ra - rh - adj) / 400))
    score = {2: 1.0, 1: 0.5, 0: 0.0}[row.result]
    k     = k_factor(row.tournament, row.is_knockout)
    delta = k * (score - exp_h)
    elo_home_list.append(rh)
    elo_away_list.append(ra)
    elo[h] = rh + delta
    elo[a] = ra - delta

df['elo_home'] = elo_home_list
df['elo_away'] = elo_away_list
df['elo_diff'] = df['elo_home'] - df['elo_away']

top5 = sorted(elo.items(), key=lambda x: -x[1])[:5]
print(f"   Top 5 ELO actual: { {k: round(v) for k,v in top5} }")

# ══════════════════════════════════════════════════════════════
# PASO 5 — Rolling features vectorizadas (rendimiento reciente)
# ══════════════════════════════════════════════════════════════
print("\n🔢 PASO 5 — Rolling features (rendimiento reciente)...")

# Expandir a formato largo: 2 filas por partido
home_r = df[['date','home_team','away_team','home_score','away_score',
             'result','is_wc','is_qualifier','is_knockout']].copy()
home_r.columns = ['date','team','opp','gf','ga','result_raw',
                  'is_wc','is_qualifier','is_knockout']
home_r['is_home'] = 1
home_r['win']     = (home_r['result_raw'] == 2).astype(float)
home_r['draw']    = (home_r['result_raw'] == 1).astype(float)

away_r = df[['date','away_team','home_team','away_score','home_score',
             'result','is_wc','is_qualifier','is_knockout']].copy()
away_r.columns = ['date','team','opp','gf','ga','result_raw',
                  'is_wc','is_qualifier','is_knockout']
away_r['is_home'] = 0
away_r['win']     = (away_r['result_raw'] == 0).astype(float)
away_r['draw']    = (away_r['result_raw'] == 1).astype(float)

long = pd.concat([home_r, away_r], ignore_index=True)
long = long.sort_values(['team', 'date']).reset_index(drop=True)
long['points']   = long['win'] + 0.5 * long['draw']
long['is_comp']  = ((long['is_wc'] == 1) | (long['is_qualifier'] == 1)).astype(float)
long['comp_pts'] = long['points'] * long['is_comp']

W = 15  # ventana de 15 partidos

def roll(col):
    return long.groupby('team')[col].transform(
        lambda x: x.shift(1).rolling(W, min_periods=3).mean()
    )

long['win_rate_r']       = roll('points').fillna(0.45)
long['goals_scored_r']   = roll('gf').fillna(1.2)
long['goals_conceded_r'] = roll('ga').fillna(1.2)
long['wc_win_rate_r']    = long.groupby('team')['comp_pts'].transform(
    lambda x: x.shift(1).rolling(W, min_periods=1).mean()
).fillna(0.40)

FEAT = ['win_rate_r', 'goals_scored_r', 'goals_conceded_r', 'wc_win_rate_r']

sh = long[long['is_home'] == 1][['date', 'team'] + FEAT].copy()
sh.columns = ['date', 'home_team',
              'win_rate_home', 'goals_scored_home',
              'goals_conceded_home', 'wc_win_rate_home']

sa = long[long['is_home'] == 0][['date', 'team'] + FEAT].copy()
sa.columns = ['date', 'away_team',
              'win_rate_away', 'goals_scored_away',
              'goals_conceded_away', 'wc_win_rate_away']

df = df.merge(sh, on=['date', 'home_team'], how='left')
df = df.merge(sa, on=['date', 'away_team'], how='left')
print(f"   Rolling features calculadas para {long['team'].nunique()} equipos")

# ══════════════════════════════════════════════════════════════
# PASO 6 — Features de goles (goalscorers.csv)
# ══════════════════════════════════════════════════════════════
print("\n⚽ PASO 6 — Features de goles (goalscorers.csv)...")
gs = descargar("goalscorers", URLS["goalscorers"])
gs['date'] = pd.to_datetime(gs['date'])
gs['team'] = gs['team'].replace(NAME_MAP)

# Goles promedio rolling por equipo
gp = gs.groupby(['date', 'team']).size().reset_index(name='gn')
gp = gp.sort_values(['team', 'date'])
gp['goals_rolling'] = gp.groupby('team')['gn'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=3).mean()
).fillna(1.2)

# % goles de penal rolling por equipo
pen  = gs.groupby(['date', 'team'])['penalty'].sum().reset_index(name='pn')
tot  = gs.groupby(['date', 'team']).size().reset_index(name='tg')
pr   = pen.merge(tot, on=['date', 'team'])
pr['pp'] = pr['pn'] / pr['tg']
pr = pr.sort_values(['team', 'date'])
pr['pen_pct_rolling'] = pr.groupby('team')['pp'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=3).mean()
).fillna(0.12)

# Merge al dataset principal (local y visitante)
gh = gp[['date','team','goals_rolling']].rename(
    columns={'team':'home_team','goals_rolling':'goals_rolling_home'})
ga_ = gp[['date','team','goals_rolling']].rename(
    columns={'team':'away_team','goals_rolling':'goals_rolling_away'})
ph = pr[['date','team','pen_pct_rolling']].rename(
    columns={'team':'home_team','pen_pct_rolling':'pen_pct_home'})
pa = pr[['date','team','pen_pct_rolling']].rename(
    columns={'team':'away_team','pen_pct_rolling':'pen_pct_away'})

df = df.merge(gh, on=['date','home_team'], how='left')
df = df.merge(ph, on=['date','home_team'], how='left')
df = df.merge(ga_, on=['date','away_team'], how='left')
df = df.merge(pa,  on=['date','away_team'], how='left')

df['goals_rolling_home'] = df['goals_rolling_home'].fillna(1.2)
df['goals_rolling_away'] = df['goals_rolling_away'].fillna(1.2)
df['pen_pct_home']       = df['pen_pct_home'].fillna(0.12)
df['pen_pct_away']       = df['pen_pct_away'].fillna(0.12)
print(f"   Features de goles añadidas")

# ══════════════════════════════════════════════════════════════
# PASO 7 — Ranking FIFA histórico (merge_asof por fecha)
# ══════════════════════════════════════════════════════════════
print("\n🏅 PASO 7 — Ranking FIFA histórico...")

rankings = []
for key in ['fifa_rank_1', 'fifa_rank_2', 'fifa_rank_3']:
    try:
        r = descargar(key, URLS[key])
        r['rank_date'] = pd.to_datetime(r['rank_date'])
        rankings.append(r)
        print(f"   {key}: {len(r):,} filas | "
              f"{r['rank_date'].min().date()} → {r['rank_date'].max().date()}")
    except Exception as e:
        print(f"   Error cargando {key}: {e}")

if rankings:
    fifa = pd.concat(rankings, ignore_index=True)
    fifa = fifa.drop_duplicates(subset=['country_full', 'rank_date'])
    fifa = fifa.sort_values('rank_date').reset_index(drop=True)

    df_s = df.sort_values('date').reset_index(drop=True)

    for side, tcol, r_col, p_col, c_col in [
        ('home', 'home_team', 'fifa_rank_home', 'fifa_pts_home', 'conf_home'),
        ('away', 'away_team', 'fifa_rank_away', 'fifa_pts_away', 'conf_away'),
    ]:
        resultados = []
        for equipo in df_s[tcol].unique():
            nombre_fifa = FIFA_NAME_MAP.get(equipo, equipo)
            sub_r = fifa[fifa['country_full'] == nombre_fifa][
                ['rank_date', 'rank', 'total_points', 'confederation']
            ].rename(columns={
                'rank_date':     'date',
                'rank':          r_col,
                'total_points':  p_col,
                'confederation': c_col,
            }).sort_values('date')

            sub_d = df_s[df_s[tcol] == equipo][['date', tcol]].sort_values('date')

            if len(sub_r) == 0:
                sub_d[r_col] = np.nan
                sub_d[p_col] = np.nan
                sub_d[c_col] = np.nan
            else:
                m = pd.merge_asof(sub_d, sub_r, on='date', direction='backward')
                sub_d[r_col] = m[r_col].values
                sub_d[p_col] = m[p_col].values
                sub_d[c_col] = m[c_col].values

            resultados.append(sub_d)

        res = pd.concat(resultados).sort_index()
        df_s[r_col] = res[r_col].values
        df_s[p_col] = res[p_col].values
        df_s[c_col] = res[c_col].values

    df = df_s.copy()

    # Diferencias de ranking
    df['fifa_rank_diff'] = df['fifa_rank_away'] - df['fifa_rank_home']
    df['fifa_pts_diff']  = df['fifa_pts_home']  - df['fifa_pts_away']
    df['same_conf']      = (df['conf_home'] == df['conf_away']).astype(int)

    # Rellenar nulos (partidos anteriores a 1992, equipos sin ranking)
    df['fifa_rank_home'] = df['fifa_rank_home'].fillna(100)
    df['fifa_rank_away'] = df['fifa_rank_away'].fillna(100)
    df['fifa_rank_diff'] = df['fifa_rank_diff'].fillna(0)
    df['fifa_pts_diff']  = df['fifa_pts_diff'].fillna(0)
    df['same_conf']      = df['same_conf'].fillna(0).astype(int)

    cob = (df['fifa_rank_home'] != 100).mean() * 100
    print(f"   Ranking FIFA integrado. Cobertura: {cob:.1f}% de partidos")
else:
    print("   Sin archivos de ranking FIFA disponibles")
    for col in ['fifa_rank_home','fifa_rank_away','fifa_rank_diff',
                'fifa_pts_home','fifa_pts_away','fifa_pts_diff',
                'conf_home','conf_away','same_conf']:
        df[col] = np.nan

# ══════════════════════════════════════════════════════════════
# PASO 8 — Decay temporal y features de diferencia
# ══════════════════════════════════════════════════════════════
print("\n📉 PASO 8 — Decay temporal y features de diferencia...")
FECHA_CORTE   = pd.Timestamp('2025-06-01')
HALF_LIFE_YRS = 4

df['decay_weight'] = np.exp(
    -np.log(2) * (FECHA_CORTE - df['date']).dt.days / (365.25 * HALF_LIFE_YRS)
).clip(0.05, 1.0).round(4)

df['win_rate_diff'] = df['win_rate_home'] - df['win_rate_away']
df['goals_diff']    = df['goals_scored_home'] - df['goals_scored_away']
df['elo_diff']      = df['elo_home'] - df['elo_away']

# ══════════════════════════════════════════════════════════════
# PASO 9 — Selección de columnas y guardado
# ══════════════════════════════════════════════════════════════
print("\n💾 PASO 9 — Guardando dataset...")
COLS = [
    # Identificadores
    'date', 'home_team', 'away_team', 'tournament', 'stage',
    # ELO calculado
    'elo_home', 'elo_away', 'elo_diff',
    # Ranking FIFA
    'fifa_rank_home', 'fifa_rank_away', 'fifa_rank_diff',
    'fifa_pts_home', 'fifa_pts_away', 'fifa_pts_diff',
    'conf_home', 'conf_away', 'same_conf',
    # Rendimiento reciente rolling (15 partidos)
    'win_rate_home', 'goals_scored_home', 'goals_conceded_home', 'wc_win_rate_home',
    'win_rate_away', 'goals_scored_away', 'goals_conceded_away', 'wc_win_rate_away',
    # Features de goles (goalscorers.csv)
    'goals_rolling_home', 'goals_rolling_away',
    'pen_pct_home', 'pen_pct_away',
    # Diferencias
    'win_rate_diff', 'goals_diff',
    # Contexto del partido
    'is_neutral', 'is_wc', 'is_qualifier', 'is_knockout',
    # Decay temporal
    'decay_weight',
    # Scores reales — solo EDA, NO usar como features del modelo
    'home_score', 'away_score',
    # TARGET
    'result',
]

COLS    = [c for c in COLS if c in df.columns]
dataset = df[COLS].copy()
num_c   = dataset.select_dtypes(include=[np.number]).columns
dataset[num_c] = dataset[num_c].round(4)

dataset.to_csv("dataset_mundial2026_v3.csv", index=False)

df_elo = pd.DataFrame(
    sorted(elo.items(), key=lambda x: -x[1]),
    columns=['team', 'elo_rating']
).round(1)
df_elo.to_csv("elo_ratings_final_v3.csv", index=False)

elapsed = time.time() - start

print(f"""
{'='*55}
✅ Pipeline completado en {elapsed:.1f} segundos

📊 dataset_mundial2026_v3.csv
   Filas:    {len(dataset):,}
   Columnas: {len(dataset.columns)}
   Período:  {dataset['date'].min().date()} → {dataset['date'].max().date()}

🎯 Distribución del target (result):
   Local gana   (2): {(dataset['result']==2).sum():,}  ({(dataset['result']==2).mean()*100:.1f}%)
   Empate       (1): {(dataset['result']==1).sum():,}  ({(dataset['result']==1).mean()*100:.1f}%)
   Local pierde (0): {(dataset['result']==0).sum():,}  ({(dataset['result']==0).mean()*100:.1f}%)

🏅 Top 10 ELO actual:
{df_elo.head(10).to_string(index=False)}

⚠️  IMPORTANTE: para eliminar el modelo eliminaremos home_score y away_score
   antes de entrenar el modelo.
{'='*55}
""")