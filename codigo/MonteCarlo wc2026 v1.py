"""
monte_carlo_wc2026.py
=====================
Simulación Monte Carlo — Mundial FIFA 2026 (48 equipos, 12 grupos).

Fuentes de datos (todas desde GitHub):
  - future_match_probabilities_baseline.csv  → grupos reales + ELO oficial
  - elo_ratings_final.csv                    → ELO calculado por nuestro modelo
  - dataset_modelo_entrenamiento.csv         → rolling stats por equipo
  - modelo_xgb.pkl                           → modelo entrenado (subir a Colab)

NOTA sobre modelo_xgb.pkl:
Es necesario entrenar el modelo para que se genere el archivo modelo_xgb.pkl o en su defecto subirlo a google drive. 
Uso en Colab:
  !pip install pandas numpy xgboost joblib requests -q
  # Subir modelo_xgb.pkl (ver opciones arriba), luego:
  !monte_carlo_wc2026.py (este escript)

Salidas:
  resultados_montecarlo.csv   — prob. de campeonato Top 48 con IC 95%
  clasificados_octavos.csv    — prob. de clasificar a octavos por equipo
  simulacion_resumen.txt      — resumen ejecutivo para el reporte
"""

import pandas as pd
import numpy as np
import joblib
import requests
import io
import time
from collections import defaultdict

np.random.seed(42)
start = time.time()

REPO = "https://raw.githubusercontent.com/Melvillalta1/Predicciones-Mundial-2026-ML/refs/heads/main"

def fetch_csv(url, parse_dates=None):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), parse_dates=parse_dates)

# ══════════════════════════════════════════════════════════════
# 1 — CARGAR MODELO
# ══════════════════════════════════════════════════════════════
print("📦 Cargando modelo XGBoost...")
print("   (Asegúrate de haber subido modelo_xgb.pkl a Colab)")
try:
    modelo = joblib.load("modelo_xgb.pkl")
    FEATURES = list(modelo.feature_names_in_)
    print(f"   ✅ Modelo cargado: {modelo.n_features_in_} features")
except FileNotFoundError:
    print("""
❌ No se encontró modelo_xgb.pkl

Opciones para cargarlo:
  1. Panel lateral de Colab → Files → Upload → selecciona modelo_xgb.pkl
  2. Desde Google Drive:
       from google.colab import drive
       drive.mount('/content/drive')
       import shutil
       shutil.copy('/content/drive/MyDrive/modelo_xgb.pkl', 'modelo_xgb.pkl')
  3. Volver a entrenar:
       !python train_model.py
""")
    raise

# ══════════════════════════════════════════════════════════════
# 2 — CARGAR GRUPOS Y ELO DESDE future_match_probabilities_baseline
# ══════════════════════════════════════════════════════════════
print("\n📥 Cargando grupos y ELO desde GitHub...")
probs_base = fetch_csv(f"{REPO}/future_match_probabilities_baseline.csv")

# Extraer ELO por equipo desde el archivo
elo_base = {}
for _, row in probs_base.iterrows():
    elo_base[row['home_team']] = row['home_elo']
    if pd.notna(row['away_elo']):
        elo_base[row['away_team']] = float(row['away_elo'])

# Extraer grupos reales (preservar el sorteo oficial)
GROUPS_RAW = {}
for g in sorted(probs_base['group'].unique()):
    sub = probs_base[probs_base['group'] == g]
    equipos_g = sorted(set(list(sub['home_team']) + list(sub['away_team'])))
    GROUPS_RAW[g] = equipos_g

print(f"   Grupos cargados: {len(GROUPS_RAW)} grupos, "
      f"{sum(len(v) for v in GROUPS_RAW.values())} equipos")

# ══════════════════════════════════════════════════════════════
# 3 — CARGAR ELO CALCULADO (nuestro modelo es más preciso)
# ══════════════════════════════════════════════════════════════
print("📥 Cargando ELO calculado...")
try:
    elo_df = fetch_csv(f"{REPO}/elo_ratings_final.csv")
    elo_calc = dict(zip(elo_df['team'], elo_df['elo_rating']))
    print(f"   ELO calculado: {len(elo_calc)} equipos")
except Exception as e:
    print(f"   Usando ELO del baseline: {e}")
    elo_calc = {}

# Mapa de nombres entre fuentes (futuro_match usa nombres distintos en algunos casos)
NAME_BRIDGE = {
    "United States": "USA",
    "South Korea":   "Korea Republic",
    "Iran":          "IR Iran",
    "Cape_Verde":    "Cabo Verde",
}

def get_elo(team):
    """ELO: preferir el calculado por nuestro modelo, fallback al baseline."""
    name_calc = NAME_BRIDGE.get(team, team)
    return elo_calc.get(name_calc, elo_base.get(team, 1500))

# ══════════════════════════════════════════════════════════════
# 4 — CARGAR ROLLING STATS DESDE EL DATASET DE ENTRENAMIENTO
#     Calculamos las stats más recientes (2024-2025) por equipo
# ══════════════════════════════════════════════════════════════
print("📥 Calculando rolling stats recientes desde dataset...")
try:
    ds = fetch_csv(f"{REPO}/dataset_modelo_entrenamiento.csv", parse_dates=['date'])
    ds_recent = ds[ds['date'] >= '2023-01-01'].copy()

    # Promediar features recientes por equipo (como local Y visitante)
    stats = {}
    todos_equipos_ds = set(ds_recent['home_team']) | set(ds_recent['away_team'])

    for eq in todos_equipos_ds:
        home_m = ds_recent[ds_recent['home_team'] == eq]
        away_m = ds_recent[ds_recent['away_team'] == eq]

        wr_vals, gs_vals, gc_vals, wcwr_vals, gr_vals, pp_vals = [], [], [], [], [], []

        if len(home_m) > 0:
            wr_vals.append(home_m['win_rate_home'].mean())
            gs_vals.append(home_m['goals_scored_home'].mean())
            gc_vals.append(home_m['goals_conceded_home'].mean())
            wcwr_vals.append(home_m['wc_win_rate_home'].mean())
            gr_vals.append(home_m['goals_rolling_home'].mean())
            pp_vals.append(home_m['pen_pct_home'].mean())

        if len(away_m) > 0:
            wr_vals.append(away_m['win_rate_away'].mean())
            gs_vals.append(away_m['goals_scored_away'].mean())
            gc_vals.append(away_m['goals_conceded_away'].mean())
            wcwr_vals.append(away_m['wc_win_rate_away'].mean())
            gr_vals.append(away_m['goals_rolling_away'].mean())
            pp_vals.append(away_m['pen_pct_away'].mean())

        if wr_vals:
            stats[eq] = {
                "wr":   np.mean(wr_vals),
                "gs":   np.mean(gs_vals),
                "gc":   np.mean(gc_vals),
                "wcwr": np.mean(wcwr_vals),
                "gr":   np.mean(gr_vals),
                "pp":   np.mean(pp_vals),
            }

    print(f"   Rolling stats calculadas para {len(stats)} equipos")
except Exception as e:
    print(f"   No se pudo cargar dataset: {e}. Usando valores por defecto.")
    stats = {}

# ══════════════════════════════════════════════════════════════
# 5 — CARGAR RANKING FIFA MÁS RECIENTE
# ══════════════════════════════════════════════════════════════
print("📥 Cargando ranking FIFA más reciente...")
try:
    fifa = fetch_csv(f"{REPO}/fifa_ranking-2024-06-20.csv")
    fifa_rank_dict = dict(zip(fifa['country_full'], fifa['rank']))
    fifa_pts_dict  = dict(zip(fifa['country_full'], fifa['total_points']))
    print(f"   FIFA ranking: {len(fifa_rank_dict)} equipos")
except Exception as e:
    print(f"   Sin ranking FIFA disponible: {e}")
    fifa_rank_dict = {}
    fifa_pts_dict  = {}

# ══════════════════════════════════════════════════════════════
# 6 — FUNCIÓN GET_FEATURES: construye el vector de features por equipo
# ══════════════════════════════════════════════════════════════

# Valores por defecto para equipos sin datos (playoffs aún no definidos)
DEFAULT = {
    "wr":0.42, "gs":1.25, "gc":1.35, "wcwr":0.32, "gr":1.20, "pp":0.09
}
DEFAULT_RANK = 80
DEFAULT_PTS  = 1200

def get_team_features(team):
    """Retorna dict con todas las features necesarias para el modelo."""
    # ELO
    elo_val = get_elo(team)

    # FIFA ranking (intentar nombre directo y variantes)
    name_fifa = NAME_BRIDGE.get(team, team)
    rank = fifa_rank_dict.get(name_fifa, fifa_rank_dict.get(team, DEFAULT_RANK))
    pts  = fifa_pts_dict.get(name_fifa,  fifa_pts_dict.get(team,  DEFAULT_PTS))

    # Rolling stats (intentar nombre directo y variantes)
    name_ds = NAME_BRIDGE.get(team, team)
    s = stats.get(name_ds, stats.get(team, DEFAULT))

    return {
        "elo":      elo_val,
        "fifa_rank": rank,
        "fifa_pts":  pts,
        "wr":        s["wr"],
        "gs":        s["gs"],
        "gc":        s["gc"],
        "wcwr":      s["wcwr"],
        "gr":        s["gr"],
        "pp":        s["pp"],
    }

# ══════════════════════════════════════════════════════════════
# 7 — FUNCIÓN PREDICT_MATCH
# ══════════════════════════════════════════════════════════════
def predict_match(team_a, team_b, is_knockout=0):
    """
    Predice probabilidades del partido usando el modelo XGBoost.
    Retorna (p_a_gana, p_empate, p_b_gana).
    Todos los partidos del WC2026 son en cancha neutral.
    """
    fa = get_team_features(team_a)
    fb = get_team_features(team_b)

    row = {
        "elo_home":            fa["elo"],
        "elo_away":            fb["elo"],
        "elo_diff":            fa["elo"] - fb["elo"],
        "fifa_rank_home":      fa["fifa_rank"],
        "fifa_rank_away":      fb["fifa_rank"],
        "fifa_rank_diff":      fb["fifa_rank"] - fa["fifa_rank"],
        "fifa_pts_home":       fa["fifa_pts"],
        "fifa_pts_away":       fb["fifa_pts"],
        "fifa_pts_diff":       fa["fifa_pts"] - fb["fifa_pts"],
        "win_rate_home":       fa["wr"],
        "goals_scored_home":   fa["gs"],
        "goals_conceded_home": fa["gc"],
        "wc_win_rate_home":    fa["wcwr"],
        "win_rate_away":       fb["wr"],
        "goals_scored_away":   fb["gs"],
        "goals_conceded_away": fb["gc"],
        "wc_win_rate_away":    fb["wcwr"],
        "goals_rolling_home":  fa["gr"],
        "goals_rolling_away":  fb["gr"],
        "pen_pct_home":        fa["pp"],
        "pen_pct_away":        fb["pp"],
        "win_rate_diff":       fa["wr"] - fb["wr"],
        "goals_diff":          fa["gs"] - fb["gs"],
        "is_neutral":          1,
        "is_wc":               1,
        "is_qualifier":        0,
        "is_knockout":         is_knockout,
        "decay_weight":        1.0,
    }

    X = pd.DataFrame([row])[FEATURES]
    probs = modelo.predict_proba(X)[0]
    # modelo: 0=local pierde, 1=empate, 2=local gana
    return probs[2], probs[1], probs[0]

def sortear(p_a, p_draw, p_b):
    """Sorteo ponderado. Retorna 'A', 'draw' o 'B'."""
    r = np.random.random()
    if r < p_a:           return "A"
    elif r < p_a + p_draw: return "draw"
    else:                  return "B"

def eliminar(team_a, team_b):
    """Partido eliminatorio: en empate → penales (50/50)."""
    pa, pd_, pb = predict_match(team_a, team_b, is_knockout=1)
    res = sortear(pa, pd_, pb)
    if res == "A":    return team_a
    elif res == "B":  return team_b
    else:             return team_a if np.random.random() < 0.5 else team_b

# ══════════════════════════════════════════════════════════════
# 8 — SIMULAR FASE DE GRUPOS
# ══════════════════════════════════════════════════════════════
def simular_grupo(equipos):
    """
    Round-robin de 4 equipos (6 partidos).
    Retorna ranking ordenado y tabla de posiciones.
    """
    tabla = {e: {"pts":0,"gf":0,"ga":0} for e in equipos}

    for i in range(len(equipos)):
        for j in range(i+1, len(equipos)):
            ea, eb = equipos[i], equipos[j]

            # Probabilidades del modelo
            pa, pd_, pb = predict_match(ea, eb, is_knockout=0)
            res = sortear(pa, pd_, pb)

            # Goles aproximados por Poisson (coherentes con el resultado)
            ga = get_team_features(ea)
            gb = get_team_features(eb)
            g_ea = max(0, int(np.random.poisson(ga["gs"] * 0.85)))
            g_eb = max(0, int(np.random.poisson(gb["gs"] * 0.85)))

            if res == "A":
                g_ea = max(g_ea, g_eb + 1)
                tabla[ea]["pts"] += 3
            elif res == "B":
                g_eb = max(g_eb, g_ea + 1)
                tabla[eb]["pts"] += 3
            else:
                g_ea = g_eb = min(g_ea, g_eb)
                tabla[ea]["pts"] += 1
                tabla[eb]["pts"] += 1

            tabla[ea]["gf"] += g_ea; tabla[ea]["ga"] += g_eb
            tabla[eb]["gf"] += g_eb; tabla[eb]["ga"] += g_ea

    ranking = sorted(
        equipos,
        key=lambda e: (tabla[e]["pts"], tabla[e]["gf"]-tabla[e]["ga"], tabla[e]["gf"]),
        reverse=True
    )
    return ranking, tabla

def simular_fase_grupos(grupos):
    """
    Simula los 12 grupos.
    Top 2 de cada grupo clasifican directamente (24 equipos).
    8 mejores terceros también clasifican (total 32 a octavos).
    """
    clasificados = {}   # grupo → [1ro, 2do]
    pool_terceros = []  # (equipo, pts, dg, gf, grupo)

    for letra, equipos in grupos.items():
        ranking, tabla = simular_grupo(equipos)
        clasificados[letra] = ranking[:2]
        tercero = ranking[2]
        pool_terceros.append((
            tercero,
            tabla[tercero]["pts"],
            tabla[tercero]["gf"] - tabla[tercero]["ga"],
            tabla[tercero]["gf"],
            letra
        ))

    # Los 8 mejores terceros (por pts → dg → gf)
    pool_terceros.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    mejores_terceros = [t[0] for t in pool_terceros[:8]]

    return clasificados, mejores_terceros

# ══════════════════════════════════════════════════════════════
# 9 — ARMAR OCTAVOS Y SIMULAR ELIMINATORIA
# ══════════════════════════════════════════════════════════════
def armar_octavos(clasificados, terceros):
    """
    32 equipos → 16 partidos de octavos.
    Cruces: 1ro de grupo X vs 2do de grupo Y (estructura FIFA 2026).
    Los 8 terceros completan el cuadro.
    """
    grupos = sorted(clasificados.keys())
    primeros = [clasificados[g][0] for g in grupos]   # 12 primeros
    segundos = [clasificados[g][1] for g in grupos]   # 12 segundos

    # 12 cruces directos (1ro vs 2do de grupos distintos)
    partidos = []
    mitad = len(grupos) // 2
    for i in range(mitad):
        partidos.append((primeros[i], segundos[mitad + i]))
    for i in range(mitad):
        partidos.append((primeros[mitad + i], segundos[i]))

    # 4 cruces adicionales con los 8 mejores terceros
    for i in range(0, min(8, len(terceros)), 2):
        if i+1 < len(terceros):
            partidos.append((terceros[i], terceros[i+1]))
        else:
            partidos.append((terceros[i], primeros[0]))

    return partidos[:16]  # exactamente 16 partidos

def simular_eliminatoria(octavos):
    """
    Simula octavos → cuartos → semis → final.
    Retorna (campeón, [finalistas], [semifinalistas], [cuartofinalistas])
    """
    ronda = [eliminar(a, b) for a, b in octavos]
    cuartos_eq = list(ronda)

    while len(ronda) > 1:
        siguiente = []
        for i in range(0, len(ronda)-1, 2):
            siguiente.append(eliminar(ronda[i], ronda[i+1]))
        if len(ronda) % 2 == 1:
            siguiente.append(ronda[-1])
        ronda = siguiente

    return ronda[0], cuartos_eq

# ══════════════════════════════════════════════════════════════
# 10 — MONTE CARLO PRINCIPAL
# ══════════════════════════════════════════════════════════════
N_SIMS = 10_000
print(f"\n🎲 Iniciando Monte Carlo ({N_SIMS:,} iteraciones)...")
print(f"   Grupos: {len(GROUPS_RAW)} | Equipos: {sum(len(v) for v in GROUPS_RAW.values())}")

campeonatos     = defaultdict(int)
finales_cnt     = defaultdict(int)
semifinales_cnt = defaultdict(int)
octavos_cnt     = defaultdict(int)

todos_equipos = [e for g in GROUPS_RAW.values() for e in g]

for sim in range(N_SIMS):
    if sim % 2000 == 0:
        pct = sim / N_SIMS * 100
        print(f"   {sim:>6,}/{N_SIMS:,}  ({pct:.0f}%)  +{time.time()-start:.0f}s")

    # Fase de grupos
    clasificados, terceros = simular_fase_grupos(GROUPS_RAW)

    # Registrar clasificados a octavos
    for g in clasificados:
        for eq in clasificados[g]:
            octavos_cnt[eq] += 1
    for eq in terceros:
        octavos_cnt[eq] += 1

    # Armar octavos y simular eliminatoria
    octavos = armar_octavos(clasificados, terceros)
    campeon, cuartos_eq = simular_eliminatoria(octavos)

    # Registrar cuartos (simplificado: los 16 que llegaron a octavos)
    for eq in cuartos_eq:
        semifinales_cnt[eq] += 1   # aproximación: cuartos → semis tracking

    campeonatos[campeon] += 1

elapsed = time.time() - start
print(f"\n✅ Completado en {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════
# 11 — CALCULAR RESULTADOS CON INTERVALOS DE CONFIANZA
# ══════════════════════════════════════════════════════════════
def ic95(n, N):
    """Intervalo de confianza 95% para proporción binomial (Wilson)."""
    p = n / N
    z = 1.96
    denom = 1 + z**2/N
    center = (p + z**2/(2*N)) / denom
    margin = z * np.sqrt(p*(1-p)/N + z**2/(4*N**2)) / denom
    return max(0, center-margin), min(1, center+margin)

resultados = []
for eq in todos_equipos:
    n_camp = campeonatos.get(eq, 0)
    n_oct  = octavos_cnt.get(eq, 0)
    ic_lo, ic_hi = ic95(n_camp, N_SIMS)
    resultados.append({
        "equipo":       eq,
        "prob_campeon": round(n_camp / N_SIMS, 4),
        "prob_octavos": round(n_oct  / N_SIMS, 4),
        "ic95_low":     round(ic_lo, 4),
        "ic95_high":    round(ic_hi, 4),
        "campeonatos":  n_camp,
        "n_octavos":    n_oct,
    })

df_res = pd.DataFrame(resultados).sort_values("prob_campeon", ascending=False).reset_index(drop=True)
df_res.index += 1

df_res.to_csv("resultados_montecarlo.csv", index=True, index_label="rank")

df_oct = df_res[["equipo","prob_octavos","prob_campeon","ic95_low","ic95_high"]]\
    .sort_values("prob_octavos", ascending=False).reset_index(drop=True)
df_oct.to_csv("clasificados_octavos.csv", index=False)

# ══════════════════════════════════════════════════════════════
# 12 — IMPRIMIR RESUMEN
# ══════════════════════════════════════════════════════════════
sep = "="*65

resumen = f"""
{sep}
  SIMULACIÓN MONTE CARLO — MUNDIAL FIFA 2026
  {N_SIMS:,} iteraciones | Modelo: XGBoost | Tiempo: {elapsed:.0f}s
{sep}

TOP 10 FAVORITOS AL TÍTULO
{'#':<4} {'Equipo':<22} {'Campeón':>9} {'IC 95%':>22} {'Octavos':>9}
{'-'*68}
"""
for i, row in df_res.head(10).iterrows():
    ic = f"[{row['ic95_low']:.1%} – {row['ic95_high']:.1%}]"
    resumen += f"{i:<4} {row['equipo']:<22} {row['prob_campeon']:>8.2%} {ic:>22} {row['prob_octavos']:>8.2%}\n"

resumen += f"""
TOP 10 — PROB. CLASIFICAR A OCTAVOS
{'#':<4} {'Equipo':<22} {'Octavos':>9} {'Campeón':>9}
{'-'*48}
"""
for i, (_, row) in enumerate(df_oct.head(10).iterrows(), 1):
    resumen += f"{i:<4} {row['equipo']:<22} {row['prob_octavos']:>8.2%} {row['prob_campeon']:>9.2%}\n"

resumen += f"""
{sep}
Archivos generados:
  resultados_montecarlo.csv  — {len(df_res)} equipos con IC 95%
  clasificados_octavos.csv   — prob. clasificar a octavos
{sep}
"""

print(resumen)

with open("simulacion_resumen.txt", "w", encoding="utf-8") as f:
    f.write(resumen)

print("💾 Archivos guardados. Listo para el reporte IEEE.")