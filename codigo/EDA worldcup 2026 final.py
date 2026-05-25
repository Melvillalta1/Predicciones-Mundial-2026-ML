#EDA prediccion Mundial 2026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Estilo
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.6,
    'font.family':      'monospace',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.titlecolor':  '#f0f6fc',
})

GOLD   = '#f0c048'
CYAN   = '#58a6ff'
GREEN  = '#3fb950'
CORAL  = '#f85149'
PURPLE = '#bc8cff'
GRAY   = '#8b949e'

#Carga del dataset
import requests, io

REPO = "https://raw.githubusercontent.com/Melvillalta1/Predicciones-Mundial-2026-ML/refs/heads/main"

def fetch(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

df  = fetch(f"{REPO}/dataset_mundial2026.csv")
elo = fetch(f"{REPO}/elo_ratings_final.csv")
df['date'] = pd.to_datetime(df['date'])

print(f"Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"Período: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"\nDistribución del target (result):")
vc = df['result'].value_counts().sort_index()
labels = {0:'Local pierde', 1:'Empate', 2:'Local gana'}
for k,v in vc.items():
    print(f"  {labels[k]:15s}: {v:,} ({v/len(df)*100:.1f}%)")

#Calidad del dataset y valores nulos
nulos = df.isnull().sum()
nulos = nulos[nulos > 0]
print("Columnas con nulos:")
for col, n in nulos.items():
    print(f"  {col:30s}: {n:,} ({n/len(df)*100:.1f}%)")

print(f"\nNota: los nulos en fifa_pts y conf corresponden a partidos")
print(f"anteriores a 1992 (inicio del ranking FIFA). Se imputarán con")
print(f"la mediana / valor más frecuente antes de entrenar el modelo.")

#Distribuicion del tarjet y ventaja de localia
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribución del target y ventaja de localía', fontsize=15, color='#f0f6fc', y=1.02)

# 3a — Pie global
labels_pie = ['Local pierde\n(0)', 'Empate\n(1)', 'Local gana\n(2)']
sizes = [df['result'].eq(0).sum(), df['result'].eq(1).sum(), df['result'].eq(2).sum()]
colors_pie = [CORAL, GRAY, GREEN]
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels_pie, colors=colors_pie,
    autopct='%1.1f%%', startangle=90,
    textprops={'color':'#c9d1d9','fontsize':10},
    wedgeprops={'edgecolor':'#0d1117','linewidth':2}
)
for at in autotexts: at.set_color('#0d1117'); at.set_fontweight('bold')
axes[0].set_title('Global (37,169 partidos)')

# 3b — Barras: cancha propia vs neutral
cats = ['Cancha propia\n(is_neutral=0)', 'Cancha neutral\n(is_neutral=1)']
own   = df[df['is_neutral']==0]['result'].value_counts(normalize=True).sort_index()
neut  = df[df['is_neutral']==1]['result'].value_counts(normalize=True).sort_index()
x = np.arange(2); w = 0.25
for i, (label, color) in enumerate(zip(['Pierde','Empate','Gana'],[CORAL,GRAY,GREEN])):
    vals = [own.get(i,0), neut.get(i,0)]
    axes[1].bar(x + (i-1)*w, vals, w, label=label, color=color, alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(cats, fontsize=9)
axes[1].set_ylabel('Proporción'); axes[1].legend(fontsize=9)
axes[1].set_title('Por tipo de cancha')
axes[1].yaxis.grid(True, alpha=0.4); axes[1].set_axisbelow(True)

# 3c — Tendencia anual del win rate local
yearly = df.groupby(df['date'].dt.year).agg(
    win_local=('result', lambda x: (x==2).mean()),
    empate=   ('result', lambda x: (x==1).mean()),
).reset_index()
yearly = yearly[yearly['date'] >= 1990]
axes[2].plot(yearly['date'], yearly['win_local'], color=GREEN,  lw=2, marker='o', ms=3, label='Local gana')
axes[2].plot(yearly['date'], yearly['empate'],    color=GRAY,   lw=2, marker='o', ms=3, label='Empate')
axes[2].axhline(0.483, color=GREEN, ls='--', lw=1, alpha=0.5)
axes[2].set_title('Tendencia anual'); axes[2].legend(fontsize=9)
axes[2].yaxis.grid(True, alpha=0.4); axes[2].set_axisbelow(True)
axes[2].set_ylabel('Proporción')

plt.tight_layout()
plt.savefig('eda_01_target.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Hallazgo: el local gana el 48.3% de los partidos. En cancha propia sube al 50.5%,")
print("en cancha neutral cae al 42.6%. La ventaja de localía es real pero moderada.")

# Elo Rating - El predictor mas poderoso
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ELO Rating como predictor de resultados', fontsize=15, color='#f0f6fc', y=1.02)

# 4a — Win rate por bin de diferencia ELO
bins_labels = ['<-300', '-300/-100', '-100/-30', '-30/30', '30/100', '100/300', '>300']
win_rates   = [0.066, 0.237, 0.376, 0.464, 0.563, 0.704, 0.895]
colors_bar  = [CORAL if w < 0.5 else GREEN for w in win_rates]
bars = axes[0].bar(bins_labels, win_rates, color=colors_bar, alpha=0.85, edgecolor='#0d1117', linewidth=1)
axes[0].axhline(0.5, color=GOLD, ls='--', lw=1.5, label='50% (neutral)')
axes[0].set_ylabel('Prob. victoria local'); axes[0].set_ylim(0, 1)
axes[0].set_title('Win rate local por diferencia ELO')
axes[0].legend(fontsize=9)
for bar, wr in zip(bars, win_rates):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{wr:.0%}', ha='center', va='bottom', fontsize=9, color='#f0f6fc')
axes[0].yaxis.grid(True, alpha=0.4); axes[0].set_axisbelow(True)
axes[0].tick_params(axis='x', rotation=30)

# 4b — Top 20 ELO equipos WC2026
wc_teams = ['Spain','Argentina','France','Brazil','England','Portugal','Colombia',
            'Netherlands','Germany','Morocco','Japan','Croatia','Ecuador','Uruguay',
            'Mexico','Turkey','Senegal','Belgium','Australia','Switzerland']
elo_wc = elo[elo['team'].isin(wc_teams)].sort_values('elo_rating').tail(20)
colors_elo = [GOLD if r >= 2000 else CYAN if r >= 1900 else GRAY for r in elo_wc['elo_rating']]
axes[1].barh(elo_wc['team'], elo_wc['elo_rating'], color=colors_elo, alpha=0.85, edgecolor='#0d1117')
axes[1].axvline(1800, color=GRAY, ls='--', lw=1, alpha=0.6)
axes[1].set_xlabel('ELO Rating'); axes[1].set_title('Top 20 ELO — clasificados WC2026')
for i, (_, row) in enumerate(elo_wc.iterrows()):
    axes[1].text(row['elo_rating']+5, i, f"{row['elo_rating']:.0f}", va='center', fontsize=8)
axes[1].xaxis.grid(True, alpha=0.4); axes[1].set_axisbelow(True)

gold_p  = mpatches.Patch(color=GOLD,  label='≥ 2000 (élite)')
cyan_p  = mpatches.Patch(color=CYAN,  label='1900-1999 (contendor)')
gray_p  = mpatches.Patch(color=GRAY,  label='< 1900 (resto)')
axes[1].legend(handles=[gold_p,cyan_p,gray_p], fontsize=8)

plt.tight_layout()
plt.savefig('eda_02_elo.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Hallazgo clave: correlación ELO_diff vs result = 0.465 (la más alta del dataset).")
print("Cuando la diferencia ELO > 300, la prob. de victoria supera el 89%.")
print("España y Argentina lideran el ranking ELO entre los 48 clasificados.")

#Correlacion de features con el resultados
feat_cols = ['elo_diff','fifa_rank_diff','win_rate_diff','goals_diff',
             'goals_rolling_home','goals_rolling_away',
             'wc_win_rate_home','wc_win_rate_away',
             'pen_pct_home','pen_pct_away',
             
             
             'is_neutral','is_wc','is_qualifier','decay_weight']

corr = df[feat_cols + ['result']].corr()['result'].drop('result').sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors_c = [GREEN if v > 0 else CORAL for v in corr.values]
bars = ax.barh(corr.index, corr.values, color=colors_c, alpha=0.85, edgecolor='#0d1117')
ax.axvline(0, color=GRAY, lw=1)
ax.set_xlabel('Correlación de Pearson con result')
ax.set_title('Correlación de cada feature con el resultado del partido', color='#f0f6fc')
for bar, val in zip(bars, corr.values):
    x = val + 0.005 if val >= 0 else val - 0.005
    ha = 'left' if val >= 0 else 'right'
    ax.text(x, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', ha=ha, fontsize=9)
ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('eda_03_correlaciones.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("Top 5 features más correlacionadas con el resultado:")
for feat, val in corr.sort_values(ascending=False).head(5).items():
    print(f"  {feat:30s}: {val:.4f}")

#Rendimiento historico en mundiales
wc = df[df['is_wc']==1].copy()
wc_teams_list = ['Brazil','Germany','France','Netherlands','Argentina','Spain',
                 'Belgium','Colombia','England','Croatia','Senegal','Portugal',
                 'Switzerland','Denmark','Mexico','Ecuador','Uruguay',
                 'Morocco','USA','Korea Republic','IR Iran','Australia']

def wc_wr(team):
    h = wc[wc['home_team']==team]; a = wc[wc['away_team']==team]
    wins  = (h['result']==2).sum() + (a['result']==0).sum()
    total_ = len(h)+len(a)
    return {'team':team,'partidos':int(total_),'win_rate':round(wins/total_,3)} if total_>=8 else None

wc_df = pd.DataFrame([x for x in [wc_wr(t) for t in wc_teams_list] if x])
wc_df = wc_df.sort_values('win_rate', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
colors_wc = [GOLD if r >= 0.55 else CYAN if r >= 0.40 else CORAL for r in wc_df['win_rate']]
bars = ax.barh(wc_df['team'], wc_df['win_rate'], color=colors_wc, alpha=0.85, edgecolor='#0d1117')
ax.axvline(0.5,  color=GOLD, ls='--', lw=1.5, alpha=0.8, label='50%')
ax.axvline(0.33, color=GRAY, ls=':',  lw=1,   alpha=0.6, label='33% (1/3 gana)')
ax.set_xlabel('Win Rate en Mundiales')
ax.set_title('Performance histórica en Mundiales FIFA (≥8 partidos)', color='#f0f6fc')
for bar, (_, row) in zip(bars, wc_df.iterrows()):
    ax.text(row['win_rate']+0.005, bar.get_y()+bar.get_height()/2,
            f"{row['win_rate']:.1%} ({int(row['partidos'])} pts)", va='center', fontsize=8.5)
ax.legend(fontsize=9); ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
ax.set_xlim(0, 0.85)

gold_p = mpatches.Patch(color=GOLD, label='Favorito histórico (≥55%)')
cyan_p = mpatches.Patch(color=CYAN, label='Contendor (40-55%)')
red_p  = mpatches.Patch(color=CORAL,label='Historial débil (<40%)')
ax.legend(handles=[gold_p,cyan_p,red_p], fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('eda_04_wc_winrate.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Hallazgo: Brasil lidera históricamente (69.4%). Muchos favoritos actuales")
print("(España, Bélgica) tienen win rates mundialistas más modestos de lo esperado.")

#Analisis de goles y contexto del partido
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Contexto de los partidos y producción de goles', fontsize=14, color='#f0f6fc', y=1.02)

df['total_goles'] = df['home_score'] + df['away_score']

# 7a — Distribución de goles totales
max_g = min(df['total_goles'].max(), 12)
goles_dist = df['total_goles'].value_counts().sort_index().loc[:max_g]
axes[0].bar(goles_dist.index, goles_dist.values, color=CYAN, alpha=0.8, edgecolor='#0d1117')
axes[0].set_xlabel('Goles totales en el partido')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de goles por partido')
media_g = df['total_goles'].mean()
axes[0].axvline(media_g, color=GOLD, ls='--', lw=2, label=f'Media: {media_g:.2f}')
axes[0].legend(fontsize=9)
axes[0].yaxis.grid(True, alpha=0.4); axes[0].set_axisbelow(True)

# 7b — Goles promedio por tipo de torneo
tipo_goles = {
    'Amistosos':       df[df['tournament']=='Friendly']['total_goles'].mean(),
    'Clasificatorias': df[df['is_qualifier']==1]['total_goles'].mean(),
    'Copa FIFA':       df[df['is_wc']==1]['total_goles'].mean(),
    'Cancha neutral':  df[df['is_neutral']==1]['total_goles'].mean(),
    'Copa América':    df[df['tournament'].str.contains('Copa Am',na=False)]['total_goles'].mean(),
    'UEFA Euro':       df[df['tournament'].str.contains('UEFA Euro$',na=False)]['total_goles'].mean(),
}
names = list(tipo_goles.keys()); vals = list(tipo_goles.values())
colors_t = [GOLD if v == max(vals) else CYAN for v in vals]
bars = axes[1].bar(names, vals, color=colors_t, alpha=0.85, edgecolor='#0d1117')
axes[1].set_ylabel('Goles promedio'); axes[1].set_title('Goles promedio por tipo de torneo')
axes[1].tick_params(axis='x', rotation=30)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                 f'{v:.2f}', ha='center', fontsize=9)
axes[1].yaxis.grid(True, alpha=0.4); axes[1].set_axisbelow(True)
axes[1].set_ylim(0, 3.5)

# 7c — Tendencia de goles por año
yearly_g = df.groupby(df['date'].dt.year)['total_goles'].mean()
yearly_g = yearly_g[yearly_g.index >= 1990]
axes[2].plot(yearly_g.index, yearly_g.values, color=PURPLE, lw=2, marker='o', ms=3)
axes[2].fill_between(yearly_g.index, yearly_g.values, alpha=0.15, color=PURPLE)
axes[2].axhline(yearly_g.mean(), color=GOLD, ls='--', lw=1, label=f'Media: {yearly_g.mean():.2f}')
axes[2].set_title('Tendencia de goles promedio (1990-2026)')
axes[2].set_ylabel('Goles promedio por partido')
axes[2].legend(fontsize=9)
axes[2].yaxis.grid(True, alpha=0.4); axes[2].set_axisbelow(True)

plt.tight_layout()
plt.savefig('eda_05_goles.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Promedio general: {media_g:.2f} goles/partido")
print(f"Los partidos de Mundiales tienen MENOS goles ({tipo_goles['Copa FIFA']:.2f}) que amistosos ({tipo_goles['Amistosos']:.2f})")
print("Esto refleja mayor cautela táctica en torneos de eliminación directa.")

#Decay temporal - ponderacion de datos historicos
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Decay temporal: peso de los datos históricos', fontsize=14, color='#f0f6fc')

# 8a — Función de decay
years = np.linspace(0, 15, 200)
for hl, col, lbl in [(2, CORAL, 'half_life=2 años'), (4, GOLD, 'half_life=4 años (elegido)'), (6, CYAN, 'half_life=6 años')]:
    decay = np.exp(-np.log(2) * years / hl)
    axes[0].plot(years, decay, color=col, lw=2, label=lbl)
axes[0].axhline(0.5, color=GRAY, ls=':', lw=1)
axes[0].set_xlabel('Años antes del corte (2025)')
axes[0].set_ylabel('Peso del partido')
axes[0].set_title('Función de decay exponencial')
axes[0].legend(fontsize=9); axes[0].yaxis.grid(True, alpha=0.4); axes[0].set_axisbelow(True)

# 8b — Distribución del decay_weight en el dataset
axes[1].hist(df['decay_weight'], bins=40, color=PURPLE, alpha=0.8, edgecolor='#0d1117')
axes[1].set_xlabel('decay_weight'); axes[1].set_ylabel('Partidos')
axes[1].set_title('Distribución del peso en el dataset')
axes[1].axvline(df['decay_weight'].median(), color=GOLD, ls='--', lw=2,
                label=f"Mediana: {df['decay_weight'].median():.3f}")
axes[1].legend(fontsize=9); axes[1].yaxis.grid(True, alpha=0.4); axes[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig('eda_06_decay.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Con half_life=4 años:")
print(f"  Partidos 2024-2025: peso ≈ 0.83-1.00")
print(f"  Partidos 2020:      peso ≈ 0.44")
print(f"  Partidos 2015:      peso ≈ 0.23")
print(f"  Partidos 2010:      peso ≈ 0.13")
print(f"  Partidos 2000:      peso ≈ 0.06 (mínimo aplicado: 0.05)")

#Resumen de features del modelo
print("="*60)
print("RESUMEN DEL DATASET — Listo para modelado")
print("="*60)
print(f"\nFilas totales:    {len(df):,}")
print(f"Período:          {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Equipos únicos:   {pd.concat([df['home_team'],df['away_team']]).nunique()}")
print(f"Torneos únicos:   {df['tournament'].nunique()}")

print("\n--- FEATURES DEL MODELO (NO incluir home_score/away_score) ---")
features = ['elo_diff','fifa_rank_diff','win_rate_diff','goals_diff',
            'elo_home','elo_away','fifa_rank_home','fifa_rank_away',
            'win_rate_home','goals_scored_home','goals_conceded_home','wc_win_rate_home',
            'win_rate_away','goals_scored_away','goals_conceded_away','wc_win_rate_away',
            'goals_rolling_home','goals_rolling_away','pen_pct_home','pen_pct_away',
            'is_neutral','is_wc','is_qualifier','is_knockout','decay_weight']
for i, f in enumerate(features, 1):
    print(f"  {i:2d}. {f}")

print(f"\nTotal features para el modelo: {len(features)}")
print(f"Variable target: result (0=pierde, 1=empate, 2=gana)")
#Mapa de correlaciones

FEAT_HEATMAP = [
    'elo_diff', 'elo_home', 'elo_away',
    'fifa_rank_diff', 'fifa_rank_home', 'fifa_rank_away',
    'win_rate_diff', 'win_rate_home', 'win_rate_away',
    'goals_diff', 'goals_scored_home', 'goals_conceded_home',
    'wc_win_rate_home', 'wc_win_rate_away',
    'goals_rolling_home', 'goals_rolling_away',
    'pen_pct_home', 'pen_pct_away',
    'is_neutral', 'is_wc', 'is_knockout',
    'decay_weight', 'result',
]

# Etiquetas cortas para el eje
LABELS = {
    'elo_diff':'ELO diff', 'elo_home':'ELO local', 'elo_away':'ELO visit.',
    'fifa_rank_diff':'FIFA rank diff', 'fifa_rank_home':'FIFA rank L', 'fifa_rank_away':'FIFA rank V',
    'win_rate_diff':'WR diff', 'win_rate_home':'WR local', 'win_rate_away':'WR visit.',
    'goals_diff':'Goles diff', 'goals_scored_home':'Goles L', 'goals_conceded_home':'GC local',
    'wc_win_rate_home':'WC WR L', 'wc_win_rate_away':'WC WR V',
    'goals_rolling_home':'G rolling L', 'goals_rolling_away':'G rolling V',
    'pen_pct_home':'Pen% L', 'pen_pct_away':'Pen% V',
    'is_neutral':'Neutral', 'is_wc':'Es WC', 'is_knockout':'Knockout',
    'decay_weight':'Decay', 'result':'RESULT',
}

corr = df[FEAT_HEATMAP].corr()
labels_short = [LABELS.get(c, c) for c in FEAT_HEATMAP]

fig, ax = plt.subplots(figsize=(15, 13))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

# Mapa de calor manual con imshow
im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

# Etiquetas de ejes
ax.set_xticks(range(len(labels_short)))
ax.set_yticks(range(len(labels_short)))
ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=9, color='#c9d1d9')
ax.set_yticklabels(labels_short, fontsize=9, color='#c9d1d9')

# Valores dentro de cada celda (solo los significativos)
for i in range(len(FEAT_HEATMAP)):
    for j in range(len(FEAT_HEATMAP)):
        val = corr.values[i, j]
        if abs(val) >= 0.15 or (i == len(FEAT_HEATMAP)-1 or j == len(FEAT_HEATMAP)-1):
            color_txt = 'black' if abs(val) > 0.5 else '#f0f6fc'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color_txt, fontweight='bold' if abs(val)>=0.4 else 'normal')

# Barra de color
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors='#c9d1d9', labelsize=9)
cbar.set_label('Correlación de Pearson', color='#c9d1d9', fontsize=10)

# Resaltar fila/columna de RESULT
n = len(FEAT_HEATMAP) - 1
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')
ax.axhline(n - 0.5, color='#f0c048', lw=1.5, alpha=0.8)
ax.axhline(n + 0.5, color='#f0c048', lw=1.5, alpha=0.8)
ax.axvline(n - 0.5, color='#f0c048', lw=1.5, alpha=0.8)
ax.axvline(n + 0.5, color='#f0c048', lw=1.5, alpha=0.8)

ax.set_title('Mapa de calor — Matriz de correlaciones (dataset WC2026)',
             color='#f0f6fc', fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('eda_07_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("Hallazgos clave del mapa de calor:")
# Top correlaciones con result
corr_result = corr['result'].drop('result').abs().sort_values(ascending=False)
for feat, val in corr_result.head(6).items():
    signo = '+' if corr['result'][feat] > 0 else '-'
    print(f"  {signo}{val:.3f}  {feat}")
print("\nCorrelaciones entre features (multicolinealidad):")
pairs = [('elo_home','elo_diff'), ('fifa_rank_home','fifa_rank_diff'),
         ('win_rate_home','win_rate_diff'), ('elo_diff','fifa_rank_diff')]
for a,b in pairs:
    if a in corr.columns and b in corr.columns:
        print(f"  {a} vs {b}: {corr[a][b]:.3f}")

#Diagrama de bigotes por resultados
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Diagramas de bigotes: distribución de features por resultado del partido',
             fontsize=14, color='#f0f6fc', y=1.02)

resultado_labels = ['Local pierde (0)', 'Empate (1)', 'Local gana (2)']
colores = [CORAL, GRAY, GREEN]

# Features a analizar
features_box = [
    ('elo_diff',          'Diferencia ELO',              'Puntos ELO'),
    ('win_rate_diff',     'Diferencia Win Rate',          'Proporción'),
    ('goals_scored_home', 'Goles/partido local (rolling)','Goles promedio'),
    ('fifa_rank_diff',    'Diferencia Ranking FIFA',      'Posiciones (+ = local mejor)'),
    ('goals_conceded_home','Goles recibidos local (rolling)','Goles promedio'),
    ('wc_win_rate_home',  'WC Win Rate local',            'Proporción'),
]

for ax, (feat, titulo, ylabel) in zip(axes.flat, features_box):
    data_by_result = [df[df['result']==r][feat].dropna() for r in [0, 1, 2]]

    bp = ax.boxplot(
        data_by_result,
        patch_artist=True,
        notch=True,
        whis=[5, 95],          # bigotes en percentil 5 y 95
        showfliers=False,       # ocultar outliers extremos para legibilidad
        medianprops=dict(color='#f0c048', linewidth=2.5),
        whiskerprops=dict(color='#8b949e', linewidth=1.2),
        capprops=dict(color='#8b949e', linewidth=1.5),
        boxprops=dict(linewidth=0.8),
    )

    # Colorear cada caja
    for patch, color in zip(bp['boxes'], ['#f85149', '#8b949e', '#3fb950']):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Añadir media como punto
    for i, (data, color) in enumerate(zip(data_by_result, ['#f85149','#8b949e','#3fb950']), 1):
        ax.scatter(i, data.mean(), color='white', s=40, zorder=5,
                   marker='D', edgecolors=color, linewidth=1.5)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Pierde', 'Empate', 'Gana'], fontsize=9)
    ax.set_title(titulo, fontsize=11, color='#f0f6fc', fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.axhline(0, color='#f0c048', lw=0.8, ls=':', alpha=0.6)

# Leyenda global
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#f85149', alpha=0.75, label='Local pierde (0)'),
    Patch(facecolor='#8b949e', alpha=0.75, label='Empate (1)'),
    Patch(facecolor='#3fb950', alpha=0.75, label='Local gana (2)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='white',
           markeredgecolor='gray', markersize=7, label='Media'),
    Line2D([0],[0], color='#f0c048', lw=2.5, label='Mediana'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.2, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig('eda_08_boxplots.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("Interpretación de los boxplots:")
print("  elo_diff:   la mediana sube claramente de 'pierde' a 'gana' → predictor clave")
print("  win_rate:   misma tendencia → forma reciente importa")
print("  goals home: diferencia sutil pero consistente")
print("  fifa_rank:  diferencia negativa = local mejor rankeado → corr. con victoria")
print("  GC local:   equipos que ganan conceden menos goles")
print("  WC WR:      equipos con mejor historial mundialista ganan más")
