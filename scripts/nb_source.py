# %% [markdown]
# # SOMPO 2026 - Sprint 3: Modelagem de Machine Learning
#
# **Disciplina:** Machine Learning & Modeling
# **Turma:** 1TIAPZ
# **Autor:** Equipe 1TIAPZ
# **Data:** 2026-08
# **Repositorio:** https://github.com/orugian/sprint3-ml
#
# ---
#
# ## Contexto do Desafio
#
# A **SOMPO Seguros** propoe uma solucao com inteligencia artificial para prever
# riscos no seguro rural. A plataforma integra dados, gera indicadores e alertas
# preventivos, alem de recomendacoes operacionais. O objetivo e reduzir sinistros,
# apoiar decisoes de subscricao e precificacao, e melhorar a gestao de carteira.
#
# Este notebook corresponde ao **3o Entregavel - Modelagem de Machine Learning**.
#
# ---
#
# ## O que muda em relacao a Sprint 2
#
# A Sprint 2 (nota 100) entregou CRISP-DM, limpeza, tratamento de categoricas,
# EDA e insights iniciais sobre a base **PSR 2025** da Sompo. Naquela sprint,
# duas limitacoes foram **declaradas explicitamente** pela propria equipe:
#
# > *"TARGET nao observado no PSR 2025: as colunas `VALOR_INDENIZACAO` e
# > `EVENTO_PREPONDERANTE` estao vazias porque o dataset contem apenas apolices
# > ativas em 2025. Para a modelagem baseline, foi criado um score de risco
# > derivado."*
#
# > *"Proximos Passos (Sprint 3), item 1: Buscar dataset PSR historico com a
# > coluna `EVENTO_PREPONDERANTE` populada para treinar com target real."*
#
# **Esta Sprint 3 cumpre exatamente esse proximo passo.** Foi obtida a base
# historica oficial **PSR 2016-2024 (SISSER/MAPA)**, da qual extraimos o recorte
# da Sompo Seguros:
#
# | | Sprint 2 (PSR 2025) | Sprint 3 (PSR 2016-2024) |
# |---|---|---|
# | Apolices Sompo | 1.414 | **18.872** |
# | Periodo | 2025 (1 safra) | **2019-2024 (6 safras)** |
# | Target | Derivado (score de pesos) | **Real (sinistro ocorrido)** |
# | Sinistros observados | 0 | **5.033 (26,7%)** |
# | Evento climatico | Ausente | **SECA, GEADA, GRANIZO, ...** |
# | Valor de indenizacao | Ausente | **Populado (R$)** |
# | Culturas | 7 (sem Soja) | **11 (com Soja, principal do Brasil)** |
# | UFs | 8 | **10** |
#
# Isso resolve a principal fragilidade metodologica da Sprint 2: como o
# `SCORE_RISCO` era funcao deterministica de tres variaveis categoricas, o
# problema era **circular** - o modelo reaprendia uma regra que nos mesmos
# tinhamos escrito. Agora o modelo preve um **desfecho real e independente**.
#
# A Secao 12 mantem a **comparacao direta** entre as duas abordagens.
#
# ---
#
# ## Atividades deste Entregavel (checklist do desafio)
#
# | # | Requisito | Onde |
# |---|---|---|
# | 1 | **Preparacao dos dados**: scaling e encoding (OHE) | Secoes 4 e 5 |
# | 2 | **Treinamento**: pelo menos 2 modelos | Secao 6 (3 modelos + baseline) |
# | 3 | **Validacao**: holdout (obrigatorio) | Secao 5.4 |
# | 3 | **Metricas de classificacao**: accuracy, precision, recall, f1, AUC | Secao 7 |
# | 3 | **Metricas de regressao**: MAE, RMSE, R2 | Secao 9 |
# | 4 | **Tuning basico**: ajuste de hiperparametros (ex: k no KNN) | Secao 8 |
# | 5 | **Interpretacao**: quais variaveis mais impactam o risco | Secao 10 |
#
# Alem do exigido, entregamos: validacao cruzada k=5, curva Precision-Recall,
# permutation importance, **validacao temporal out-of-time** (Secao 11) e
# persistencia do modelo em disco (Secao 13).

# %% [markdown]
# ## 1. CRISP-DM: Posicionamento desta Sprint
#
# O **CRISP-DM** estrutura o projeto em seis fases. A Sprint 2 cobriu as fases
# 1 a 3. Esta Sprint 3 concentra-se nas fases **4 (Modelagem)** e **5 (Avaliacao)**.
#
# | Fase | Status | Nesta sprint |
# |---|---|---|
# | 1. Entendimento do Negocio | Sprint 2 | Revisado: objetivo passa a ser prever sinistro **real** |
# | 2. Entendimento dos Dados | Sprint 2 | Ampliado: nova base historica com desfecho observado |
# | 3. Preparacao dos Dados | Sprint 2 | **Refeito** para a base historica (Secoes 4-5) |
# | 4. **Modelagem** | **Sprint 3** | **Secoes 6, 8 e 9** |
# | 5. **Avaliacao** | **Sprint 3** | **Secoes 7, 10 e 11** |
# | 6. Implantacao | Sprint 4 | Modelo serializado na Secao 13 |
#
# ### 1.1 Entendimento do Negocio (revisado)
#
# - **Problema de negocio:** dada uma proposta de seguro rural, estimar (a) a
#   **probabilidade** de ela gerar sinistro e (b) a **severidade** esperada em R$.
# - **Por que os dois:** e a decomposicao atuarial classica do premio puro,
#   `Premio Puro = Frequencia x Severidade`. Um modelo so de frequencia nao
#   distingue um sinistro de R$ 5 mil de um de R$ 1,8 milhao.
# - **Uso pratico:** priorizacao de vistorias, alertas preventivos ao produtor,
#   apoio a subscricao e definicao de taxa.
# - **Criterio de sucesso:** AUC-ROC substancialmente acima do acaso (0,50) em
#   holdout nao visto, com metricas reportadas de forma honesta.
#
# ### 1.2 Formulacao das duas tarefas de ML
#
# | | Tarefa A - Frequencia | Tarefa B - Severidade |
# |---|---|---|
# | Tipo | Classificacao binaria | Regressao |
# | Target | `TEVE_SINISTRO` (0/1) | `VL_INDENIZACAO` (R$) |
# | Amostra | 18.872 apolices | 5.033 apolices sinistradas |
# | Metricas | accuracy, precision, recall, f1, AUC | MAE, RMSE, R2 |

# %% [markdown]
# ## 2. Setup e Imports

# %%
import sys, os, time, json, warnings, unicodedata
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Python     : {sys.version.split()[0]}")
print(f"Working dir: {os.getcwd()}")

for d in ['data/principal', 'data/complementar', 'data/complementar/inmet',
          'figures', 'models']:
    os.makedirs(d, exist_ok=True)
    n = len(os.listdir(d))
    print(f"  {d:28s} -> {n} arquivo(s)")

# %%
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     cross_val_score, GridSearchCV)
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             precision_recall_curve, average_precision_score,
                             mean_absolute_error, mean_squared_error, r2_score)
import sklearn, joblib
from scipy.stats import chi2_contingency

# Configuracao visual e de exibicao
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 220)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_style('whitegrid')

RANDOM_STATE = 42          # semente unica para todo o notebook (reprodutibilidade)
TEST_SIZE    = 0.20        # holdout 80/20
CV_FOLDS     = 5           # validacao cruzada k=5

np.random.seed(RANDOM_STATE)

print("Ambiente pronto:")
print(f"  pandas       = {pd.__version__}")
print(f"  numpy        = {np.__version__}")
print(f"  scikit-learn = {sklearn.__version__}")
print(f"  seaborn      = {sns.__version__}")
print(f"\nRANDOM_STATE={RANDOM_STATE} | TEST_SIZE={TEST_SIZE} | CV_FOLDS={CV_FOLDS}")

# %% [markdown]
# ## 3. Carga dos Dados
#
# ### 3.1 Base Principal: PSR 2016-2024 (SISSER/MAPA) - recorte Sompo
#
# **Fonte oficial:** Ministerio da Agricultura e Pecuaria (MAPA) - Dados Abertos
# Dataset `sisser3` - *Sistema de Subvencao Economica ao Premio do Seguro Rural*
# URL: https://dados.agricultura.gov.br/dataset/sisser3
# Recurso: **"PSR - 2016 a 2024"** (CSV, ~297 MB, 1.048.565 apolices de todo o mercado)
#
# O script `scripts/prepare_data.py` documenta e reproduz o download e o filtro
# `NM_RAZAO_SOCIAL == "Sompo Seguros S/A"`, que gera o arquivo usado aqui.
# Mantemos apenas o recorte Sompo (5,4 MB) versionado no repositorio.

# %%
PATH_PSR_HIST = 'data/principal/psr_2016a2024_sompo.csv'

df_raw = pd.read_csv(PATH_PSR_HIST, sep=';', encoding='utf-8', dtype=str)
df_raw.columns = [c.strip() for c in df_raw.columns]

print(f"PSR 2016-2024 (Sompo): {df_raw.shape[0]:,} apolices x {df_raw.shape[1]} colunas")
print(f"Memoria: {df_raw.memory_usage(deep=True).sum()/1024**2:.1f} MB")
print(f"\nSeguradora(s): {df_raw['NM_RAZAO_SOCIAL'].unique()}")
df_raw.head(3)

# %% [markdown]
# ### 3.2 A diferenca decisiva: o target existe nesta base
#
# Comparamos lado a lado a coluna `EVENTO_PREPONDERANTE` na base da Sprint 2
# (PSR 2025) e nesta base historica.

# %%
df_2025 = pd.read_excel('data/principal/psr_2025_sompo.xlsx')

print("=" * 74)
print("COMPARACAO DO TARGET: Sprint 2 (PSR 2025) vs Sprint 3 (PSR 2016-2024)")
print("=" * 74)

COL_IND = 'VALOR_INDENIZAÇÃO'
for rotulo, d in [("Sprint 2 - PSR 2025   ", df_2025),
                  ("Sprint 3 - PSR 2016-24", df_raw)]:
    ev = d['EVENTO_PREPONDERANTE'].astype(str).str.strip()
    preenchidos = int((ev != '-').sum())
    ind = d[COL_IND].astype(str).str.strip()
    print(f"\n{rotulo} | {len(d):,} apolices")
    print(f"  EVENTO_PREPONDERANTE preenchido : {preenchidos:,} ({preenchidos/len(d)*100:.1f}%)")
    print(f"  valores distintos de evento     : {ev.nunique()}")
    print(f"  VALOR_INDENIZACAO preenchido    : {int((ind != '-').sum()):,}")

print("\n" + "=" * 74)
print("Eventos climaticos observados na base historica (o target real):")
print("=" * 74)
ev_hist = df_raw['EVENTO_PREPONDERANTE'].astype(str).str.strip()
print(ev_hist[ev_hist != '-'].value_counts().to_string())

# %% [markdown]
# **Confirmado.** A base da Sprint 2 tem 0% de eventos preenchidos (apolices
# ainda vigentes). A base historica tem 26,7% de sinistros observados, com a
# causa climatica identificada. E este o desfecho que vamos prever.

# %% [markdown]
# ### 3.3 Datasets Complementares (continuidade com a Sprint 2)
#
# Mantemos os tres complementares validados na Sprint 2, agora aplicados a base
# historica:
#
# 1. **IBGE - RUR_MUNI.DBF** (5.656 municipios) - validacao do codigo municipal
# 2. **CONAB - Serie Historica Cana-de-Acucar** - proxy do perfil agricola por UF
# 3. **INMET - 22 estacoes meteorologicas** - metadados geograficos das estacoes

# %%
from dbfread import DBF

# --- IBGE ---
ibge = pd.DataFrame(iter(DBF('data/complementar/ibge_rur_muni.DBF', load=True)))
print(f"IBGE RUR_MUNI  : {ibge.shape[0]:,} municipios | UFs: {ibge['UF'].nunique()}")

# --- CONAB ---
SHEET_AREA = 'Área'
conab_area = pd.read_excel('data/complementar/conab_cana_serie_historica.xls',
                           sheet_name=SHEET_AREA, header=4)
conab_prod = pd.read_excel('data/complementar/conab_cana_serie_historica.xls',
                           sheet_name='Produtividade', header=4)
for _d in (conab_area, conab_prod):
    _d.rename(columns={_d.columns[0]: 'REGIAO_UF'}, inplace=True)
print(f"CONAB cana     : {conab_area.shape[0]} regioes x {conab_area.shape[1]-1} safras")

# --- INMET ---
inmet_dir = 'data/complementar/inmet'
inmet_files = sorted(f for f in os.listdir(inmet_dir) if f.endswith('.csv'))
_meta = []
for f in inmet_files:
    with open(os.path.join(inmet_dir, f), encoding='utf-8') as fp:
        m = {}
        for line in fp:
            if ':' in line and not line.startswith('Data Medicao'):
                k, _, v = line.partition(':')
                m[k.strip()] = v.strip()
    m['codigo_estacao'] = f.split('_')[1]
    _meta.append(m)
inmet = pd.DataFrame(_meta)
print(f"INMET estacoes : {inmet.shape[0]} estacoes")
print(f"\nColunas INMET: {list(inmet.columns)}")

# %% [markdown]
# ## 4. Preparacao dos Dados
#
# ### 4.1 Conversao de tipos e normalizacao
#
# O CSV do MAPA vem inteiramente como texto, no padrao brasileiro
# (`1.234,56` para numeros, `dd/mm/aaaa` para datas) e com acentuacao.
# Padronizamos tudo aqui.

# %%
def to_num(serie):
    """Converte string no formato numerico brasileiro (1.234,56) para float."""
    return pd.to_numeric(
        serie.astype(str).str.strip()
             .str.replace('.', '', regex=False)   # separador de milhar
             .str.replace(',', '.', regex=False)  # separador decimal
             .replace({'-': np.nan, '': np.nan, 'nan': np.nan}),
        errors='coerce')


def normalize_str(texto):
    """Remove acentos e ordinais, e padroniza para maiuscula sem espacos nas bordas."""
    if not isinstance(texto, str):
        return texto
    decomposto = unicodedata.normalize('NFKD', texto)
    sem_acento = ''.join(c for c in decomposto if not unicodedata.combining(c))
    return (sem_acento.replace('ª', 'A').replace('º', 'O')
                      .replace('°', ' ').upper().strip())


df = df_raw.copy()

COLS_NUMERICAS = ['NR_AREA_TOTAL', 'NR_PRODUTIVIDADE_ESTIMADA',
                  'NR_PRODUTIVIDADE_SEGURADA', 'NivelDeCobertura',
                  'VL_LIMITE_GARANTIA', 'VL_PREMIO_LIQUIDO', 'PE_TAXA',
                  'VL_SUBVENCAO_FEDERAL', 'NR_GRAU_LAT', 'NR_MIN_LAT',
                  'NR_SEG_LAT', 'NR_GRAU_LONG', 'NR_MIN_LONG', 'NR_SEG_LONG',
                  COL_IND]
COLS_TEXTO = ['NM_CULTURA_GLOBAL', 'SG_UF_PROPRIEDADE', 'NM_CLASSIF_PRODUTO',
              'NM_MUNICIPIO_PROPRIEDADE', 'EVENTO_PREPONDERANTE']
COLS_DATA = ['DT_PROPOSTA', 'DT_INICIO_VIGENCIA', 'DT_FIM_VIGENCIA', 'DT_APOLICE']

for c in COLS_NUMERICAS:
    df[c] = to_num(df[c])
for c in COLS_TEXTO:
    df[c] = df[c].apply(normalize_str)
for c in COLS_DATA:
    df[c] = pd.to_datetime(df[c], format='%d/%m/%Y', errors='coerce')
df['ANO_APOLICE'] = pd.to_numeric(df['ANO_APOLICE'], errors='coerce').astype('Int64')

print("Conversao concluida.\n")
print("Nulos nas colunas numericas criticas:")
print(df[COLS_NUMERICAS[:8]].isnull().sum().to_string())
print(f"\nDatas invalidas: {df[COLS_DATA].isnull().sum().sum()}")
print("\nExemplo de normalizacao de texto:")
print(f"  {df_raw['NM_CULTURA_GLOBAL'].iloc[0]!r} -> {df['NM_CULTURA_GLOBAL'].iloc[0]!r}")

# %% [markdown]
# ### 4.2 Construcao dos targets
#
# Os dois targets vem **diretamente do dado observado**, sem nenhuma regra
# criada por nos - essa e a diferenca central em relacao a Sprint 2.
#
# - `TEVE_SINISTRO` = 1 se `EVENTO_PREPONDERANTE` foi registrado (Tarefa A)
# - `VL_INDENIZACAO` = valor pago em R$ (Tarefa B)

# %%
df['TEVE_SINISTRO'] = (df['EVENTO_PREPONDERANTE'].fillna('-') != '-').astype(int)
df['VL_INDENIZACAO'] = df[COL_IND].fillna(0.0)
df['LOSS_RATIO'] = df['VL_INDENIZACAO'] / df['VL_LIMITE_GARANTIA']

n_sin = int(df['TEVE_SINISTRO'].sum())
print("=" * 70)
print("TARGET A (classificacao) - TEVE_SINISTRO")
print("=" * 70)
print(f"  Com sinistro (1) : {n_sin:,} ({df['TEVE_SINISTRO'].mean()*100:.2f}%)")
print(f"  Sem sinistro (0) : {len(df)-n_sin:,} ({(1-df['TEVE_SINISTRO'].mean())*100:.2f}%)")
print(f"  Razao de desbalanceamento: 1 : {(len(df)-n_sin)/n_sin:.2f}")

print("\n" + "=" * 70)
print("TARGET B (regressao) - VL_INDENIZACAO, nas apolices sinistradas")
print("=" * 70)
sev = df.loc[df['TEVE_SINISTRO'] == 1, 'VL_INDENIZACAO']
print(sev.describe().apply(lambda v: f"{v:,.2f}").to_string())
print(f"\n  Indenizacao = R$ 0 mesmo com evento: {int((sev == 0).sum()):,} apolices")
print("  (evento registrado, mas sem pagamento apurado - caso legitimo)")

# Checagem de coerencia interna entre os dois targets
incoerentes = int(((df['VL_INDENIZACAO'] > 0) & (df['TEVE_SINISTRO'] == 0)).sum())
print(f"\n[SANITY CHECK] indenizacao > 0 sem evento registrado: {incoerentes} (esperado: 0)")
assert incoerentes == 0, "Incoerencia entre VL_INDENIZACAO e EVENTO_PREPONDERANTE"
print("[SANITY CHECK] OK - os dois targets sao mutuamente coerentes.")

# %% [markdown]
# ### 4.3 Cruzamento com os datasets complementares
#
# Repetimos os joins validados na Sprint 2, agora sobre a base historica.

# %%
# --- Join PSR x IBGE: o CD_GEOCMU tem 7 digitos; o RUR_MUNI usa os 5 primeiros
df['CD_MUNIC_5'] = df['CD_GEOCMU'].astype(str).str[:5]
ibge_join = ibge.rename(columns={'CODIGO': 'CD_MUNIC_5',
                                 'MUNIC': 'NM_MUNICIPIO_IBGE',
                                 'UF': 'UF_IBGE'})[['CD_MUNIC_5', 'NM_MUNICIPIO_IBGE', 'UF_IBGE']]
df = df.merge(ibge_join.drop_duplicates('CD_MUNIC_5'), on='CD_MUNIC_5', how='left')
match_ibge = int(df['NM_MUNICIPIO_IBGE'].notna().sum())
print(f"Join PSR x IBGE : {match_ibge:,}/{len(df):,} apolices ({match_ibge/len(df)*100:.1f}%)")

# --- Join PSR x CONAB: perfil agricola regional (proxy de infraestrutura/clima)
UF_TO_CONAB = {'SP': 'SÃO PAULO', 'PR': 'PARANÁ', 'MG': 'MINAS GERAIS',
               'GO': 'GOIÁS', 'MS': 'MATO GROSSO DO SUL', 'RS': 'RIO GRANDE DO SUL',
               'SC': 'SANTA CATARINA', 'MT': 'MATO GROSSO', 'TO': 'TOCANTINS',
               'DF': 'DISTRITO FEDERAL'}
safras = [c for c in conab_area.columns if c != 'REGIAO_UF']
ultima_safra = safras[-1]

conab_map = {}
for uf, nome in UF_TO_CONAB.items():
    linha = conab_area[conab_area['REGIAO_UF'].astype(str).str.strip().str.upper() == nome]
    if len(linha):
        conab_map[uf] = pd.to_numeric(linha[ultima_safra].iloc[0], errors='coerce')
df['CONAB_AREA_CANA_UF'] = df['SG_UF_PROPRIEDADE'].map(conab_map)
print(f"Join PSR x CONAB: {int(df['CONAB_AREA_CANA_UF'].notna().sum()):,} apolices "
      f"com perfil regional (safra {ultima_safra})")

# --- Join PSR x INMET: estacao meteorologica mais proxima (distancia haversine)
inmet_geo = inmet.copy()
inmet_geo['lat'] = pd.to_numeric(inmet_geo['Latitude'], errors='coerce')
inmet_geo['lon'] = pd.to_numeric(inmet_geo['Longitude'], errors='coerce')
inmet_geo = inmet_geo.dropna(subset=['lat', 'lon'])
print(f"Join PSR x INMET: {len(inmet_geo)} estacoes com coordenada valida "
      f"(distancia calculada apos derivar lat/lon do PSR, na secao 4.4)")

# %% [markdown]
# ### 4.4 Feature Engineering
#
# Derivamos variaveis que capturam **exposicao, intensidade de cobertura e
# sazonalidade** - todas conhecidas no momento da subscricao, portanto
# legitimamente utilizaveis para previsao.

# %%
def dms_para_decimal(grau, minuto, segundo, hemisferio_negativo=True):
    """Converte coordenada em graus/minutos/segundos para decimal.
    No PSR todas as apolices sao S (latitude) e W (longitude) -> sempre negativo."""
    dec = grau + minuto / 60 + segundo / 3600
    return -dec if hemisferio_negativo else dec


# --- Geograficas
df['LAT_DECIMAL'] = dms_para_decimal(df['NR_GRAU_LAT'], df['NR_MIN_LAT'], df['NR_SEG_LAT'])
df['LON_DECIMAL'] = dms_para_decimal(df['NR_GRAU_LONG'], df['NR_MIN_LONG'], df['NR_SEG_LONG'])

# --- Temporais (sazonalidade do plantio e janela de exposicao ao clima)
df['MES_INICIO'] = df['DT_INICIO_VIGENCIA'].dt.month
df['MES_FIM'] = df['DT_FIM_VIGENCIA'].dt.month
df['DURACAO_VIGENCIA_DIAS'] = (df['DT_FIM_VIGENCIA'] - df['DT_INICIO_VIGENCIA']).dt.days
df['DIAS_PROPOSTA_APOLICE'] = (df['DT_APOLICE'] - df['DT_PROPOSTA']).dt.days

# --- Razoes economicas (intensidade de risco precificada pela seguradora)
df['PE_TAXA_EFETIVA'] = df['VL_PREMIO_LIQUIDO'] / df['VL_LIMITE_GARANTIA']
df['PE_SUBVENCAO'] = df['VL_SUBVENCAO_FEDERAL'] / df['VL_PREMIO_LIQUIDO']
df['VL_LIMITE_POR_HA'] = df['VL_LIMITE_GARANTIA'] / df['NR_AREA_TOTAL']
df['GAP_PRODUTIVIDADE'] = ((df['NR_PRODUTIVIDADE_ESTIMADA'] - df['NR_PRODUTIVIDADE_SEGURADA'])
                           / df['NR_PRODUTIVIDADE_ESTIMADA'])

# --- Transformacoes log (area e valores sao fortemente assimetricos a direita)
df['LOG_AREA'] = np.log1p(df['NR_AREA_TOTAL'])
df['LOG_LIMITE'] = np.log1p(df['VL_LIMITE_GARANTIA'])

# --- Frequency encoding do municipio (975 categorias: OHE seria inviavel)
#     Nao usa o target => nao gera vazamento.
freq_muni = df['NM_MUNICIPIO_PROPRIEDADE'].value_counts(normalize=True)
df['FREQ_MUNICIPIO'] = df['NM_MUNICIPIO_PROPRIEDADE'].map(freq_muni)

# --- Distancia a estacao INMET mais proxima (km, formula de haversine)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


est_lat = inmet_geo['lat'].to_numpy()
est_lon = inmet_geo['lon'].to_numpy()
dist = haversine_km(df['LAT_DECIMAL'].to_numpy()[:, None], df['LON_DECIMAL'].to_numpy()[:, None],
                    est_lat[None, :], est_lon[None, :])
df['DIST_ESTACAO_INMET_KM'] = dist.min(axis=1)

print("Features derivadas:")
novas = ['LAT_DECIMAL', 'LON_DECIMAL', 'MES_INICIO', 'MES_FIM', 'DURACAO_VIGENCIA_DIAS',
         'DIAS_PROPOSTA_APOLICE', 'PE_TAXA_EFETIVA', 'PE_SUBVENCAO', 'VL_LIMITE_POR_HA',
         'GAP_PRODUTIVIDADE', 'LOG_AREA', 'LOG_LIMITE', 'FREQ_MUNICIPIO',
         'DIST_ESTACAO_INMET_KM', 'CONAB_AREA_CANA_UF']
print(df[novas].describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
print(f"\nValidacao geografica: LAT [{df['LAT_DECIMAL'].min():.1f}, {df['LAT_DECIMAL'].max():.1f}] "
      f"| LON [{df['LON_DECIMAL'].min():.1f}, {df['LON_DECIMAL'].max():.1f}]")
print("Esperado para o Brasil: LAT [-35, 5], LON [-75, -30]  -> OK")

# %% [markdown]
# ### 4.5 Auditoria anti-vazamento (data leakage)
#
# Esta e a etapa mais critica do notebook. Uma coluna so pode virar feature se
# for **conhecida no momento da subscricao**, antes de o sinistro existir.
# Auditamos as 38 colunas originais uma a uma.

# %%
auditoria = [
    # (coluna, decisao, motivo)
    ('EVENTO_PREPONDERANTE',      'EXCLUIR', 'E o proprio target A (define TEVE_SINISTRO)'),
    ('VALOR_INDENIZAÇÃO',         'EXCLUIR', 'E o proprio target B (severidade)'),
    ('NM_RAZAO_SOCIAL',           'EXCLUIR', 'Constante (todo o recorte e Sompo)'),
    ('NR_ANIMAL',                 'EXCLUIR', 'Constante = 0 (nao ha seguro pecuario no recorte)'),
    ('LATITUDE',                  'EXCLUIR', "Constante 'S'; ja codificada em LAT_DECIMAL"),
    ('LONGITUDE',                 'EXCLUIR', "Constante 'W'; ja codificada em LON_DECIMAL"),
    ('NR_DECIMAL_LATITUDE',       'EXCLUIR', "Vazia ('-') em toda a base"),
    ('NR_DECIMAL_LONGITUDE',      'EXCLUIR', "Vazia ('-') em toda a base"),
    ('NM_SEGURADO',               'EXCLUIR', 'Dado pessoal (LGPD); alta cardinalidade'),
    ('NR_DOCUMENTO_SEGURADO',     'EXCLUIR', 'Dado pessoal mascarado (LGPD)'),
    ('NR_PROPOSTA',               'EXCLUIR', 'Identificador unico - memorizacao, sem generalizacao'),
    ('ID_PROPOSTA',               'EXCLUIR', 'Identificador unico'),
    ('NR_APOLICE',                'EXCLUIR', 'Identificador unico'),
    ('CD_PROCESSO_SUSEP',         'EXCLUIR', 'Identificador administrativo do produto'),
    ('ANO_APOLICE',               'EXCLUIR', 'Ver nota metodologica abaixo (choque climatico anual)'),
    ('DT_PROPOSTA/VIGENCIA/APOLICE', 'DERIVAR', 'Viram MES_*, DURACAO_* e DIAS_PROPOSTA_APOLICE'),
    ('NR_GRAU/MIN/SEG_LAT/LONG',  'DERIVAR', 'Viram LAT_DECIMAL e LON_DECIMAL'),
    ('CD_GEOCMU',                 'DERIVAR', 'Usado no join com IBGE; vira FREQ_MUNICIPIO'),
    ('NM_MUNICIPIO_PROPRIEDADE',  'DERIVAR', 'Vira FREQ_MUNICIPIO (975 categorias)'),
    ('NM_CULTURA_GLOBAL',         'MANTER',  'Conhecida na subscricao - One-Hot Encoding'),
    ('SG_UF_PROPRIEDADE',         'MANTER',  'Conhecida na subscricao - One-Hot Encoding'),
    ('NM_CLASSIF_PRODUTO',        'MANTER',  'Conhecida na subscricao - One-Hot Encoding'),
    ('NR_AREA_TOTAL',             'MANTER',  'Exposicao fisica declarada na proposta'),
    ('NR_PRODUTIVIDADE_ESTIMADA', 'MANTER',  'Estimativa feita na subscricao'),
    ('NR_PRODUTIVIDADE_SEGURADA', 'MANTER',  'Produtividade contratada na apolice'),
    ('NivelDeCobertura',          'MANTER',  'Parametro contratual'),
    ('VL_LIMITE_GARANTIA',        'MANTER',  'Exposicao financeira contratada'),
    ('VL_PREMIO_LIQUIDO',         'MANTER',  'Premio cobrado - reflete a taxacao da seguradora'),
    ('PE_TAXA',                   'MANTER',  'Taxa aplicada na subscricao'),
    ('VL_SUBVENCAO_FEDERAL',      'MANTER',  'Subvencao concedida na contratacao'),
]
aud = pd.DataFrame(auditoria, columns=['coluna', 'decisao', 'motivo'])
print("AUDITORIA ANTI-VAZAMENTO")
print("=" * 110)
print(aud.to_string(index=False))
print("=" * 110)
print(aud['decisao'].value_counts().to_string())

# %% [markdown]
# #### Nota metodologica: por que `ANO_APOLICE` fica de fora
#
# O ano e altamente preditivo nesta base - 2021 concentrou 71,4% de sinistros
# por causa da seca e das geadas historicas daquela safra. Incluir o ano faria
# o modelo aprender *"2021 foi ruim"*, o que:
#
# 1. **infla artificialmente** as metricas no holdout aleatorio (o teste contem
#    apolices de 2021, cujo ano o modelo ja viu no treino);
# 2. **nao serve na pratica** - ao precificar uma apolice de 2027, ninguem sabe
#    ainda se sera um ano de seca;
# 3. **empobrece a interpretacao** exigida no item 5 do desafio: a resposta
#    viraria "o ano", em vez de atributos acionaveis como cultura e regiao.
#
# Excluindo o ano, o modelo aprende risco **estrutural** (cultura, geografia,
# perfil contratual). O efeito do choque climatico anual e medido separadamente
# na Secao 11 (validacao temporal).

# %% [markdown]
# ### 4.6 EDA orientada a modelagem
#
# A Sprint 2 ja fez a EDA exploratoria ampla. Aqui olhamos apenas o que
# **informa decisoes de modelagem**: forca do sinal, desbalanceamento e
# assimetria das distribuicoes.

# %%
fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# (a) Distribuicao do target
vc = df['TEVE_SINISTRO'].value_counts().sort_index()
axes[0].bar(['Sem sinistro', 'Com sinistro'], vc.values,
            color=['#2E7D32', '#C62828'], edgecolor='black')
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 200, f"{v:,}\n({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')
axes[0].set_title('Target A: TEVE_SINISTRO', fontweight='bold')
axes[0].set_ylabel('Apolices')
axes[0].set_ylim(0, len(df) * 0.90)

# (b) Taxa de sinistro por cultura
tx_cult = (df.groupby('NM_CULTURA_GLOBAL')
             .agg(taxa=('TEVE_SINISTRO', 'mean'), n=('TEVE_SINISTRO', 'size'))
             .query('n >= 50').sort_values('taxa'))
axes[1].barh(tx_cult.index, tx_cult['taxa'],
             color=['#C62828' if t > df['TEVE_SINISTRO'].mean() else '#2E7D32'
                    for t in tx_cult['taxa']], edgecolor='black')
axes[1].axvline(df['TEVE_SINISTRO'].mean(), color='black', ls='--',
                label=f"media {df['TEVE_SINISTRO'].mean()*100:.1f}%")
for i, (t, n) in enumerate(zip(tx_cult['taxa'], tx_cult['n'])):
    axes[1].text(t + 0.008, i, f"{t*100:.1f}% (n={n:,})", va='center', fontsize=9)
axes[1].set_title('Taxa de sinistro por cultura (n>=50)', fontweight='bold')
axes[1].set_xlabel('Taxa de sinistro')
axes[1].set_xlim(0, 0.62)
axes[1].legend(loc='lower right')

# (c) Severidade (escala log, fortemente assimetrica)
axes[2].hist(np.log10(sev[sev > 0]), bins=45, color='#1565C0', edgecolor='black', alpha=0.8)
axes[2].set_title('Target B: severidade (log10 de R$)', fontweight='bold')
axes[2].set_xlabel('log10(indenizacao em R$)')
axes[2].set_ylabel('Frequencia')
axes[2].axvline(np.log10(sev[sev > 0].median()), color='red', ls='--',
                label=f"mediana R$ {sev[sev>0].median():,.0f}")
axes[2].legend()

plt.suptitle('EDA orientada a modelagem - PSR 2019-2024 Sompo', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_eda_targets.png', dpi=110, bbox_inches='tight')
plt.show()

# %%
# Forca do sinal das categoricas: teste qui-quadrado contra o target
print("TESTE QUI-QUADRADO - categorica vs TEVE_SINISTRO")
print("=" * 78)
print(f"{'Variavel':30s} {'Chi2':>12s} {'p-valor':>12s} {'Cramer V':>10s} {'Sig':>6s}")
print("-" * 78)
linhas_chi = []
for col in ['NM_CULTURA_GLOBAL', 'SG_UF_PROPRIEDADE', 'NM_CLASSIF_PRODUTO', 'MES_INICIO']:
    tab = pd.crosstab(df[col], df['TEVE_SINISTRO'])
    chi2, p, dof, _ = chi2_contingency(tab)
    cramer = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    linhas_chi.append((col, chi2, p, cramer, sig))
    print(f"{col:30s} {chi2:>12.1f} {p:>12.2e} {cramer:>10.3f} {sig:>6s}")
print("=" * 78)
print("Todas as categoricas candidatas tem associacao significativa com o target.")

# %% [markdown]
# ## 5. Preparacao para Machine Learning
#
# > **Requisito 1 do desafio:** *scaling (quando necessario)* e *encoding (OHE, etc.)*
#
# ### 5.1 Definicao da matriz de features

# %%
FEATURES_NUMERICAS = [
    # Exposicao fisica e produtiva
    'NR_AREA_TOTAL', 'LOG_AREA', 'NR_PRODUTIVIDADE_ESTIMADA',
    'NR_PRODUTIVIDADE_SEGURADA', 'GAP_PRODUTIVIDADE',
    # Exposicao e parametros financeiros
    'VL_LIMITE_GARANTIA', 'LOG_LIMITE', 'VL_LIMITE_POR_HA', 'VL_PREMIO_LIQUIDO',
    'PE_TAXA', 'PE_TAXA_EFETIVA', 'VL_SUBVENCAO_FEDERAL', 'PE_SUBVENCAO',
    'NivelDeCobertura',
    # Sazonalidade e janela de exposicao
    'MES_INICIO', 'MES_FIM', 'DURACAO_VIGENCIA_DIAS', 'DIAS_PROPOSTA_APOLICE',
    # Geografia (inclui os complementares INMET e CONAB)
    'LAT_DECIMAL', 'LON_DECIMAL', 'FREQ_MUNICIPIO',
    'DIST_ESTACAO_INMET_KM', 'CONAB_AREA_CANA_UF',
]
FEATURES_CATEGORICAS = ['NM_CULTURA_GLOBAL', 'SG_UF_PROPRIEDADE', 'NM_CLASSIF_PRODUTO']
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

X = df[FEATURES].copy()
y = df['TEVE_SINISTRO'].copy()

print(f"Matriz de features : {X.shape[0]:,} linhas x {X.shape[1]} colunas")
print(f"  numericas   : {len(FEATURES_NUMERICAS)}")
print(f"  categoricas : {len(FEATURES_CATEGORICAS)}")
print(f"Target             : {y.sum():,} positivos ({y.mean()*100:.2f}%)")
print("\nCardinalidade das categoricas:")
for c in FEATURES_CATEGORICAS:
    print(f"  {c:26s} {X[c].nunique():3d} categorias -> {X[c].nunique()} colunas apos OHE")
print(f"\nNulos por feature (top 5):")
print(X.isnull().sum().sort_values(ascending=False).head(5).to_string())

# %% [markdown]
# ### 5.2 Por que scaling e encoding, e por que dentro de um Pipeline
#
# **Scaling (`StandardScaler`).** As features estao em escalas radicalmente
# diferentes: `NR_AREA_TOTAL` chega a milhares de hectares e `VL_LIMITE_GARANTIA`
# a milhoes de reais, enquanto `PE_TAXA` vive entre 0 e 1. O **KNN** mede
# distancia euclidiana/manhattan: sem padronizacao, o valor da garantia
# dominaria a distancia e as demais variaveis seriam praticamente ignoradas.
# Random Forest e Gradient Boosting sao invariantes a escala (particionam por
# limiar), mas aplicamos o scaler a todos para manter um pipeline unico e
# comparavel - sem prejuizo para as arvores.
#
# **Encoding (`OneHotEncoder`).** Cultura, UF e classe de produto sao nominais,
# sem ordem. Codifica-las como inteiros criaria uma ordem falsa (`SOJA=1 < TRIGO=2`).
# O OHE cria uma coluna binaria por categoria. Usamos `handle_unknown='ignore'`
# para que uma categoria vista so no teste nao quebre a predicao.
# Para `NM_MUNICIPIO_PROPRIEDADE` (975 categorias) o OHE seria inviavel, entao
# aplicamos **frequency encoding** (Secao 4.4).
#
# **Por que dentro de um `Pipeline`.** Se ajustassemos o `StandardScaler` no
# dataset inteiro antes do split, a media e o desvio-padrao carregariam
# informacao do conjunto de teste - um **vazamento sutil** que infla as metricas.
# Encapsulando pre-processamento e modelo num `Pipeline`, o `fit` do scaler
# ocorre **apenas no treino**, inclusive dentro de cada fold da validacao cruzada.

# %%
def criar_preprocessador():
    """Devolve um pre-processador NOVO a cada chamada.

    Usar uma fabrica em vez de um objeto unico e proposital: o `Pipeline` do
    scikit-learn ajusta seus passos *no proprio objeto*, sem clonar. Se todos os
    pipelines compartilhassem a mesma instancia, treinar um modelo em outro
    recorte de dados (por exemplo, o split temporal da Secao 11) reajustaria o
    OneHotEncoder e mudaria o numero de colunas por baixo dos modelos ja
    treinados. Cada pipeline recebe, portanto, a sua propria instancia.
    """
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
            ]), FEATURES_NUMERICAS),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), FEATURES_CATEGORICAS),
        ],
        remainder='drop',
    )


print(criar_preprocessador())

# %% [markdown]
# ### 5.3 Holdout estratificado 80/20
#
# > **Requisito 3 do desafio:** *holdout (obrigatorio)*
#
# Separamos 20% das apolices que **nao participam de nenhuma etapa de treino ou
# tuning**. A estratificacao (`stratify=y`) preserva a proporcao de 26,7% de
# sinistros nos dois conjuntos - essencial em base desbalanceada.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print("HOLDOUT ESTRATIFICADO 80/20")
print("=" * 60)
print(f"  Treino : {X_train.shape[0]:,} apolices | {y_train.sum():,} sinistros "
      f"({y_train.mean()*100:.2f}%)")
print(f"  Teste  : {X_test.shape[0]:,} apolices | {y_test.sum():,} sinistros "
      f"({y_test.mean()*100:.2f}%)")
print(f"\n  Diferenca de proporcao treino vs teste: "
      f"{abs(y_train.mean()-y_test.mean())*100:.3f} p.p. -> estratificacao OK")

# Dimensionalidade final apos o encoding
_insp = criar_preprocessador().fit(X_train)   # instancia usada so para inspecao
nomes_features = list(_insp.get_feature_names_out())
print(f"\n  Colunas apos scaling + OHE: {len(nomes_features)}")
print(f"  ({len(FEATURES_NUMERICAS)} numericas padronizadas + "
      f"{len(nomes_features)-len(FEATURES_NUMERICAS)} binarias do OHE)")
print("\n  Exemplo de colunas geradas pelo OHE:")
for n in [n for n in nomes_features if n.startswith('cat__')][:6]:
    print(f"    {n}")

# %% [markdown]
# ## 6. Treinamento dos Modelos (Classificacao)
#
# > **Requisito 2 do desafio:** *pelo menos 2 modelos*
#
# Treinamos **tres** algoritmos de familias distintas, mais um baseline de
# referencia. A diversidade e proposital: se familias diferentes convergem para
# as mesmas variaveis importantes, a conclusao da Secao 10 fica mais solida.
#
# | Modelo | Familia | Por que esta aqui |
# |---|---|---|
# | `DummyClassifier` | Referencia | Piso de comparacao - qualquer modelo precisa superar |
# | **KNN** | Baseado em distancia | Exigido pelo item de tuning (`k`); mede similaridade entre apolices |
# | **Random Forest** | Ensemble bagging | Robusto a outliers e nao-linearidades; da feature importance |
# | **Gradient Boosting** | Ensemble boosting | Corrige erros sequencialmente; costuma liderar em dados tabulares |
#
# `class_weight='balanced'` compensa o desbalanceamento de 1:2,75 nos modelos
# que suportam o parametro.

# %%
modelos = {
    'Baseline (Dummy)': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
}


def avaliar_classificador(nome, pipe, X_te, y_te, tempo, cv_auc=None):
    """Calcula o conjunto de metricas exigido pelo desafio."""
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    reg = {
        'modelo':    nome,
        'accuracy':  accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall':    recall_score(y_te, y_pred, zero_division=0),
        'f1':        f1_score(y_te, y_pred, zero_division=0),
        'auc_roc':   roc_auc_score(y_te, y_prob),
        'avg_prec':  average_precision_score(y_te, y_prob),
        'tempo_s':   tempo,
    }
    if cv_auc is not None:
        reg['cv_auc_mean'] = cv_auc.mean()
        reg['cv_auc_std'] = cv_auc.std()
    return reg


resultados, pipelines = [], {}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print("=" * 78)
print("TREINAMENTO - holdout 80/20 + validacao cruzada k=5 no treino")
print("=" * 78)
for nome, estimador in modelos.items():
    pipe = Pipeline([('pre', criar_preprocessador()), ('clf', estimador)])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    tempo = time.time() - t0

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    reg = avaliar_classificador(nome, pipe, X_test, y_test, tempo, cv_auc)
    resultados.append(reg)
    pipelines[nome] = pipe

    print(f"\n>>> {nome}")
    print(f"    acc={reg['accuracy']:.4f}  prec={reg['precision']:.4f}  "
          f"rec={reg['recall']:.4f}  f1={reg['f1']:.4f}  AUC={reg['auc_roc']:.4f}")
    print(f"    CV AUC (k=5) = {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}  |  "
          f"treino em {tempo:.2f}s")

df_result = pd.DataFrame(resultados).set_index('modelo')

# %% [markdown]
# ## 7. Validacao e Metricas
#
# > **Requisito 3 do desafio:** *metricas de classificacao - accuracy, precision,
# > recall, f1-score, AUC*
#
# ### 7.1 Quadro comparativo

# %%
cols_show = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc',
             'cv_auc_mean', 'cv_auc_std', 'tempo_s']
print("=" * 100)
print("COMPARATIVO NO HOLDOUT (ordenado por AUC-ROC)")
print("=" * 100)
print(df_result[cols_show].sort_values('auc_roc', ascending=False).round(4).to_string())
print("=" * 100)

melhor_nome = df_result.drop(index='Baseline (Dummy)')['auc_roc'].idxmax()
melhor_pipe = pipelines[melhor_nome]
print(f"\n>>> MELHOR MODELO (pre-tuning): {melhor_nome} "
      f"(AUC = {df_result.loc[melhor_nome, 'auc_roc']:.4f})")

ganho = (df_result.loc[melhor_nome, 'auc_roc'] - df_result.loc['Baseline (Dummy)', 'auc_roc'])
print(f">>> Ganho sobre o baseline aleatorio: +{ganho:.4f} de AUC")
print(f">>> Diferenca CV vs teste no melhor modelo: "
      f"{abs(df_result.loc[melhor_nome,'cv_auc_mean']-df_result.loc[melhor_nome,'auc_roc']):.4f} "
      f"(valores proximos indicam ausencia de overfitting grosseiro)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# (a) Metricas lado a lado
metricas = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
xs = np.arange(len(metricas))
largura = 0.2
for i, nome in enumerate(df_result.index):
    axes[0].bar(xs + i * largura, [df_result.loc[nome, m] for m in metricas],
                largura, label=nome, edgecolor='black', alpha=0.85)
axes[0].set_xticks(xs + largura * 1.5)
axes[0].set_xticklabels([m.upper().replace('_', '-') for m in metricas])
axes[0].set_ylim(0, 1.02)
axes[0].set_ylabel('Score')
axes[0].set_title('Metricas no conjunto de teste (holdout)', fontweight='bold')
axes[0].legend(loc='upper left', fontsize=9)

# (b) CV vs teste - deteccao de overfitting
sub = df_result.drop(index='Baseline (Dummy)')
x2 = np.arange(len(sub))
axes[1].bar(x2 - 0.2, sub['cv_auc_mean'], 0.4, yerr=sub['cv_auc_std'], capsize=6,
            label='CV AUC (treino, k=5)', color='#1976D2', edgecolor='black')
axes[1].bar(x2 + 0.2, sub['auc_roc'], 0.4, label='AUC no holdout',
            color='#388E3C', edgecolor='black')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(sub.index, rotation=12, ha='right')
axes[1].set_ylim(0, 1.02)
axes[1].set_ylabel('AUC-ROC')
axes[1].set_title('Validacao cruzada vs holdout', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/s3_comparativo_modelos.png', dpi=110, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 7.2 Curva ROC, curva Precision-Recall e matriz de confusao
#
# Em base desbalanceada a curva **Precision-Recall** e mais informativa que a
# ROC, por isso reportamos as duas.

# %%
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# (a) ROC
for nome, pipe in pipelines.items():
    prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(fpr, tpr, lw=2, label=f"{nome} (AUC={roc_auc_score(y_test, prob):.3f})")
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Aleatorio (0.500)')
axes[0].set_xlabel('Taxa de falso positivo')
axes[0].set_ylabel('Taxa de verdadeiro positivo')
axes[0].set_title('Curva ROC', fontweight='bold')
axes[0].legend(fontsize=8, loc='lower right')

# (b) Precision-Recall
for nome, pipe in pipelines.items():
    prob = pipe.predict_proba(X_test)[:, 1]
    pr, rc, _ = precision_recall_curve(y_test, prob)
    axes[1].plot(rc, pr, lw=2,
                 label=f"{nome} (AP={average_precision_score(y_test, prob):.3f})")
axes[1].axhline(y_test.mean(), color='k', ls='--', lw=1,
                label=f'Prevalencia ({y_test.mean():.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Curva Precision-Recall', fontweight='bold')
axes[1].legend(fontsize=8, loc='upper right')

# (c) Matriz de confusao do melhor modelo
cm = confusion_matrix(y_test, melhor_pipe.predict(X_test))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=axes[2],
            xticklabels=['Sem sinistro', 'Com sinistro'],
            yticklabels=['Sem sinistro', 'Com sinistro'],
            annot_kws={'fontsize': 13, 'fontweight': 'bold'})
axes[2].set_title(f'Matriz de confusao - {melhor_nome}', fontweight='bold')
axes[2].set_xlabel('Predito')
axes[2].set_ylabel('Real')

plt.tight_layout()
plt.savefig('figures/s3_roc_pr_confusao.png', dpi=110, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"Classification report - {melhor_nome}")
print("=" * 60)
print(classification_report(y_test, melhor_pipe.predict(X_test),
                            target_names=['Sem sinistro', 'Com sinistro'], digits=4))
print("Leitura de negocio da matriz de confusao:")
print(f"  Verdadeiro positivo (sinistro previsto e ocorrido) : {tp:,}")
print(f"  Falso negativo (sinistro NAO previsto)             : {fn:,}  <- risco nao antecipado")
print(f"  Falso positivo (alarme falso)                      : {fp:,}  <- custo de vistoria")
print(f"  Verdadeiro negativo                                : {tn:,}")

# %% [markdown]
# ## 8. Tuning de Hiperparametros
#
# > **Requisito 4 do desafio:** *ajuste de hiperparametros (ex: k no KNN)*
#
# Usamos `GridSearchCV` com **validacao cruzada estratificada k=5 sobre o conjunto
# de treino**. O conjunto de teste continua intocado - qualquer escolha de
# hiperparametro feita olhando o teste invalidaria a avaliacao final.
# A metrica otimizada e o **AUC-ROC**, apropriada para base desbalanceada por
# avaliar o ordenamento de risco independentemente do limiar de decisao.
#
# ### 8.1 KNN - o efeito do `k` (o exemplo citado no enunciado)

# %%
grade_knn = {
    'clf__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 41],
    'clf__weights':     ['uniform', 'distance'],
    'clf__metric':      ['euclidean', 'manhattan'],
}
gs_knn = GridSearchCV(
    Pipeline([('pre', criar_preprocessador()), ('clf', KNeighborsClassifier())]),
    grade_knn, scoring='roc_auc', cv=cv, n_jobs=-1, return_train_score=True)

t0 = time.time()
gs_knn.fit(X_train, y_train)
print(f"KNN - {len(gs_knn.cv_results_['params'])} combinacoes x {CV_FOLDS} folds "
      f"= {len(gs_knn.cv_results_['params'])*CV_FOLDS} treinos em {time.time()-t0:.0f}s")
print(f"\nMelhores hiperparametros : {gs_knn.best_params_}")
print(f"Melhor AUC na CV         : {gs_knn.best_score_:.4f}")

# %%
# Curva do k: o classico compromisso vies-variancia
res_knn = pd.DataFrame(gs_knn.cv_results_)

fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))

for metrica in ['euclidean', 'manhattan']:
    for peso in ['uniform', 'distance']:
        m = (res_knn['param_clf__metric'] == metrica) & (res_knn['param_clf__weights'] == peso)
        sub = res_knn[m].sort_values('param_clf__n_neighbors')
        axes[0].plot(sub['param_clf__n_neighbors'].astype(int), sub['mean_test_score'],
                     marker='o', lw=2, label=f'{metrica} / {peso}')
axes[0].axvline(gs_knn.best_params_['clf__n_neighbors'], color='red', ls='--',
                label=f"k otimo = {gs_knn.best_params_['clf__n_neighbors']}")
axes[0].set_xlabel('k (numero de vizinhos)')
axes[0].set_ylabel('AUC-ROC medio na validacao cruzada')
axes[0].set_title('Tuning do KNN: efeito do k', fontweight='bold')
axes[0].legend(fontsize=9)

# Vies-variancia: treino vs validacao na melhor configuracao de metrica/peso
mb = ((res_knn['param_clf__metric'] == gs_knn.best_params_['clf__metric']) &
      (res_knn['param_clf__weights'] == gs_knn.best_params_['clf__weights']))
sb = res_knn[mb].sort_values('param_clf__n_neighbors')
ks = sb['param_clf__n_neighbors'].astype(int)
axes[1].plot(ks, sb['mean_train_score'], marker='s', lw=2, color='#C62828', label='AUC no treino')
axes[1].plot(ks, sb['mean_test_score'], marker='o', lw=2, color='#1565C0', label='AUC na validacao')
axes[1].fill_between(ks, sb['mean_test_score'] - sb['std_test_score'],
                     sb['mean_test_score'] + sb['std_test_score'], alpha=0.2, color='#1565C0')
axes[1].axvline(gs_knn.best_params_['clf__n_neighbors'], color='red', ls='--')
axes[1].set_xlabel('k (numero de vizinhos)')
axes[1].set_ylabel('AUC-ROC')
axes[1].set_title(f"Vies x variancia ({gs_knn.best_params_['clf__metric']} / "
                  f"{gs_knn.best_params_['clf__weights']})", fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/s3_tuning_knn.png', dpi=110, bbox_inches='tight')
plt.show()

print("AUC por k (melhor combinacao de metrica e peso):")
print(pd.DataFrame({'k': ks.values,
                    'AUC_treino': sb['mean_train_score'].round(4).values,
                    'AUC_validacao': sb['mean_test_score'].round(4).values,
                    'desvio': sb['std_test_score'].round(4).values}).to_string(index=False))

# %% [markdown]
# **Leitura do grafico.** A curva de **validacao** (azul) descreve o compromisso
# vies-variancia de manual. Com `k` pequeno cada predicao se apoia em pouquissimos
# vizinhos e absorve o ruido das apolices individuais - **alta variancia**, e o
# AUC de validacao e o mais baixo da serie. Conforme `k` cresce, a vizinhanca
# suaviza esse ruido e o AUC sobe ate um pico. Depois dele, vizinhancas grandes
# demais passam a misturar apolices dissimilares e o modelo perde poder
# discriminante - **alto vies**. O `k` escolhido pela busca esta exatamente no pico.
#
# **Por que a curva de treino (vermelha) fica achatada em 1,0.** E um artefato
# esperado de `weights='distance'`: ao pontuar um ponto do proprio treino, ele
# aparece como vizinho de si mesmo a distancia zero e recebe peso infinito,
# dominando a votacao. O acerto no treino e, portanto, trivialmente perfeito para
# **qualquer** `k`. Isso mostra por que a validacao cruzada e indispensavel aqui:
# a metrica de treino nao carrega informacao alguma sobre generalizacao neste
# modelo, e escolher `k` por ela seria impossivel.

# %% [markdown]
# ### 8.2 Tuning do Random Forest e do Gradient Boosting

# %%
grades = {
    'Random Forest': (
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        {'clf__n_estimators':     [200, 400],
         'clf__max_depth':        [None, 12, 20],
         'clf__min_samples_leaf': [1, 5],
         'clf__max_features':     ['sqrt', 0.5]},
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'clf__n_estimators':  [200, 300],
         'clf__learning_rate': [0.05, 0.1],
         'clf__max_depth':     [3, 5],
         'clf__subsample':     [1.0, 0.8]},
    ),
}

buscas = {'KNN': gs_knn}
for nome, (estimador, grade) in grades.items():
    gs = GridSearchCV(Pipeline([('pre', criar_preprocessador()), ('clf', estimador)]),
                      grade, scoring='roc_auc', cv=cv, n_jobs=-1)
    t0 = time.time()
    gs.fit(X_train, y_train)
    buscas[nome] = gs
    print(f"{nome:20s} | {len(gs.cv_results_['params']):3d} combinacoes | "
          f"CV AUC = {gs.best_score_:.4f} | {time.time()-t0:.0f}s")
    print(f"{'':20s}   {gs.best_params_}")

# %% [markdown]
# ### 8.3 Impacto do tuning: antes x depois
#
# Reavaliamos cada modelo otimizado **no mesmo holdout**, nunca usado no tuning.

# %%
linhas_tuning = []
pipelines_tuned = {}
for nome, gs in buscas.items():
    best = gs.best_estimator_
    pipelines_tuned[nome] = best
    prob = best.predict_proba(X_test)[:, 1]
    pred = best.predict(X_test)
    linhas_tuning.append({
        'modelo': nome,
        'auc_antes':  df_result.loc[nome, 'auc_roc'],
        'auc_depois': roc_auc_score(y_test, prob),
        'f1_antes':   df_result.loc[nome, 'f1'],
        'f1_depois':  f1_score(y_test, pred, zero_division=0),
        'accuracy':   accuracy_score(y_test, pred),
        'precision':  precision_score(y_test, pred, zero_division=0),
        'recall':     recall_score(y_test, pred, zero_division=0),
        'cv_auc':     gs.best_score_,
    })

df_tuning = pd.DataFrame(linhas_tuning).set_index('modelo')
df_tuning['ganho_auc'] = df_tuning['auc_depois'] - df_tuning['auc_antes']

print("=" * 104)
print("IMPACTO DO TUNING - avaliado no holdout")
print("=" * 104)
print(df_tuning[['auc_antes', 'auc_depois', 'ganho_auc', 'f1_antes', 'f1_depois',
                 'accuracy', 'precision', 'recall']].round(4).to_string())
print("=" * 104)

melhor_final = df_tuning['auc_depois'].idxmax()
modelo_final = pipelines_tuned[melhor_final]
print(f"\n>>> MODELO FINAL: {melhor_final}")
print(f"    AUC no holdout : {df_tuning.loc[melhor_final,'auc_depois']:.4f}")
print(f"    F1 no holdout  : {df_tuning.loc[melhor_final,'f1_depois']:.4f}")
print(f"    Hiperparametros: {buscas[melhor_final].best_params_}")

# %%
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(df_tuning))
ax.bar(x - 0.2, df_tuning['auc_antes'], 0.4, label='Antes do tuning',
       color='#90A4AE', edgecolor='black')
ax.bar(x + 0.2, df_tuning['auc_depois'], 0.4, label='Depois do tuning',
       color='#2E7D32', edgecolor='black')
for i, (a, d) in enumerate(zip(df_tuning['auc_antes'], df_tuning['auc_depois'])):
    ax.text(i - 0.2, a + 0.008, f'{a:.3f}', ha='center', fontsize=9)
    ax.text(i + 0.2, d + 0.008, f'{d:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_tuning.index)
ax.set_ylabel('AUC-ROC no holdout')
ax.set_ylim(0, 1.05)
ax.set_title('Efeito do tuning de hiperparametros', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('figures/s3_impacto_tuning.png', dpi=110, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Tarefa B - Regressao da Severidade
#
# > **Requisito 3 do desafio:** *metricas de regressao - MAE, RMSE, R2*
#
# Prever *se* havera sinistro nao basta para precificar: e preciso estimar
# **quanto** custa. Modelamos aqui `VL_INDENIZACAO` (em R$) sobre as
# **5.033 apolices que efetivamente sinistraram**.
#
# Este e o segundo fator da decomposicao atuarial
# `Premio Puro = Frequencia (Secao 6) x Severidade (esta secao)`.

# %%
df_sev = df[df['TEVE_SINISTRO'] == 1].copy()
X_sev = df_sev[FEATURES].copy()
y_sev = df_sev['VL_INDENIZACAO'].copy()

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_sev, y_sev, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print("HOLDOUT DA REGRESSAO (apolices sinistradas)")
print("=" * 62)
print(f"  Treino : {Xs_train.shape[0]:,} | Teste : {Xs_test.shape[0]:,}")
print(f"\n  Indenizacao no treino: media R$ {ys_train.mean():,.0f} | "
      f"mediana R$ {ys_train.median():,.0f}")
print(f"  Indenizacao no teste : media R$ {ys_test.mean():,.0f} | "
      f"mediana R$ {ys_test.median():,.0f}")
print(f"  Assimetria (skew) do target: {y_sev.skew():.2f} "
      f"-> cauda longa a direita, tipica de severidade em seguros")

# %%
regressores = {
    'Baseline (media)':  DummyRegressor(strategy='mean'),
    'KNN':               KNeighborsRegressor(n_neighbors=10, weights='distance'),
    'Random Forest':     RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
}

linhas_reg, pipelines_reg = [], {}
print("=" * 88)
print("REGRESSAO DA SEVERIDADE - metricas no holdout")
print("=" * 88)
for nome, reg in regressores.items():
    pipe = Pipeline([('pre', criar_preprocessador()), ('reg', reg)])
    t0 = time.time()
    pipe.fit(Xs_train, ys_train)
    tempo = time.time() - t0
    pred = pipe.predict(Xs_test)

    mae = mean_absolute_error(ys_test, pred)
    rmse = float(np.sqrt(mean_squared_error(ys_test, pred)))
    r2 = r2_score(ys_test, pred)
    linhas_reg.append({'modelo': nome, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'tempo_s': tempo})
    pipelines_reg[nome] = pipe
    print(f"  {nome:20s} MAE = R$ {mae:>11,.0f} | RMSE = R$ {rmse:>11,.0f} | R2 = {r2:>7.4f}")

df_reg = pd.DataFrame(linhas_reg).set_index('modelo')
melhor_reg = df_reg.drop(index='Baseline (media)')['R2'].idxmax()
print("=" * 88)
print(f">>> MELHOR REGRESSOR: {melhor_reg} (R2 = {df_reg.loc[melhor_reg,'R2']:.4f})")
print(f">>> Reducao de MAE sobre o baseline: "
      f"R$ {df_reg.loc['Baseline (media)','MAE'] - df_reg.loc[melhor_reg,'MAE']:,.0f} "
      f"({(1 - df_reg.loc[melhor_reg,'MAE']/df_reg.loc['Baseline (media)','MAE'])*100:.1f}%)")

# %%
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# (a) MAE e RMSE
sub_r = df_reg.drop(index='Baseline (media)')
x = np.arange(len(sub_r))
axes[0].bar(x - 0.2, sub_r['MAE'] / 1000, 0.4, label='MAE', color='#1565C0', edgecolor='black')
axes[0].bar(x + 0.2, sub_r['RMSE'] / 1000, 0.4, label='RMSE', color='#C62828', edgecolor='black')
axes[0].axhline(df_reg.loc['Baseline (media)', 'MAE'] / 1000, color='gray', ls='--',
                label='MAE do baseline')
axes[0].set_xticks(x); axes[0].set_xticklabels(sub_r.index, rotation=12, ha='right')
axes[0].set_ylabel('Erro (R$ mil)')
axes[0].set_title('Erro de previsao da severidade', fontweight='bold')
axes[0].legend()

# (b) R2
cores = ['#2E7D32' if v > 0 else '#C62828' for v in df_reg['R2']]
axes[1].bar(df_reg.index, df_reg['R2'], color=cores, edgecolor='black')
axes[1].axhline(0, color='black', lw=1)
for i, v in enumerate(df_reg['R2']):
    axes[1].text(i, v + 0.012, f'{v:.3f}', ha='center', fontweight='bold')
axes[1].set_ylabel('R2')
axes[1].set_title('Coeficiente de determinacao (R2)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=12)

# (c) Previsto vs real
pred_best = pipelines_reg[melhor_reg].predict(Xs_test)
axes[2].scatter(ys_test / 1000, pred_best / 1000, alpha=0.3, s=14, color='#1565C0')
lim = max(ys_test.max(), pred_best.max()) / 1000
axes[2].plot([0, lim], [0, lim], 'r--', lw=2, label='Previsao perfeita')
axes[2].set_xlabel('Indenizacao real (R$ mil)')
axes[2].set_ylabel('Indenizacao prevista (R$ mil)')
axes[2].set_title(f'Previsto vs real - {melhor_reg}', fontweight='bold')
axes[2].legend()

plt.suptitle('Tarefa B - Regressao da severidade', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_regressao_severidade.png', dpi=110, bbox_inches='tight')
plt.show()

print(f"\nInterpretacao: um R2 de {df_reg.loc[melhor_reg,'R2']:.2f} significa que o modelo")
print(f"explica {df_reg.loc[melhor_reg,'R2']*100:.0f}% da variancia do valor indenizado.")
print("Severidade e reconhecidamente mais dificil que frequencia: o valor pago depende")
print("da INTENSIDADE do evento climatico (quantos mm de chuva faltaram, quantos graus")
print("de geada), informacao que nao esta na apolice. Ainda assim, o modelo reduz o erro")
print("medio de forma relevante frente ao baseline que sempre chuta a media.")

# %% [markdown]
# ## 10. Interpretacao: quais variaveis mais impactam o risco?
#
# > **Requisito 5 do desafio:** *quais variaveis mais impactam o risco?*
#
# Nenhuma metrica de importancia isolada e confiavel: cada uma tem um vies
# conhecido. Por isso usamos **tres leituras complementares**.
#
# 1. **Feature importance (impureza)** - nativa das arvores e barata, mas
#    tendenciosa a favor de variaveis continuas de alta cardinalidade e calculada
#    sobre o conjunto de treino.
# 2. **Permutation importance individual** - embaralha uma coluna por vez **no
#    holdout** e mede quanto o AUC cai. E agnostica ao modelo e avalia o impacto
#    sobre a metrica que interessa, mas **subestima variaveis redundantes**: se
#    duas colunas carregam a mesma informacao, destruir uma nao machuca.
# 3. **Permutation importance por bloco** - embaralha grupos de colunas
#    correlacionadas de uma vez, corrigindo justamente essa diluicao.
#
# A conclusao final (Secao 14.2) sai da convergencia entre as tres, cruzada com a
# evidencia descritiva da Secao 10.4.
#
# ### 10.1 Feature importance das arvores

# %%
def nomes_legiveis(nomes):
    """Remove os prefixos gerados pelo ColumnTransformer."""
    return [n.replace('num__', '').replace('cat__', '') for n in nomes]


nomes_pos = nomes_legiveis(list(modelo_final.named_steps['pre'].get_feature_names_out()))

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
importancias = {}
for i, nome in enumerate(['Random Forest', 'Gradient Boosting']):
    pipe = pipelines_tuned[nome]
    imp = pipe.named_steps['clf'].feature_importances_
    cols = nomes_legiveis(list(pipe.named_steps['pre'].get_feature_names_out()))
    s = pd.Series(imp, index=cols).sort_values(ascending=False)
    importancias[nome] = s
    top = s.head(15)[::-1]
    axes[i].barh(top.index, top.values, color='#1565C0', edgecolor='black')
    axes[i].set_title(f'Top 15 - {nome} (tuned)', fontweight='bold')
    axes[i].set_xlabel('Importancia (reducao de impureza)')

plt.suptitle('Feature importance dos modelos de arvore', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_feature_importance.png', dpi=110, bbox_inches='tight')
plt.show()

# Ranking lado a lado, para conferencia numerica (o grafico mostra so o top 15)
comp_imp = pd.DataFrame({
    'Random Forest': importancias['Random Forest'],
    'Gradient Boosting': importancias['Gradient Boosting'],
}).fillna(0)
comp_imp['media'] = comp_imp.mean(axis=1)
print("Top 15 por importancia de impureza (media dos dois modelos):")
print(comp_imp.sort_values('media', ascending=False).head(15).round(4).to_string())

# %% [markdown]
# ### 10.2 Permutation importance (no holdout)

# %%
t0 = time.time()
perm = permutation_importance(modelo_final, X_test, y_test, n_repeats=10,
                              random_state=RANDOM_STATE, scoring='roc_auc', n_jobs=-1)
print(f"Permutation importance calculada em {time.time()-t0:.0f}s "
      f"({len(FEATURES)} features x 10 repeticoes)")

perm_s = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
perm_sd = pd.Series(perm.importances_std, index=FEATURES)

fig, ax = plt.subplots(figsize=(11, 8))
top_p = perm_s.head(15)[::-1]
ax.barh(top_p.index, top_p.values, xerr=perm_sd[top_p.index].values,
        color='#2E7D32', edgecolor='black', capsize=3)
ax.set_xlabel('Queda media no AUC-ROC ao embaralhar a variavel')
ax.set_title(f'Permutation importance no holdout - {melhor_final}', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_permutation_importance.png', dpi=110, bbox_inches='tight')
plt.show()

print("Ranking completo (queda de AUC ao embaralhar):")
print(pd.DataFrame({'queda_AUC': perm_s.round(4),
                    'desvio': perm_sd[perm_s.index].round(4)}).to_string())

# %% [markdown]
# ### 10.3 Importancia por bloco de informacao
#
# #### Por que o ranking acima precisa de um cuidado: diluicao entre variaveis correlacionadas
#
# A permutation importance responde *"quanto o modelo perde se eu destruir**apenas
# esta** coluna?"*. Quando duas colunas carregam a mesma informacao, embaralhar uma
# delas nao machuca - o modelo simplesmente recorre a outra. As duas aparecem entao
# como irrelevantes, mesmo sendo importantes em conjunto.
#
# Nossa matriz tem redundancia **proposital**, e ela aparece com clareza no ranking:
#
# - `NR_AREA_TOTAL` e `LOG_AREA` sao a mesma variavel em escalas diferentes:
#   ambas ficam em torno de zero.
# - `VL_LIMITE_GARANTIA` e `LOG_LIMITE`: mesma situacao.
# - `CONAB_AREA_CANA_UF` marca exatamente 0,0000 - e constante dentro de cada UF,
#   ou seja, totalmente redundante com `SG_UF_PROPRIEDADE`. **O complementar da
#   CONAB nao agrega poder preditivo**, e o registramos honestamente.
# - `NM_CULTURA_GLOBAL` aparece com importancia baixa por um motivo mais sutil:
#   a cultura **determina o calendario agricola**. Soja, milho 2a safra e trigo tem
#   janelas de vigencia e meses de plantio distintos, entao `DURACAO_VIGENCIA_DIAS`,
#   `MES_INICIO` e `MES_FIM` ja carregam quase toda a informacao da cultura.
#
# Para medir o efeito real de cada **bloco de informacao**, embaralhamos grupos
# inteiros de colunas de uma vez.

# %%
GRUPOS = {
    'Cultura e calendario agricola': ['NM_CULTURA_GLOBAL', 'MES_INICIO', 'MES_FIM',
                                      'DURACAO_VIGENCIA_DIAS'],
    'Geografia':                     ['LAT_DECIMAL', 'LON_DECIMAL', 'SG_UF_PROPRIEDADE',
                                      'FREQ_MUNICIPIO', 'DIST_ESTACAO_INMET_KM',
                                      'CONAB_AREA_CANA_UF'],
    'Precificacao e cobertura':      ['PE_TAXA', 'PE_TAXA_EFETIVA', 'PE_SUBVENCAO',
                                      'NivelDeCobertura'],
    'Exposicao financeira':          ['VL_LIMITE_GARANTIA', 'LOG_LIMITE', 'VL_LIMITE_POR_HA',
                                      'VL_PREMIO_LIQUIDO', 'VL_SUBVENCAO_FEDERAL'],
    'Produtividade contratada':      ['NR_PRODUTIVIDADE_ESTIMADA', 'NR_PRODUTIVIDADE_SEGURADA',
                                      'GAP_PRODUTIVIDADE'],
    'Escala da operacao':            ['NR_AREA_TOTAL', 'LOG_AREA'],
    'Processo e tipo de produto':    ['DIAS_PROPOSTA_APOLICE', 'NM_CLASSIF_PRODUTO'],
}

auc_base = roc_auc_score(y_test, modelo_final.predict_proba(X_test)[:, 1])
rng = np.random.default_rng(RANDOM_STATE)
N_REPETICOES = 10

linhas_grupo = []
for grupo, colunas in GRUPOS.items():
    quedas = []
    for _ in range(N_REPETICOES):
        X_emb = X_test.copy()
        ordem = rng.permutation(len(X_emb))          # a MESMA permutacao para o grupo,
        for c in colunas:                            # preservando a correlacao interna
            X_emb[c] = X_emb[c].to_numpy()[ordem]
        quedas.append(auc_base - roc_auc_score(y_test, modelo_final.predict_proba(X_emb)[:, 1]))
    linhas_grupo.append({'grupo': grupo, 'n_variaveis': len(colunas),
                         'queda_AUC': np.mean(quedas), 'desvio': np.std(quedas)})

df_grupos = pd.DataFrame(linhas_grupo).set_index('grupo').sort_values('queda_AUC', ascending=False)
print(f"AUC do modelo intacto: {auc_base:.4f}\n")
print("IMPORTANCIA POR BLOCO DE INFORMACAO (grupo embaralhado em conjunto)")
print("=" * 78)
print(df_grupos.round(4).to_string())
print("=" * 78)

fig, ax = plt.subplots(figsize=(11, 5.5))
d = df_grupos.sort_values('queda_AUC')
ax.barh(d.index, d['queda_AUC'], xerr=d['desvio'], color='#6A1B9A',
        edgecolor='black', capsize=4)
for i, v in enumerate(d['queda_AUC']):
    ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)
ax.set_xlabel('Queda no AUC-ROC ao embaralhar o bloco inteiro')
ax.set_title('Importancia por bloco de informacao (holdout)', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_importancia_grupos.png', dpi=110, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 10.4 Evidencia de negocio por tras das variaveis do topo
#
# As metricas de importancia dizem *quais* variaveis pesam. Abaixo mostramos
# *como* elas se comportam, para que a leitura seja acionavel pela area tecnica.

# %%
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# (a) Taxa de sinistro por cultura
g = (df.groupby('NM_CULTURA_GLOBAL')
       .agg(taxa=('TEVE_SINISTRO', 'mean'), n=('TEVE_SINISTRO', 'size'))
       .query('n >= 200').sort_values('taxa'))
axes[0].barh(g.index, g['taxa'], color='#C62828', edgecolor='black')
axes[0].axvline(df['TEVE_SINISTRO'].mean(), color='black', ls='--', label='media da carteira')
for i, (t, n) in enumerate(zip(g['taxa'], g['n'])):
    axes[0].text(t + 0.006, i, f'{t*100:.1f}%', va='center', fontsize=9)
axes[0].set_title('Risco por cultura (n>=200)', fontweight='bold')
axes[0].set_xlabel('Taxa de sinistro'); axes[0].legend(fontsize=8)

# (b) Taxa de sinistro por UF
g2 = (df.groupby('SG_UF_PROPRIEDADE')
        .agg(taxa=('TEVE_SINISTRO', 'mean'), n=('TEVE_SINISTRO', 'size'))
        .query('n >= 200').sort_values('taxa'))
axes[1].barh(g2.index, g2['taxa'], color='#1565C0', edgecolor='black')
axes[1].axvline(df['TEVE_SINISTRO'].mean(), color='black', ls='--')
for i, (t, n) in enumerate(zip(g2['taxa'], g2['n'])):
    axes[1].text(t + 0.006, i, f'{t*100:.1f}% (n={n:,})', va='center', fontsize=9)
axes[1].set_title('Risco por estado (n>=200)', fontweight='bold')
axes[1].set_xlabel('Taxa de sinistro')

# (c) Taxa de sinistro por decil da taxa cobrada
df['_decil_taxa'] = pd.qcut(df['PE_TAXA'], 10, labels=False, duplicates='drop')
g3 = df.groupby('_decil_taxa')['TEVE_SINISTRO'].mean()
axes[2].plot(g3.index + 1, g3.values, marker='o', lw=2.5, color='#6A1B9A')
axes[2].axhline(df['TEVE_SINISTRO'].mean(), color='black', ls='--', label='media da carteira')
axes[2].set_xlabel('Decil de PE_TAXA (1 = taxa mais baixa)')
axes[2].set_ylabel('Taxa de sinistro observada')
axes[2].set_title('A taxa cobrada acompanha o risco real?', fontweight='bold')
axes[2].set_xticks(range(1, 11)); axes[2].legend(fontsize=8)

plt.suptitle('Leitura de negocio das variaveis mais importantes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_insights_negocio.png', dpi=110, bbox_inches='tight')
plt.show()

corr_taxa = df[['PE_TAXA', 'TEVE_SINISTRO']].corr().iloc[0, 1]
print(f"Correlacao entre a taxa cobrada (PE_TAXA) e o sinistro observado: {corr_taxa:.4f}")
print(f"Risco no decil de menor taxa : {g3.iloc[0]*100:.1f}%")
print(f"Risco no decil de maior taxa : {g3.iloc[-1]*100:.1f}%")
df.drop(columns=['_decil_taxa'], inplace=True)

# %% [markdown]
# ## 11. Validacao Temporal (out-of-time)
#
# O holdout aleatorio exigido pelo desafio responde: *"o modelo generaliza para
# apolices novas do mesmo periodo historico?"*. Mas o uso real e outro: treinar
# com o passado e prever a **safra seguinte**. Testamos esse cenario treinando
# com 2019-2023 e prevendo 2024.
#
# Esta secao vai alem do exigido e produz o achado mais importante do trabalho.

# %%
mask_tr = df['ANO_APOLICE'] <= 2023
mask_te = df['ANO_APOLICE'] == 2024

print("SPLIT TEMPORAL")
print("=" * 66)
print(f"  Treino (2019-2023): {int(mask_tr.sum()):,} apolices | "
      f"taxa de sinistro {y[mask_tr].mean()*100:.1f}%")
print(f"  Teste  (2024)     : {int(mask_te.sum()):,} apolices | "
      f"taxa de sinistro {y[mask_te].mean()*100:.1f}%")

linhas_temp = []
for nome, gs in buscas.items():
    # clone() devolve uma copia NAO treinada com os mesmos hiperparametros.
    # Reutilizar gs.best_estimator_ diretamente retreinaria, por referencia,
    # o modelo final da Secao 8.
    pipe = clone(gs.best_estimator_)
    pipe.fit(X[mask_tr], y[mask_tr])
    prob = pipe.predict_proba(X[mask_te])[:, 1]
    pred = pipe.predict(X[mask_te])
    linhas_temp.append({
        'modelo': nome,
        'auc_holdout_aleatorio': df_tuning.loc[nome, 'auc_depois'],
        'auc_temporal_2024': roc_auc_score(y[mask_te], prob),
        'f1_temporal_2024': f1_score(y[mask_te], pred, zero_division=0),
    })

df_temp = pd.DataFrame(linhas_temp).set_index('modelo')
df_temp['queda_auc'] = df_temp['auc_holdout_aleatorio'] - df_temp['auc_temporal_2024']
print("\n" + "=" * 88)
print("HOLDOUT ALEATORIO vs VALIDACAO TEMPORAL")
print("=" * 88)
print(df_temp.round(4).to_string())
print("=" * 88)

# %%
# Por que a performance cai? O ranking de risco se INVERTE entre as safras.
top_cult = df['NM_CULTURA_GLOBAL'].value_counts().head(4).index
pivot_ano = (df[df['NM_CULTURA_GLOBAL'].isin(top_cult)]
             .pivot_table(index='ANO_APOLICE', columns='NM_CULTURA_GLOBAL',
                          values='TEVE_SINISTRO', aggfunc='mean'))

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# (a) Taxa de sinistro por safra
taxa_ano = df.groupby('ANO_APOLICE')['TEVE_SINISTRO'].agg(['mean', 'size'])
axes[0].bar(taxa_ano.index.astype(str), taxa_ano['mean'], color='#C62828', edgecolor='black')
axes[0].axhline(df['TEVE_SINISTRO'].mean(), color='black', ls='--', label='media do periodo')
for i, (m, n) in enumerate(zip(taxa_ano['mean'], taxa_ano['size'])):
    axes[0].text(i, m + 0.015, f'{m*100:.0f}%\n(n={n:,})', ha='center', fontsize=8.5)
axes[0].set_title('Taxa de sinistro por safra', fontweight='bold')
axes[0].set_ylabel('Taxa de sinistro'); axes[0].set_ylim(0, 0.85); axes[0].legend(fontsize=8)

# (b) Inversao do ranking entre culturas
for c in pivot_ano.columns:
    axes[1].plot(pivot_ano.index.astype(int), pivot_ano[c], marker='o', lw=2.5, label=c)
axes[1].set_title('O ranking de risco se inverte entre safras', fontweight='bold')
axes[1].set_xlabel('Safra'); axes[1].set_ylabel('Taxa de sinistro')
axes[1].legend(fontsize=8.5)

# (c) Holdout aleatorio vs temporal
x = np.arange(len(df_temp))
axes[2].bar(x - 0.2, df_temp['auc_holdout_aleatorio'], 0.4,
            label='Holdout aleatorio', color='#2E7D32', edgecolor='black')
axes[2].bar(x + 0.2, df_temp['auc_temporal_2024'], 0.4,
            label='Temporal (prever 2024)', color='#EF6C00', edgecolor='black')
axes[2].axhline(0.5, color='red', ls='--', label='Acaso (0.50)')
axes[2].set_xticks(x); axes[2].set_xticklabels(df_temp.index, rotation=12, ha='right')
axes[2].set_ylabel('AUC-ROC'); axes[2].set_ylim(0, 1.0)
axes[2].set_title('Queda ao prever uma safra futura', fontweight='bold')
axes[2].legend(fontsize=8)

plt.suptitle('Validacao temporal: o efeito do choque climatico anual',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/s3_validacao_temporal.png', dpi=110, bbox_inches='tight')
plt.show()

print("Taxa de sinistro por safra e cultura (%):")
print((pivot_ano * 100).round(1).to_string())

# %%
# O sinal existe DENTRO de cada safra? Holdout 80/20 isolado por ano.
print("AUC intra-safra (holdout 80/20 dentro de cada ano, Random Forest)")
print("=" * 66)
linhas_intra = []
for ano in sorted(df['ANO_APOLICE'].dropna().unique()):
    sub = df[df['ANO_APOLICE'] == ano]
    if len(sub) < 400 or sub['TEVE_SINISTRO'].nunique() < 2:
        continue
    Xa, Xb, ya, yb = train_test_split(sub[FEATURES], sub['TEVE_SINISTRO'],
                                      test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                      stratify=sub['TEVE_SINISTRO'])
    p = Pipeline([('pre', criar_preprocessador()),
                  ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                                 random_state=RANDOM_STATE, n_jobs=-1))])
    p.fit(Xa, ya)
    auc_ano = roc_auc_score(yb, p.predict_proba(Xb)[:, 1])
    linhas_intra.append({'safra': int(ano), 'n': len(sub),
                         'taxa_sinistro': sub['TEVE_SINISTRO'].mean(), 'auc': auc_ano})
    print(f"  {int(ano)} | n={len(sub):5,} | taxa={sub['TEVE_SINISTRO'].mean()*100:5.1f}% "
          f"| AUC = {auc_ano:.4f}")

df_intra = pd.DataFrame(linhas_intra)
print("=" * 66)
print(f"AUC intra-safra medio: {df_intra['auc'].mean():.4f} "
      f"(min {df_intra['auc'].min():.4f} | max {df_intra['auc'].max():.4f})")

# %% [markdown]
# ### 11.1 O achado
#
# Os numeros acima contam uma historia consistente:
#
# - **Dentro de cada safra** o modelo discrimina risco muito bem (AUC alto e
#   estavel em todas as seis safras, inclusive nas de baixa sinistralidade).
# - **Entre safras** a performance despenca para perto do acaso.
#
# A causa esta no grafico do meio: o **ranking de risco se inverte**. Milho 2a
# safra foi a cultura mais sinistrada em 2021 (seca historica) e uma das menos
# sinistradas em 2023. Um modelo treinado ate 2023 aprende uma ordenacao que
# **deixa de valer** na safra seguinte.
#
# **Conclusao tecnica.** Os atributos da apolice (cultura, area, valores,
# geografia) explicam bem o risco **relativo** dentro de um mesmo contexto
# climatico, mas nao capturam o **choque climatico anual**, que e o fator
# dominante do seguro agricola. Para previsao out-of-time seria indispensavel
# incorporar variaveis climaticas com valores efetivos (precipitacao acumulada,
# dias de deficit hidrico, ocorrencia de geada) - exatamente o dado que a
# Sprint 2 registrou como indisponivel (estacoes INMET com metadados, porem
# series mensais nulas).
#
# Esse resultado transforma uma limitacao herdada em **requisito tecnico
# justificado por evidencia** para a proxima sprint.

# %% [markdown]
# ## 12. Comparacao com a Sprint 2 (target derivado vs target real)
#
# Reproduzimos aqui o baseline da Sprint 2 sobre a base PSR 2025 para deixar
# explicito **por que a mudanca de target foi necessaria**.

# %%
# --- Reproducao do target derivado da Sprint 2 sobre a base PSR 2025 ---
d2 = df_2025.copy()
for c in ['NM_CULTURA_GLOBAL', 'SG_UF_PROPRIEDADE', 'NM_CLASSIF_PRODUTO']:
    d2[c] = d2[c].apply(normalize_str)

PESOS_CULTURA = {'CEBOLA': .85, 'TOMATE': .80, 'MACA': .75, 'BATATA': .70,
                 'SORGO': .65, 'TRIGO': .55, 'MILHO 2A SAFRA': .50}
PESOS_ESTADO = {'SC': .75, 'PR': .70, 'RS': .60, 'MS': .55,
                'SP': .50, 'GO': .55, 'MG': .50, 'MT': .55}
PESOS_CLASSE = {'RECEITA': .80, 'CUSTEIO': .60, 'PRODUTIVIDADE': .50}

d2['SCORE_RISCO'] = (d2['NM_CULTURA_GLOBAL'].map(PESOS_CULTURA) * .40 +
                     d2['SG_UF_PROPRIEDADE'].map(PESOS_ESTADO) * .30 +
                     d2['NM_CLASSIF_PRODUTO'].map(PESOS_CLASSE) * .30)
d2['ALTO_RISCO'] = (d2['SCORE_RISCO'] > d2['SCORE_RISCO'].median()).astype(int)

print("Sprint 2 - o target derivado e funcao deterministica da cultura:")
print(d2.groupby('NM_CULTURA_GLOBAL')['ALTO_RISCO'].mean().round(3).to_string())
print("\nQuase todas as culturas caem em 0.0 ou 1.0: conhecer a cultura equivale")
print("a conhecer o target. O modelo nao aprendia risco - reaprendia nossa regra.")

# %%
feat_s2 = ['PE_TAXA', 'VL_SUBVENCAO_FEDERAL']
d2['PE_TAXA_EFETIVA'] = d2['VL_PREMIO_LIQUIDO'] / d2['VL_LIMITE_GARANTIA']
d2['PE_SUBVENCAO'] = d2['VL_SUBVENCAO_FEDERAL'] / d2['VL_PREMIO_LIQUIDO']
feat_s2 += ['PE_TAXA_EFETIVA', 'PE_SUBVENCAO']

X2 = d2[feat_s2].fillna(d2[feat_s2].median())
y2 = d2['ALTO_RISCO']
X2tr, X2te, y2tr, y2te = train_test_split(X2, y2, test_size=TEST_SIZE,
                                          random_state=RANDOM_STATE, stratify=y2)
rf2 = Pipeline([('sc', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                               random_state=RANDOM_STATE, n_jobs=-1))])
rf2.fit(X2tr, y2tr)
auc_s2 = roc_auc_score(y2te, rf2.predict_proba(X2te)[:, 1])

comparativo = pd.DataFrame([
    {'aspecto': 'Base de dados',            'Sprint 2': 'PSR 2025',              'Sprint 3': 'PSR 2016-2024'},
    {'aspecto': 'Apolices',                 'Sprint 2': f'{len(df_2025):,}',     'Sprint 3': f'{len(df):,}'},
    {'aspecto': 'Safras cobertas',          'Sprint 2': '1 (2025)',              'Sprint 3': '6 (2019-2024)'},
    {'aspecto': 'Natureza do target',       'Sprint 2': 'Derivado (regra nossa)','Sprint 3': 'Real (observado)'},
    {'aspecto': 'Sinistros observados',     'Sprint 2': '0',                     'Sprint 3': f'{n_sin:,}'},
    {'aspecto': 'Culturas',                 'Sprint 2': '7 (sem Soja)',          'Sprint 3': '11 (com Soja)'},
    {'aspecto': 'Features usadas',          'Sprint 2': '4 numericas',           'Sprint 3': f'{len(FEATURES)} (num + cat)'},
    {'aspecto': 'AUC do melhor modelo',     'Sprint 2': f'{auc_s2:.4f}',         'Sprint 3': f"{df_tuning.loc[melhor_final,'auc_depois']:.4f}"},
    {'aspecto': 'Tarefa de regressao',      'Sprint 2': 'Nao aplicavel',         'Sprint 3': f"R2 = {df_reg.loc[melhor_reg,'R2']:.3f}"},
    {'aspecto': 'Circularidade do target',  'Sprint 2': 'SIM (limitacao)',       'Sprint 3': 'NAO'},
])
print("=" * 96)
print("SPRINT 2 vs SPRINT 3")
print("=" * 96)
print(comparativo.to_string(index=False))
print("=" * 96)
print(f"""
O AUC da Sprint 2 ({auc_s2:.3f}) e MAIOR que o desta sprint
({df_tuning.loc[melhor_final,'auc_depois']:.3f}) - e isso e esperado e desejavel.
Prever uma regra deterministica que nos mesmos escrevemos e facil; prever um
sinistro climatico real e dificil. O numero menor desta sprint mede um problema
de verdade, e por isso vale mais.""")

# %% [markdown]
# ## 13. Persistencia dos Modelos
#
# Serializamos os pipelines completos (pre-processamento + modelo) para uso na
# API da proxima sprint. Como o `Pipeline` carrega o `StandardScaler` e o
# `OneHotEncoder` ja ajustados, o consumo em producao recebe o dado bruto e o
# pre-processamento e reaplicado de forma identica ao treino.

# %%
os.makedirs('models', exist_ok=True)

# compress=3 reduz os arquivos em cerca de 5x sem qualquer perda de precisao:
# ensembles de arvores sao muito redundantes e comprimem bem.
joblib.dump(modelo_final, 'models/classificador_sinistro.pkl', compress=3)
joblib.dump(pipelines_reg[melhor_reg], 'models/regressor_severidade.pkl', compress=3)

metadados = {
    'projeto': 'SOMPO 2026 - Sprint 3 - Modelagem de ML',
    'base': 'PSR 2016-2024 (SISSER/MAPA) - recorte Sompo Seguros S/A',
    'periodo': '2019-2024',
    'n_apolices': int(len(df)),
    'classificacao': {
        'modelo': melhor_final,
        'target': 'TEVE_SINISTRO',
        'hiperparametros': {k: str(v) for k, v in buscas[melhor_final].best_params_.items()},
        'auc_roc': float(df_tuning.loc[melhor_final, 'auc_depois']),
        'f1': float(df_tuning.loc[melhor_final, 'f1_depois']),
        'accuracy': float(df_tuning.loc[melhor_final, 'accuracy']),
        'precision': float(df_tuning.loc[melhor_final, 'precision']),
        'recall': float(df_tuning.loc[melhor_final, 'recall']),
    },
    'regressao': {
        'modelo': melhor_reg,
        'target': 'VL_INDENIZACAO',
        'mae': float(df_reg.loc[melhor_reg, 'MAE']),
        'rmse': float(df_reg.loc[melhor_reg, 'RMSE']),
        'r2': float(df_reg.loc[melhor_reg, 'R2']),
    },
    'features_numericas': FEATURES_NUMERICAS,
    'features_categoricas': FEATURES_CATEGORICAS,
    'random_state': RANDOM_STATE,
    'sklearn_version': sklearn.__version__,
}
with open('models/metadata.json', 'w', encoding='utf-8') as fp:
    json.dump(metadados, fp, indent=2, ensure_ascii=False)

# Tabela consolidada de resultados, para auditoria externa
df_tuning.to_csv('models/resultados_classificacao.csv')
df_reg.to_csv('models/resultados_regressao.csv')
df_temp.to_csv('models/resultados_validacao_temporal.csv')

print("Artefatos gravados em models/:")
for f in sorted(os.listdir('models')):
    print(f"  {f:38s} {os.path.getsize(os.path.join('models', f))/1024:8.1f} KB")

# %%
# Verificacao: recarregar do disco e confirmar que a predicao e identica
modelo_recarregado = joblib.load('models/classificador_sinistro.pkl')
prob_orig = modelo_final.predict_proba(X_test)[:, 1]
prob_novo = modelo_recarregado.predict_proba(X_test)[:, 1]

print(f"[VERIFICACAO] diferenca maxima entre o modelo em memoria e o recarregado: "
      f"{np.abs(prob_orig - prob_novo).max():.2e}")
assert np.allclose(prob_orig, prob_novo), "O modelo serializado nao reproduz as predicoes"
print("[VERIFICACAO] OK - o modelo persistido reproduz exatamente as predicoes.")

# Exemplo de uso: escoragem de 5 apolices do holdout
exemplo = X_test.head(5).copy()
exemplo_out = pd.DataFrame({
    'cultura': exemplo['NM_CULTURA_GLOBAL'].values,
    'UF': exemplo['SG_UF_PROPRIEDADE'].values,
    'area_ha': exemplo['NR_AREA_TOTAL'].round(1).values,
    'limite_R$': exemplo['VL_LIMITE_GARANTIA'].round(0).values,
    'prob_sinistro': modelo_recarregado.predict_proba(exemplo)[:, 1].round(4),
    'sinistro_real': y_test.head(5).values,
})
print("\nExemplo de escoragem (5 apolices do holdout):")
print(exemplo_out.to_string(index=False))

# %% [markdown]
# ## 14. Conclusoes
#
# ### 14.1 Cumprimento dos requisitos do desafio

# %%
checklist = pd.DataFrame([
    ['1. Preparacao - scaling',        'StandardScaler dentro do Pipeline (essencial para o KNN)', 'Secao 5.2'],
    ['1. Preparacao - encoding',       f'OneHotEncoder em {len(FEATURES_CATEGORICAS)} categoricas + frequency encoding do municipio', 'Secoes 4.4 e 5.2'],
    ['2. Treinamento - >= 2 modelos',  'KNN, Random Forest e Gradient Boosting (+ baseline Dummy)', 'Secao 6'],
    ['3. Validacao - holdout',         f'Holdout estratificado 80/20 ({len(X_test):,} apolices reservadas)', 'Secao 5.3'],
    ['3. Metricas - classificacao',    'accuracy, precision, recall, f1-score, AUC-ROC (+ AP e CV k=5)', 'Secao 7'],
    ['3. Metricas - regressao',        'MAE, RMSE, R2 na severidade em R$', 'Secao 9'],
    ['4. Tuning de hiperparametros',   f'GridSearchCV; k do KNN otimizado ({gs_knn.best_params_["clf__n_neighbors"]}) + RF e GB', 'Secao 8'],
    ['5. Interpretacao',               'Feature importance + permutation importance + leitura de negocio', 'Secao 10'],
    ['Extra - validacao temporal',     'Teste out-of-time treinando ate 2023 e prevendo 2024', 'Secao 11'],
    ['Extra - comparativo Sprint 2',   'Target derivado vs target real, lado a lado', 'Secao 12'],
    ['Extra - persistencia',           'Pipelines .pkl + metadata.json + CSVs de resultados', 'Secao 13'],
], columns=['Requisito', 'Como foi atendido', 'Onde'])

print("=" * 122)
print("CHECKLIST DO 3o ENTREGAVEL")
print("=" * 122)
print(checklist.to_string(index=False))
print("=" * 122)

# %%
print("=" * 78)
print("RESUMO EXECUTIVO DOS RESULTADOS")
print("=" * 78)
print(f"""
BASE
  PSR 2016-2024 (SISSER/MAPA), recorte Sompo Seguros S/A
  {len(df):,} apolices | 2019-2024 | {n_sin:,} sinistros reais ({df['TEVE_SINISTRO'].mean()*100:.1f}%)
  {len(FEATURES)} features ({len(FEATURES_NUMERICAS)} numericas + {len(FEATURES_CATEGORICAS)} categoricas
  -> {len(nomes_features)} colunas apos scaling e OHE)

TAREFA A - CLASSIFICACAO (havera sinistro?)
  Modelo campeao : {melhor_final}
  AUC-ROC        : {df_tuning.loc[melhor_final,'auc_depois']:.4f}   (acaso = 0.5000)
  Accuracy       : {df_tuning.loc[melhor_final,'accuracy']:.4f}
  Precision      : {df_tuning.loc[melhor_final,'precision']:.4f}
  Recall         : {df_tuning.loc[melhor_final,'recall']:.4f}
  F1-score       : {df_tuning.loc[melhor_final,'f1_depois']:.4f}

TAREFA B - REGRESSAO (quanto custa o sinistro?)
  Modelo campeao : {melhor_reg}
  MAE            : R$ {df_reg.loc[melhor_reg,'MAE']:,.0f}
  RMSE           : R$ {df_reg.loc[melhor_reg,'RMSE']:,.0f}
  R2             : {df_reg.loc[melhor_reg,'R2']:.4f}

TUNING
  k otimo do KNN : {gs_knn.best_params_['clf__n_neighbors']} vizinhos ({gs_knn.best_params_['clf__metric']}, {gs_knn.best_params_['clf__weights']})
  Ganho de AUC no KNN com o tuning: {df_tuning.loc['KNN','ganho_auc']:+.4f}

ROBUSTEZ
  AUC intra-safra medio        : {df_intra['auc'].mean():.4f}
  AUC prevendo a safra seguinte: {df_temp['auc_temporal_2024'].max():.4f}
""")
print("=" * 78)

# %% [markdown]
# ### 14.2 Quais variaveis mais impactam o risco (resposta ao item 5)
#
# A resposta abaixo sai da convergencia entre as tres leituras da Secao 10
# (importancia de impureza, permutation importance individual e por bloco) e a
# evidencia descritiva.
#
# **1. A janela de exposicao climatica e o fator dominante.**
# Na permutation importance individual, `DURACAO_VIGENCIA_DIAS` e a variavel cujo
# embaralhamento mais derruba o AUC. Junto com `MES_INICIO` e `MES_FIM`, ela descreve
# *quanto tempo* e *em qual estacao* a lavoura fica exposta. Somadas a cultura, essas
# variaveis formam o bloco mais importante do modelo por larga margem. E um resultado
# com sentido agronomico direto: mais dias de vigencia significam mais chances de
# atravessar uma seca, uma geada ou um granizo.
#
# **2. A cultura age atraves do calendario, nao ao lado dele.**
# Tomada isoladamente, `NM_CULTURA_GLOBAL` tem permutation importance quase nula -
# mas isso **nao** significa que a cultura seja irrelevante. A cultura *determina* a
# janela de plantio, entao sua informacao ja esta embutida nas variaveis de
# calendario. Por isso a analise por bloco (Secao 10.3) e a leitura correta: o bloco
# *"Cultura e calendario agricola"* e o mais importante do modelo. Descritivamente,
# Milho 2a safra e Sorgo - plantados na janela seca do inverno - sinistram cerca de
# duas vezes mais que a Soja, plantada no periodo chuvoso.
#
# **3. Geografia e o segundo bloco mais forte.**
# `LON_DECIMAL` lidera a importancia de impureza nos dois modelos de arvore e fica em
# segundo na permutation importance individual - as duas leituras concordam, e ambas
# a colocam acima de `LAT_DECIMAL` e da propria UF. A longitude separa a fronteira
# agricola do Centro-Oeste (MT com 49,6% e MS com 41,8% de sinistralidade) das
# lavouras do Sul. As coordenadas discriminam melhor que a UF porque capturam
# microrregiao, e nao apenas a divisa administrativa.
#
# **4. Precificacao e cobertura entram com peso moderado.**
# `VL_LIMITE_POR_HA`, `PE_TAXA`, `PE_TAXA_EFETIVA` e `PE_SUBVENCAO` aparecem no meio
# do ranking. A taxa cobrada correlaciona-se **positivamente, porem fracamente**, com
# o sinistro observado: o decil de menor taxa registra cerca de 21% de sinistralidade
# contra cerca de 30% no decil de maior taxa. A subscricao atual aponta na direcao
# certa, mas com poder de separacao modesto - exatamente a lacuna que um modelo
# preditivo se propoe a cobrir.
#
# **5. Escala e valor absoluto praticamente nao discriminam.**
# `NR_AREA_TOTAL` e `VL_LIMITE_GARANTIA` ficam proximos de zero. Faz sentido: o
# sinistro climatico atinge a lavoura independentemente do tamanho dela. O que pesa e
# o valor segurado *por hectare* (intensidade da cobertura), nao o valor total.
#
# **6. Um complementar que nao agregou.**
# `CONAB_AREA_CANA_UF` marcou importancia exatamente zero. Por ser constante dentro
# de cada UF, e inteiramente redundante com `SG_UF_PROPRIEDADE`. Registramos o
# resultado negativo em vez de omiti-lo: nem todo enriquecimento de base se converte
# em poder preditivo.
#
# ### 14.3 Limitacoes
#
# 1. **Ausencia de variaveis climaticas efetivas.** E a limitacao dominante,
#    demonstrada empiricamente na Secao 11. As series mensais do INMET
#    disponiveis estao nulas; usamos apenas metadados geograficos das estacoes.
# 2. **Generalizacao temporal limitada.** O modelo ordena risco muito bem dentro
#    de uma safra, mas nao antecipa o choque climatico da safra seguinte.
# 3. **Severidade parcialmente explicada.** O R2 da regressao reflete que o
#    valor pago depende da intensidade do evento, ausente na base.
# 4. **Escopo da carteira.** O recorte cobre 10 UFs e 11 culturas, com forte
#    concentracao em Soja e Milho 2a safra no Sul e Sudeste. Culturas e regioes
#    pouco representadas tem estimativas menos confiaveis.
# 5. **Vies de selecao do PSR.** A base cobre apenas apolices com subvencao
#    federal; contratos privados sem subvencao nao aparecem.
# 6. **Dominio.** O desafio menciona maquinas agricolas, enquanto o PSR cobre
#    culturas. O dominio de risco climatico rural e analogo, porem nao identico -
#    limitacao ja declarada na Sprint 2 e mantida aqui por transparencia.
#
# ### 14.4 Proximos passos (Sprint 4)
#
# 1. **Integrar dados climaticos com valores efetivos** - NASA POWER ou series
#    historicas completas do INMET, agregando precipitacao acumulada, deficit
#    hidrico e dias de geada na janela de vigencia de cada apolice. A Secao 11
#    fornece a justificativa empirica para priorizar esse item.
# 2. **Revalidar o ganho out-of-time** - repetir o teste temporal apos incluir
#    o clima e medir a recuperacao do AUC na previsao da safra seguinte.
# 3. **Modelo de premio puro** - combinar as duas tarefas
#    (`probabilidade x severidade`) num unico indicador monetario de risco
#    por apolice.
# 4. **Explicabilidade individual** - SHAP values para justificar cada score no
#    dashboard, atendendo ao requisito de relatorios explicaveis do desafio.
# 5. **Servir o modelo** - expor `models/classificador_sinistro.pkl` e
#    `models/regressor_severidade.pkl` na API e alimentar os dashboards e
#    rankings de risco previstos no projeto.

# %% [markdown]
# ---
#
# ## Referencias
#
# - **MAPA - Dados Abertos, SISSER** (base principal):
#   https://dados.agricultura.gov.br/dataset/sisser3
#   Recurso *"PSR - 2016 a 2024"*, 1.048.565 apolices de todo o mercado segurador.
# - **MAPA - Atlas do Seguro Rural**:
#   https://www.gov.br/agricultura/pt-br/assuntos/riscos-seguro
# - **CONAB - Serie Historica de Safras**: https://www.conab.gov.br/
# - **IBGE - Codigos de Municipios**: https://www.ibge.gov.br/
# - **INMET - Banco de Dados Meteorologicos**: https://bdmep.inmet.gov.br/
# - Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12.
# - Chapman et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
#
# ---
#
# **Repositorio:** https://github.com/orugian/sprint3-ml
# **Reproducao:** `pip install -r requirements.txt` e executar este notebook do
# inicio ao fim. O script `scripts/prepare_data.py` documenta a obtencao da base
# bruta a partir da fonte oficial do MAPA.
