# SOMPO 2026 — Sprint 3: Modelagem de Machine Learning

**Disciplina:** Machine Learning & Modeling · **Turma:** 1TIAPZ
**Entregável:** 3º — Modelagem de Machine Learning
**Notebook:** [`Sprint3-ML.ipynb`](Sprint3-ML.ipynb)

Previsão de risco de sinistro no seguro rural da **Sompo Seguros**, a partir de
dados públicos oficiais do Programa de Subvenção ao Prêmio do Seguro Rural (PSR).

---

## O que foi entregue

| # | Requisito do desafio | Como foi atendido | Seção |
|---|---|---|---|
| 1 | Preparação — *scaling* | `StandardScaler` dentro do `Pipeline` (indispensável para o KNN) | 5.2 |
| 1 | Preparação — *encoding* | `OneHotEncoder` nas 3 categóricas + *frequency encoding* do município | 4.4 e 5.2 |
| 2 | Treinamento — ≥ 2 modelos | **KNN**, **Random Forest** e **Gradient Boosting** (+ baseline `Dummy`) | 6 |
| 3 | Validação — **holdout** | Holdout estratificado 80/20, intocado durante todo o tuning | 5.3 |
| 3 | Métricas de classificação | accuracy, precision, recall, f1-score, AUC-ROC (+ AP e CV k=5) | 7 |
| 3 | Métricas de regressão | MAE, RMSE, R² sobre a severidade em R$ | 9 |
| 4 | Tuning de hiperparâmetros | `GridSearchCV`; **k do KNN** otimizado, mais RF e GB | 8 |
| 5 | Interpretação do risco | Feature importance + permutation importance + leitura de negócio | 10 |

**Além do exigido:** validação temporal *out-of-time* (Seção 11), comparativo
direto com a Sprint 2 (Seção 12) e persistência dos modelos (Seção 13).

---

## A mudança central em relação à Sprint 2

A Sprint 2 declarou duas limitações e as registrou como próximo passo:

> *"TARGET não observado no PSR 2025 (...) foi criado um score de risco derivado."*
> *"Próximos Passos (Sprint 3), item 1: buscar dataset PSR histórico com a coluna
> `EVENTO_PREPONDERANTE` populada para treinar com target real."*

**Esta sprint cumpre esse passo.** Foi obtida a base histórica oficial
**PSR 2016–2024 (SISSER/MAPA)** e dela extraído o recorte da Sompo:

| | Sprint 2 (PSR 2025) | Sprint 3 (PSR 2016–2024) |
|---|---|---|
| Apólices Sompo | 1.414 | **18.872** |
| Período | 2025 (1 safra) | **2019–2024 (6 safras)** |
| Target | Derivado (regra nossa) | **Real (sinistro observado)** |
| Sinistros observados | 0 | **5.033 (26,7%)** |
| Evento climático | Ausente | **SECA, GEADA, GRANIZO, …** |
| Valor de indenização | Ausente | **Populado (R$)** |
| Culturas | 7 (sem Soja) | **11 (com Soja)** |
| UFs | 8 | **10** |

Isso elimina a **circularidade** do target da Sprint 2, em que o `SCORE_RISCO`
era função determinística de três variáveis categóricas — conhecer a cultura
equivalia a conhecer o rótulo.

---

## As duas tarefas de ML

Modelamos a decomposição atuarial clássica do prêmio puro,
`Prêmio Puro = Frequência × Severidade`:

| | **Tarefa A — Frequência** | **Tarefa B — Severidade** |
|---|---|---|
| Tipo | Classificação binária | Regressão |
| Pergunta | *Esta apólice terá sinistro?* | *Quanto custará o sinistro?* |
| Target | `TEVE_SINISTRO` (0/1) | `VL_INDENIZACAO` (R$) |
| Amostra | 18.872 apólices | 5.033 apólices sinistradas |
| Métricas | accuracy, precision, recall, f1, AUC | MAE, RMSE, R² |

### Resultados

**Tarefa A — classificação**, holdout de 3.775 apólices (após tuning):

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Baseline (`Dummy`) | 0,614 | 0,275 | 0,272 | 0,273 | 0,505 |
| KNN (k=21, manhattan, distance) | 0,817 | 0,752 | 0,470 | 0,578 | 0,838 |
| Gradient Boosting | 0,845 | 0,793 | 0,564 | 0,659 | 0,873 |
| **Random Forest** ← campeão | 0,844 | 0,709 | 0,703 | **0,706** | **0,890** |

O critério de escolha foi o **AUC-ROC**, que mede o ordenamento de risco
independentemente do limiar — apropriado para base desbalanceada. Note o trade-off:
o Gradient Boosting tem precision maior (0,793 contra 0,709), mas deixa passar
bem mais sinistros (recall 0,564 contra 0,703). Para uso em alerta preventivo,
o recall do Random Forest é o comportamento desejado.

**Tarefa B — regressão da severidade**, holdout de 1.007 apólices sinistradas:

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (média) | R$ 61.397 | R$ 115.218 | −0,000 |
| KNN | R$ 44.665 | R$ 96.909 | 0,293 |
| Gradient Boosting | R$ 45.998 | R$ 94.718 | 0,324 |
| **Random Forest** ← campeão | **R$ 44.072** | **R$ 91.676** | **0,367** |

**Efeito do tuning** (ganho de AUC no holdout): KNN **+0,036**, Gradient Boosting
**+0,044**, Random Forest **+0,006**. O `k` ótimo do KNN foi **21** — a curva
completa de bias-variância está na Seção 8.1.

Os números completos ficam nas saídas do notebook e em
[`models/metadata.json`](models/metadata.json),
[`models/resultados_classificacao.csv`](models/resultados_classificacao.csv) e
[`models/resultados_regressao.csv`](models/resultados_regressao.csv).

### Quais variáveis mais impactam o risco

Medido por *permutation importance* aplicada a blocos de variáveis correlacionadas
(queda de AUC ao embaralhar o bloco inteiro):

| Bloco de informação | Queda de AUC |
|---|---|
| **Cultura e calendário agrícola** | **0,252** |
| Geografia (lat/lon, UF, município) | 0,158 |
| Precificação e cobertura | 0,112 |
| Exposição financeira | 0,038 |
| Produtividade contratada | 0,025 |
| Processo e tipo de produto | 0,003 |
| Escala da operação (área) | −0,000 |

A leitura por blocos é necessária porque a *permutation importance* individual
subestima variáveis redundantes: isoladamente, `NM_CULTURA_GLOBAL` marca apenas
0,0015, já que a cultura **determina** o calendário agrícola e sua informação está
absorvida por `DURACAO_VIGENCIA_DIAS`, `MES_INICIO` e `MES_FIM`.

Dois resultados que registramos por transparência: a **área total praticamente não
discrimina** risco (o evento climático atinge a lavoura independentemente do tamanho
— o que pesa é o valor segurado *por hectare*), e o complementar
`CONAB_AREA_CANA_UF` teve importância **exatamente zero**, por ser constante dentro
de cada UF e portanto redundante com `SG_UF_PROPRIEDADE`.

---

## O achado principal

A Seção 11 testa o cenário de uso real: treinar com o passado (2019–2023) e
prever a safra seguinte (2024). O resultado é revelador:

- **Dentro de cada safra**, o modelo ordena risco muito bem — AUC alto e estável
  nas seis safras.
- **Entre safras**, a performance cai para perto do acaso.

A causa é que o **ranking de risco se inverte** entre anos. Milho 2ª safra foi a
cultura mais sinistrada em 2021 (seca histórica, 95,1%) e uma das menos
sinistradas em 2023 (5,6%). Os atributos da apólice explicam o risco **relativo**
dentro de um mesmo contexto climático, mas não capturam o **choque climático
anual**, que é o fator dominante do seguro agrícola.

Isso converte a limitação herdada da Sprint 2 (dados climáticos do INMET com
séries nulas) em **requisito técnico justificado por evidência** para a Sprint 4.

---

## Estrutura do repositório

```
Sprint3-ML/
├── Sprint3-ML.ipynb                    # notebook do 3º entregável
├── README.md
├── requirements.txt
├── data/
│   ├── principal/
│   │   ├── psr_2016a2024_sompo.csv     # base histórica com target real (18.872)
│   │   └── psr_2025_sompo.xlsx         # base da Sprint 2, usada no comparativo
│   └── complementar/
│       ├── conab_cana_serie_historica.xls
│       ├── ibge_rur_muni.DBF
│       └── inmet/                      # 22 estações meteorológicas
├── figures/                            # gráficos gerados pelo notebook
├── models/                             # pipelines .pkl + métricas + metadata
├── scripts/
│   ├── prepare_data.py                 # reproduz o download da fonte oficial
│   ├── nb_source.py                    # fonte do notebook (marcadores # %%)
│   └── build_notebook.py               # gera e executa o .ipynb
└── Sprint-2/                           # entregável anterior, para referência
```

---

## Como reproduzir

```bash
pip install -r requirements.txt
```

Abrir e executar [`Sprint3-ML.ipynb`](Sprint3-ML.ipynb) do início ao fim
(cerca de 15 minutos, majoritariamente no `GridSearchCV`).

Alternativamente, pela linha de comando:

```bash
python scripts/build_notebook.py --run
```

Para refazer a base bruta a partir da fonte oficial do MAPA (download de ~297 MB):

```bash
python scripts/prepare_data.py --force
```

---

## Fontes de dados

Todas públicas, oficiais e brasileiras.

| Base | Órgão | Uso |
|---|---|---|
| [PSR 2016–2024 (SISSER)](https://dados.agricultura.gov.br/dataset/sisser3) | MAPA | **Base principal** — 18.872 apólices Sompo com sinistro real |
| [PSR 2025](https://dados.agricultura.gov.br/dataset/sisser3) | MAPA | Comparativo com a Sprint 2 |
| [Série Histórica de Safras](https://www.conab.gov.br/) | CONAB | Perfil agrícola regional por UF |
| [Códigos de Municípios](https://www.ibge.gov.br/) | IBGE | Validação do código municipal |
| [Banco de Dados Meteorológicos](https://bdmep.inmet.gov.br/) | INMET | Metadados geográficos das estações |

---

## Limitações declaradas

1. **Ausência de variáveis climáticas efetivas** — limitação dominante,
   demonstrada empiricamente na Seção 11. As séries mensais do INMET disponíveis
   estão nulas; usamos apenas os metadados geográficos das estações.
2. **Generalização temporal limitada** — o modelo ordena risco bem dentro de uma
   safra, mas não antecipa o choque climático da safra seguinte.
3. **Severidade parcialmente explicada** — o valor pago depende da intensidade do
   evento climático, informação ausente na base.
4. **Escopo da carteira** — 10 UFs e 11 culturas, com forte concentração em Soja
   e Milho 2ª safra no Sul e Sudeste.
5. **Viés de seleção do PSR** — cobre apenas apólices com subvenção federal.
6. **Domínio** — o desafio menciona máquinas agrícolas, enquanto o PSR cobre
   culturas. O domínio de risco climático rural é análogo, porém não idêntico —
   limitação já declarada na Sprint 2 e mantida aqui por transparência.

---

## Próximos passos (Sprint 4)

1. Integrar dados climáticos com valores efetivos (NASA POWER ou séries completas
   do INMET), agregando precipitação, déficit hídrico e geada na janela de
   vigência de cada apólice.
2. Revalidar o ganho *out-of-time* após incluir o clima.
3. Combinar as duas tarefas num indicador monetário único de prêmio puro.
4. SHAP values para explicabilidade individual no dashboard.
5. Servir os modelos persistidos via API.

---

**Sprint anterior:** https://github.com/orugian/sprint2-ml
