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
| 5 | Interpretação do risco | 3 técnicas: importância de impureza + permutation individual + **permutation por bloco** (corrige a diluição entre variáveis correlacionadas) | 10 |

**Além do exigido:** auditoria anti-vazamento coluna a coluna das 38 originais
(Seção 4.5), validação temporal *out-of-time* (Seção 11), **decomposição do AUC** em
efeito de coorte, vazamento de grupo e falha real de transferência (Seção 11.2),
comparativo direto com a Sprint 2 (Seção 12) e persistência dos modelos (Seção 13).

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

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC | CV AUC (treino) |
|---|---|---|---|---|---|---|
| Baseline (`Dummy`) | 0,614 | 0,275 | 0,272 | 0,273 | 0,505 | 0,498 |
| KNN (k=21, manhattan, distance) | 0,818 | 0,753 | 0,473 | 0,581 | 0,837 | 0,833 |
| Gradient Boosting | 0,847 | 0,799 | 0,571 | 0,666 | 0,879 | 0,867 |
| **Random Forest** ← campeão | 0,847 | 0,715 | 0,712 | **0,713** | **0,892** | **0,883** |

O campeão é eleito pelo **AUC-ROC da validação cruzada no treino**, nunca pelo
holdout — que serve apenas para confirmar a ordem. Sobre o trade-off precision/recall:
o Gradient Boosting tem precision maior mas deixa passar bem mais sinistros. A
comparação não é inteiramente *ceteris paribus* — o Random Forest usa
`class_weight='balanced'` e o Gradient Boosting não o suporta, e nenhum dos dois teve
o limiar de decisão otimizado (ambos usam 0,5). Parte da diferença de recall vem daí.

**Tarefa B — regressão da severidade**, holdout de 1.007 apólices sinistradas:

| Modelo | MAE | RMSE | R² (holdout) | CV R² (treino) |
|---|---|---|---|---|
| Baseline (média) | R$ 61.397 | R$ 115.218 | −0,000 | −0,000 |
| KNN | R$ 44.080 | R$ 95.389 | 0,315 | 0,422 ± 0,059 |
| Gradient Boosting | R$ 46.256 | R$ 96.347 | 0,301 | 0,383 ± 0,045 |
| **Random Forest** ← campeão | **R$ 43.608** | **R$ 91.138** | **0,374** | **0,434 ± 0,081** |

O R² de um split único é instável num alvo de cauda pesada — o desvio da validação
cruzada (±0,08) é a medida honesta dessa incerteza, e o valor pontual do holdout não
deve ser lido como precisão de três casas.

**Efeito do tuning** (ganho de AUC no holdout): Gradient Boosting **+0,050**,
KNN **+0,034**, Random Forest **+0,006**. O `k` ótimo do KNN foi **21** — a curva
completa de bias-variância está na Seção 8.1.

Os números completos ficam nas saídas do notebook e em
[`models/metadata.json`](models/metadata.json),
[`models/resultados_classificacao.csv`](models/resultados_classificacao.csv),
[`models/resultados_regressao.csv`](models/resultados_regressao.csv) e
[`models/resultados_validacao_temporal.csv`](models/resultados_validacao_temporal.csv)
— este último contém os resultados *out-of-time*, que são os menos favoráveis do
trabalho e estão publicados junto com os demais.

### Quais variáveis mais impactam o risco

Medido por *permutation importance* aplicada a blocos de variáveis correlacionadas
(queda de AUC ao embaralhar o bloco inteiro):

| Bloco de informação | Queda de AUC |
|---|---|
| **Cultura e calendário agrícola** | **0,242** |
| Geografia (lat/lon, UF, município) | 0,171 |
| Precificação e cobertura | 0,110 |
| Exposição financeira | 0,036 |
| Produtividade contratada | 0,024 |
| Processo e tipo de produto | 0,004 |
| Escala da operação (área) | −0,000 |

A leitura por blocos é necessária porque a *permutation importance* individual
subestima variáveis redundantes: isoladamente, `NM_CULTURA_GLOBAL` marca apenas
**0,0011**, já que a cultura **determina** o calendário agrícola e sua informação
está absorvida por `DURACAO_VIGENCIA_DIAS`, `MES_INICIO` e `MES_FIM`. O mesmo vale
para `CONAB_AREA_CANA_UF` e `SG_UF_PROPRIEDADE`, que marcam **0,0034 cada** — por ser
constante dentro da UF, a variável da CONAB é redundante com a própria UF, e
embaralhar uma não machuca enquanto a outra permanece.

Um resultado que registramos por transparência: a **área total praticamente não
discrimina** risco (`NR_AREA_TOTAL` e `LOG_AREA` ficam em −0,000; o bloco inteiro de
escala também). O evento climático atinge a lavoura independentemente do tamanho dela
— o que pesa é o valor segurado *por hectare*, não o valor total.

---

## O achado principal

A Seção 11 testa o cenário de uso real: treinar com o passado (2019–2023) e
prever a safra seguinte (2024). O resultado é revelador:

- **Dentro de cada safra**, o modelo ordena risco muito bem — AUC alto e estável
  nas seis safras (média 0,859).
- **Entre safras**, a performance cai drasticamente, retendo apenas sinal residual:
  **Random Forest 0,604 · Gradient Boosting 0,584 · KNN 0,563**, contra 0,892 / 0,879
  / 0,837 no holdout aleatório.

A causa é que o **ranking de risco se inverte** entre anos. Milho 2ª safra foi a
cultura mais sinistrada em 2021 (seca histórica, 95,1%) e uma das menos
sinistradas em 2023 (5,6%). Os atributos da apólice explicam o risco **relativo**
dentro de um mesmo contexto climático, mas não capturam o **choque climático
anual**, que é o fator dominante do seguro agrícola.

A Seção 11.2 **decompõe essa queda** e mostra que ela não é artefato de amostragem:

| Cenário | AUC | Δ |
|---|---|---|
| (a) Holdout aleatório (número reportado) | 0,892 | — |
| (b) …medindo só **dentro de cada safra** | 0,854 | −0,039 (efeito de coorte) |
| (c) …com split por **segurado** | 0,882 | −0,010 (vazamento de grupo) |
| (d) …por segurado **e** dentro da safra | 0,844 | −0,049 |
| (e) **Out-of-time** (treino ≤2023 → 2024) | 0,604 | −0,288 |

Ou seja: **16,9% da queda vem do desenho amostral e 83,1% é falha genuína de
transferência entre safras.** Descobrimos, medindo, que remover a coluna
`ANO_APOLICE` **não remove o ano** — ele continua recuperável a 97,1% a partir das
features mantidas (e a 95,5% mesmo descartando todas as variáveis de calendário).
A afirmação honesta é "reduzimos o efeito de coorte e o quantificamos", não
"eliminamos o ano".

A ordem entre os três modelos, porém, **se mantém** nos dois protocolos: o campeão
eleito pela validação cruzada continua sendo o melhor também fora da amostra. O que
a validação temporal derruba é a expectativa de performance, não a escolha do
algoritmo.

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

| Base | Órgão | Uso | Vira feature? |
|---|---|---|---|
| [PSR 2016–2024 (SISSER)](https://dados.agricultura.gov.br/dataset/sisser3) | MAPA | **Base principal** — 18.872 apólices Sompo com sinistro real | sim |
| [PSR 2025](https://dados.agricultura.gov.br/dataset/sisser3) | MAPA | Comparativo com a Sprint 2 | — |
| [Banco de Dados Meteorológicos](https://bdmep.inmet.gov.br/) | INMET | Distância à estação mais próxima | **sim** |
| [Série Histórica de Safras](https://www.conab.gov.br/) | CONAB | Perfil agrícola regional por UF | sim, mas redundante com a UF |
| [Códigos de Municípios](https://www.ibge.gov.br/) | IBGE | Validação cadastral do município | não |

**Nota sobre os complementares.** A auditoria desta sprint descobriu que **os dois
joins herdados da Sprint 2 estavam incorretos** e ambos foram refeitos:

- **CONAB** — o cabeçalho da planilha estava sendo lido na linha errada (`header=4`
  em vez de `header=5`), o que transformava os rótulos de safra em `Unnamed: N`; além
  disso o arquivo identifica os estados por **sigla**, não por nome por extenso. O
  join casava **0 apólices** e a coluna ficava 100% nula. Corrigido: **94,2%** de
  cobertura.
- **IBGE** — o `CODIGO` do `RUR_MUNI.DBF` não é o prefixo do geocódigo de 7 dígitos,
  e sim um código sequencial próprio do arquivo. O join por prefixo casava **8,9%**,
  por coincidência numérica. Trocado pela chave (município normalizado + UF):
  **98,9%** de cobertura.

---

## Limitações declaradas

1. **Ausência de variáveis climáticas efetivas** — limitação dominante,
   demonstrada empiricamente na Seção 11. As séries mensais do INMET disponíveis
   estão nulas; usamos apenas os metadados geográficos das estações.
2. **Generalização temporal limitada** — o modelo ordena risco bem dentro de uma
   safra, mas não antecipa o choque climático da safra seguinte.
3. **O AUC do holdout aleatório não é o AUC de produção** — ele embute efeito de
   coorte anual e vazamento de grupo por segurado. A Seção 11.2 mede as duas parcelas
   e reporta o valor corrigido.
4. **Severidade parcialmente explicada** — além da falta de dados de intensidade do
   evento, **18% das apólices sinistradas têm indenização de R$ 0** (evento sem
   pagamento apurado) e foram mantidas no alvo. O R² de um split único também é
   instável num alvo de cauda pesada; o desvio da validação cruzada é a medida
   honesta dessa incerteza.
5. **Qualidade do dado de origem** — a base do MAPA traz erros de digitação (ex.: uma
   apólice com fim de vigência em 5207, saneada na Seção 4.4). Não houve auditoria
   exaustiva de todos os campos.
6. **Complementares com contribuição desigual** — os joins com IBGE e CONAB herdados
   da Sprint 2 estavam **ambos incorretos** e foram refeitos nesta sprint (ver abaixo).
   Mesmo corrigidos, contribuem pouco: o único complementar com efeito direto no
   modelo é a distância à estação INMET mais próxima.
7. **Escopo da carteira** — 10 UFs e 11 culturas, com forte concentração em Soja
   e Milho 2ª safra no Sul e Sudeste.
8. **Viés de seleção do PSR** — cobre apenas apólices com subvenção federal.
9. **Domínio** — o desafio menciona máquinas agrícolas, enquanto o PSR cobre
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
