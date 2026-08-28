"""
Reproduz a obtencao da base principal da Sprint 3 a partir da fonte oficial.

FONTE
    Ministerio da Agricultura e Pecuaria (MAPA) - Portal de Dados Abertos
    Dataset : sisser3 - Sistema de Subvencao Economica ao Premio do Seguro Rural
    Pagina  : https://dados.agricultura.gov.br/dataset/sisser3
    Recurso : "PSR - 2016 a 2024" (CSV, ~297 MB, 1.048.565 apolices do mercado)

O QUE ESTE SCRIPT FAZ
    1. baixa o CSV completo do PSR 2016-2024 (todas as seguradoras);
    2. filtra apenas as apolices da Sompo Seguros S/A;
    3. grava data/principal/psr_2016a2024_sompo.csv (~5,4 MB, 18.872 apolices).

O arquivo filtrado ja esta versionado no repositorio; este script existe para
tornar a origem do dado auditavel e o pipeline reproduzivel do zero.

Uso:
    python scripts/prepare_data.py                 # usa o cache se ja existir
    python scripts/prepare_data.py --force         # rebaixa o CSV bruto
"""
import argparse
import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URL_PSR = ('https://dados.agricultura.gov.br/dataset/'
           'baefdc68-9bad-4204-83e8-f2888b79ab48/resource/'
           '54e04a6b-15b3-4bda-a330-b8e805deabe4/download/'
           'dados_abertos_psr_2016a2024csv.csv')

BRUTO = os.path.join(RAIZ, 'data', 'raw', 'dados_abertos_psr_2016a2024.csv')
SAIDA = os.path.join(RAIZ, 'data', 'principal', 'psr_2016a2024_sompo.csv')

SEGURADORA = 'Sompo'
CHUNK = 200_000


def baixar(force=False):
    """Baixa o CSV bruto do MAPA, se ainda nao estiver em disco."""
    if os.path.exists(BRUTO) and not force:
        print(f'CSV bruto ja existe ({os.path.getsize(BRUTO)/1024**2:.0f} MB): {BRUTO}')
        return BRUTO

    os.makedirs(os.path.dirname(BRUTO), exist_ok=True)
    print(f'Baixando de {URL_PSR}\n(cerca de 297 MB, pode levar alguns minutos)...')

    import urllib.request
    req = urllib.request.Request(URL_PSR, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(BRUTO, 'wb') as fp:
        baixados = 0
        while True:
            bloco = resp.read(1 << 20)
            if not bloco:
                break
            fp.write(bloco)
            baixados += len(bloco)
            print(f'\r  {baixados/1024**2:7.1f} MB', end='', flush=True)
    print(f'\nGravado em {BRUTO}')
    return BRUTO


def filtrar_sompo(caminho_bruto):
    """Le o CSV em blocos e mantem apenas as apolices da Sompo."""
    print(f'\nFiltrando NM_RAZAO_SOCIAL contendo "{SEGURADORA}"...')
    partes, total = [], 0
    for bloco in pd.read_csv(caminho_bruto, sep=';', encoding='latin-1',
                             dtype=str, chunksize=CHUNK, low_memory=False):
        bloco.columns = [c.strip() for c in bloco.columns]
        total += len(bloco)
        partes.append(bloco[bloco['NM_RAZAO_SOCIAL']
                            .fillna('')
                            .str.contains(SEGURADORA, case=False, na=False)])
        print(f'\r  {total:,} linhas lidas', end='', flush=True)

    df = pd.concat(partes, ignore_index=True)
    print(f'\n  Mercado total : {total:,} apolices')
    print(f'  Recorte Sompo : {len(df):,} apolices ({len(df)/total*100:.2f}%)')

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    df.to_csv(SAIDA, sep=';', index=False, encoding='utf-8')
    print(f'\nGravado: {SAIDA} ({os.path.getsize(SAIDA)/1024**2:.1f} MB)')

    ev = df['EVENTO_PREPONDERANTE'].astype(str).str.strip()
    n_sin = int((ev != '-').sum())
    print(f'  Sinistros observados: {n_sin:,} ({n_sin/len(df)*100:.1f}%)')
    print(f'  Safras              : {sorted(df["ANO_APOLICE"].dropna().unique())}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='rebaixa o CSV bruto mesmo se ja estiver em cache')
    args = parser.parse_args()

    if os.path.exists(SAIDA) and not args.force:
        print(f'O recorte Sompo ja existe: {SAIDA}')
        print('Use --force para refazer o download e o filtro a partir da fonte.')
        sys.exit(0)

    filtrar_sompo(baixar(force=args.force))
