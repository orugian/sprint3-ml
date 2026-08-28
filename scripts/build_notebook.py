"""
Constroi o notebook Sprint3-ML.ipynb a partir de scripts/nb_source.py.

O arquivo-fonte usa marcadores no estilo Jupytter/VSCode:
    # %%              -> inicia uma celula de codigo
    # %% [markdown]   -> inicia uma celula de markdown (linhas prefixadas por '# ')

Uso:
    python scripts/build_notebook.py            # gera o .ipynb sem executar
    python scripts/build_notebook.py --run      # gera e executa (grava as saidas)
"""
import argparse
import os
import re
import sys

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTE = os.path.join(RAIZ, 'scripts', 'nb_source.py')
SAIDA = os.path.join(RAIZ, 'Sprint3-ML.ipynb')

TITULO = 'SOMPO 2026 - Sprint 3 - Modelagem de Machine Learning'


def dividir_em_celulas(texto):
    """Quebra o fonte nos marcadores '# %%' e devolve [(tipo, conteudo), ...]."""
    linhas = texto.splitlines()
    celulas, tipo_atual, buffer = [], None, []

    def fechar():
        if tipo_atual is None:
            return
        corpo = '\n'.join(buffer).strip('\n')
        if corpo.strip():
            celulas.append((tipo_atual, corpo))

    for linha in linhas:
        if re.match(r'^# %%\s*\[markdown\]\s*$', linha):
            fechar()
            tipo_atual, buffer = 'markdown', []
        elif re.match(r'^# %%\s*$', linha):
            fechar()
            tipo_atual, buffer = 'code', []
        else:
            if tipo_atual is not None:
                buffer.append(linha)
    fechar()
    return celulas


def limpar_markdown(corpo):
    """Remove o prefixo de comentario '# ' das linhas de uma celula markdown."""
    saida = []
    for linha in corpo.split('\n'):
        if linha.startswith('# '):
            saida.append(linha[2:])
        elif linha.strip() == '#':
            saida.append('')
        else:
            saida.append(linha)
    return '\n'.join(saida).strip('\n')


def construir():
    with open(FONTE, encoding='utf-8') as fp:
        texto = fp.read()

    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        'title': TITULO,
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': sys.version.split()[0],
                          'file_extension': '.py', 'mimetype': 'text/x-python',
                          'nbconvert_exporter': 'python', 'pygments_lexer': 'ipython3'},
    })

    n_md = n_code = 0
    for tipo, corpo in dividir_em_celulas(texto):
        if tipo == 'markdown':
            nb.cells.append(nbf.v4.new_markdown_cell(limpar_markdown(corpo)))
            n_md += 1
        else:
            nb.cells.append(nbf.v4.new_code_cell(corpo))
            n_code += 1

    with open(SAIDA, 'w', encoding='utf-8') as fp:
        nbf.write(nb, fp)

    print(f'Notebook gerado: {SAIDA}')
    print(f'  celulas markdown : {n_md}')
    print(f'  celulas de codigo: {n_code}')
    print(f'  total            : {n_md + n_code}')
    return SAIDA


def executar(caminho):
    """Executa o notebook no diretorio raiz do projeto e regrava com as saidas."""
    from nbclient import NotebookClient

    nb = nbf.read(caminho, as_version=4)
    cliente = NotebookClient(nb, timeout=3600, kernel_name='python3',
                             resources={'metadata': {'path': RAIZ}})
    print('Executando o notebook (pode levar alguns minutos)...')
    cliente.execute()
    nbf.write(nb, caminho)
    print(f'Notebook executado e salvo com as saidas: {caminho}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true',
                        help='executa o notebook apos gera-lo')
    args = parser.parse_args()

    caminho = construir()
    if args.run:
        executar(caminho)
