"""
===============================================================================
ETAPA 3 — XGBoost global com quantis empiricos (validacao com origem movel)
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE SCRIPT FAZ
---------------------
1. Monta as variaveis explicativas (scripts/etapa2_variaveis.py)
2. Treina UM modelo para todas as 5.748 series (modelo global): ele aprende
   padroes comuns entre produtos e lojas, o que ajuda series intermitentes
3. Preve cada dia do teste (d_1824 a d_1913) com origem movel, como no
   Holt-Winters: reestima a cada 7 dias; dentro da semana o modelo fica fixo,
   mas as variaveis de venda sao atualizadas todo dia com a venda de ontem
4. Constroi os quantis pelo MESMO metodo do Holt-Winters (modulo comum):
   previsao + quantil dos erros dos ultimos 180 dias
5. Avalia pelas mesmas funcoes e salva tabelas, importancia das variaveis e
   figura

POR QUE HA UM PERIODO DE "CALIBRACAO" ANTES DO TESTE
----------------------------------------------------
Os quantis precisam dos erros que o modelo comete ao prever dias que ele
NAO viu. No Holt-Winters, os erros dentro do treino ja sao praticamente isso
(o modelo tem so 2 parametros). No XGBoost, nao: com milhares de arvores ele
"decora" parte do treino, e os erros dentro do treino ficam pequenos demais —
os quantis sairiam estreitos e o estoque, curto.

Solucao: a mesma origem movel e estendida 26 semanas (182 dias) para TRAS do
teste (d_1642 a d_1823). Essas previsoes nao sao avaliadas; servem so para
fornecer 180 dias de erros genuinos de previsao no primeiro dia do teste.
Sao 39 treinos no total (26 de calibracao + 13 de teste).

ESCOLHAS DO MODELO (feitas so com dados anteriores ao teste)
------------------------------------------------------------
- Janela de treino: os ultimos 3 anos (1.092 dias) antes de cada origem.
  Limita memoria e tempo; tres ciclos anuais bastam para a sazonalidade anual.
- Funcao objetivo e numero de arvores: escolhidos com --ajustar, numa
  validacao interna (28 dias imediatamente antes da 1a origem de calibracao).
  O resultado fica em resultados/tabelas/xgb_ajuste.txt e nas constantes
  OBJETIVO e N_ARVORES abaixo.

COMO USAR
---------
    python scripts/etapa3_xgboost.py --ajustar    (escolhe objetivo e arvores)
    python scripts/etapa3_xgboost.py              (roda tudo, ~40 min)
    python scripts/etapa3_xgboost.py --so-tabelas (refaz so as tabelas)

SAIDAS
------
dados/previsoes_xgboost.parquet              previsoes e quantis por serie-dia
resultados/tabelas/xgb_*.csv e .tex          mesmas tabelas do Holt-Winters
resultados/tabelas/xgb_importancia*.csv/.tex importancia das variaveis
resultados/figuras/fig_xgb_importancia.png
resultados/tabelas/xgb_resumo_execucao.txt
dados/previsoes_pontuais_xgboost.npz         previsoes de 1 passo, calibracao + teste (Etapa 5B)
===============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import (
    PASTA, PASTA_TAB, ARQUIVO_SAIDA, COR_PRINCIPAL, COR_TEXTO_SEC,
    estilo_eixos, salvar_figura, num_pt,
)
import etapa2_variaveis as ev
import etapa3_avaliacao as av

# -----------------------------------------------------------------------------
# CONFIGURACAO
# -----------------------------------------------------------------------------
PASSO_REESTIMACAO = 7                                  # igual ao Holt-Winters
SEMANAS_CALIBRACAO = 26                                # 182 dias >= 180 de erros
INICIO_CALIBRACAO = av.INICIO_TESTE - PASSO_REESTIMACAO * SEMANAS_CALIBRACAO  # d_1642
JANELA_TREINO = 1092                                   # 3 anos (156 semanas)
DIAS_VALIDACAO = 28                                    # so para --ajustar

# Definidos por --ajustar em 21/10/2026 (ver resultados/tabelas/xgb_ajuste.txt):
# erro quadratico teve RMSE de validacao 2,578 (90 arvores) contra 2,595 da
# Tweedie (546 arvores). A Tweedie teve MAE menor, mas o criterio fixado antes
# do ajuste era o RMSE, porque a previsao pontual deve representar a media.
OBJETIVO = "reg:squarederror"
N_ARVORES = 90

PARAMETROS = dict(
    eta=0.1, max_depth=8, min_child_weight=20, subsample=0.7,
    colsample_bytree=0.7, reg_lambda=1.0, max_bin=256, tree_method="hist",
    max_cat_to_onehot=1, nthread=16, seed=0,
)
OBJETIVOS = {"reg:tweedie": dict(objective="reg:tweedie", tweedie_variance_power=1.1),
             "reg:squarederror": dict(objective="reg:squarederror")}

ARQ_PREVISOES = PASTA / "previsoes_xgboost.parquet"
ORIGEM = "etapa3_xgboost.py"


def parametros(objetivo=OBJETIVO):
    return {**PARAMETROS, **OBJETIVOS[objetivo]}


def matriz(grade, mascara, y=None, quantil=False):
    """Matriz do XGBoost com as variaveis das celulas marcadas."""
    X = grade.tabela(mascara)[ev.VARIAVEIS]
    if quantil:
        return xgb.QuantileDMatrix(X, y, enable_categorical=True, max_bin=PARAMETROS["max_bin"])
    return xgb.DMatrix(X, y, enable_categorical=True)


# -----------------------------------------------------------------------------
# DADOS
# -----------------------------------------------------------------------------
def carregar():
    print("\n[1/5] Lendo a base e montando as variaveis...")
    base = pd.read_parquet(PASTA / ARQUIVO_SAIDA)
    classe = av.classificar_sbc(base)
    grade, disp, fechada, vendas = ev.montar_variaveis(base)
    teste = base.loc[(base["dia"] >= av.INICIO_TESTE) & base["disponivel"],
                     ["id", "store_id", "dept_id", "dia", "date", "vendas", "loja_fechada"]]
    del base
    print(f"      {len(ev.VARIAVEIS)} variaveis, {len(grade.ids):,} series")
    return grade, disp, fechada, vendas, teste, classe


# -----------------------------------------------------------------------------
# AJUSTE: objetivo e numero de arvores (so dados anteriores ao teste)
# -----------------------------------------------------------------------------
def ajustar(grade, disp, vendas):
    print("\n[ajuste] Validacao interna antes da 1a origem de calibracao...")
    dias = grade.dias[None, :]
    fim_treino = INICIO_CALIBRACAO - DIAS_VALIDACAO
    tr = disp & (dias >= fim_treino - JANELA_TREINO) & (dias < fim_treino)
    va = disp & (dias >= fim_treino) & (dias < INICIO_CALIBRACAO)
    dtr = matriz(grade, tr, vendas[tr], quantil=True)
    dva = matriz(grade, va, vendas[va])
    y = vendas[va]
    linhas = [f"Treino: d_{fim_treino - JANELA_TREINO} a d_{fim_treino - 1} "
              f"({int(tr.sum()):,} linhas) | validacao: d_{fim_treino} a "
              f"d_{INICIO_CALIBRACAO - 1} ({int(va.sum()):,} linhas)",
              f"Parametros fixos: {PARAMETROS}", ""]
    for nome in OBJETIVOS:
        t0 = time.time()
        m = xgb.train(parametros(nome), dtr, num_boost_round=1500,
                      evals=[(dva, "validacao")], early_stopping_rounds=30,
                      verbose_eval=False)
        p = np.maximum(m.predict(dva, iteration_range=(0, m.best_iteration + 1)), 0)
        linhas.append(f"{nome:<18} melhor n. de arvores: {m.best_iteration + 1:>5} | "
                      f"MAE {np.abs(y - p).mean():.4f} | RMSE {np.sqrt(((y - p) ** 2).mean()):.4f}"
                      f" | {time.time() - t0:.0f} s")
        print("      " + linhas[-1])
    linhas.append("\nCriterio de escolha: menor RMSE (a previsao pontual deve ser a media).")
    (PASTA_TAB / "xgb_ajuste.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("      resultados/tabelas/xgb_ajuste.txt")


# -----------------------------------------------------------------------------
# ORIGEM MOVEL
# -----------------------------------------------------------------------------
def rodar(grade, disp, vendas):
    origens = list(range(INICIO_CALIBRACAO, av.FIM_TESTE + 1, PASSO_REESTIMACAO))
    print(f"\n[2/5] Origem movel: {len(origens)} treinos "
          f"({SEMANAS_CALIBRACAO} de calibracao + "
          f"{sum(o >= av.INICIO_TESTE for o in origens)} de teste)...")
    dias = grade.dias[None, :]
    prev = np.full(vendas.shape, np.nan, dtype=np.float64)
    importancias = []
    for o in tqdm(origens, unit="origem"):
        tr = disp & (dias >= o - JANELA_TREINO) & (dias < o)
        dtr = matriz(grade, tr, vendas[tr], quantil=True)
        modelo = xgb.train(parametros(), dtr, num_boost_round=N_ARVORES)
        del dtr
        alvo = disp & (dias >= o) & (dias < o + PASSO_REESTIMACAO)
        prev[alvo] = modelo.predict(matriz(grade, alvo))
        if o >= av.INICIO_TESTE:
            importancias.append(modelo.get_score(importance_type="total_gain"))
    return prev, importancias


def montar_quantis(grade, disp, fechada, vendas, prev):
    """Erros genuinos de previsao (calibracao + teste ja observado) -> quantis
    dos dias de teste, pela mesma regra do Holt-Winters."""
    print("\n[3/5] Construindo os quantis (erros dos ultimos 180 dias)...")
    c0 = np.searchsorted(grade.dias, INICIO_CALIBRACAO)
    p = prev[:, c0:]
    erros = np.where(disp[:, c0:] & ~fechada[:, c0:], vendas[:, c0:] - p, np.nan)
    pos_teste = np.searchsorted(grade.dias[c0:], av.INICIO_TESTE)
    nomes = av.colunas_quantis()
    partes = []
    for i, sid in enumerate(grade.ids):
        alvo = pos_teste + np.flatnonzero(disp[i, c0 + pos_teste:])
        if len(alvo) == 0:
            continue
        emp, norm = av.quantis_janela(p[i], erros[i], alvo)
        df = pd.DataFrame(np.hstack([emp, norm]), columns=nomes)
        df.insert(0, "prev", p[i, alvo])
        df.insert(0, "dia", grade.dias[c0 + alvo])
        df.insert(0, "id", sid)
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


# -----------------------------------------------------------------------------
# IMPORTANCIA DAS VARIAVEIS
# -----------------------------------------------------------------------------
def importancia(importancias):
    """Participacao de cada variavel no ganho total das arvores, media dos 13
    modelos do teste. Ganho = quanto a variavel reduziu o erro ao ser usada."""
    print("\n[5/5] Importancia das variaveis...")
    part = pd.DataFrame(importancias).reindex(columns=ev.VARIAVEIS).fillna(0)
    part = part.div(part.sum(axis=1), axis=0).mean() * 100
    familia = {v: f for f, vs in ev.FAMILIAS.items() for v in vs}
    tab = pd.DataFrame({"Família": [familia[v] for v in part.index],
                        "Participação no ganho": part.values}, index=part.index)
    tab = tab.sort_values("Participação no ganho", ascending=False).rename_axis("Variável")
    fam = (tab.groupby("Família")["Participação no ganho"].sum()
           .sort_values(ascending=False).to_frame())
    fam["Variáveis"] = tab.groupby("Família").size()
    fam = fam[["Variáveis", "Participação no ganho"]]

    nota = ("participação de cada variável no ganho total das árvores (redução do erro "
            "de treino), em \\%, média dos 13 modelos reestimados no teste.")
    av.salvar_tabela(tab, "xgb_importancia", "XGBoost — importância das variáveis", ORIGEM,
                     formatos={"Participação no ganho": 1}, nota=nota)
    av.salvar_tabela(fam.rename_axis("Família"), "xgb_importancia_familias",
                     "XGBoost — importância por família de variáveis", ORIGEM,
                     formatos={"Variáveis": "int", "Participação no ganho": 1}, nota=nota)
    figura_importancia(tab, fam)
    return tab, fam


def figura_importancia(tab, fam, n=15):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6),
                                 gridspec_kw=dict(width_ratios=[1, 1.15]))
    for ax, serie, titulo in (
            (a1, fam["Participação no ganho"], "Por família de variáveis"),
            (a2, tab["Participação no ganho"].head(n), f"As {n} variáveis mais importantes")):
        s = serie.iloc[::-1]                           # maior no topo
        barras = ax.barh(s.index, s.values, color=COR_PRINCIPAL, height=0.7,
                         edgecolor="white", linewidth=1.5)
        ax.bar_label(barras, labels=[num_pt(v, 1) + "%" for v in s.values], padding=3,
                     fontsize=8, color=COR_TEXTO_SEC)
        ax.set_xlim(0, s.max() * 1.18)
        ax.set_title(titulo, fontsize=10, loc="left")
        ax.set_xlabel("% do ganho total")
        ax.tick_params(axis="y", labelsize=8, length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        estilo_eixos(ax, grade="x")
    fig.suptitle("XGBoost — importância das variáveis (média dos 13 modelos do teste)",
                 fontsize=11, x=0.01, ha="left")
    salvar_figura(fig, "fig_xgb_importancia.png")


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ajustar", action="store_true",
                        help="so escolhe objetivo e numero de arvores (validacao interna)")
    parser.add_argument("--so-tabelas", action="store_true",
                        help="nao treina: refaz so as tabelas de metricas a partir das previsoes")
    args = parser.parse_args()

    print("=" * 70)
    print("ETAPA 3 — XGBOOST GLOBAL COM QUANTIS EMPIRICOS (ORIGEM MOVEL)")
    print("=" * 70)

    if args.so_tabelas:
        prev_df = pd.read_parquet(ARQ_PREVISOES)
    else:
        inicio = time.time()
        grade, disp, fechada, vendas, teste, classe = carregar()
        if args.ajustar:
            ajustar(grade, disp, vendas)
            return
        prev, importancias = rodar(grade, disp, vendas)
        # Para a Etapa 5B: previsoes de 1 passo fora da amostra (calibracao + teste)
        c0 = np.searchsorted(grade.dias, INICIO_CALIBRACAO)
        np.savez_compressed(PASTA / "previsoes_pontuais_xgboost.npz", ids=grade.ids,
                            dias=grade.dias[c0:], prev=prev[:, c0:].astype(np.float32))
        print("      previsoes pontuais (Etapa 5B): dados/previsoes_pontuais_xgboost.npz")
        q = montar_quantis(grade, disp, fechada, vendas, prev)
        prev_df = teste.merge(q, on=["id", "dia"], how="inner", validate="1:1")
        assert len(prev_df) == len(teste), (len(prev_df), len(teste))
        assert prev_df["prev"].notna().all()
        prev_df["classe"] = prev_df["id"].map(classe)
        prev_df.to_parquet(ARQ_PREVISOES, index=False)
        print(f"      previsoes salvas em: dados/{ARQ_PREVISOES.name}")
        duracao = time.time() - inicio

    aval = prev_df[~prev_df["loja_fechada"]]
    print("\n[4/5] Tabelas de avaliacao...")
    resultados = av.tabelas_modelo(aval, "xgb", "XGBoost", ORIGEM)
    av.tabelas_modelo(aval, "xgb", "XGBoost", ORIGEM, regra="cima", sufixo="_sensibilidade")

    if not args.so_tabelas:
        tab, fam = importancia(importancias)
        resumo = [
            f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: {duracao / 60:.1f} min",
            f"Objetivo: {OBJETIVO} | arvores: {N_ARVORES} | janela de treino: "
            f"{JANELA_TREINO} dias | parametros: {PARAMETROS}",
            f"Origens: {SEMANAS_CALIBRACAO} de calibracao (d_{INICIO_CALIBRACAO} a "
            f"d_{av.INICIO_TESTE - 1}) + 13 de teste, a cada {PASSO_REESTIMACAO} dias",
            f"Dias-serie previstos no teste: {len(prev_df):,} | series: "
            f"{prev_df['id'].nunique():,}",
            f"Previsao media no teste: {prev_df['prev'].mean():.3f} | demanda media: "
            f"{prev_df['vendas'].mean():.3f}",
            "", "Importancia por familia (% do ganho):",
            *[f"  {f:<28} {v:5.1f}%" for f, v in fam["Participação no ganho"].items()],
        ]
        (PASTA_TAB / "xgb_resumo_execucao.txt").write_text("\n".join(resumo) + "\n",
                                                           encoding="utf-8")
        print("\n" + "\n".join("      " + r for r in resumo))
        print("\nTOP 15 VARIAVEIS\n" + tab.head(15).to_string())
    av.imprimir(*resultados)


if __name__ == "__main__":
    main()
