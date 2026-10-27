"""
===============================================================================
ETAPA 4 — Mecanismo da inversao de ordem entre a Etapa 3 e a Etapa 4
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

A PERGUNTA
----------
No nivel de 80% (razao 1:4), o XGBoost e mais barato que o Holt-Winters na
avaliacao de periodo unico (Etapa 3), mas mais caro na simulacao com
perecibilidade (Etapa 4). Por que?

HIPOTESES TESTADAS
------------------
1. PERSISTENCIA: o erro de um modelo se repete em dias seguidos (ex.: erra
   para cima varios dias), acumulando estoque que apodrece.
   Medida: autocorrelacao de 1 dia do erro de previsao e da sobra, e duracao
   media das sequencias de superestimacao.
2. ASSIMETRIA: o mesmo erro medio, mas dividido de forma diferente entre
   errar para cima (sobra) e errar para baixo (falta).
   Medida: vies, proporcao e tamanho dos erros para cima e para baixo, na
   previsao pontual e na decisao de estoque.
3. DECOMPOSICAO DO CUSTO: quanto da diferenca de custo vem de sobra/perda e
   quanto vem de falta, em cada etapa. Mostra QUAL parcela muda de peso.
4. NIVEL ERRADO: a inversao persiste se cada modelo usar o seu nivel de
   servico otimo (da varredura da Etapa 4) em vez do newsvendor?

PRE-REQUISITO: etapa3_comparacao.py e etapa4_simulacao.py ja rodados
SAIDAS: resultados/tabelas/mec_*.csv e .tex
===============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import PASTA, PASTA_TAB, num_pt
import etapa3_avaliacao as av

ORIGEM = "etapa4_mecanismo.py"
MODELOS = {"Holt-Winters": "previsoes_holt_winters.parquet",
           "XGBoost": "previsoes_xgboost.parquet",
           "Combinação": "previsoes_combinacao.parquet"}
PARES = [("XGBoost", "Holt-Winters"), ("Combinação", "Holt-Winters")]
THETA_BASE = 0.10
BASE = ("teste de 90 dias (26/01 a 24/04/2016), 5.748 séries, quantis empíricos, "
        "estoque = quantil arredondado para o inteiro mais próximo.")


def carregar():
    cols = (["id", "dia", "vendas", "loja_fechada", "prev"]
            + [f"q_emp_{av.pct(t)}" for t in av.NIVEIS])
    out = {}
    for nome, arq in MODELOS.items():
        df = pd.read_parquet(PASTA / arq, columns=cols).sort_values(["id", "dia"],
                                                                    ignore_index=True)
        out[nome] = df[~df["loja_fechada"]].reset_index(drop=True)
    return out


def grade(df, valores):
    """Vetor por serie-dia -> matriz serie x dia (NaN onde nao ha dado)."""
    ids, lin = np.unique(df["id"].to_numpy(), return_inverse=True)
    m = np.full((len(ids), av.FIM_TESTE - av.INICIO_TESTE + 1), np.nan)
    m[lin, df["dia"].to_numpy() - av.INICIO_TESTE] = valores
    return m


def autocorrelacao(m):
    """Autocorrelacao de 1 dia dentro de cada serie (desvios em relacao a
    media da propria serie). Devolve (agrupada, mediana por serie)."""
    x = m - np.nanmean(m, axis=1, keepdims=True)
    par = ~np.isnan(x[:, 1:]) & ~np.isnan(x[:, :-1])
    num = np.where(par, x[:, 1:] * x[:, :-1], 0).sum(axis=1)
    den = np.where(~np.isnan(x), x ** 2, 0).sum(axis=1)
    ok = den > 0
    return num[ok].sum() / den[ok].sum(), np.median(num[ok] / den[ok])


def duracao_media(ind):
    """Duracao media (dias) das sequencias consecutivas em que ind e verdadeiro."""
    ind = np.nan_to_num(ind, nan=0).astype(bool)
    inicios = ind[:, 0].sum() + (ind[:, 1:] & ~ind[:, :-1]).sum()
    return ind.sum() / inicios


# -----------------------------------------------------------------------------
# HIPOTESE 2 — ASSIMETRIA
# -----------------------------------------------------------------------------
def assimetria(prev):
    pontual, decisao = {}, {}
    for nome, df in prev.items():
        d = df["vendas"].to_numpy(float)
        p = np.maximum(df["prev"].to_numpy(float), 0)
        e = p - d                                          # > 0: previu demais
        cima, baixo = e > 0, e < 0
        pontual[nome] = {
            "Viés (previsão − demanda)": e.mean(),
            "% dias previu demais": cima.mean(),
            "Erro médio para cima": e[cima].mean(),
            "% dias previu de menos": baixo.mean(),
            "Erro médio para baixo": -e[baixo].mean(),
            "Parcela do MAE para cima": e[cima].sum() / np.abs(e).sum(),
        }
        for tau in av.NIVEIS:
            S = av.decisao(np.maximum(df[f"q_emp_{av.pct(tau)}"].to_numpy(float), 0))
            sobra, falta = np.maximum(S - d, 0), np.maximum(d - S, 0)
            decisao[(f"{av.pct(tau)}%", nome)] = {
                "Alvo médio": S.mean(),
                "Folga do alvo": (S - p).mean(),
                "% dias com sobra": (sobra > 0).mean(),
                "Sobra média": sobra.mean(),
                "% dias com falta": (falta > 0).mean(),
                "Falta média": falta.mean(),
            }
    t1 = pd.DataFrame(pontual).T.rename_axis("Modelo")
    t2 = pd.DataFrame(decisao).T
    t2.index.names = ["Nível", "Modelo"]
    av.salvar_tabela(
        t1, "mec_erro_pontual", "Assimetria do erro da previsão pontual", ORIGEM,
        formatos={"% dias previu demais": "pct", "% dias previu de menos": "pct",
                  "Parcela do MAE para cima": "pct"},
        nota=(f"{BASE} Erro = previsão − demanda (positivo = previu demais), em "
              "unidades por série-dia. Os dias restantes são acertos exatos."))
    av.salvar_tabela(
        t2, "mec_decisao", "Assimetria da decisão de estoque (período único)", ORIGEM,
        formatos={"% dias com sobra": "pct", "% dias com falta": "pct"},
        nota=(f"{BASE} Alvo: estoque-alvo do dia. Folga: alvo − previsão pontual "
              "(a margem de segurança). Sobra e falta em unidades por série-dia, "
              "sem carregar estoque de um dia para o outro (como na Etapa 3)."))
    return t1, t2


# -----------------------------------------------------------------------------
# HIPOTESE 1 — PERSISTENCIA
# -----------------------------------------------------------------------------
def persistencia(prev):
    linhas = {}
    for nome, df in prev.items():
        d = df["vendas"].to_numpy(float)
        e = np.maximum(df["prev"].to_numpy(float), 0) - d
        ac_e, med_e = autocorrelacao(grade(df, e))
        S = av.decisao(np.maximum(df["q_emp_80"].to_numpy(float), 0))
        ac_s, med_s = autocorrelacao(grade(df, np.maximum(S - d, 0)))
        linhas[nome] = {
            "AC1 do erro (agrupada)": ac_e,
            "AC1 do erro (mediana)": med_e,
            "AC1 da sobra 80% (agrupada)": ac_s,
            "AC1 da sobra 80% (mediana)": med_s,
            "Sequência média previu demais (dias)": duracao_media(grade(df, (e > 0) * 1.0)),
            "Sequência média com sobra 80% (dias)": duracao_media(grade(df, (S > d) * 1.0)),
        }
    t = pd.DataFrame(linhas).T.rename_axis("Modelo")
    av.salvar_tabela(
        t, "mec_persistencia", "Persistência do erro de um dia para o outro", ORIGEM,
        nota=(f"{BASE} AC1: autocorrelação de 1 dia dentro de cada série (agrupada: "
              "todas as séries juntas; mediana: mediana das séries). Sequência média: "
              "duração média, em dias, das sequências consecutivas de dias com "
              "previsão acima da demanda (ou com sobra no nível de 80\\%)."))
    return t


# -----------------------------------------------------------------------------
# DECOMPOSICAO DA DIFERENCA DE CUSTO: ETAPA 3 x ETAPA 4
# -----------------------------------------------------------------------------
def decomposicao(prev, sim):
    e3 = {nome: av.metricas(df, "emp") for nome, df in prev.items()}
    linhas, longa = {}, []
    for tau, cr in av.CUSTO_FALTA.items():
        k = av.pct(tau)
        for a, b in PARES:
            lin = {
                "E3: sobra": e3[a][f"Sobra {k}%"] - e3[b][f"Sobra {k}%"],
                "E3: falta × c_r": cr * (e3[a][f"Falta {k}%"] - e3[b][f"Falta {k}%"]),
            }
            lin["E3: custo"] = lin["E3: sobra"] + lin["E3: falta × c_r"]
            for th in sorted(sim["theta"].unique()):
                s = sim[(sim["theta"] == th) & (sim["Nível"] == k)].set_index("Modelo")
                dp = s.loc[a, "Perdidas"] - s.loc[b, "Perdidas"]
                df_ = cr * (s.loc[a, "Faltas"] - s.loc[b, "Faltas"])
                longa.append({"razao": f"1:{cr}", "comparacao": f"{a} vs {b}", "theta": th,
                              "e4_perda": dp, "e4_falta_cr": df_, "e4_custo": dp + df_,
                              **{c: v for c, v in lin.items()}})
                if th == THETA_BASE:
                    lin.update({"E4: perda": dp, "E4: falta × c_r": df_,
                                "E4: custo": dp + df_})
            linhas[(f"1:{cr} ({k}%)", f"{a} − {b}")] = lin
    t = pd.DataFrame(linhas).T
    t.index.names = ["Razão (nível)", "Comparação"]
    av.salvar_tabela(
        t, "mec_decomposicao", "De onde vem a diferença de custo entre os modelos", ORIGEM,
        nota=(f"{BASE} Diferenças em unidades de custo por série-dia (negativo = primeiro "
              "modelo mais barato). E3: período único (Etapa 3), custo = sobra + $c_r$ × "
              "falta. E4: simulação com perecibilidade θ = 0,10 (Etapa 4), custo = perda "
              "por validade + $c_r$ × falta. A soma das duas parcelas é a diferença de custo."))
    pd.DataFrame(longa).to_csv(PASTA_TAB / "mec_decomposicao_numerico.csv", index=False,
                               encoding="utf-8-sig")
    return t


# -----------------------------------------------------------------------------
# HIPOTESE 4 — CADA MODELO NO SEU NIVEL OTIMO
# -----------------------------------------------------------------------------
def nivel_otimo(sim):
    linhas = {}
    s = sim[sim["theta"] == THETA_BASE]
    for tau, cr in av.CUSTO_FALTA.items():
        col = f"Custo 1:{cr}"
        for nome in MODELOS:
            g = s[s["Modelo"] == nome]
            i = g[col].idxmin()
            linhas[(f"1:{cr} ({av.pct(tau)}%)", nome)] = {
                "Custo no newsvendor": g.loc[g["Nível"] == av.pct(tau), col].iloc[0],
                "Nível ótimo": int(g.loc[i, "Nível"]),
                "Custo no ótimo": g.loc[i, col],
            }
    t = pd.DataFrame(linhas).T
    t.index.names = ["Razão (nível)", "Modelo"]
    for c in ("Custo no newsvendor", "Custo no ótimo"):
        t[f"Ordem ({c.split(' no ')[1]})"] = t.groupby(level=0, sort=False)[c].rank(
            method="min").astype(int)
    av.salvar_tabela(
        t, "mec_nivel_otimo", "Ordem dos modelos no nível do newsvendor e no nível ótimo",
        ORIGEM, formatos={"Nível ótimo": "int", "Ordem (newsvendor)": "int",
                          "Ordem (ótimo)": "int"},
        nota=(f"{BASE} Simulação com θ = 0,10; custo = perda por validade + $c_r$ × falta. "
              "Nível ótimo: o que minimiza o custo de cada modelo na varredura de 70\\% "
              "a 99\\%."))
    return t


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inicio = time.time()
    print("=" * 70)
    print("ETAPA 4 — MECANISMO DA INVERSAO DE ORDEM")
    print("=" * 70)
    prev = carregar()
    sim = pd.read_csv(PASTA_TAB / "sim_resultados_completos.csv")
    pd.set_option("display.width", 220)
    f = lambda v: f"{v:.4f}"
    t1, t2 = assimetria(prev)
    print("\nERRO PONTUAL\n" + t1.to_string(float_format=f))
    print("\nDECISAO\n" + t2.to_string(float_format=f))
    print("\nPERSISTENCIA\n" + persistencia(prev).to_string(float_format=f))
    print("\nDECOMPOSICAO\n" + decomposicao(prev, sim).to_string(float_format=f))
    print("\nNIVEL OTIMO\n" + nivel_otimo(sim).to_string(float_format=f))
    df = next(iter(prev.values()))
    (PASTA_TAB / "mec_resumo_execucao.txt").write_text(
        f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: "
        f"{(time.time() - inicio) / 60:.1f} min\n"
        f"Series: {df['id'].nunique():,} | dias-serie avaliados por modelo: {len(df):,}\n",
        encoding="utf-8")
    print("      resultados/tabelas/mec_resumo_execucao.txt")


if __name__ == "__main__":
    main()
