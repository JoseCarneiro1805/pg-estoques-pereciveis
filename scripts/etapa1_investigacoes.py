"""
===============================================================================
ETAPA 1 — Investigacoes complementares da base
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

Tres investigacoes levantadas na verificacao de disponibilidade:

A. EVENTO NA LOJA CA_2 EM MAIO/2015
   Muitas series da CA_2 tem a primeira venda exatamente em d_1579. Isso e
   uma mudanca estrutural (reforma, reabertura, ampliacao de sortimento)?
   Afeta a comparacao entre lojas na politica de transbordo.

B. SERIES COM PRECO MUITO ANTES DA PRIMEIRA VENDA
   Ex.: FOODS_2_101 na CA_2 tem preco desde o dia 1 e so vende 4 anos depois.
   Quantos casos assim existem? Se forem muitos, o criterio de disponibilidade
   (presenca de preco) precisa ser revisto.

C. VALOR EXTREMO DE 763 UNIDADES
   FOODS_3_090 na CA_3 em 14/09/2013. O valor e MANTIDO na base (venda real;
   a cauda importa para previsao por quantis). Aqui so verificamos se houve
   queda de preco naquela semana.

PRE-REQUISITO: rodar antes scripts/etapa1_preparar_dados.py
SAIDAS: resultados/tabelas/investigacoes_etapa1.txt
        resultados/figuras/fig_vendas_mensais_lojas.png
===============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import (
    PASTA, PASTA_FIG, PASTA_TAB, LOJAS, ARQUIVO_SAIDA, COR_TEXTO_SEC,
    num_pt, estilo_eixos, salvar_figura,
)

DIA_EVENTO = 1579                 # dia com o pico de primeiras vendas na CA_2
JANELA = 14                       # dias antes/depois considerados "proximos"

# Cores das 4 lojas (ordem fixa da paleta categorica de referencia)
CORES_LOJAS = {"CA_1": "#2a78d6", "CA_2": "#eb6834",
               "CA_3": "#1baf7a", "CA_4": "#eda100"}

relatorio = []


def escreve(texto=""):
    print(texto)
    relatorio.append(texto)


def titulo(texto):
    escreve("\n" + "=" * 70)
    escreve(texto)
    escreve("=" * 70)


# -----------------------------------------------------------------------------
# Resumo por serie: primeiro preco e primeira venda
# -----------------------------------------------------------------------------
def resumo_series(base):
    g = base.groupby("id")
    r = pd.DataFrame({
        "store_id": g["store_id"].first(),
        "primeiro_preco": base[base["disponivel"]].groupby("id")["dia"].min(),
        "primeira_venda": base[base["vendas"] > 0].groupby("id")["dia"].min(),
    })
    r["atraso"] = r["primeira_venda"] - r["primeiro_preco"]
    return r


# -----------------------------------------------------------------------------
# A. Evento da CA_2
# -----------------------------------------------------------------------------
def investigar_ca2(base, series, datas):
    titulo("A. EVENTO NA LOJA CA_2 — MAIO/2015")
    data_evento = datas[DIA_EVENTO]
    dia_de = lambda d: int(datas.index[datas >= pd.Timestamp(d)][0])   # data -> d_
    ini_2014, ini_2016 = dia_de("2014-01-01"), dia_de("2016-01-01")
    escreve(f"d_{DIA_EVENTO} = {data_evento.date()} ({data_evento.day_name()})")

    # --- A1. primeiras vendas no dia e ao redor, por loja ---
    escreve(f"\nA1. Series com PRIMEIRA VENDA em d_{DIA_EVENTO} e ao redor, por loja")
    escreve(f"  {'loja':<6}{'no dia':>8}{'+-7 dias':>10}{'+-14 dias':>11}"
            f"{'+-28 dias':>11}{'   entrada tipica*':>18}")
    for loja in LOJAS:
        pv = series.loc[series["store_id"] == loja, "primeira_venda"]
        dist = (pv - DIA_EVENTO).abs()
        # referencia: media de primeiras vendas por janela de 29 dias em 2014-2015
        tipica = ((pv >= ini_2014) & (pv < ini_2016)).sum() / ((ini_2016 - ini_2014) / 29)
        escreve(f"  {loja:<6}{int((dist == 0).sum()):>8}{int((dist <= 7).sum()):>10}"
                f"{int((dist <= 14).sum()):>11}{int((dist <= 28).sum()):>11}"
                f"{tipica:>18.1f}")
    escreve("  * media de series que estreiam por janela de 29 dias em 2014-2015")

    # Das series da CA_2 que estreiam no dia, quantas ja tinham preco antes?
    ca2_dia = series[(series["store_id"] == "CA_2")
                     & (series["primeira_venda"] == DIA_EVENTO)]
    escreve(f"\n  Series da CA_2 que estreiam exatamente em d_{DIA_EVENTO}: {len(ca2_dia)}")
    escreve(f"    ja tinham preco ANTES do dia (atraso > 6): "
            f"{int((ca2_dia['atraso'] > 6).sum())}")
    escreve(f"    preco surge na mesma semana ..............: "
            f"{int((ca2_dia['atraso'] <= 6).sum())}")

    # --- A2. volume de vendas mensal por loja e variacao anual ---
    disp = base[base["disponivel"]]
    mensal = (disp.groupby(["store_id", pd.Grouper(key="date", freq="MS")])["vendas"]
              .sum().unstack(0))
    mensal = mensal.iloc[1:-1]           # jan/2011 (3 dias) e abr/2016 incompletos
    variacao = mensal.pct_change(12) * 100

    escreve("\nA2. Vendas totais mensais por loja (unidades) e variacao sobre o")
    escreve("    mesmo mes do ano anterior (remove a sazonalidade anual)")
    escreve(f"  {'mes':<9}" + "".join(f"{l:>17}" for l in LOJAS))
    periodo = mensal.loc["2014-10-01":"2015-12-01"]
    for mes, lin in periodo.iterrows():
        cel = "".join(f"{num_pt(lin[l]):>9} {variacao.loc[mes, l]:+5.0f}%  "
                      for l in LOJAS)
        escreve(f"  {mes:%Y-%m}  {cel}")

    # --- A3. dias com a loja praticamente sem venda (fechamento?) ---
    diario = disp.groupby(["store_id", "date"])["vendas"].sum().unstack(0)
    mediana_loja = diario.rolling(28, center=True, min_periods=7).median()
    quase_zero = diario < 0.10 * mediana_loja
    escreve("\nA3. Dias com vendas da loja abaixo de 10% da mediana movel "
            "(possivel loja fechada)")
    for loja in LOJAS:
        dias = diario.index[quase_zero[loja]]
        natal = [d for d in dias if d.month == 12 and d.day == 25]
        outros = [d for d in dias if not (d.month == 12 and d.day == 25)]
        escreve(f"  {loja}: {len(natal)} Natais"
                + (f"; outros: {', '.join(str(d.date()) for d in outros)}"
                   if outros else "; nenhum outro dia"))

    # --- A4. comportamento da CA_2 em torno do evento ---
    escreve(f"\nA4. CA_2 — vendas totais por semana em torno de d_{DIA_EVENTO}")
    ca2 = disp[disp["store_id"] == "CA_2"]
    sem = ca2[(ca2["dia"] >= DIA_EVENTO - 70) & (ca2["dia"] < DIA_EVENTO + 70)].copy()
    sem["semana"] = (sem["dia"] - DIA_EVENTO) // 7
    tab = sem.groupby("semana").agg(
        inicio=("date", "min"), vendas=("vendas", "sum"),
        series=("id", "nunique"),
        zeros=("vendas", lambda v: (v == 0).mean()),
    )
    escreve(f"  {'semana':>7}  {'inicio':<11}{'vendas':>9}{'series ativas':>15}"
            f"{'% zeros':>9}")
    for s, lin in tab.iterrows():
        marca = "  <- d_1579" if s == 0 else ""
        escreve(f"  {s:>+7}  {lin['inicio'].date()!s:<11}{num_pt(lin['vendas']):>9}"
                f"{lin['series']:>15,}{lin['zeros']:>9.1%}{marca}")

    # --- A5. o salto vem dos itens novos ou dos que ja existiam? ---
    # Itens "antigos": ja vendiam antes de 2014 em cada loja. Compara as vendas
    # desses mesmos itens em jun-dez/2015 com jun-dez/2014 (mesma estacao).
    escreve("\nA5. Vendas jun-dez/2015 x jun-dez/2014, so itens que ja vendiam "
            "antes de 2014")
    antigos = series.index[series["primeira_venda"] < ini_2014]
    b_ant = disp[disp["id"].isin(antigos)]
    for loja in LOJAS:
        b = b_ant[b_ant["store_id"] == loja]
        v14 = b.loc[b["date"].between("2014-06-01", "2014-12-31"), "vendas"].sum()
        v15 = b.loc[b["date"].between("2015-06-01", "2015-12-31"), "vendas"].sum()
        n = b["id"].nunique()
        escreve(f"  {loja}: {n:>5,} itens | 2014: {num_pt(v14):>9} | "
                f"2015: {num_pt(v15):>9} | variacao {(v15 / v14 - 1) * 100:+.0f}%")

    figura_mensal(mensal, data_evento)
    return mensal


def figura_mensal(mensal, data_evento):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for loja in LOJAS:
        ax.plot(mensal.index, mensal[loja], color=CORES_LOJAS[loja],
                linewidth=2, label=loja)
        ax.annotate(loja, xy=(mensal.index[-1], mensal[loja].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#0b0b0b")
    ax.axvline(data_evento, color=COR_TEXTO_SEC, linewidth=1, linestyle="--")
    ax.text(data_evento, ax.get_ylim()[1], " maio/2015 (d_1579)", va="top",
            fontsize=9, color=COR_TEXTO_SEC)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Unidades vendidas no mês")
    ax.set_xlabel("Data")
    ax.set_title("Vendas mensais por loja — FOODS (dias disponíveis)",
                 fontsize=11, loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: num_pt(v)))
    ax.legend(frameon=False, ncol=4, loc="lower left", fontsize=9)
    estilo_eixos(ax)
    salvar_figura(fig, "fig_vendas_mensais_lojas.png")


# -----------------------------------------------------------------------------
# B. Series com preco muito antes da primeira venda
# -----------------------------------------------------------------------------
def investigar_atrasos(series, datas):
    titulo("B. SERIES COM PRECO MUITO ANTES DA PRIMEIRA VENDA")
    n = len(series)
    a = series["atraso"]
    escreve("Atraso = dias entre o primeiro preco e a primeira venda positiva.")
    escreve("Ate 6 dias e esperado (o preco e semanal).\n")
    for lim in (6, 28, 90, 365):
        k = int((a > lim).sum())
        escreve(f"  atraso > {lim:>3} dias: {k:>4} series ({k / n:.2%} do total)")

    longos = series[a > 6].sort_values("atraso", ascending=False)
    dias_perdidos = int((longos["atraso"] - 6).clip(lower=0).sum())
    escreve(f"\n  Dias-serie com preco mas antes da 1a venda (acima de 6 dias): "
            f"{dias_perdidos:,}")
    escreve(f"  Isso equivale a {dias_perdidos / 8_423_403:.3%} dos dias disponiveis.")

    escreve("\n  Casos (atraso > 6 dias):")
    escreve(f"  {'serie':<30}{'1o preco':>12}{'1a venda':>12}{'atraso':>8}")
    for sid, lin in longos.iterrows():
        escreve(f"  {sid.replace('_validation', ''):<30}"
                f"{datas[lin['primeiro_preco']].date()!s:>12}"
                f"{datas[lin['primeira_venda']].date()!s:>12}{lin['atraso']:>8,}")
    escreve("\n  Por loja: " + ", ".join(
        f"{l} {int((longos['store_id'] == l).sum())}" for l in LOJAS))


# -----------------------------------------------------------------------------
# C. Valor extremo de 763 unidades
# -----------------------------------------------------------------------------
def investigar_extremo(base):
    titulo("C. VALOR EXTREMO (MANTIDO NA BASE)")
    i = base["vendas"].idxmax()
    pico = base.loc[i]
    item, loja, data = pico["item_id"], pico["store_id"], pico["date"]
    escreve(f"{item} na loja {loja} em {data.date()} ({data.day_name()}): "
            f"{pico['vendas']:.0f} unidades")
    escreve(f"Evento no calendario: {pico['event_name_1'] if pd.notna(pico['event_name_1']) else 'nenhum'}"
            f" | SNAP: {'sim' if pico['snap'] == 1 else 'nao'}")

    serie = base[(base["item_id"] == item) & (base["store_id"] == loja)]
    semana = pico["wm_yr_wk"]

    # preco semanal em torno do pico
    precos = serie.groupby("wm_yr_wk").agg(
        inicio=("date", "min"), preco=("sell_price", "first"),
        vendas=("vendas", "sum"))
    pos = precos.index.get_loc(semana)
    escreve("\nPreco e vendas semanais em torno do pico:")
    escreve(f"  {'semana':<8}{'inicio':<12}{'preco':>8}{'vendas':>8}")
    for wk, lin in precos.iloc[max(pos - 5, 0):pos + 4].iterrows():
        marca = "  <- semana do pico" if wk == semana else ""
        escreve(f"  {wk:<8}{lin['inicio'].date()!s:<12}{lin['preco']:>8.2f}"
                f"{lin['vendas']:>8.0f}{marca}")
    ref = precos["preco"].iloc[max(pos - 8, 0):pos]
    escreve(f"\n  preco na semana do pico: {precos['preco'].iloc[pos]:.2f} | "
            f"mediana das 8 semanas anteriores: {ref.median():.2f}")
    escreve(f"  preco minimo historico da serie: {serie['sell_price'].min():.2f} | "
            f"maximo: {serie['sell_price'].max():.2f}")

    # vendas diarias ao redor
    d0 = pico["dia"]
    viz = serie[(serie["dia"] >= d0 - 7) & (serie["dia"] <= d0 + 7)]
    escreve("\nVendas diarias de 7 dias antes a 7 dias depois:")
    escreve("  " + " ".join(f"{v:.0f}" for v in viz["vendas"]))
    escreve(f"  media da serie (dias disponiveis): "
            f"{serie.loc[serie['disponivel'], 'vendas'].mean():.1f} | "
            f"percentil 99: {serie.loc[serie['disponivel'], 'vendas'].quantile(0.99):.0f}")

    # mesmo item nas outras lojas no mesmo dia
    outras = base[(base["item_id"] == item) & (base["date"] == data)]
    escreve("\nMesmo item no mesmo dia em todas as lojas:")
    for _, lin in outras.sort_values("store_id").iterrows():
        escreve(f"  {lin['store_id']}: {lin['vendas']:.0f} unidades, "
                f"preco {lin['sell_price']:.2f}")


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    base = pd.read_parquet(PASTA / ARQUIVO_SAIDA, columns=[
        "id", "item_id", "store_id", "dia", "date", "wm_yr_wk", "vendas",
        "sell_price", "disponivel", "snap", "event_name_1",
    ])
    datas = base.drop_duplicates("dia").set_index("dia")["date"].sort_index()
    series = resumo_series(base)

    investigar_ca2(base, series, datas)
    investigar_atrasos(series, datas)
    investigar_extremo(base)

    PASTA_TAB.mkdir(parents=True, exist_ok=True)
    arq = PASTA_TAB / "investigacoes_etapa1.txt"
    arq.write_text("\n".join(relatorio) + "\n", encoding="utf-8")
    print("\nRelatorio salvo em: resultados/tabelas/investigacoes_etapa1.txt")


if __name__ == "__main__":
    main()
