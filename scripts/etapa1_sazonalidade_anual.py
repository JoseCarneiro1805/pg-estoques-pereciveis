"""
===============================================================================
ETAPA 1 — Evidencia de sazonalidade anual e efeito de eventos de calendario
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

POR QUE ESTE SCRIPT EXISTE
--------------------------
O Capitulo 3 declara que as series tem sazonalidade semanal E anual. A
analise exploratoria original (etapa1_preparar_dados.py) so mediu a semanal.
Este script mede a anual e o efeito dos eventos de calendario.

RECORTE
-------
FOODS, lojas da California, dias disponiveis e de loja aberta (o recorte
adotado no texto). A unidade e a DEMANDA MEDIA POR SERIE-DIA, para nao
confundir sazonalidade com a entrada de novos produtos ao longo dos anos.

O QUE E MEDIDO
--------------
A. Demanda media por serie-dia, por mes e ano (tabela bruta).
   Indice sazonal mensal: demanda do mes / media movel centrada de 12 meses
   (2x12), o que remove a tendencia e mudancas de nivel (ex.: reforma da
   CA_2). Indice 1,05 = mes 5% acima da tendencia.
B. Estabilidade: correlacao de Spearman entre os perfis mensais (indices) de
   cada par de anos completos, e coeficiente de concordancia de Kendall (W)
   entre esses anos, com teste qui-quadrado.
C. Eventos de calendario (calendar.csv, eventos 1 e 2): demanda no dia do
   evento, na media dos 3 dias anteriores e dos 3 posteriores, contra duas
   referencias do mesmo mes e ano, fora da janela de -3 a +3 dias:
     - media do mes (pedido original)
     - media do MESMO DIA DA SEMANA no mes, que separa o efeito do evento do
       efeito do dia da semana (o Super Bowl cai sempre num domingo, p. ex.)
     - referencia AJUSTADA (principal): mesmo dia da semana E mesma condicao de
       SNAP (dias 1 a 10 do mes na California), ate 6 semanas antes ou depois.
       Varios eventos caem nos dias de SNAP (Dia do Trabalho, Ano Novo, Super
       Bowl), e sem esse ajuste o efeito do SNAP seria atribuido ao evento.
D. Autocorrelacao da serie diaria agregada na defasagem de 364 dias (52
   semanas, multiplo de 7): bruta; sem tendencia (razao para a media movel
   centrada de 365 dias); e sem tendencia e sem efeito semanal. A tendencia
   infla qualquer defasagem longa, e o padrao semanal infla toda defasagem
   multipla de 7 — por isso 364 e comparado com 357 e 371 (51 e 53 semanas)
   e com 182 (meio ano).

SAIDAS
------
resultados/tabelas/saz_*.csv e .tex
resultados/figuras/fig_saz_mensal.png, fig_saz_eventos.png, fig_saz_autocorrelacao.png
===============================================================================
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import (
    PASTA, PASTA_TAB, ARQUIVO_SAIDA, COR_PRINCIPAL, COR_TEXTO_SEC, estilo_eixos, salvar_figura, num_pt,
)
import etapa3_avaliacao as av

ORIGEM = "etapa1_sazonalidade_anual.py"
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
JANELA = 3                                   # dias antes e depois do evento
# Tabela enxuta (corpo do texto): efeito ajustado de modulo >= 5% que se repete
# no mesmo sentido em pelo menos 4 ocorrencias
EFEITO_MINIMO = 0.05
OCORRENCIAS_MINIMAS = 4
TRADUCAO_EVENTOS = {
    "SuperBowl": "Super Bowl", "ValentinesDay": "Dia dos Namorados",
    "PresidentsDay": "Dia dos Presidentes", "LentStart": "Início da Quaresma",
    "LentWeek2": "2ª semana da Quaresma", "StPatricksDay": "Dia de São Patrício",
    "Purim End": "Fim do Purim", "OrthodoxEaster": "Páscoa Ortodoxa",
    "Pesach End": "Fim do Pessach", "Cinco De Mayo": "Cinco de Mayo",
    "Mother's day": "Dia das Mães", "MemorialDay": "Memorial Day",
    "NBAFinalsStart": "Início das finais da NBA", "NBAFinalsEnd": "Fim das finais da NBA",
    "Father's day": "Dia dos Pais", "IndependenceDay": "Dia da Independência",
    "Ramadan starts": "Início do Ramadã", "Eid al-Fitr": "Eid al-Fitr",
    "LaborDay": "Dia do Trabalho", "ColumbusDay": "Dia de Colombo",
    "Halloween": "Halloween", "EidAlAdha": "Eid al-Adha",
    "VeteransDay": "Dia dos Veteranos", "Thanksgiving": "Ação de Graças",
    "Christmas": "Natal", "Chanukah End": "Fim do Chanucá", "NewYear": "Ano Novo",
    "OrthodoxChristmas": "Natal Ortodoxo", "MartinLutherKingDay": "Martin Luther King",
    "Easter": "Páscoa",
}
TRADUCAO_TIPOS = {"National": "Nacional", "Cultural": "Cultural",
                  "Religious": "Religioso", "Sporting": "Esportivo"}
RAIO_AJUSTE = 42                             # dias (6 semanas) da referencia ajustada
DEFASAGENS = (7, 182, 357, 364, 371)          # 364 = 52 semanas
# Paleta de referencia (categorica, ordem fixa)
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BASE = ("FOODS, lojas da Califórnia, dias disponíveis e de loja aberta; demanda "
        "média por série-dia.")


# -----------------------------------------------------------------------------
# Serie diaria agregada
# -----------------------------------------------------------------------------
def carregar():
    b = pd.read_parquet(PASTA / ARQUIVO_SAIDA,
                        columns=["date", "vendas", "disponivel", "loja_fechada"])
    b = b[b["disponivel"] & ~b["loja_fechada"]]
    diaria = b.groupby("date")["vendas"].mean()
    # Grade completa de datas: dias sem loja aberta (Natal) ficam ausentes
    diaria = diaria.reindex(pd.date_range(diaria.index.min(), diaria.index.max()))
    cal = pd.read_csv(PASTA / "calendar.csv", parse_dates=["date"])
    return diaria, cal


# -----------------------------------------------------------------------------
# A e B — perfil mensal e estabilidade
# -----------------------------------------------------------------------------
def mensal(diaria):
    m = diaria.groupby([diaria.index.year, diaria.index.month]).mean()
    n = diaria.groupby([diaria.index.year, diaria.index.month]).count()
    m.index.names = n.index.names = ["Ano", "Mês"]
    bruta = m.unstack("Ano")
    bruta.index = [MESES[i - 1] for i in bruta.index]
    bruta.index.name = "Mês"

    # Indice: so meses completos (o 1o mes tem 3 dias e o ultimo 24)
    dias_no_mes = pd.Series([pd.Timestamp(a, mm, 1).days_in_month for a, mm in n.index],
                            index=n.index)
    fech = 1                                  # o Natal fica de fora em dezembro
    completo = n >= dias_no_mes - fech
    serie = m[completo]
    serie.index = pd.PeriodIndex([pd.Period(year=a, month=mm, freq="M")
                                  for a, mm in serie.index])
    assert (np.diff(serie.index.asi8) == 1).all(), "meses faltando na serie mensal"
    # Media movel centrada 2x12: pesos 1/24 nas pontas (t-6, t+6) e 1/12 no meio
    pesos = np.r_[0.5, np.ones(11), 0.5] / 12
    mm2x12 = pd.Series(np.nan, index=serie.index)
    mm2x12.iloc[6:-6] = np.convolve(serie.to_numpy(), pesos, mode="valid")
    indice = (serie / mm2x12).dropna()
    ind = pd.DataFrame({"Ano": indice.index.year, "Mês": indice.index.month,
                        "i": indice.values}).pivot(index="Mês", columns="Ano", values="i")
    ind.index = [MESES[i - 1] for i in ind.index]
    ind.index.name = "Mês"

    completos = [a for a in ind.columns if ind[a].notna().sum() == 12]
    pares = {}
    for a, b in combinations(completos, 2):
        rho, p = stats.spearmanr(ind[a], ind[b])
        pares[f"{a} × {b}"] = {"Spearman": rho, "p-valor": p}
    est = pd.DataFrame(pares).T.rename_axis("Par de anos")
    # Concordancia de Kendall entre os anos completos
    ranks = ind[completos].rank()
    k, n_m = len(completos), 12
    s = ((ranks.sum(axis=1) - ranks.sum(axis=1).mean()) ** 2).sum()
    w = 12 * s / (k ** 2 * (n_m ** 3 - n_m))
    chi2 = k * (n_m - 1) * w
    p_w = stats.chi2.sf(chi2, n_m - 1)
    est.loc[f"Kendall W ({', '.join(map(str, completos))})"] = {"Spearman": w, "p-valor": p_w}
    medio = ind[completos].mean(axis=1)
    return bruta, ind, est, medio, completos


# -----------------------------------------------------------------------------
# C — eventos
# -----------------------------------------------------------------------------
def eventos(diaria, cal):
    ev = pd.concat([cal[["date", "event_name_1", "event_type_1"]].set_axis(
                        ["date", "evento", "tipo"], axis=1),
                    cal[["date", "event_name_2", "event_type_2"]].set_axis(
                        ["date", "evento", "tipo"], axis=1)]).dropna()
    ev = ev[(ev["date"] >= diaria.index.min()) & (ev["date"] <= diaria.index.max())]
    dsem = diaria.index.dayofweek
    snap = cal.set_index("date")["snap_CA"].reindex(diaria.index).to_numpy()
    linhas = []
    for _, r in ev.iterrows():
        d0 = r["date"]
        janela = (diaria.index >= d0 - pd.Timedelta(days=JANELA)) & \
                 (diaria.index <= d0 + pd.Timedelta(days=JANELA))
        mes = (diaria.index.year == d0.year) & (diaria.index.month == d0.month) & ~janela
        ref_mes = diaria[mes].mean()
        ref_dsem = diaria[mes & (dsem == d0.dayofweek)].mean()
        # Referencia ajustada: mesmo dia da semana E mesma condicao de SNAP,
        # num raio de 6 semanas em torno do evento (fora da janela -3..+3)
        perto = (np.abs((diaria.index - d0).days) <= RAIO_AJUSTE) & ~janela
        snap0 = snap[diaria.index.get_loc(d0)]
        ref_aj = diaria[perto & (dsem == d0.dayofweek) & (snap == snap0)].mean()
        antes = diaria[d0 - pd.Timedelta(days=JANELA): d0 - pd.Timedelta(days=1)].mean()
        depois = diaria[d0 + pd.Timedelta(days=1): d0 + pd.Timedelta(days=JANELA)].mean()
        dia = diaria.get(d0, np.nan)
        linhas.append({"evento": r["evento"], "tipo": r["tipo"], "data": d0.date(),
                       "dia_vs_mes": dia / ref_mes - 1, "antes_vs_mes": antes / ref_mes - 1,
                       "depois_vs_mes": depois / ref_mes - 1,
                       "dia_vs_mesmo_dia_semana": dia / ref_dsem - 1,
                       "dia_ajustado": dia / ref_aj - 1, "snap_no_dia": int(snap0)})
    occ = pd.DataFrame(linhas)
    g = occ.groupby(["evento", "tipo"])
    t = pd.DataFrame({
        "Ocorrências": g.size(),
        "Dia vs mês": g["dia_vs_mes"].mean(),
        "3 dias antes vs mês": g["antes_vs_mes"].mean(),
        "3 dias depois vs mês": g["depois_vs_mes"].mean(),
        "Dia vs mesmo dia da semana": g["dia_vs_mesmo_dia_semana"].mean(),
        "Dia ajustado": g["dia_ajustado"].mean(),
        "Anos com aumento (ajustado)": g["dia_ajustado"].apply(lambda s: int((s > 0).sum())),
        "Anos com SNAP no dia": g["snap_no_dia"].sum(),
    })
    t = t.reset_index().set_index("evento").rename_axis("Evento")
    t = t.rename(columns={"tipo": "Tipo"})
    ordem = t["Dia ajustado"].abs().fillna(
        t["3 dias antes vs mês"].abs()).sort_values(ascending=False).index
    return t.loc[ordem], occ


def eventos_resumo(occ):
    """Versao enxuta para o corpo do texto: so o efeito ajustado (controla dia
    da semana e SNAP), so eventos com efeito de modulo >= 5% que se repetem no
    mesmo sentido em >= 4 ocorrencias, nomes em portugues."""
    g = occ.groupby(["evento", "tipo"])["dia_ajustado"]
    t = pd.DataFrame({"efeito": g.mean(), "n": g.count(),
                      "aumento": g.apply(lambda s: int((s > 0).sum())),
                      "queda": g.apply(lambda s: int((s < 0).sum()))}).reset_index()
    # Ocorrencias no sentido do efeito: aumento nos positivos, queda nos negativos
    t["mesmo_sentido"] = np.where(t["efeito"] > 0, t["aumento"], t["queda"])
    t = t[(t["efeito"].abs() >= EFEITO_MINIMO) & (t["mesmo_sentido"] >= OCORRENCIAS_MINIMAS)]
    t = t.sort_values("efeito", ascending=False)
    faltando = set(t["evento"]) - set(TRADUCAO_EVENTOS)
    assert not faltando, f"eventos sem traducao: {faltando}"
    return pd.DataFrame({
        "Tipo": t["tipo"].map(TRADUCAO_TIPOS).to_numpy(),
        "Efeito ajustado": (t["efeito"] * 100).to_numpy(),
        "Ocorrências no mesmo sentido": [f"{m} de {n}"
                                         for m, n in zip(t["mesmo_sentido"], t["n"])],
    }, index=pd.Index(t["evento"].map(TRADUCAO_EVENTOS), name="Evento"))


# -----------------------------------------------------------------------------
# D — autocorrelacao
# -----------------------------------------------------------------------------
def autocorrelacao(diaria):
    tendencia = diaria.rolling(365, center=True, min_periods=300).mean()
    sem_tend = diaria / tendencia
    # Todas as defasagens testadas sao multiplas de 7: o padrao semanal infla
    # todas. Para isolar o anual, divide-se tambem pelo fator do dia da semana.
    fator = sem_tend.groupby(sem_tend.index.dayofweek).transform("mean")
    sem_semana = sem_tend / fator
    linhas = {}
    for lag in DEFASAGENS:
        linhas[f"{lag} dias"] = {"Série bruta": diaria.autocorr(lag),
                                 "Sem tendência": sem_tend.autocorr(lag),
                                 "Sem tendência e sem efeito semanal": sem_semana.autocorr(lag)}
    t = pd.DataFrame(linhas).T.rename_axis("Defasagem")
    n = sem_semana.notna().sum()
    acf = pd.Series({lag: sem_semana.autocorr(lag) for lag in range(1, 401)})
    return t, acf, n


# -----------------------------------------------------------------------------
# FIGURAS
# -----------------------------------------------------------------------------
def figura_mensal(bruta, ind, completos):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    x = np.arange(12)
    for k, ano in enumerate(bruta.columns):
        a1.plot(x, bruta[ano].values, color=CORES[k], linewidth=2, marker="o",
                markersize=4, label=str(ano))
    a1.set_title("Demanda média por série-dia, por mês e ano", fontsize=10, loc="left")
    a1.set_ylabel("Unidades por série-dia")
    a1.legend(frameon=False, fontsize=8, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), handlelength=1.5, columnspacing=1)
    for k, ano in enumerate(ind.columns):
        cor = CORES[list(bruta.columns).index(ano)]
        a2.plot(x, ind[ano].values, color=cor, linewidth=2 if ano in completos else 1,
                linestyle="-" if ano in completos else "--", marker="o", markersize=4,
                label=str(ano) + ("" if ano in completos else " (parcial)"))
    a2.axhline(1, color=COR_TEXTO_SEC, linewidth=1)
    a2.set_title("Índice sazonal (mês ÷ tendência de 12 meses)", fontsize=10, loc="left")
    a2.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), handlelength=1.5, columnspacing=1)
    for ax in (a1, a2):
        ax.set_xticks(x, MESES, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: num_pt(v, 2)))
        estilo_eixos(ax)
    fig.suptitle("Sazonalidade anual — FOODS, lojas da Califórnia", fontsize=11, x=0.01,
                 ha="left")
    salvar_figura(fig, "fig_saz_mensal.png")


def figura_eventos(t):
    s = t.sort_values("Dia ajustado", na_position="first")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(s) + 1.4))
    series = [("3 dias antes vs mês", "3 dias antes (vs mês)", CORES[1]),
              ("Dia ajustado", "Dia do evento (ajustado: dia da semana e SNAP)", CORES[0]),
              ("3 dias depois vs mês", "3 dias depois (vs mês)", CORES[2])]
    for col, rot, cor in series:
        ax.scatter(s[col] * 100, y, color=cor, s=36, label=rot, zorder=3,
                   edgecolor="white", linewidth=1.5)
    ax.axvline(0, color=COR_TEXTO_SEC, linewidth=1)
    ax.set_yticks(y, s.index, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel("Diferença da demanda média por série-dia")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1,
              bbox_to_anchor=(0.4, -0.06))
    ax.set_title("Efeito dos eventos de calendário (média das ocorrências 2011–2016)",
                 fontsize=10, loc="left")
    estilo_eixos(ax, grade="x")
    salvar_figura(fig, "fig_saz_eventos.png")


def figura_acf(acf, n):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.vlines(acf.index, 0, acf.values, color=COR_PRINCIPAL, linewidth=0.8)
    banda = 1.96 / np.sqrt(n)
    ax.axhspan(-banda, banda, color="#e5e4df", zorder=0)
    ax.scatter([364], [acf[364]], color=CORES[1], s=40, zorder=3, edgecolor="white",
               linewidth=1.5)
    ax.annotate(f"364 dias: {num_pt(acf[364], 2)}", xy=(364, acf[364]), xytext=(250, acf[364]),
                fontsize=9, color=COR_TEXTO_SEC, va="center",
                arrowprops=dict(arrowstyle="-", color=COR_TEXTO_SEC, linewidth=0.8))
    ax.axhline(0, color=COR_TEXTO_SEC, linewidth=0.8)
    ax.set_xlabel("Defasagem (dias)")
    ax.set_ylabel("Autocorrelação")
    ax.set_title("Autocorrelação da demanda diária agregada, sem tendência e sem efeito "
                 "semanal\n(faixa cinza: ±1,96/√n)", fontsize=10, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: num_pt(v, 1)))
    estilo_eixos(ax)
    salvar_figura(fig, "fig_saz_autocorrelacao.png")


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 70)
    print("ETAPA 1 — SAZONALIDADE ANUAL E EVENTOS DE CALENDARIO")
    print("=" * 70)
    diaria, cal = carregar()
    bruta, ind, est, medio, completos = mensal(diaria)
    t_ev, occ = eventos(diaria, cal)
    t_acf, acf, n = autocorrelacao(diaria)

    av.salvar_tabela(bruta, "saz_mensal_por_ano", "Demanda média por série-dia, por mês e ano",
                     ORIGEM, nota=(f"{BASE} 2011 começa em 29/01 e 2016 termina em 24/04; "
                                   "dezembro sem o dia 25 (lojas fechadas)."))
    ind_t = ind.copy()
    ind_t["Média dos anos completos"] = medio
    av.salvar_tabela(ind_t, "saz_indice_mensal", "Índice sazonal mensal", ORIGEM,
                     nota=(f"{BASE} Índice = demanda do mês ÷ média móvel centrada 2×12 "
                           "meses (remove tendência e mudanças de nível). 1,05 = 5\\% acima "
                           "da tendência. Só meses completos; anos nas pontas ficam parciais."))
    av.salvar_tabela(est, "saz_estabilidade", "Estabilidade do perfil mensal entre anos",
                     ORIGEM, formatos={"p-valor": 4},
                     nota=(f"{BASE} Correlação de Spearman entre os índices mensais de cada "
                           "par de anos completos; última linha: coeficiente de concordância "
                           "de Kendall (W) entre esses anos, na coluna Spearman, com p-valor "
                           "do teste qui-quadrado."))
    pct = {c: "pct" for c in t_ev.columns if "vs" in c or c == "Dia ajustado"}
    av.salvar_tabela(t_ev, "saz_eventos", "Efeito dos eventos de calendário sobre a demanda",
                     ORIGEM, largura_total=True,
                     formatos={**pct, "Ocorrências": "int",
                               "Anos com aumento (ajustado)": "int",
                               "Anos com SNAP no dia": "int"},
                     nota=(f"{BASE} Média das ocorrências de cada evento. Referências fora "
                           "dos dias −3 a +3 do evento: média do mês (mesmo mês e ano); "
                           "média do mesmo dia da semana no mês; e referência ajustada "
                           "(mesmo dia da semana e mesma condição de SNAP, até 6 semanas "
                           "antes ou depois do evento). Anos com aumento: ocorrências com o "
                           "dia do evento acima da referência ajustada. Natal: lojas "
                           "fechadas no dia (sem dado)."))
    occ.to_csv(PASTA_TAB / "saz_eventos_ocorrencias.csv", index=False, encoding="utf-8-sig")
    resumo = eventos_resumo(occ)
    av.salvar_tabela(
        resumo, "saz_eventos_resumo", "Eventos de calendário com efeito consistente sobre "
        "a demanda", ORIGEM, formatos={"Efeito ajustado": "sinal"},
        nota=(f"{BASE} Efeito ajustado: diferença \\% entre a demanda no dia do evento e "
              "a média dos dias de mesmo dia da semana e mesma condição de SNAP, numa "
              "janela de até 6 semanas antes ou depois do evento (excluídos os 3 dias "
              "anteriores e posteriores); média das ocorrências de 2011 a 2016. Incluídos "
              "os eventos com efeito de pelo menos 5\\% em módulo que se repete no mesmo "
              "sentido em pelo menos 4 ocorrências. Tabela completa no apêndice."))
    av.salvar_tabela(t_acf, "saz_autocorrelacao", "Autocorrelação da demanda diária agregada",
                     ORIGEM, nota=(f"{BASE} Série diária da demanda média por série-dia "
                                   f"(n = {num_pt(n)} dias com tendência estimada). Sem "
                                   "tendência: razão para a média móvel centrada de 365 dias. "
                                   "Limite de significância aproximado: "
                                   f"$\\pm${num_pt(1.96 / np.sqrt(n), 3)} "
                                   "($\\pm 1{,}96/\\sqrt{n}$)."))
    figura_mensal(bruta, ind, completos)
    figura_eventos(t_ev)
    figura_acf(acf, n)

    pd.set_option("display.width", 220)
    f = lambda v: f"{v:.4f}"
    print("\nINDICE\n" + ind_t.to_string(float_format=f))
    print("\nESTABILIDADE\n" + est.to_string(float_format=f))
    print("\nEVENTOS\n" + t_ev.to_string(float_format=f))
    print("\nAUTOCORRELACAO\n" + t_acf.to_string(float_format=f), f"\nn = {n}")


if __name__ == "__main__":
    main()
