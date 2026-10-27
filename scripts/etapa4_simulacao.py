"""
===============================================================================
ETAPA 4 — Simulacao multiperiodo de estoque perecivel
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE SCRIPT FAZ
---------------------
Para cada serie, simula dia a dia o estoque da loja nos 90 dias de teste,
repondo ate o quantil previsto por cada modelo. Calcula perdas por validade,
faltas e custo. Faz isso para:
  - 3 modelos (Holt-Winters, XGBoost, combinacao)
  - 30 niveis de servico (70% a 99%), que incluem os do newsvendor
    (80%, 90%, 95%)
  - 4 taxas de perecibilidade (theta = 0,05 / 0,10 / 0,15 / 0,20)
e identifica, em cada caso, o nivel de servico que minimiza o custo.

A aleatoriedade vem da DEMANDA OBSERVADA (as vendas reais do teste); a
deterioracao e deterministica (fracao theta da sobra, inclusive fracoes de
unidade, que representam o valor esperado da perda).

REGRAS DA SIMULACAO — aprovadas em 21/10/2026 (justificativa na monografia)
-------------------------------------------------------------
Politica: todo dia, repor ate o alvo S(t) = quantil do nivel de servico,
arredondado para o inteiro mais proximo (regra da Etapa 3). Pedido feito a
noite e entregue antes da abertura (prazo zero). Sem devolucao.

Ordem dos eventos em cada dia t:
  1. a sobra da vespera perde a fracao theta (perda por validade)
  2. chega a reposicao: Q(t) = max(S(t) - estoque disponivel, 0)
  3. acontece a demanda; o que falta e venda perdida
  4. o que sobra passa para o dia seguinte
Equivale a I(t+1) = max((1 - theta) * I(t) + Q(t) - d(t), 0).

- Estoque inicial zero (o 1o pedido leva ao alvo); sem aquecimento.
- Fim do horizonte: a sobra final nao tem custo extra.
- Loja fechada: sem demanda e sem reposicao; a perecibilidade continua.
  (Nao ha dia fechado no periodo de teste; a regra fica implementada.)
- Serie que entra no meio do teste: simulada a partir do 1o dia disponivel.

CUSTO
-----
Principal: c_h x unidades PERDIDAS por validade + c_r x unidades em FALTA,
com c_h = 1 e c_r = 4, 9 ou 19 (razoes 1:4, 1:9, 1:19). A sobra que nao
estraga sera vendida depois e por isso nao e prejuizo.
Sensibilidades: (a) c_h sobre TODA a sobra do fim do dia (definicao da
Etapa 3, equivalente a tratar a sobra como perda total, como no newsvendor);
(b) descartar os 7 primeiros dias; (c) a sobra final vira perda.

NIVEL DE SERVICO OTIMO — por que pode diferir do newsvendor
-----------------------------------------------------------
O newsvendor (tau = c_r / (c_r + c_h)) supoe que toda sobra se perde. Aqui so
a fracao theta da sobra se perde por dia. Um argumento marginal simples: uma
unidade a mais em estoque, se sobrar, custa so theta * c_h, e nao c_h. O nivel
critico passaria a ser aproximadamente c_r / (c_r + theta * c_h). As tabelas
mostram essa aproximacao ao lado do otimo encontrado na simulacao.

PRE-REQUISITO: rodar antes os scripts da Etapa 3 (inclusive etapa3_comparacao)
INTERVALOS DE CONFIANCA: metodo unico de etapa3_avaliacao.diferenca_ic (item
como unidade), o mesmo de todas as tabelas de diferencas do projeto.

SAIDAS: resultados/tabelas/sim_*.csv e .tex
        resultados/figuras/fig_sim_*.png
===============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import (
    PASTA, PASTA_TAB, COR_TEXTO_SEC, estilo_eixos, salvar_figura, num_pt,
)
import etapa3_avaliacao as av

ORIGEM = "etapa4_simulacao.py"
MODELOS = {"Holt-Winters": "previsoes_holt_winters.parquet",
           "XGBoost": "previsoes_xgboost.parquet",
           "Combinação": "previsoes_combinacao.parquet"}
THETAS = (0.05, 0.10, 0.15, 0.20)          # taxa de perecibilidade (theta_perec)
THETA_BASE = 0.10
NIVEIS = av.NIVEIS_QUANTIS                 # 70% a 99%
DIAS_AQUECIMENTO_SENS = 7                  # sensibilidade (b)

# Paleta de referencia (categorica, ordem fixa) e rampa azul para theta
CORES_MODELOS = {"Holt-Winters": "#2a78d6", "XGBoost": "#eb6834", "Combinação": "#1baf7a"}
CORES_THETA = {0.05: "#86b6ef", 0.10: "#3987e5", 0.15: "#1c5cab", 0.20: "#0d366b"}

INDICADORES = ["perdido", "recebido", "vendido", "falta", "demanda", "estoque",
               "sobra", "dias", "dias_sem_falta"]


def rotulo_theta(th):
    return f"θ = {num_pt(th, 2)}"


def rotulo_razao(tau):
    return f"1:{av.CUSTO_FALTA[tau]} ({av.pct(tau)}%)"


# -----------------------------------------------------------------------------
# DADOS: alvos de estoque e demanda em grade serie x dia
# -----------------------------------------------------------------------------
def carregar():
    print("\n[1/4] Lendo previsoes e montando alvos de estoque...")
    cols_q = [f"q_emp_{av.pct(t)}" for t in NIVEIS]
    alvos, ref = {}, None
    for nome, arq in MODELOS.items():
        df = pd.read_parquet(PASTA / arq,
                             columns=["id", "dia", "vendas", "loja_fechada"] + cols_q)
        if ref is None:
            ids = np.sort(df["id"].unique())
            dias = np.arange(av.INICIO_TESTE, av.FIM_TESTE + 1)
            lin = np.searchsorted(ids, df["id"].to_numpy())
            col = df["dia"].to_numpy() - av.INICIO_TESTE
            d = np.full((len(ids), len(dias)), np.nan)
            d[lin, col] = df["vendas"].to_numpy(float)
            fechado = np.zeros(d.shape, bool)
            fechado[lin, col] = df["loja_fechada"].to_numpy(bool)
            ref = (df[["id", "dia"]].sort_values(["id", "dia"]).to_numpy(), lin, col)
        else:
            atual = df[["id", "dia"]].sort_values(["id", "dia"]).to_numpy()
            assert np.array_equal(atual, ref[0]), f"{nome}: series-dias diferentes"
            lin = np.searchsorted(ids, df["id"].to_numpy())
            col = df["dia"].to_numpy() - av.INICIO_TESTE
        S = np.zeros((len(NIVEIS),) + d.shape, np.float32)
        for k, c in enumerate(cols_q):
            S[k, lin, col] = av.decisao(np.maximum(df[c].to_numpy(float), 0))
        alvos[nome] = S
        print(f"      {nome}: {len(df):,} dias-série")
    item = pd.Series(ids).str.rsplit("_", n=3).str[0].to_numpy()
    return ids, item, d, fechado, alvos


# -----------------------------------------------------------------------------
# SIMULACAO (todas as series e todos os niveis ao mesmo tempo)
# -----------------------------------------------------------------------------
def simular(S, d, fechado, theta, inicio_medicao=0):
    """S: alvos (niveis x series x dias); d: demanda (series x dias, NaN antes
    de a serie existir). Devolve totais por nivel e serie de cada indicador,
    contando so os dias a partir de inicio_medicao, e a sobra no fim."""
    L, n, D = S.shape
    sobra = np.zeros((L, n))
    tot = {k: np.zeros((L, n)) for k in INDICADORES}
    for t in range(D):
        ativa = ~np.isnan(d[:, t])                     # serie ja disponivel
        aberta = ativa & ~fechado[:, t]
        perdido = theta * sobra                        # 1. perecibilidade
        disponivel = sobra - perdido
        pedido = np.where(aberta, np.maximum(S[:, :, t] - disponivel, 0), 0)   # 2.
        estoque = disponivel + pedido
        dem = np.where(aberta, np.nan_to_num(d[:, t]), 0)                       # 3.
        vendido = np.minimum(estoque, dem)
        falta = dem - vendido
        sobra = estoque - vendido                                               # 4.
        if t >= inicio_medicao:
            tot["perdido"] += perdido
            tot["recebido"] += pedido
            tot["vendido"] += vendido
            tot["falta"] += falta
            tot["demanda"] += dem
            tot["estoque"] += np.where(aberta, estoque, 0)
            tot["sobra"] += np.where(aberta, sobra, 0)
            tot["dias"] += aberta
            tot["dias_sem_falta"] += aberta & (falta == 0)
    return tot, sobra


def resumir(tot, sobra_final):
    """Indicadores agregados por nivel (todas as series somadas)."""
    s = {k: v.sum(axis=1) for k, v in tot.items()}
    r = pd.DataFrame({
        "Perdidas": s["perdido"] / s["dias"],
        "Perda % do recebido": s["perdido"] / s["recebido"],
        "Faltas": s["falta"] / s["dias"],
        "Atendimento": s["vendido"] / s["demanda"],
        "Cobertura": s["dias_sem_falta"] / s["dias"],
        "Estoque médio": s["estoque"] / s["dias"],
        "Sobra média": s["sobra"] / s["dias"],
        "Sobra final": sobra_final.sum(axis=1) / s["dias"],
    }, index=pd.Index([av.pct(t) for t in NIVEIS], name="Nível"))
    for tau, cr in av.CUSTO_FALTA.items():
        r[f"Custo 1:{cr}"] = (s["perdido"] + cr * s["falta"]) / s["dias"]
        r[f"Custo 1:{cr} toda sobra"] = (s["sobra"] + cr * s["falta"]) / s["dias"]
        r[f"Custo 1:{cr} sobra final perdida"] = r[f"Custo 1:{cr}"] + r["Sobra final"]
    return r


def rodar(d, fechado, alvos):
    print("\n[2/4] Simulando 3 modelos x 30 niveis x 4 thetas...")
    res, por_serie = [], {}
    for nome, S in alvos.items():
        for th in THETAS:
            tot, sf = simular(S, d, fechado, th)
            r = resumir(tot, sf)
            tot7, sf7 = simular(S, d, fechado, th, inicio_medicao=DIAS_AQUECIMENTO_SENS)
            r7 = resumir(tot7, sf7)
            for cr in av.CUSTO_FALTA.values():
                r[f"Custo 1:{cr} sem 7 primeiros dias"] = r7[f"Custo 1:{cr}"]
            r.insert(0, "theta", th)
            r.insert(0, "Modelo", nome)
            res.append(r.reset_index())
            por_serie[(nome, th)] = tot
        print(f"      {nome}: ok")
    res = pd.concat(res, ignore_index=True)
    res.to_csv(PASTA_TAB / "sim_resultados_completos.csv", index=False, encoding="utf-8-sig")
    print("      resultados/tabelas/sim_resultados_completos.csv")
    return res, por_serie


# -----------------------------------------------------------------------------
# TABELAS
# -----------------------------------------------------------------------------
def no_newsvendor(res, coluna):
    """Valor de uma coluna de custo/indicador no nivel do newsvendor de cada
    razao: linhas (razao, modelo), colunas theta."""
    partes = {}
    for tau, cr in av.CUSTO_FALTA.items():
        col = coluna.format(cr=cr)
        sel = res[res["Nível"] == av.pct(tau)]
        t = sel.pivot(index="Modelo", columns="theta", values=col).reindex(list(MODELOS))
        partes[rotulo_razao(tau)] = t
    t = pd.concat(partes, names=["Razão (nível)", "Modelo"])
    t.columns = [rotulo_theta(th) for th in t.columns]
    return t


def otimos(res, sufixo=""):
    """Nivel que minimiza o custo, por razao, modelo e theta."""
    linhas = []
    for tau, cr in av.CUSTO_FALTA.items():
        col = f"Custo 1:{cr}{sufixo}"
        for (nome, th), g in res.groupby(["Modelo", "theta"], sort=False):
            i = g[col].idxmin()
            ref = g.loc[g["Nível"] == av.pct(tau), col].iloc[0]
            linhas.append({"razao": rotulo_razao(tau), "tau": tau, "Modelo": nome,
                           "theta": th, "nivel_otimo": int(g.loc[i, "Nível"]),
                           "custo_otimo": g.loc[i, col], "custo_newsvendor": ref,
                           "economia_%": (1 - g.loc[i, col] / ref) * 100})
    return pd.DataFrame(linhas)


def tabela_otimos(ot, aproximacao):
    """Linhas (razao, modelo), colunas: newsvendor e otimo por theta (texto)."""
    linhas = {}
    for (razao, nome), g in ot.groupby(["razao", "Modelo"], sort=False):
        tau = g["tau"].iloc[0]
        lin = {"Newsvendor": f"{av.pct(tau)}%"}
        for _, r in g.iterrows():
            marca = "*" if r["nivel_otimo"] == av.pct(NIVEIS[-1]) else ""
            lin[rotulo_theta(r["theta"])] = f"{r['nivel_otimo']}%{marca}"
        linhas[(razao, nome)] = lin
        if nome == list(MODELOS)[-1] and aproximacao:
            linhas[(razao, "Aproximação marginal")] = {
                "Newsvendor": f"{av.pct(tau)}%",
                **{rotulo_theta(th): num_pt(100 * av.CUSTO_FALTA[tau]
                                            / (av.CUSTO_FALTA[tau] + th), 1) + "%"
                   for th in THETAS}}
    t = pd.DataFrame(linhas).T
    t.index.names = ["Razão (nível)", "Modelo"]
    return t


def diferencas(por_serie, item, nivel=None):
    """Diferenca % de custo entre modelos, com IC de 95% (item como unidade;
    metodo unico de av.diferenca_ic). nivel(tau, modelo, theta) da o nivel de
    servico em que cada modelo e avaliado; sem ele, o nivel do newsvendor.
    Com nivel, acrescenta por razao uma linha com os niveis usados."""
    pares = [("XGBoost", "Holt-Winters"), ("Combinação", "Holt-Winters"),
             ("Combinação", "XGBoost")]
    otimo = nivel is not None
    nivel = nivel or (lambda tau, modelo, th: tau)
    texto, longa = {}, []
    for tau, cr in av.CUSTO_FALTA.items():
        if otimo:
            usados = {}
            for th in THETAS:
                ns = [av.pct(nivel(tau, m, th)) for m in MODELOS]
                usados[rotulo_theta(th)] = (f"{ns[0]}%" if len(set(ns)) == 1
                                            else " / ".join(f"{x}%" for x in ns))
            texto[(rotulo_razao(tau), "Nível ótimo usado")] = usados
        for a, b in pares:
            lin = {}
            for th in THETAS:
                ka, kb = NIVEIS.index(nivel(tau, a, th)), NIVEIS.index(nivel(tau, b, th))
                ta, tb = por_serie[(a, th)], por_serie[(b, th)]
                ca = ta["perdido"][ka] + cr * ta["falta"][ka]
                cb = tb["perdido"][kb] + cr * tb["falta"][kb]
                ref = cb.sum() / tb["dias"][kb].sum()
                v = av.diferenca_ic(ca - cb, item, ref, n=tb["dias"][kb])
                lin[rotulo_theta(th)] = av.texto_ic(v)
                longa.append({"razao": rotulo_razao(tau), "comparacao": f"{a} vs {b}",
                              "theta": th, "nivel_a": av.pct(NIVEIS[ka]),
                              "nivel_b": av.pct(NIVEIS[kb]), "dif_%": v[0],
                              "ic_inf": v[1], "ic_sup": v[2]})
            texto[(rotulo_razao(tau), f"{a} vs {b}")] = lin
    t = pd.DataFrame(texto).T
    t.index.names = ["Razão (nível)", "Comparação"]
    return t, pd.DataFrame(longa)


def tabelas(res, por_serie, item):
    print("\n[3/4] Tabelas...")
    base = ("simulação de 90 dias (26/01 a 24/04/2016), 5.748 séries; reposição diária "
            "até o quantil empírico do nível de serviço, arredondado para o inteiro mais "
            "próximo; prazo zero; venda perdida.")
    custo = ("custo por série-dia = unidades perdidas por validade + $c_r$ × unidades em "
             "falta, com $c_r$ = 4, 9 e 19.")

    t = no_newsvendor(res, "Custo 1:{cr}")
    av.salvar_tabela(t, "sim_custo", "Custo da operação por modelo e taxa de perecibilidade",
                     ORIGEM, nota=f"{base} Nível de serviço do newsvendor. {custo}")
    ordem = t.groupby(level=0, sort=False).rank(method="min").astype(int)
    av.salvar_tabela(ordem, "sim_ordem", "Ordem dos modelos pelo custo (1 = menor custo)",
                     ORIGEM, formatos={c: "int" for c in ordem.columns},
                     nota=f"{base} Posição de cada modelo dentro de cada razão de custos.")

    t = no_newsvendor(res, "Perda % do recebido")
    av.salvar_tabela(t, "sim_perda", "Perda por validade (\\% das unidades recebidas)",
                     ORIGEM, formatos={c: "pct" for c in t.columns},
                     nota=f"{base} Nível de serviço do newsvendor.")

    sel = res[(res["theta"] == THETA_BASE)]
    ind = {}
    for tau, cr in av.CUSTO_FALTA.items():
        g = sel[sel["Nível"] == av.pct(tau)].set_index("Modelo").reindex(list(MODELOS))
        g = g[["Perdidas", "Perda % do recebido", "Faltas", "Atendimento", "Cobertura",
               "Estoque médio", f"Custo 1:{cr}", f"Custo 1:{cr} toda sobra"]]
        g.columns = ["Perdidas", "Perda % receb.", "Faltas", "Atendimento", "Cobertura",
                     "Estoque médio", "Custo", "Custo (toda sobra)"]
        ind[rotulo_razao(tau)] = g
    t = pd.concat(ind, names=["Razão (nível)", "Modelo"])
    av.salvar_tabela(
        t, "sim_indicadores", "Indicadores da simulação (θ = 0,10)", ORIGEM,
        formatos={"Perda % receb.": "pct", "Atendimento": "pct", "Cobertura": "pct"},
        nota=(f"{base} Nível do newsvendor; θ = 0,10. Perdidas, faltas e estoque médio "
              "(na abertura, após a reposição) em unidades por série-dia. Custo: "
              "perdas + $c_r$ × faltas. Custo (toda sobra): sobra do fim do dia + "
              "$c_r$ × faltas (sensibilidade)."))

    sens = {}
    for tau, cr in av.CUSTO_FALTA.items():
        g = sel[sel["Nível"] == av.pct(tau)].set_index("Modelo").reindex(list(MODELOS))
        g = g[[f"Custo 1:{cr}", f"Custo 1:{cr} sem 7 primeiros dias",
               f"Custo 1:{cr} sobra final perdida", f"Custo 1:{cr} toda sobra"]]
        g.columns = ["Base", "Sem 7 primeiros dias", "Sobra final perdida", "Toda sobra"]
        sens[rotulo_razao(tau)] = g
    t = pd.concat(sens, names=["Razão (nível)", "Modelo"])
    av.salvar_tabela(
        t, "sim_sensibilidade", "Sensibilidade do custo às regras da simulação (θ = 0,10)",
        ORIGEM, nota=(f"{base} Nível do newsvendor; θ = 0,10. Base: {custo} Sem 7 primeiros "
                      "dias: mede a partir do 8º dia. Sobra final perdida: a sobra do último "
                      "dia conta como perda. Toda sobra: custo de sobra sobre toda a sobra "
                      "do fim do dia."))

    tdif, longa = diferencas(por_serie, item)
    av.salvar_tabela(
        tdif, "sim_diferencas", "Diferenças de custo entre modelos, com IC de 95\\%", ORIGEM,
        nota=(f"{base} Nível do newsvendor. Diferença \\% do primeiro modelo em relação "
              "ao segundo (negativo = primeiro mais barato), intervalo de 95\\% de "
              "confiança entre colchetes, item como unidade."))
    longa.to_csv(PASTA_TAB / "sim_diferencas_numerico.csv", index=False, encoding="utf-8-sig")

    resultados_otimo = {}
    for sufixo, nome, legenda, aprox in (
            ("", "sim_nivel_otimo", "Nível de serviço que minimiza o custo", True),
            (" toda sobra", "sim_nivel_otimo_toda_sobra",
             "Nível de serviço que minimiza o custo — custo sobre toda a sobra", False)):
        ot = otimos(res, sufixo)
        ot.to_csv(PASTA_TAB / f"{nome}_numerico.csv", index=False, encoding="utf-8-sig")
        av.salvar_tabela(
            tabela_otimos(ot, aprox), nome, legenda, ORIGEM,
            nota=(f"{base} Varredura de 70\\% a 99\\%, passo de 1 ponto. * = no limite "
                  "da varredura (o ótimo pode estar acima de 99\\%). "
                  + ("Aproximação marginal: $c_r / (c_r + \\theta c_h)$, supondo que a sobra só "
                     "perde a fração θ. " if aprox else
                     "Custo de sobra sobre toda a sobra do fim do dia, como no newsvendor. ")))
        eco = ot.pivot_table(index=["razao", "Modelo"], columns="theta",
                             values="economia_%", sort=False)
        eco.index.names = ["Razão (nível)", "Modelo"]
        eco.columns = [rotulo_theta(th) for th in eco.columns]
        av.salvar_tabela(
            eco, f"{nome}_economia",
            "Redução de custo ao usar o nível ótimo em vez do newsvendor"
            + (" — custo sobre toda a sobra" if sufixo else ""), ORIGEM,
            formatos={c: "sinal" for c in eco.columns},
            nota=f"{base} Redução \\% do custo no nível ótimo em relação ao nível do newsvendor.")
        resultados_otimo[nome] = ot

    # Diferencas entre modelos com cada modelo no SEU nivel otimo (custo principal)
    ot = resultados_otimo["sim_nivel_otimo"]
    mapa = {(r["tau"], r["Modelo"], r["theta"]): r["nivel_otimo"] / 100
            for _, r in ot.iterrows()}
    tdo, longa = diferencas(por_serie, item, nivel=lambda tau, m, th: mapa[(tau, m, th)])
    av.salvar_tabela(
        tdo, "sim_diferencas_otimo",
        "Diferenças de custo entre modelos no nível de serviço ótimo, com IC de 95\\%",
        ORIGEM,
        nota=(f"{base} Cada modelo avaliado no seu nível ótimo (tabela de nível ótimo; "
              f"linha \"Nível ótimo usado\"). {custo} Diferença \\% do primeiro modelo em "
              "relação ao segundo (negativo = primeiro mais barato), intervalo de 95\\% "
              "de confiança entre colchetes, item como unidade."))
    longa.to_csv(PASTA_TAB / "sim_diferencas_otimo_numerico.csv", index=False,
                 encoding="utf-8-sig")
    return resultados_otimo


# -----------------------------------------------------------------------------
# FIGURAS
# -----------------------------------------------------------------------------
def _fmt_pct(ax, eixo="x"):
    f = mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(f)


def _fmt_num(ax, casas=1):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: num_pt(v, casas)))


def figuras(res, ot):
    print("\n[4/4] Figuras...")
    razoes = list(av.CUSTO_FALTA.items())

    # --- 1. Custo x theta (nivel do newsvendor) ---
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (tau, cr) in zip(eixos, razoes):
        sel = res[res["Nível"] == av.pct(tau)]
        for nome in MODELOS:
            g = sel[sel["Modelo"] == nome]
            ax.plot(g["theta"], g[f"Custo 1:{cr}"], color=CORES_MODELOS[nome],
                    linewidth=2, marker="o", markersize=5, label=nome)
        ax.set_title(f"Razão 1:{cr} — nível {av.pct(tau)}%", fontsize=10, loc="left")
        ax.set_xticks(THETAS, [num_pt(t, 2) for t in THETAS])
        ax.set_xlabel("Taxa de perecibilidade θ")
        _fmt_num(ax, 2)
        estilo_eixos(ax)
    eixos[0].set_ylabel("Custo por série-dia")
    eixos[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Custo da operação × taxa de perecibilidade (nível do newsvendor)",
                 fontsize=11, x=0.01, ha="left")
    salvar_figura(fig, "fig_sim_custo_theta.png")

    # --- 2. Custo x nivel de servico (theta base), com o minimo marcado ---
    sel = res[res["theta"] == THETA_BASE]
    fig, eixos = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True)
    for linha, (sufixo, rot) in enumerate((("", "Custo: perdas + faltas"),
                                           (" toda sobra", "Custo: toda sobra + faltas"))):
        for ax, (tau, cr) in zip(eixos[linha], razoes):
            col = f"Custo 1:{cr}{sufixo}"
            for nome in MODELOS:
                g = sel[sel["Modelo"] == nome]
                ax.plot(g["Nível"], g[col], color=CORES_MODELOS[nome], linewidth=2,
                        label=nome)
                i = g[col].idxmin()
                ax.plot(g.loc[i, "Nível"], g.loc[i, col], marker="o", markersize=8,
                        color=CORES_MODELOS[nome], markeredgecolor="white",
                        markeredgewidth=2, zorder=3)
            ax.axvline(av.pct(tau), color=COR_TEXTO_SEC, linewidth=1, linestyle="--")
            ax.text(av.pct(tau), 1.0, " newsvendor", transform=ax.get_xaxis_transform(),
                    fontsize=8, color=COR_TEXTO_SEC, va="top")
            ax.set_title(f"{rot} — razão 1:{cr}", fontsize=9, loc="left")
            _fmt_pct(ax)
            _fmt_num(ax, 1)
            estilo_eixos(ax)
            if linha == 1:
                ax.set_xlabel("Nível de serviço do alvo de estoque")
        eixos[linha][0].set_ylabel("Custo por série-dia")
    eixos[0][0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Custo × nível de serviço (θ = 0,10); círculo = mínimo de cada modelo",
                 fontsize=11, x=0.01, ha="left")
    salvar_figura(fig, "fig_sim_custo_nivel.png")

    # --- 3. Perda por validade x nivel de servico, por theta ---
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, nome in zip(eixos, MODELOS):
        g0 = res[res["Modelo"] == nome]
        for th in THETAS:
            g = g0[g0["theta"] == th]
            ax.plot(g["Nível"], g["Perda % do recebido"] * 100, color=CORES_THETA[th],
                    linewidth=2, label=rotulo_theta(th))
        for tau in av.NIVEIS:
            ax.axvline(av.pct(tau), color=COR_TEXTO_SEC, linewidth=0.8, linestyle=":")
        ax.set_title(nome, fontsize=10, loc="left")
        ax.set_xlabel("Nível de serviço do alvo de estoque")
        _fmt_pct(ax)
        _fmt_pct(ax, "y")
        estilo_eixos(ax)
    eixos[0].set_ylabel("Perda por validade\n(% das unidades recebidas)")
    eixos[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Perda por validade × nível de serviço (linhas pontilhadas: 80%, 90% e 95%)",
                 fontsize=11, x=0.01, ha="left")
    salvar_figura(fig, "fig_sim_perda_nivel.png")


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inicio = time.time()
    print("=" * 70)
    print("ETAPA 4 — SIMULACAO MULTIPERIODO DE ESTOQUE PERECIVEL")
    print("=" * 70)
    ids, item, d, fechado, alvos = carregar()
    res, por_serie = rodar(d, fechado, alvos)
    ot = tabelas(res, por_serie, item)
    figuras(res, ot)
    (PASTA_TAB / "sim_resumo_execucao.txt").write_text(
        f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: "
        f"{(time.time() - inicio) / 60:.1f} min\n"
        f"Series: {len(ids):,} | dias-serie por modelo: {int((~np.isnan(d)).sum()):,} | "
        f"cenarios: {len(MODELOS)} modelos x {len(NIVEIS)} niveis x {len(THETAS)} thetas\n",
        encoding="utf-8")
    print("      resultados/tabelas/sim_resumo_execucao.txt")

    pd.set_option("display.width", 220)
    print("\nCUSTO (nivel do newsvendor)\n"
          + no_newsvendor(res, "Custo 1:{cr}").to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nPERDA % DO RECEBIDO\n"
          + no_newsvendor(res, "Perda % do recebido").to_string(float_format=lambda v: f"{v:.4f}"))
    for nome, o in ot.items():
        print(f"\n{nome.upper()}\n" + o.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
