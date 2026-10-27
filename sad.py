"""
===============================================================================
SISTEMA DE APOIO A DECISAO (SAD) — estoques pereciveis no varejo alimentar
Projeto de Graduacao — Engenharia de Producao (UnB)
===============================================================================

Duas abas, com os parametros na barra lateral:

OPERACAO — a tela do dia: para uma loja e uma data do periodo de teste
(26/01 a 24/04/2016), uma linha por item com estoque disponivel, previsao,
alvo de reposicao, pedido ao fornecedor e situacao; abaixo, os transbordos
sugeridos. Le dados/sad_operacao.npz (gerado por scripts/etapa6_sad_apoio.py,
nao versionado: exige os dados do M5) e reproduz a simulacao da Etapa 5B num
unico cenario (nivel da aba Calibracao, theta, L_f e, se compensar, L_t), dia a
dia ate a data — o motor e conferido contra a Etapa 5B ao gerar o arquivo.

CALIBRACAO — qual nivel de servico usar. NAO refaz simulacao: le a tabela
resultados/tabelas/tbr_resultados_completos.csv, gerada por
scripts/etapa5_transbordo.py --etapa-5b (modelo: combinacao Holt-Winters +
XGBoost; 1.437 itens x 4 lojas; 90 dias de teste). Funciona sem os dados do M5.

COMO RODAR (na pasta do projeto):
    streamlit run sad.py

COMO OS NUMEROS SAO OBTIDOS
- Nivel recomendado: tau* = c_r / (c_r + theta * c_h), o nivel que minimiza o
  custo quando so a fracao theta da sobra se perde por dia. O newsvendor
  (tau = c_r / (c_r + c_h)) trata toda sobra como perda.
- A simulacao usou niveis inteiros de 70% a 99%; o nivel aplicado e tau*
  arredondado para o inteiro mais proximo, limitado a essa faixa. E esse nivel
  que a aba de Operacao usa.
- Custo por serie-dia = perda por validade + (c_r/c_h) x faltas [+ c_t x volume
  transbordado], em unidades de c_h. Como a tabela guarda perda e faltas
  separadas, o custo vale para qualquer razao c_r/c_h.
- theta foi simulado em 0,05 / 0,10 / 0,15 / 0,20; entre esses valores os
  indicadores sao interpolados linearmente; fora deles, o SAD nao estima.
- Operacao: o transbordo so e sugerido quando a aba Calibracao conclui que
  ele compensa no cenario informado (decisao de menor custo esperado).
===============================================================================
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "scripts"))
from etapa6_sad_apoio import ARQ_APOIO, trajetoria, pares_transbordo   # so numpy

ARQ = RAIZ / "resultados" / "tabelas" / "tbr_resultados_completos.csv"
THETA_MIN, THETA_MAX = 0.05, 0.20
NIVEL_MIN, NIVEL_MAX = 70, 99
COLUNAS = ["Perda total", "Perda % do recebido", "Faltas", "Atendimento", "Estoque médio",
           "Volume enviado", "Volume % do fluxo"]
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#52514e"
RUPTURA, EXCESSO, NORMAL = "Risco de ruptura", "Excesso", "Normal"
COR_SITUACAO = {RUPTURA: "color: #c0392b; font-weight: 600",
                EXCESSO: "color: #b8650a; font-weight: 600", NORMAL: ""}


# -----------------------------------------------------------------------------
# Dados e contas
# -----------------------------------------------------------------------------
@st.cache_data
def carregar():
    return pd.read_csv(ARQ)


def curva(df, lf, lt, theta):
    """Indicadores por nivel no cenario (L_f, L_t), interpolados linearmente em
    theta entre os valores simulados. None se theta estiver fora da faixa."""
    if not THETA_MIN - 1e-9 <= theta <= THETA_MAX + 1e-9:
        return None
    sel = df[(df["L_f"] == lf) & (df["L_t"] == lt)]
    linhas = {}
    for nivel, g in sel.groupby("Nível"):
        g = g.sort_values("theta")
        linhas[nivel] = {c: float(np.interp(theta, g["theta"], g[c])) for c in COLUNAS}
    return pd.DataFrame(linhas).T


def custo(c, razao, ct=0.0):
    return c["Perda total"] + razao * c["Faltas"] + ct * c["Volume enviado"]


def nivel_aplicado(tau):
    return int(min(max(np.floor(100 * tau + 0.5), NIVEL_MIN), NIVEL_MAX))


def avaliar_transbordo(df, lf, lt, theta, razao, c_t, nivel):
    """Recomendacao sobre o transbordo no nivel dado (usada pelas duas abas).
    None se theta estiver fora da faixa simulada."""
    sem, com = curva(df, lf, 0, theta), curva(df, lf, lt, theta)
    if sem is None:
        return None
    cs, cc = custo(sem.loc[nivel], razao), custo(com.loc[nivel], razao)
    vol = com.loc[nivel, "Volume enviado"]
    ct_eq = (cs - cc) / vol if vol > 0 else float("nan")
    rapido = lt < lf + 1
    return dict(rapido=rapido, ct_eq=ct_eq, vol=vol, vol_pct=com.loc[nivel, "Volume % do fluxo"],
                efeito=custo(com.loc[nivel], razao, c_t) / cs - 1,
                compensa=bool(rapido and ct_eq > 0 and c_t < ct_eq))


def br(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, casas=1):
    return br(100 * x, casas) + "%"


# --- Operacao: arquivo de apoio e trajetoria ---
@st.cache_resource
def carregar_apoio():
    if not ARQ_APOIO.exists():
        return None
    z = np.load(ARQ_APOIO)
    return {k: z[k] for k in ("itens", "lojas", "datas", "niveis", "demanda", "fechada",
                              "previsao")}


@st.cache_resource
def alvos(lf):
    """Alvos arredondados (niveis x itens x lojas x dias) do prazo L_f."""
    return np.load(ARQ_APOIO)[f"alvo_lf{lf}"]


@st.cache_data(max_entries=6, show_spinner="Reproduzindo a trajetória de estoque até a data...")
def simular(nivel, theta, lf, lt):
    ap = carregar_apoio()
    k = list(ap["niveis"]).index(nivel)
    return trajetoria(alvos(lf)[k].astype(float), alvos(0)[k].astype(float), ap["previsao"],
                      ap["demanda"], ap["fechada"], theta, lf, lt)


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SAD — estoques perecíveis", layout="wide")
st.title("Sistema de apoio à decisão — estoques perecíveis")
st.caption("Recomendações a partir da simulação do trabalho (categoria FOODS, 4 lojas da "
           "Califórnia, base M5; previsão pela combinação Holt-Winters + XGBoost).")

with st.sidebar:
    st.header("Parâmetros")
    c_r = st.number_input("Custo de ruptura por unidade (c_r)", min_value=0.01, value=9.0, step=0.5)
    c_h = st.number_input("Custo por unidade perdida por validade (c_h)", min_value=0.01,
                          value=1.0, step=0.5,
                          help="No modelo, o custo de sobra incide sobre a unidade que se "
                               "perde por validade; a sobra que não estraga é vendida depois.")
    theta = st.number_input("Taxa de perecibilidade θ (fração da sobra perdida por dia)",
                            min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f")
    c_t = st.number_input("Custo de transbordo por unidade (em unidades de c_h)",
                          min_value=0.0, value=0.10, step=0.05, format="%.2f")
    lf = st.selectbox("Prazo do fornecedor L_f (dias além da entrega noturna)", [0, 1, 2, 3])
    lt = st.selectbox("Prazo do transbordo L_t (noites em trânsito)", [1, 2])

df = carregar()
razao = c_r / c_h
tau_nv = razao / (razao + 1)
tau_rec = razao / (razao + theta) if theta > 0 else 1.0
n_rec, n_nv = nivel_aplicado(tau_rec), nivel_aplicado(tau_nv)
sem = curva(df, lf, 0, theta)
tb = avaliar_transbordo(df, lf, lt, theta, razao, c_t, n_rec)

aba_op, aba_cal = st.tabs(["Operação", "Calibração"])

# =============================================================================
# ABA 1 — OPERACAO
# =============================================================================
with aba_op:
    ap = carregar_apoio()
    if ap is None:
        st.warning("Arquivo de apoio `dados/sad_operacao.npz` não encontrado. Ele é gerado por "
                   "`python scripts/etapa6_sad_apoio.py` e exige os dados do M5 e as previsões "
                   "das etapas anteriores (ver README). A aba Calibração funciona sem ele.")
    else:
        transb = tb is not None and tb["compensa"]
        lt_op = lt if transb else None
        if tb is None:
            motivo = (f"o transbordo não é avaliado com θ fora da faixa simulada "
                      f"({br(THETA_MIN)} a {br(THETA_MAX)})")
        elif not tb["rapido"]:
            motivo = (f"o transbordo (L_t = {lt}) não é mais rápido que o fornecedor "
                      f"({lf + 1} noite(s)) e não compensa")
        elif not transb:
            motivo = "o transbordo não compensa com o custo de transporte informado"
        st.markdown(
            f"Nível de serviço aplicado: **{n_rec}%** (definido na aba Calibração) · "
            f"θ = {br(theta)} · prazo do fornecedor L_f = {lf} · transbordo: "
            + (f"**ativo** (L_t = {lt})" if transb else f"**desligado** — {motivo}"))

        ini = date.fromisoformat(str(ap["datas"][0]))
        fim = date.fromisoformat(str(ap["datas"][-1]))
        lojas = [str(x) for x in ap["lojas"]]
        e1, e2 = st.columns(2)
        loja = e1.selectbox("Loja", lojas)
        dia = e2.date_input("Data", value=fim, min_value=ini, max_value=fim, format="DD/MM/YYYY")
        t, j = (dia - ini).days, lojas.index(loja)

        r = simular(n_rec, float(theta), int(lf), lt_op)
        k = list(ap["niveis"]).index(n_rec)
        ativa = ~np.isnan(ap["demanda"][:, j, t]) & ~ap["fechada"][:, j, t]
        if not ativa.any():
            st.info(f"A loja {loja} não opera em {dia:%d/%m/%Y}.")
        else:
            alvo = alvos(lf)[k][:, j, t].astype(float)
            alvo1 = alvos(0)[k][:, j, t].astype(float)
            g = lambda c: r[c][:, j, t].astype(float)
            disp, caminho, pedido, estoque = g("abertura"), g("caminho"), g("pedido"), g("estoque")
            ruptura = estoque <= alvo1 - 1                      # abaixo do quantil do dia
            excesso = ~ruptura & (disp + caminho >= alvo + 1)   # acima do alvo, sem pedido
            situacao = np.where(ruptura, RUPTURA, np.where(excesso, EXCESSO, NORMAL))
            real = np.nan_to_num(ap["demanda"][:, j, t].astype(float))
            col_alvo = "Alvo de reposição" + (f" ({lf + 1} dias)" if lf > 0 else "")
            tab = pd.DataFrame({
                "Item": ap["itens"], "Estoque disponível": disp, "A caminho": caminho,
                "Demanda prevista": ap["previsao"][:, j, t].astype(float), col_alvo: alvo,
                "Pedir ao fornecedor": pedido, "Situação": situacao,
                "Demanda real": real, "Desvio (real − estoque do dia)": real - estoque,
                "_ordem": np.select([ruptura, excesso], [0, 1], 2)})[ativa]
            if lf == 0:
                tab = tab.drop(columns="A caminho")
            tab = (tab.sort_values(["_ordem", "Demanda prevista"], ascending=[True, False])
                   .drop(columns="_ordem").reset_index(drop=True))

            pares = []
            if transb and t > 0:                               # decididos na noite anterior
                pares = [p for p in pares_transbordo(r["envio"][:, :, t - 1].astype(float),
                                                     r["recebe"][:, :, t - 1].astype(float))
                         if j in (p[1], p[2])]
            vol_tb = sum(p[3] for p in pares)

            # --- Resumo ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Itens", br(len(tab), 0))
            m2.metric("Total a pedir (un.)", br(tab["Pedir ao fornecedor"].sum(), 1))
            m3.metric("Em risco de ruptura", br(int((tab["Situação"] == RUPTURA).sum()), 0))
            m4.metric("Em excesso", br(int((tab["Situação"] == EXCESSO).sum()), 0))
            m5.metric("Transbordo (un.)", br(vol_tb, 1),
                      help="Unidades enviadas e recebidas pela loja nos transbordos sugeridos.")

            # --- Tabela de reposicao ---
            st.subheader(f"Reposição — {loja}, {dia:%d/%m/%Y}")
            st.info("**Demanda real** e **Desvio** existem só para demonstração e validação: o "
                    "sistema opera sobre o período histórico de teste (26/01 a 24/04/2016), em "
                    "que a venda do dia já é conhecida. Numa operação real, essas colunas não "
                    "existiriam no momento da decisão.")
            formato = {c: (lambda v: br(v, 2)) if c == "Demanda prevista" else
                       (lambda v: br(v, 0)) if c in (col_alvo, "Demanda real") else
                       (lambda v: br(v, 1))
                       for c in tab.columns if c not in ("Item", "Situação")}
            estilo = (tab.style.format(formato)
                      .map(lambda s: COR_SITUACAO[s], subset=["Situação"]))
            st.dataframe(estilo, hide_index=True, width="stretch", height=420)
            entrega = ("O pedido chega antes da abertura (L_f = 0)." if lf == 0 else
                       f"O pedido chega em {lf + 1} noites e cobre {lf + 1} dias; pedido = alvo − "
                       "estoque disponível − a caminho.")
            st.caption(
                "Estoque disponível: sobra da véspera menos a perda da noite (θ), mais o que "
                "chegou hoje (transbordo e pedidos anteriores); não inclui o pedido de hoje. "
                f"Alvo: quantil da demanda no nível de {n_rec}% (previsão + quantil empírico dos "
                f"erros dos 180 dias anteriores), arredondado. {entrega} "
                "**Risco de ruptura**: o estoque do dia fica pelo menos 1 unidade abaixo do "
                "quantil da demanda do dia (chance de falta maior que a tolerada"
                + ("; com L_f = 0 não ocorre, porque a entrega chega antes da abertura" if lf == 0
                   else "") + "). **Excesso**: estoque (com o que está a caminho) pelo menos "
                "1 unidade acima do alvo; nada a pedir. Desvio: positivo = faltou, negativo = "
                "sobrou. Unidades fracionárias vêm da perda proporcional θ.")

            # --- Transbordos ---
            st.subheader("Transbordos sugeridos")
            if not transb:
                st.write(f"Nenhum transbordo sugerido: {motivo} (ver aba Calibração).")
            elif not pares:
                st.write(f"Nenhum transbordo sugerido envolvendo a loja {loja} em "
                         f"{dia:%d/%m/%Y}.")
            else:
                ttb = pd.DataFrame([(str(ap["itens"][i]), lojas[o], lojas[dd], q)
                                    for i, o, dd, q in pares],
                                   columns=["Item", "Loja de origem", "Loja de destino",
                                            "Quantidade enviada"])
                ttb = ttb.sort_values("Quantidade enviada", ascending=False)
                st.dataframe(ttb.style.format({"Quantidade enviada": lambda v: br(v, 1)}),
                             hide_index=True, width="stretch")
            if transb:
                chega = dia + timedelta(days=lt - 1)
                st.caption(f"Decididos na noite anterior, junto com o pedido ao fornecedor: "
                           f"excedente acima do alvo na origem vai para lojas abaixo do alvo, "
                           f"em rateio proporcional. Chegam em {chega:%d/%m/%Y} valendo "
                           f"(1 − θ)^{lt} da quantidade enviada (perda em trânsito). Com "
                           f"L_t = 1, o recebido já está no estoque disponível do destino.")
        st.caption(f"Estoques reproduzidos pela simulação da Etapa 5B desde {ini:%d/%m/%Y}, "
                   "dia a dia, com a regra de reposição até o alvo e venda perdida.")

# =============================================================================
# ABA 2 — CALIBRACAO
# =============================================================================
with aba_cal:
    # --- 1. Nivel de servico recomendado ---
    st.subheader("1. Nível de serviço recomendado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Newsvendor (sobra = perda total)", pct(tau_nv))
    c2.metric("Recomendado (com perecibilidade θ)", pct(tau_rec))
    if sem is not None and NIVEL_MIN <= 100 * tau_nv:
        economia = 1 - custo(sem.loc[n_rec], razao) / custo(sem.loc[n_nv], razao)
        c3.metric("Economia esperada sobre o newsvendor", pct(economia))
    else:
        c3.metric("Economia esperada sobre o newsvendor", "—")
    st.caption(f"Fórmula: τ* = c_r / (c_r + θ·c_h) = {br(c_r)} / ({br(c_r)} + {br(theta)}·{br(c_h)}). "
               f"Nível aplicado na simulação e na aba Operação: **{n_rec}%** (níveis inteiros de "
               f"{NIVEL_MIN}% a {NIVEL_MAX}%).")
    if 100 * tau_rec > NIVEL_MAX + 0.5:
        st.info(f"O nível recomendado passa de {NIVEL_MAX}%, o limite da faixa simulada; os "
                f"indicadores abaixo usam {NIVEL_MAX}%.")
    if 100 * tau_nv < NIVEL_MIN - 0.5:
        st.info(f"O nível do newsvendor ({pct(tau_nv)}) está abaixo da faixa simulada "
                f"({NIVEL_MIN}%–{NIVEL_MAX}%); a economia não pode ser estimada.")
    if sem is None:
        st.warning(f"θ = {br(theta)} está fora da faixa examinada no trabalho "
                   f"({br(THETA_MIN)} a {br(THETA_MAX)}). Só a fórmula do nível pode ser "
                   "aplicada; os indicadores, a curva e a recomendação de transbordo não são "
                   "estimados.")
        st.stop()
    otimo = int(custo(sem, razao).idxmin())
    st.caption(f"Na simulação (L_f = {lf}, sem transbordo), o nível de menor custo foi {otimo}%"
               + (" (limite da faixa)." if otimo == NIVEL_MAX else "."))

    # --- 2. Indicadores esperados ---
    st.subheader(f"2. Indicadores esperados no nível de {n_rec}%")
    r = sem.loc[n_rec]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Perda por validade (un./série-dia)", br(r["Perda total"], 3))
    k2.metric("Perda (% do recebido)", pct(r["Perda % do recebido"]))
    k3.metric("Faltas (un./série-dia)", br(r["Faltas"], 3))
    k4.metric("Taxa de atendimento", pct(r["Atendimento"]))
    k5.metric("Estoque médio (un./série)", br(r["Estoque médio"], 2))
    if NIVEL_MIN <= n_nv <= NIVEL_MAX:
        linhas = [("Perda (un./série-dia)", "Perda total", False),
                  ("Perda (% do recebido)", "Perda % do recebido", True),
                  ("Faltas (un./série-dia)", "Faltas", False),
                  ("Atendimento", "Atendimento", True),
                  ("Estoque médio (un./série)", "Estoque médio", False)]
        comp = pd.DataFrame({
            rotulo: [pct(sem.loc[n, col]) if eh_pct else br(sem.loc[n, col], 3)
                     for rotulo_l, col, eh_pct in linhas]
            for rotulo, n in ((f"Newsvendor ({n_nv}%)", n_nv), (f"Recomendado ({n_rec}%)", n_rec))
        }, index=[l[0] for l in linhas])
        st.dataframe(comp, width="stretch")

    # --- 3. Curva de compromisso ---
    st.subheader("3. Curva de compromisso: perda e falta por nível de serviço")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(sem.index, sem["Perda total"], color=AZUL, linewidth=2, label="Perda por validade")
    ax.plot(sem.index, sem["Faltas"], color=LARANJA, linewidth=2, label="Faltas")
    for col, cor in (("Perda total", AZUL), ("Faltas", LARANJA)):
        ax.plot(n_rec, sem.loc[n_rec, col], marker="o", markersize=9, color=cor,
                markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax.axvline(n_rec, color=CINZA, linewidth=1, linestyle="--")
    ax.text(n_rec, 1.0, f" recomendado ({n_rec}%)", transform=ax.get_xaxis_transform(),
            fontsize=8, color=CINZA, va="top", ha="right" if n_rec > 90 else "left")
    if NIVEL_MIN <= n_nv <= NIVEL_MAX:
        ax.axvline(n_nv, color=CINZA, linewidth=1, linestyle=":")
        ax.text(n_nv, 0.9, f" newsvendor ({n_nv}%)", transform=ax.get_xaxis_transform(),
                fontsize=8, color=CINZA, va="top")
    ax.set_xlabel("Nível de serviço do alvo de estoque")
    ax.set_ylabel("Unidades por série-dia")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: br(v, 1)))
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    st.pyplot(fig)

    # --- 4. Transbordo ---
    st.subheader("4. Recomendação sobre o transbordo entre lojas")
    ct_eq, efeito = tb["ct_eq"], tb["efeito"]
    velocidade = (f"O fornecedor entrega em {lf + 1} noite(s); o transbordo leva {lt}. "
                  + ("O transbordo é **mais rápido** que o fornecedor." if tb["rapido"] else
                     "O transbordo **não é mais rápido** que o fornecedor."))
    if not tb["rapido"]:
        st.error(f"**Não compensa.** {velocidade} Sem essa vantagem de prazo, o destino seria "
                 "abastecido pelo fornecedor de qualquer forma, e o transbordo só tira proteção da "
                 f"loja de origem. Custo de equilíbrio neste cenário: {br(ct_eq, 3)}·c_h.")
    elif not ct_eq > 0:
        st.error(f"**Não compensa, nem com transporte gratuito.** {velocidade} Custo de equilíbrio "
                 f"negativo: {br(ct_eq, 3)}·c_h.")
    elif c_t < ct_eq:
        st.success(f"**Compensa.** {velocidade} O custo informado ({br(c_t, 2)}·c_h) está abaixo "
                   f"do custo de equilíbrio ({br(ct_eq, 3)}·c_h), com margem de "
                   f"{br(ct_eq - c_t, 3)}·c_h por unidade. Efeito esperado no custo total: "
                   f"{pct(efeito)}.")
    else:
        st.error(f"**Não compensa.** {velocidade} O custo informado ({br(c_t, 2)}·c_h) passa do "
                 f"custo de equilíbrio ({br(ct_eq, 3)}·c_h) em {br(c_t - ct_eq, 3)}·c_h por "
                 f"unidade. Efeito esperado no custo total: {pct(efeito)}.")
    st.caption(f"Avaliado no nível de {n_rec}%. Volume transbordado: {br(tb['vol'], 3)} "
               f"un./série-dia ({pct(tb['vol_pct'])} das unidades que entram nas lojas). "
               "Custo de equilíbrio = redução de custo com transporte gratuito ÷ volume "
               "transbordado. A aba Operação só sugere transbordos quando esta seção conclui "
               "que compensa.")

    with st.expander("Como os números são obtidos e limites do sistema"):
        st.markdown(
            f"- Fonte: `resultados/tabelas/tbr_resultados_completos.csv` (Etapa 5B do trabalho). "
            "Esta aba não refaz simulação.\n"
            "- Período avaliado: 90 dias (26/01 a 24/04/2016), 1.437 itens × 4 lojas.\n"
            f"- θ simulado em 0,05; 0,10; 0,15 e 0,20; valores intermediários são interpolados "
            "linearmente. Fora dessa faixa, só a fórmula do nível é aplicada.\n"
            f"- Níveis simulados: inteiros de {NIVEL_MIN}% a {NIVEL_MAX}%.\n"
            "- Premissas do modelo: reposição diária até o alvo; venda perdida; perda por validade "
            "como fração θ da sobra a cada noite; carga do fornecedor chega fresca; transbordo "
            "perde θ por noite em trânsito; sem custo de manter estoque além da perda.\n"
            "- A fórmula do nível foi validada para prazo zero do fornecedor; com prazo, compare "
            "com o nível de menor custo informado na seção 1.")
