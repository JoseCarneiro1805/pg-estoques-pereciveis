"""
===============================================================================
ETAPA 5 — Transbordo lateral preventivo entre as quatro lojas
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE SCRIPT FAZ
---------------------
Repete a simulacao multiperiodo da Etapa 4 para a COMBINACAO, agora com as
quatro lojas de cada item ligadas por uma regra de transbordo, e compara com
a simulacao sem transbordo nas mesmas condicoes (mesma grade: niveis de 70% a
99%, theta = 0,05/0,10/0,15/0,20, razoes 1:4, 1:9, 1:19).

ESCOPO (PCP minimo, orientacao do professor) — premissas a declarar
---------------------------------------------------------------------
Uma regra de transbordo; sem lote, sem custo fixo, sem classificacao ABC, sem
capacidade, sem roteirizacao. Nenhum par de lojas e privilegiado (o M5 nao
informa localizacao). A componente fixa do transporte nao e modelada
(consolidacao de carga entre os 1.437 itens). Unidades fracionarias, como nos
estoques da Etapa 4.

A REGRA — decisoes aprovadas em 26/10/2026 (justificativa na monografia)
----------------------------------------------------------
Preventiva: ao fim do dia t, depois da demanda, com base no alvo de t+1.
  excedente da loja i = max(sobra_i - S_i(t+1), 0)       (antes da perda da noite)
  falta da loja j     = max(S_j(t+1) - [(1-theta)*sobra_j + chegada_j(t+1)], 0)
                        so para lojas sem excedente
  volume do item      = min(soma dos excedentes, soma das faltas)
  rateio proporcional ao excedente (origens) e a falta (destinos)
Premissas da Etapa 4 mantidas: todo dia cada loja e reposta pelo fornecedor
ate o alvo, com prazo zero e sem limite. Por isso o transbordo ABATE o pedido
ao fornecedor do destino; nao muda o servico no destino, so o desperdicio (e,
na origem, retira a protecao acima do alvo — efeito medido aqui).

PRAZO E PERDA EM TRANSITO
-------------------------
O numero de aplicacoes de (1-theta) e igual ao prazo L: a perda incide a cada
virada de dia, na prateleira ou no caminhao.
  L = 2 (base):          chega em t+2 valendo (1-theta)^2
  L = 1 (sensibilidade): chega em t+1 valendo (1-theta)
Na chegada nao ha perda adicional: a carga entra no estoque do destino depois
da perda da noite daquele dia. Com L = 2, a necessidade em t+2 e aproximada
pelo alvo de t+1 (os modelos so preveem 1 dia a frente); o tamanho dessa
aproximacao e medido (tabelas tb_aproximacao_*). Nao ha envio que so chegaria
depois do fim do horizonte.

CUSTO
-----
c_h x (perda em loja + perda em transito) + c_r x faltas + c_t x volume enviado
c_h = 1; c_r = 4, 9, 19; c_t em {0; 0,05; 0,10; 0,20; 0,50} unidades de c_h.
c_t entra so na conta: a regra transporta sempre que ha excedente e falta.
c_t de equilibrio = (custo sem transbordo - custo com transbordo e c_t = 0)
                    / volume enviado: acima dele, o transbordo nao compensa.

CONFERENCIAS
------------
- Um caso de 2 lojas x 3 dias calculado a mao (L = 1, L = 2 e sem transbordo).
- Sem transbordo, a simulacao em rede reproduz exatamente a Etapa 4.

COMO USAR
---------
    python scripts/etapa5_transbordo.py                (Etapa 5, prazo zero do fornecedor)
    python scripts/etapa5_transbordo.py --etapa-5b     (Etapa 5B, prazo do fornecedor
                                                        L_f = 0 a 3; ver bloco ETAPA 5B)
    opcoes da 5B: --so-verificacao (para apos a verificacao de cobertura)
                  --so-figuras     (refaz so as figuras a partir das tabelas)

PRE-REQUISITO: etapa3_comparacao.py (previsoes da combinacao) e etapa4_simulacao.py;
               a 5B usa tambem dados/previsoes_pontuais_*.npz (gerados pelos modelos)
SAIDAS: Etapa 5:  resultados/tabelas/tb_*.csv e .tex, tb_resumo_execucao.txt
                  resultados/figuras/fig_tb_fronteira.png, fig_tb_ganho_nivel.png
        Etapa 5B: resultados/tabelas/tbr_*.csv e .tex, tbr_resumo_execucao.txt
                  resultados/figuras/fig_tbr_equilibrio.png, fig_tbr_fronteira.png
===============================================================================
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import (
    PASTA, PASTA_TAB, ARQUIVO_SAIDA, COR_TEXTO_SEC, estilo_eixos, salvar_figura, num_pt,
)
import etapa3_avaliacao as av

ORIGEM = "etapa5_transbordo.py"
ARQ_PREVISOES = "previsoes_combinacao.parquet"
THETAS = (0.05, 0.10, 0.15, 0.20)
THETA_BASE = 0.10
NIVEIS = av.NIVEIS_QUANTIS                              # 70% a 99%
CENARIOS = {"Sem transbordo": None, "Transbordo, L = 2": 2, "Transbordo, L = 1": 1}
CUSTOS_TRANSBORDO = (0.0, 0.05, 0.10, 0.20, 0.50)       # unidades de c_h
NIVEIS_APROX = (0.80, 0.90, 0.95, 0.98, 0.99)
ATENDIMENTOS_FRONTEIRA = (0.90, 0.95, 0.98)
DIAS_SEMANA = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
CORES = {"Sem transbordo": "#2a78d6", "Transbordo, L = 2": "#eb6834",
         "Transbordo, L = 1": "#1baf7a"}
INDICADORES = ["perda_loja", "perda_transito", "fornecedor", "chegada", "enviado",
               "vendido", "falta", "demanda", "estoque", "dias", "dias_sem_falta",
               "dias_envio", "dias_cobre_alvo"]
BASE = ("simulação de 90 dias (26/01 a 24/04/2016), combinação, 1.437 itens × 4 lojas; "
        "reposição diária pelo fornecedor até o quantil do nível de serviço, "
        "arredondado para o inteiro mais próximo; venda perdida.")


def rot_theta(th):
    return f"θ = {num_pt(th, 2)}"


def rot_razao(tau):
    return f"1:{av.CUSTO_FALTA[tau]} ({av.pct(tau)}%)"


def rot_ct(ct):
    return f"c_t = {num_pt(ct, 2)}"


# -----------------------------------------------------------------------------
# DADOS: grade nivel x item x loja x dia
# -----------------------------------------------------------------------------
def carregar():
    print("\n[1/5] Lendo previsoes da combinacao...")
    cols_q = [f"q_emp_{av.pct(t)}" for t in NIVEIS]
    df = pd.read_parquet(PASTA / ARQ_PREVISOES,
                         columns=["id", "store_id", "dia", "date", "vendas", "loja_fechada"]
                         + cols_q)
    item = av.item_de(df["id"])
    itens, lojas = np.unique(item), np.sort(df["store_id"].unique())
    i = np.searchsorted(itens, item)
    j = np.searchsorted(lojas, df["store_id"].to_numpy())
    t = df["dia"].to_numpy() - av.INICIO_TESTE
    D = av.FIM_TESTE - av.INICIO_TESTE + 1
    assert len(df) == len(set(zip(i, j, t))), "serie-dia repetido"
    d = np.full((len(itens), len(lojas), D), np.nan)
    d[i, j, t] = df["vendas"].to_numpy(float)
    fech = np.zeros(d.shape, bool)
    fech[i, j, t] = df["loja_fechada"].to_numpy(bool)
    S = np.zeros((len(NIVEIS),) + d.shape, np.float32)
    for k, c in enumerate(cols_q):
        S[k, i, j, t] = av.decisao(np.maximum(df[c].to_numpy(float), 0))
    datas = pd.to_datetime(df.drop_duplicates("dia").set_index("dia")["date"]).sort_index()
    print(f"      {len(itens):,} itens x {len(lojas)} lojas x {D} dias")
    return itens, lojas, d, fech, S, datas.to_numpy()


# -----------------------------------------------------------------------------
# SIMULACAO EM REDE
# -----------------------------------------------------------------------------
def simular_rede(S, d, fech, theta, prazo=None, falta_base=None):
    """S: alvos (niveis x itens x lojas x dias); d: demanda (itens x lojas x
    dias; NaN antes de a serie existir). prazo=None: sem transbordo.
    Devolve totais por (nivel, item, loja), a falta diaria (so sem transbordo,
    para comparar depois) e o efeito na origem (so com transbordo, comparando
    com falta_base nos dias seguintes a um envio)."""
    Lv, I, J, D = S.shape
    ativa = ~np.isnan(d)
    aberta = ativa & ~fech
    sobra = np.zeros((Lv, I, J))
    pendentes = {}                                   # dia de chegada -> quantidade
    tot = {k: np.zeros((Lv, I, J)) for k in INDICADORES}
    falta_dia = np.zeros(S.shape, np.float32) if prazo is None else None
    orig = {k: np.zeros(Lv) for k in ("dias", "falta_com", "falta_sem", "demanda",
                                      "cobre_alvo")}
    enviou = np.zeros((Lv, I, J), bool)
    fator = (1 - theta) ** prazo if prazo else 1.0
    for t in range(D):
        a = aberta[:, :, t]
        perdido = theta * sobra                                       # 1. perda da noite
        chega = pendentes.pop(t, 0.0)
        disponivel = sobra - perdido + chega                          # carga chega sem perda extra
        pedido = np.where(a, np.maximum(S[:, :, :, t] - disponivel, 0), 0)   # 2. fornecedor
        estoque = disponivel + pedido
        dem = np.where(a, np.nan_to_num(d[:, :, t]), 0)               # 3. demanda
        vendido = np.minimum(estoque, dem)
        falta = dem - vendido
        sobra = estoque - vendido                                     # 4. sobra
        tot["perda_loja"] += perdido
        tot["fornecedor"] += pedido
        tot["chegada"] += chega
        tot["vendido"] += vendido
        tot["falta"] += falta
        tot["demanda"] += dem
        tot["estoque"] += np.where(a, estoque, 0)
        tot["dias"] += a
        tot["dias_sem_falta"] += a & (falta == 0)
        # Cobertura do proprio alvo: a demanda do dia cabe no alvo? (independe do transbordo)
        cobre = a & (dem <= S[:, :, :, t] + 1e-9)
        tot["dias_cobre_alvo"] += cobre
        if prazo is None:
            falta_dia[..., t] = falta
        else:                                     # servico da loja no dia seguinte ao envio
            m = enviou & a
            orig["dias"] += m.sum(axis=(1, 2))
            orig["cobre_alvo"] += (m & cobre).sum(axis=(1, 2))
            orig["falta_com"] += np.where(m, falta, 0).sum(axis=(1, 2))
            orig["falta_sem"] += np.where(m, falta_base[..., t], 0).sum(axis=(1, 2))
            orig["demanda"] += np.where(m, dem, 0).sum(axis=(1, 2))
            enviou = np.zeros_like(enviou)
        # 5. transbordo preventivo ao fim do dia
        if prazo and t + prazo <= D - 1:
            alvo = S[:, :, :, t + 1]
            excedente = np.where(ativa[:, :, t] & ativa[:, :, t + 1],
                                 np.maximum(sobra - alvo, 0), 0)
            projetado = (1 - theta) * sobra + pendentes.get(t + 1, 0.0)
            precisa = np.where(aberta[:, :, t + 1] & (excedente == 0),
                               np.maximum(alvo - projetado, 0), 0)
            E = excedente.sum(axis=2, keepdims=True)
            F = precisa.sum(axis=2, keepdims=True)
            volume = np.minimum(E, F)
            with np.errstate(invalid="ignore", divide="ignore"):
                envio = np.where(E > 0, excedente * volume / E, 0)
                recebe = np.where(F > 0, precisa * volume / F, 0)
            sobra = sobra - envio
            pendentes[t + prazo] = pendentes.get(t + prazo, 0.0) + recebe * fator
            tot["enviado"] += envio
            tot["perda_transito"] += recebe * (1 - fator)     # atribuida ao destino
            enviou = envio > 1e-12
            tot["dias_envio"] += enviou
    assert not pendentes, "carga em transito apos o horizonte"
    return tot, falta_dia, orig


def resumir(tot):
    """Indicadores agregados por nivel (todas as series somadas)."""
    s = {k: v.sum(axis=(1, 2)) for k, v in tot.items()}
    dias = s["dias"]
    perda = s["perda_loja"] + s["perda_transito"]
    return pd.DataFrame({
        "Perda em loja": s["perda_loja"] / dias,
        "Perda em trânsito": s["perda_transito"] / dias,
        "Perda total": perda / dias,
        "Perda % do recebido": perda / s["fornecedor"],
        "Faltas": s["falta"] / dias,
        "Atendimento": s["vendido"] / s["demanda"],
        "Cobertura": s["dias_sem_falta"] / dias,
        "Estoque médio": s["estoque"] / dias,
        "Volume enviado": s["enviado"] / dias,
        "Volume % do fluxo": s["enviado"] / (s["fornecedor"] + s["chegada"]),
        "Loja-dias com envio": s["dias_envio"] / dias,
    }, index=pd.Index([av.pct(t) for t in NIVEIS], name="Nível"))


def custo(r, cr, ct):
    """Custo por serie-dia (c_h = 1) a partir de uma linha de resumir()."""
    return r["Perda total"] + cr * r["Faltas"] + ct * r["Volume enviado"]


# -----------------------------------------------------------------------------
# CONFERENCIAS
# -----------------------------------------------------------------------------
def conferencia_manual():
    """2 lojas x 3 dias, theta = 0,10, calculado a mao (valores esperados abaixo)."""
    S = np.array([[[[5, 2, 2], [1, 3, 3]]]], np.float32)       # loja A, loja B
    d = np.array([[[1, 0, 1], [1, 1, 1]]], float)
    f = np.zeros(d.shape, bool)
    esperado = {
        None: dict(perda_loja=0.96, perda_transito=0, fornecedor=10.2, enviado=0),
        1: dict(perda_loja=0.6, perda_transito=0.2, fornecedor=8.8, enviado=2, chegada=1.8),
        2: dict(perda_loja=0.6, perda_transito=0.38, fornecedor=9.4, enviado=2, chegada=1.62),
    }
    for prazo, esp in esperado.items():
        tot, _, _ = simular_rede(S, d, f, 0.10, prazo,
                                 falta_base=np.zeros(S.shape, np.float32))
        for k, v in esp.items():
            assert abs(tot[k].sum() - v) < 1e-9, (prazo, k, tot[k].sum(), v)
    print("      conferencia manual (2 lojas x 3 dias): ok")


def conferir_etapa4(base):
    """Sem transbordo, a simulacao em rede tem de reproduzir a Etapa 4."""
    e4 = pd.read_csv(PASTA_TAB / "sim_resultados_completos.csv")
    e4 = e4[e4["Modelo"] == "Combinação"]
    maior = 0.0
    for th, r in base.items():
        ref = e4[np.isclose(e4["theta"], th)].set_index("Nível")
        for c_e5, c_e4 in (("Perda em loja", "Perdidas"), ("Faltas", "Faltas"),
                           ("Atendimento", "Atendimento"), ("Cobertura", "Cobertura"),
                           ("Estoque médio", "Estoque médio"),
                           ("Perda % do recebido", "Perda % do recebido")):
            maior = max(maior, float(np.abs(r[c_e5] - ref[c_e4].reindex(r.index)).max()))
    assert maior < 1e-9, f"sem transbordo nao reproduz a Etapa 4 (dif. {maior:.2e})"
    print(f"      sem transbordo reproduz a Etapa 4 (maior diferenca {maior:.1e})")


# -----------------------------------------------------------------------------
# APROXIMACAO DO ALVO DE t+2 PELO DE t+1
# -----------------------------------------------------------------------------
def aproximacao(S, d, fech, datas):
    print("\n[2/5] Tamanho da aproximacao alvo(t+1) ~ alvo(t+2)...")
    ok = ~np.isnan(d) & ~fech
    par = ok[:, :, :-1] & ok[:, :, 1:]                        # dias k e k+1 validos
    dsem = pd.DatetimeIndex(datas).dayofweek
    geral, semana = {}, {}
    for tau in NIVEIS_APROX:
        k = NIVEIS.index(tau)
        a, b = S[k, :, :, :-1].astype(float), S[k, :, :, 1:].astype(float)
        dif = np.where(par, b - a, 0)
        base = np.where(par, a, 0)
        rot = f"{av.pct(tau)}%"
        geral[rot] = {"Alvo médio": base.sum() / par.sum(),
                      "Diferença com sinal": dif.sum() / base.sum(),
                      "Diferença absoluta": np.abs(dif).sum() / base.sum(),
                      "Pares com alvo diferente": (np.where(par, b != a, False)).sum() / par.sum()}
        for w in range(7):
            col = dsem[:-1] == w                               # dia k (t+1) com dia da semana w
            pw = par[:, :, col]
            semana.setdefault(f"{DIAS_SEMANA[w]} → {DIAS_SEMANA[(w + 1) % 7]}", {})
            semana[f"{DIAS_SEMANA[w]} → {DIAS_SEMANA[(w + 1) % 7]}"].update({
                f"Com sinal {rot}": dif[:, :, col].sum() / base[:, :, col].sum(),
                f"Absoluta {rot}": np.abs(dif[:, :, col]).sum() / base[:, :, col].sum()})
    g = pd.DataFrame(geral).T.rename_axis("Nível")
    w = pd.DataFrame(semana).T.rename_axis("Transição (t+1 → t+2)")
    cols = [c for c in w.columns if c.endswith(("90%", "98%"))]
    nota = (f"{BASE} Diferença entre o alvo de t+2 e o de t+1 (o usado como "
            "aproximação no prazo de 2 dias), somada sobre séries e dias e dividida pela "
            "soma dos alvos de t+1. Com sinal: positivo = alvo de t+2 maior (a "
            "aproximação subestima a necessidade).")
    av.salvar_tabela(g, "tb_aproximacao_alvo", "Tamanho da aproximação do alvo de t+2 "
                     "pelo de t+1", ORIGEM,
                     formatos={"Alvo médio": 3, "Diferença com sinal": "pct",
                               "Diferença absoluta": "pct", "Pares com alvo diferente": "pct"},
                     nota=nota + " Pares com alvo diferente: \\% dos pares de dias consecutivos "
                     "com alvos distintos.")
    av.salvar_tabela(w[cols], "tb_aproximacao_alvo_semana", "Aproximação do alvo de t+2 "
                     "pelo de t+1, por transição de dia da semana", ORIGEM,
                     formatos={c: "pct" for c in cols}, nota=nota)
    w.to_csv(PASTA_TAB / "tb_aproximacao_alvo_semana_completa.csv", encoding="utf-8-sig")
    return g, w


# -----------------------------------------------------------------------------
# RODADAS
# -----------------------------------------------------------------------------
def rodar(S, d, fech):
    print("\n[3/5] Simulando sem e com transbordo (30 niveis x 4 thetas x 3 cenarios)...")
    res, tots, origem = {}, {}, {}
    for th in THETAS:
        tot0, falta0, _ = simular_rede(S, d, fech, th)
        res[("Sem transbordo", th)] = resumir(tot0)
        tots[("Sem transbordo", th)] = tot0
        for nome, prazo in CENARIOS.items():
            if prazo is None:
                continue
            tot, _, orig = simular_rede(S, d, fech, th, prazo, falta_base=falta0)
            res[(nome, th)] = resumir(tot)
            tots[(nome, th)] = tot
            origem[(nome, th)] = orig
        del falta0
        print(f"      theta {th:.2f}: ok")
    conferir_etapa4({th: res[("Sem transbordo", th)] for th in THETAS})
    longa = pd.concat({k: v for k, v in res.items()}, names=["Cenário", "theta"]).reset_index()
    for tau, cr in av.CUSTO_FALTA.items():
        for ct in CUSTOS_TRANSBORDO:
            longa[f"Custo 1:{cr}, c_t = {ct:.2f}"] = custo(longa, cr, ct)
    longa.to_csv(PASTA_TAB / "tb_resultados_completos.csv", index=False, encoding="utf-8-sig")
    print("      resultados/tabelas/tb_resultados_completos.csv")
    return res, tots, origem


# -----------------------------------------------------------------------------
# TABELAS
# -----------------------------------------------------------------------------
def otimo(r, cr, ct):
    c = custo(r, cr, ct)
    return int(c.idxmin()), float(c.min())


def tabelas(res, tots, origem, itens):
    print("\n[4/5] Tabelas...")
    nomes = list(CENARIOS)
    nota_custo = ("Custo por série-dia = perda (em loja e em trânsito) + $c_r$ × faltas + "
                  "$c_t$ × volume enviado, com $c_h$ = 1.")

    # Custo e perda no nivel do newsvendor, por theta (c_t = 0)
    for nome_tab, titulo, col, fmt in (
            ("tb_custo", "Custo com e sem transbordo (nível do newsvendor, $c_t$ = 0)",
             None, 3),
            ("tb_perda", "Perda por validade com e sem transbordo (\\% das unidades "
             "recebidas do fornecedor; nível do newsvendor)", "Perda % do recebido", "pct"),
            ("tb_atendimento", "Atendimento com e sem transbordo (nível do newsvendor)",
             "Atendimento", "pct")):
        linhas = {}
        for tau, cr in av.CUSTO_FALTA.items():
            for nome in nomes:
                linhas[(rot_razao(tau), nome)] = {
                    rot_theta(th): (custo(res[(nome, th)].loc[av.pct(tau)], cr, 0.0)
                                    if col is None else res[(nome, th)].loc[av.pct(tau), col])
                    for th in THETAS}
        t = pd.DataFrame(linhas).T
        t.index.names = ["Razão (nível)", "Cenário"]
        av.salvar_tabela(t, nome_tab, titulo, ORIGEM,
                         formatos={c: fmt for c in t.columns},
                         nota=f"{BASE} {nota_custo if col is None else ''}".strip())

    # Indicadores detalhados, theta base
    linhas = {}
    for tau, cr in av.CUSTO_FALTA.items():
        for nome in nomes:
            r = res[(nome, THETA_BASE)].loc[av.pct(tau)]
            linhas[(rot_razao(tau), nome)] = {
                "Perda em loja": r["Perda em loja"], "Perda em trânsito": r["Perda em trânsito"],
                "Perda % receb.": r["Perda % do recebido"], "Faltas": r["Faltas"],
                "Atendimento": r["Atendimento"], "Volume enviado": r["Volume enviado"],
                "Volume % do fluxo": r["Volume % do fluxo"],
                "Custo (c_t = 0)": custo(r, cr, 0.0), "Custo (c_t = 0,10)": custo(r, cr, 0.10)}
    t = pd.DataFrame(linhas).T
    t.index.names = ["Razão (nível)", "Cenário"]
    av.salvar_tabela(
        t, "tb_indicadores", "Indicadores com e sem transbordo (θ = 0,10)", ORIGEM,
        largura_total=True,
        formatos={"Perda % receb.": "pct", "Atendimento": "pct", "Volume % do fluxo": "pct"},
        nota=(f"{BASE} Nível do newsvendor; θ = 0,10. Perdas, faltas e volume em unidades "
              "por série-dia. Perda \\% receb.: perda total (loja + trânsito) sobre as "
              "unidades recebidas do fornecedor. Volume \\% do fluxo: unidades enviadas "
              "sobre o total que entrou nas lojas (fornecedor + transbordo). " + nota_custo))

    # Nivel otimo e custo no otimo, por cenario e c_t
    lin_n, lin_c, longa = {}, {}, []
    for tau, cr in av.CUSTO_FALTA.items():
        for nome in nomes:
            for ct in ([0.0] if CENARIOS[nome] is None else CUSTOS_TRANSBORDO):
                cen = nome if CENARIOS[nome] is None else f"{nome}, {rot_ct(ct)}"
                ln, lc = {}, {}
                for th in THETAS:
                    n, c = otimo(res[(nome, th)], cr, ct)
                    ln[rot_theta(th)] = f"{n}%" + ("*" if n == av.pct(NIVEIS[-1]) else "")
                    lc[rot_theta(th)] = c
                    longa.append({"razao": rot_razao(tau), "cenario": nome, "c_t": ct,
                                  "theta": th, "nivel_otimo": n, "custo_otimo": c})
                lin_n[(rot_razao(tau), cen)] = ln
                lin_c[(rot_razao(tau), cen)] = lc
    for nome_tab, linhas, titulo, fmt in (
            ("tb_nivel_otimo", lin_n, "Nível de serviço ótimo com e sem transbordo", None),
            ("tb_custo_otimo", lin_c, "Custo no nível ótimo com e sem transbordo", 3)):
        t = pd.DataFrame(linhas).T
        t.index.names = ["Razão (nível)", "Cenário"]
        av.salvar_tabela(t, nome_tab, titulo, ORIGEM,
                         formatos={c: fmt for c in t.columns} if fmt else None,
                         nota=(f"{BASE} Varredura de 70\\% a 99\\%, passo de 1 ponto; * = no "
                               "limite da varredura. $c_t$ em unidades de $c_h$. " + nota_custo))
    ot = pd.DataFrame(longa)
    ot.to_csv(PASTA_TAB / "tb_nivel_otimo_numerico.csv", index=False, encoding="utf-8-sig")

    # c_t de equilibrio
    lin_e, longa_e = {}, []
    for ref in ("Nível do newsvendor", "Nível ótimo sem transbordo"):
        for tau, cr in av.CUSTO_FALTA.items():
            for nome in nomes[1:]:
                le = {}
                for th in THETAS:
                    sem = res[("Sem transbordo", th)]
                    nivel = av.pct(tau) if ref.startswith("Nível do") else otimo(sem, cr, 0.0)[0]
                    rs, rc = sem.loc[nivel], res[(nome, th)].loc[nivel]
                    ganho = custo(rs, cr, 0.0) - custo(rc, cr, 0.0)
                    eq = ganho / rc["Volume enviado"] if rc["Volume enviado"] > 0 else np.nan
                    le[rot_theta(th)] = eq
                    longa_e.append({"referencia": ref, "razao": rot_razao(tau), "cenario": nome,
                                    "theta": th, "nivel": nivel, "ganho_c_t0": ganho,
                                    "volume": rc["Volume enviado"], "c_t_equilibrio": eq})
                lin_e[(ref, rot_razao(tau), nome.replace("Transbordo, ", ""))] = le
    t = pd.DataFrame(lin_e).T
    t.index.names = ["Referência", "Razão (nível)", "Prazo"]
    av.salvar_tabela(
        t, "tb_ct_equilibrio", "Custo de transbordo de equilíbrio ($c_t$ acima do qual o "
        "transbordo não compensa)", ORIGEM,
        nota=(f"{BASE} $c_t$ de equilíbrio = (custo sem transbordo − custo com transbordo e "
              "$c_t$ = 0) ÷ volume enviado, por série-dia, em unidades de $c_h$, no mesmo "
              "nível de serviço. Negativo: o transbordo não compensa nem com transporte "
              "gratuito."))
    pd.DataFrame(longa_e).to_csv(PASTA_TAB / "tb_ct_equilibrio_numerico.csv", index=False,
                                 encoding="utf-8-sig")

    # Efeito no servico da ORIGEM (dia seguinte ao envio), theta base
    lin_o = {}
    for tau in av.CUSTO_FALTA:
        k = NIVEIS.index(tau)
        for nome in nomes[1:]:
            o, rs, rc = (origem[(nome, THETA_BASE)], res[("Sem transbordo", THETA_BASE)],
                         res[(nome, THETA_BASE)])
            lin_o[(rot_razao(tau), nome.replace("Transbordo, ", ""))] = {
                "Loja-dias de origem": o["dias"][k] / tots[(nome, THETA_BASE)]["dias"][k].sum(),
                "Faltas na origem, com": o["falta_com"][k] / o["dias"][k],
                "Faltas na origem, sem": o["falta_sem"][k] / o["dias"][k],
                "Atendimento na origem, com": 1 - o["falta_com"][k] / o["demanda"][k],
                "Atendimento na origem, sem": 1 - o["falta_sem"][k] / o["demanda"][k],
                "Faltas na rede, com": rc.loc[av.pct(tau), "Faltas"],
                "Faltas na rede, sem": rs.loc[av.pct(tau), "Faltas"],
                "Cobertura do alvo na origem": o["cobre_alvo"][k] / o["dias"][k],
                "Cobertura do alvo na rede": (tots[(nome, THETA_BASE)]["dias_cobre_alvo"][k].sum()
                                              / tots[(nome, THETA_BASE)]["dias"][k].sum()),
            }
    t = pd.DataFrame(lin_o).T
    t.index.names = ["Razão (nível)", "Prazo"]
    av.salvar_tabela(
        t, "tb_servico_origem", "Serviço na loja de origem no dia seguinte ao envio "
        "(θ = 0,10)", ORIGEM, largura_total=True,
        formatos={"Loja-dias de origem": "pct", "Atendimento na origem, com": "pct",
                  "Atendimento na origem, sem": "pct", "Cobertura do alvo na origem": "pct",
                  "Cobertura do alvo na rede": "pct"},
        nota=(f"{BASE} Nível do newsvendor; θ = 0,10. Loja-dias de origem: \\% dos "
              "loja-dias que vêm logo após um envio. Faltas em unidades por loja-dia; "
              "\"sem\": as mesmas loja-dias na simulação sem transbordo. Faltas na rede: "
              "média de todas as séries-dia. Cobertura do alvo: \\% dos loja-dias em que "
              "a demanda coube no alvo do dia (se a loja tivesse exatamente o alvo); na "
              "origem, só os loja-dias logo após um envio."))

    # Diferencas com IC (item como unidade), nivel do newsvendor, c_t = 0
    lin_d, longa_d = {}, []
    item_serie = np.repeat(itens, 4)
    for tau, cr in av.CUSTO_FALTA.items():
        k = NIVEIS.index(tau)
        for nome in nomes[1:]:
            for ind in ("Custo", "Perda total", "Faltas"):
                ld = {}
                for th in THETAS:
                    ts, tc = tots[("Sem transbordo", th)], tots[(nome, th)]
                    val = lambda tt: {"Custo": tt["perda_loja"][k] + tt["perda_transito"][k]
                                      + cr * tt["falta"][k],
                                      "Perda total": tt["perda_loja"][k] + tt["perda_transito"][k],
                                      "Faltas": tt["falta"][k]}[ind].ravel()
                    vs, vc = val(ts), val(tc)
                    n = ts["dias"][k].ravel()
                    v = av.diferenca_ic(vc - vs, item_serie, vs.sum() / n.sum(), n=n)
                    ld[rot_theta(th)] = av.texto_ic(v)
                    longa_d.append({"razao": rot_razao(tau), "cenario": nome, "indicador": ind,
                                    "theta": th, "dif_%": v[0], "ic_inf": v[1], "ic_sup": v[2]})
                lin_d[(rot_razao(tau), nome.replace("Transbordo, ", ""), ind)] = ld
    t = pd.DataFrame(lin_d).T
    t.index.names = ["Razão (nível)", "Prazo", "Indicador"]
    av.salvar_tabela(
        t, "tb_diferencas", "Efeito do transbordo, com intervalo de 95\\% de confiança", ORIGEM,
        nota=(f"{BASE} Nível do newsvendor; $c_t$ = 0. Diferença \\% com transbordo em "
              "relação a sem transbordo (negativo = transbordo reduz), intervalo de 95\\% "
              "entre colchetes, item como unidade (1.437 itens). Perda total = loja + trânsito."))
    pd.DataFrame(longa_d).to_csv(PASTA_TAB / "tb_diferencas_numerico.csv", index=False,
                                 encoding="utf-8-sig")

    # Fronteira desperdicio x servico: perda no mesmo atendimento
    lin_f = {}
    for th in THETAS:
        for nome in nomes:
            r = res[(nome, th)].sort_values("Atendimento")
            lin_f[(rot_theta(th), nome)] = {
                f"Atendimento {av.pct(a)}%": np.interp(a, r["Atendimento"], r["Perda % do recebido"],
                                                       left=np.nan, right=np.nan)
                for a in ATENDIMENTOS_FRONTEIRA}
    t = pd.DataFrame(lin_f).T
    t.index.names = ["θ", "Cenário"]
    av.salvar_tabela(
        t, "tb_fronteira", "Perda por validade para o mesmo atendimento (fronteira "
        "desperdício × serviço)", ORIGEM, formatos={c: "pct" for c in t.columns},
        nota=(f"{BASE} Perda (\\% das unidades recebidas do fornecedor) quando o "
              "atendimento é 90\\%, 95\\% ou 98\\%, interpolada linearmente ao longo da "
              "varredura de níveis (70\\% a 99\\%). Se, no mesmo atendimento, a perda com "
              "transbordo é menor, o transbordo desloca a fronteira."))
    return ot


# -----------------------------------------------------------------------------
# FIGURAS
# -----------------------------------------------------------------------------
def figuras(res):
    print("\n[5/5] Figuras...")
    fig, eixos = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    for ax, th in zip(eixos, THETAS):
        for nome in CENARIOS:
            r = res[(nome, th)]
            ax.plot(r["Atendimento"] * 100, r["Perda % do recebido"] * 100,
                    color=CORES[nome], linewidth=2, label=nome)
        ax.set_title(rot_theta(th), fontsize=10, loc="left")
        ax.set_xlabel("Atendimento")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        estilo_eixos(ax)
    eixos[0].set_ylabel("Perda por validade\n(% do recebido)")
    eixos[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Fronteira desperdício × serviço, com e sem transbordo (níveis de 70% a 99%)",
                 fontsize=11, x=0.01, ha="left")
    salvar_figura(fig, "fig_tb_fronteira.png")

    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.6))
    sem = res[("Sem transbordo", THETA_BASE)]
    for ax, (tau, cr) in zip(eixos, av.CUSTO_FALTA.items()):
        for nome in list(CENARIOS)[1:]:
            com = res[(nome, THETA_BASE)]
            ganho = (custo(com, cr, 0.0) / custo(sem, cr, 0.0) - 1) * 100
            ax.plot(com.index, ganho, color=CORES[nome], linewidth=2, label=nome)
        ax.axhline(0, color=COR_TEXTO_SEC, linewidth=1)
        ax.axvline(av.pct(tau), color=COR_TEXTO_SEC, linewidth=1, linestyle="--")
        ax.text(av.pct(tau), 1.0, " newsvendor", transform=ax.get_xaxis_transform(),
                fontsize=8, color=COR_TEXTO_SEC, va="top")
        ax.set_title(f"Razão 1:{cr}", fontsize=10, loc="left")
        ax.set_xlabel("Nível de serviço do alvo de estoque")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}%".replace(".", ",")))
        estilo_eixos(ax)
    eixos[0].set_ylabel("Custo com ÷ sem transbordo − 1")
    eixos[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Efeito do transbordo no custo por nível de serviço (θ = 0,10; $c_t$ = 0)",
                 fontsize=11, x=0.01, ha="left")
    salvar_figura(fig, "fig_tb_ganho_nivel.png")


# =============================================================================
# ETAPA 5B — TRANSBORDO SOB REPOSICAO RESTRITA (prazo do fornecedor L_f)
#
# Decisoes aprovadas em 26/10/2026 (justificativa na monografia):
# 1. O pedido feito ao fim do dia t chega no inicio de t+1+L_f e cobre os dias
#    t+1 a t+1+L_f (L_f = 0 e a Etapa 5). O transbordo e mais rapido que o
#    fornecedor quando L_t < L_f + 1.
# 2. Alvo de L_f+1 dias construido POR MODELO e combinado pela media (logica da
#    Etapa 3): alvo = (L_f+1) x previsao de 1 dia + quantil do erro acumulado,
#    erro_acum(d) = demanda de d a d+L_f - (L_f+1) x previsao feita para d.
#    Versao alternativa: erros so dos dias com o mesmo dia da semana de partida.
# 3. Posicao = estoque em maos (apos a perda da noite) + o que chega dentro do
#    horizonte do pedido (fornecedor e transbordo).
# 4. Excedente = min(sobra, max(sobra + pedidos do fornecedor a caminho dentro
#    do horizonte - alvo, 0)); o transbordo a caminho nao entra (sem vai-e-volta).
# 5. A carga do fornecedor chega fresca (premissa conservadora).
# 6. Janela: periodos de L_f+1 dias que comecam nos ultimos 180 dias e ja
#    terminaram (180 - L_f), sem dia fechado ou indisponivel.
# 7. Estado inicial com L_f > 0: no 1o dia da serie, loja abastecida ate o
#    alvo de 1 dia e pedidos a caminho iguais a previsao diaria.
# =============================================================================
PRAZOS_FORNECEDOR = (0, 1, 2, 3)
PRAZOS_TRANSBORDO = (1, 2)
DIA0_5B = 1600                                   # inicio da grade de dias da 5B
TOLERANCIA_COBERTURA = 0.02                      # verificacao obrigatoria
VERSOES = {"incondicional": "Incondicional", "dia_semana": "Por dia da semana de partida"}
CORES_LT = {1: "#1baf7a", 2: "#eb6834"}


def nome_cen(lt):
    return "Sem transbordo" if lt is None else f"Transbordo, L_t = {lt}"


def simular_rede_lf(S, S1, prev0, d, fech, theta, prazo_f, prazo=None, falta_base=None,
                    inicio_medicao=0):
    """Como simular_rede, com prazo de fornecedor prazo_f (chegada em t+prazo_f a
    partir do pedido feito na manha de t, equivalente ao fim de t-1). S: alvo do
    horizonte; S1: alvo de 1 dia (estado inicial); prev0: previsao diaria (itens x
    lojas x dias), usada so no estado inicial."""
    Lv, I, J, D = S.shape
    ativa = ~np.isnan(d)
    aberta = ativa & ~fech
    primeiro = ativa & ~np.concatenate([np.zeros((I, J, 1), bool), ativa[:, :, :-1]], axis=2)
    sobra = np.zeros((Lv, I, J))
    pend_f, pend_t = {}, {}                          # dia de chegada -> quantidade
    tot = {k: np.zeros((Lv, I, J)) for k in INDICADORES}
    falta_dia = np.zeros(S.shape, np.float32) if prazo is None else None
    orig = {k: np.zeros(Lv) for k in ("dias", "falta_com", "falta_sem", "demanda",
                                      "cobre_alvo")}
    enviou = np.zeros((Lv, I, J), bool)
    fator = (1 - theta) ** prazo if prazo else 1.0
    soma = lambda dic, dias: sum((dic.get(k, 0.0) for k in dias), 0.0)
    for t in range(D):
        a = aberta[:, :, t]
        perdido = theta * sobra
        chega_f, chega_t = pend_f.pop(t, 0.0), pend_t.pop(t, 0.0)
        if prazo_f > 0 and primeiro[:, :, t].any():           # estado inicial (decisao 7)
            p0 = primeiro[:, :, t]
            chega_f = chega_f + np.where(p0, S1[:, :, :, t], 0)
            for k in range(1, prazo_f + 1):
                if t + k <= D - 1:
                    pend_f[t + k] = pend_f.get(t + k, 0.0) + np.where(
                        p0, np.maximum(np.nan_to_num(prev0[:, :, t]), 0), 0)
        disponivel = sobra - perdido + chega_f + chega_t
        if prazo_f == 0:                                       # Etapa 5: chega na hora
            pedido = np.where(a, np.maximum(S[:, :, :, t] - disponivel, 0), 0)
            recebido = pedido
            estoque = disponivel + pedido
        else:                                                  # chega em t + prazo_f
            horizonte = range(t + 1, t + prazo_f + 1)
            posicao = disponivel + soma(pend_f, horizonte) + soma(pend_t, horizonte)
            pedido = np.where(a, np.maximum(S[:, :, :, t] - posicao, 0), 0)
            if t + prazo_f <= D - 1:
                pend_f[t + prazo_f] = pend_f.get(t + prazo_f, 0.0) + pedido
            recebido = chega_f
            estoque = disponivel
        dem = np.where(a, np.nan_to_num(d[:, :, t]), 0)
        vendido = np.minimum(estoque, dem)
        falta = dem - vendido
        sobra = estoque - vendido
        if t >= inicio_medicao:
            tot["perda_loja"] += perdido
            tot["fornecedor"] += recebido
            tot["chegada"] += chega_t
            tot["vendido"] += vendido
            tot["falta"] += falta
            tot["demanda"] += dem
            tot["estoque"] += np.where(a, estoque, 0)
            tot["dias"] += a
            tot["dias_sem_falta"] += a & (falta == 0)
        if prazo is None:
            falta_dia[..., t] = falta
        else:
            m = enviou & a
            orig["dias"] += m.sum(axis=(1, 2))
            orig["falta_com"] += np.where(m, falta, 0).sum(axis=(1, 2))
            orig["falta_sem"] += np.where(m, falta_base[..., t], 0).sum(axis=(1, 2))
            orig["demanda"] += np.where(m, dem, 0).sum(axis=(1, 2))
            enviou = np.zeros_like(enviou)
        if prazo and t + prazo <= D - 1:
            alvo = S[:, :, :, t + 1]
            horizonte = range(t + 1, t + 2 + prazo_f)          # dias t+1 .. t+1+L_f
            pipe_f = soma(pend_f, horizonte)
            excedente = np.where(ativa[:, :, t] & ativa[:, :, t + 1],
                                 np.minimum(sobra, np.maximum(sobra + pipe_f - alvo, 0)), 0)
            projetado = (1 - theta) * sobra + pipe_f + soma(pend_t, horizonte)
            precisa = np.where(aberta[:, :, t + 1] & (excedente == 0),
                               np.maximum(alvo - projetado, 0), 0)
            E = excedente.sum(axis=2, keepdims=True)
            F = precisa.sum(axis=2, keepdims=True)
            volume = np.minimum(E, F)
            with np.errstate(invalid="ignore", divide="ignore"):
                envio = np.where(E > 0, excedente * volume / E, 0)
                recebe = np.where(F > 0, precisa * volume / F, 0)
            sobra = sobra - envio
            pend_t[t + prazo] = pend_t.get(t + prazo, 0.0) + recebe * fator
            if t >= inicio_medicao:
                tot["enviado"] += envio
                tot["perda_transito"] += recebe * (1 - fator)
                tot["dias_envio"] += envio > 1e-12
            enviou = envio > 1e-12
    orig["sobra_final"] = sobra.sum(axis=(1, 2))
    return tot, falta_dia, orig


def conferir_conservacao(S, S0, prev0, d, fech):
    """Conservacao de unidades no motor com prazo: tudo o que entrou na loja
    (fornecedor + transbordo recebido) = vendido + perdido em loja + enviado +
    sobra final."""
    maior = 0.0
    for lf, Sx in S.items():
        for lt in (None,) + PRAZOS_TRANSBORDO:
            f0 = np.zeros(Sx.shape, np.float32)
            tot, _, o = simular_rede_lf(Sx, S0, prev0, d, fech, THETA_BASE, lf, lt, falta_base=f0)
            entrou = tot["fornecedor"].sum(axis=(1, 2)) + tot["chegada"].sum(axis=(1, 2))
            saiu = sum(tot[k].sum(axis=(1, 2)) for k in ("vendido", "perda_loja", "enviado"))
            maior = max(maior, float(np.abs(entrou - saiu - o["sobra_final"]).max()))
    assert maior < 1e-6, f"conservacao de unidades violada (dif. {maior:.2e})"
    print(f"      conservacao de unidades ok em todos os L_f e L_t (maior diferenca {maior:.1e})")


# -----------------------------------------------------------------------------
# Alvo de L_f + 1 dias
# -----------------------------------------------------------------------------
def carregar_5b(itens, lojas):
    """Previsoes pontuais salvas pelos modelos e demanda na grade de dias."""
    # Arquivos gerados pelos proprios scripts do projeto (ids gravados como objeto)
    hw = dict(np.load(PASTA / "previsoes_pontuais_holt_winters.npz", allow_pickle=True))
    xg = dict(np.load(PASTA / "previsoes_pontuais_xgboost.npz", allow_pickle=True))
    hw["ids"], xg["ids"] = hw["ids"].astype(str), xg["ids"].astype(str)
    ids = np.sort(hw["ids"])
    assert np.array_equal(ids, np.sort(xg["ids"]))
    G = av.FIM_TESTE - DIA0_5B + 1
    base = pd.read_parquet(PASTA / ARQUIVO_SAIDA,
                           columns=["id", "dia", "vendas", "disponivel", "loja_fechada"])
    base = base[base["dia"] >= DIA0_5B]
    s = np.searchsorted(ids, base["id"].to_numpy())
    y = np.full((len(ids), G), np.nan)
    ok = base["disponivel"].to_numpy() & ~base["loja_fechada"].to_numpy()
    y[s[ok], base["dia"].to_numpy()[ok] - DIA0_5B] = base["vendas"].to_numpy(float)[ok]
    ini = base[base["disponivel"]].groupby("id")["dia"].min().reindex(ids).to_numpy()
    ordem_hw = np.searchsorted(ids, hw["ids"])
    janelas = np.empty_like(hw["janelas"])
    janelas[ordem_hw] = hw["janelas"]
    usou_media = np.empty_like(hw["usou_media"])
    usou_media[ordem_hw] = hw["usou_media"]
    px = np.full((len(ids), G), np.nan)
    px[np.searchsorted(ids, xg["ids"])[:, None], xg["dias"][None, :] - DIA0_5B] = xg["prev"]
    prev_teste = {}
    for nome, arq in (("hw", "previsoes_holt_winters.parquet"),
                      ("xgb", "previsoes_xgboost.parquet"),
                      ("comb", "previsoes_combinacao.parquet")):
        df = pd.read_parquet(PASTA / arq, columns=["id", "dia", "prev"])
        p = np.full((len(ids), G), np.nan)
        p[np.searchsorted(ids, df["id"].to_numpy()), df["dia"].to_numpy() - DIA0_5B] = df["prev"]
        prev_teste[nome] = p
    # posicao de cada serie na grade item x loja da Etapa 5
    partes = pd.Series(ids).str.rsplit("_", n=3)
    loja = (partes.str[1] + "_" + partes.str[2]).to_numpy()
    gi = np.searchsorted(itens, av.item_de(ids))
    gj = np.searchsorted(lojas, loja)
    return dict(ids=ids, y=y, ini=ini, janelas=janelas, usou_media=usou_media,
                origens=hw["origens"], px=px, prev_teste=prev_teste, gi=gi, gj=gj)


def _quantis(win, colunas=None):
    """Quantis (niveis x linhas) dos erros validos de cada linha; sem erro -> 0."""
    if colunas is not None:
        win = win[..., colunas]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanquantile(win, NIVEIS, axis=-1)
    return np.nan_to_num(q, nan=0.0)


def alvos_modelo(dd, lf, versao):
    """Alvos de L_f+1 dias de HW e XGBoost (niveis x series x dias de teste)."""
    y, G = dd["y"], dd["y"].shape[1]
    n = lf + 1
    largura = 180 - lf
    colunas = (np.flatnonzero((180 - np.arange(largura)) % 7 == 0)
               if versao == "dia_semana" else None)
    # soma da demanda de d a d+L_f (NaN se algum dia do periodo nao conta)
    ysum = np.full_like(y, np.nan)
    ysum[:, :G - lf] = np.lib.stride_tricks.sliding_window_view(y, n, axis=1).sum(axis=2)
    Dt = av.FIM_TESTE - av.INICIO_TESTE + 1
    out = {m: np.zeros((len(NIVEIS), len(dd["ids"]), Dt)) for m in ("hw", "xgb")}
    g_teste = av.INICIO_TESTE - DIA0_5B
    # XGBoost: uma unica serie de previsoes fora da amostra
    ex = ysum - n * dd["px"]
    for tt in range(Dt):
        g = g_teste + tt
        q = _quantis(ex[:, g - 180: g - 180 + largura], colunas)
        out["xgb"][:, :, tt] = n * dd["prev_teste"]["xgb"][:, g] + q
    # Holt-Winters: previsoes da origem da semana do dia previsto (a ultima semana
    # tem 6 dias; a grade ganha dias vazios no fim para caber a janela de 187)
    ysum_hw = np.concatenate([ysum, np.full((ysum.shape[0], 7), np.nan)], axis=1)
    for k, o in enumerate(dd["origens"]):
        g_ini = o - 180 - DIA0_5B
        ehw = ysum_hw[:, g_ini: g_ini + 187] - n * dd["janelas"][:, k, :]
        for off in range(7):
            tt = o - av.INICIO_TESTE + off
            if tt >= Dt:
                break
            q = _quantis(ehw[:, off: off + largura], colunas)
            out["hw"][:, :, tt] = n * dd["prev_teste"]["hw"][:, g_teste + tt] + q
        # origens em que a serie usou media historica (pouco historico): historico todo
        for s in np.flatnonzero(dd["usou_media"][:, k]):
            for off in range(7):
                tt = o - av.INICIO_TESTE + off
                p = o + off
                if tt >= Dt or p < dd["ini"][s]:
                    continue
                g0, gp = dd["ini"][s] - DIA0_5B, p - DIA0_5B
                h = y[s, g0:gp]
                media = np.nanmean(h) if np.isfinite(h).any() else 0.0
                e = ysum[s, g0: max(g0, gp - lf)] - n * media
                if versao == "dia_semana":
                    e = e[(gp - np.arange(g0, max(g0, gp - lf))) % 7 == 0]
                e = e[np.isfinite(e)]
                q = np.quantile(e, NIVEIS) if len(e) else np.zeros(len(NIVEIS))
                out["hw"][:, s, tt] = n * media + q
    return out


def alvo_combinado(dd, lf, versao, forma):
    """Decisao de estoque da combinacao (media dos alvos dos modelos, negativos
    levados a zero), arredondada, na grade niveis x itens x lojas x dias."""
    m = alvos_modelo(dd, lf, versao)
    comb = (np.maximum(m["hw"], 0) + np.maximum(m["xgb"], 0)) / 2
    S = np.zeros(forma, np.float32)
    S[:, dd["gi"], dd["gj"], :] = np.nan_to_num(av.decisao(comb))   # dia sem serie -> 0
    return S, m


def conferir_alvo_lf0(dd):
    """Com L_f = 0, a construcao do alvo tem de devolver os quantis da Etapa 3."""
    m = alvos_modelo(dd, 0, "incondicional")
    maior = 0.0
    for nome, arq in (("hw", "previsoes_holt_winters.parquet"),
                      ("xgb", "previsoes_xgboost.parquet")):
        df = pd.read_parquet(PASTA / arq, columns=["id", "dia"]
                             + [f"q_emp_{av.pct(t)}" for t in NIVEIS])
        s = np.searchsorted(dd["ids"], df["id"].to_numpy())
        tt = df["dia"].to_numpy() - av.INICIO_TESTE
        for k, t in enumerate(NIVEIS):
            maior = max(maior, float(np.abs(m[nome][k, s, tt]
                                            - df[f"q_emp_{av.pct(t)}"].to_numpy()).max()))
    assert maior < 1e-3, f"alvo com L_f = 0 nao reproduz a Etapa 3 (dif. {maior:.2e})"
    print(f"      alvo com L_f = 0 reproduz os quantis da Etapa 3 (maior diferenca {maior:.1e};"
          " previsoes salvas em precisao simples)")


def coberturas_ciclo(dd, lf, m):
    """Cobertura de CICLO na linha de base: fracao das janelas de L_f+1 dias do
    teste em que a demanda acumulada nao excedeu o alvo — com o alvo continuo
    (antes de arredondar) e com a decisao arredondada. Independe da simulacao."""
    comb = (np.maximum(m["hw"], 0) + np.maximum(m["xgb"], 0)) / 2     # niveis x series x dias
    g0 = av.INICIO_TESTE - DIA0_5B
    Dt = comb.shape[2]
    ysum = np.lib.stride_tricks.sliding_window_view(dd["y"][:, g0:], lf + 1, axis=1).sum(axis=2)
    n = ysum.shape[1]                                     # janelas inteiras dentro do teste
    ok = np.isfinite(ysum) & np.isfinite(comb[0, :, :n])
    out = {}
    for k, tau in enumerate(NIVEIS):
        if tau not in av.NIVEIS:
            continue
        c = comb[k, :, :n]
        out[av.pct(tau)] = {"continuo": (ysum[ok] <= c[ok] + 1e-9).mean(),
                            "arredondado": (ysum[ok] <= av.decisao(c[ok]) + 1e-9).mean()}
    return out


def vies_aproximacao(dd):
    """Quanto (L_f+1) x previsao diaria difere da demanda acumulada observada."""
    y, p = dd["y"], dd["prev_teste"]["comb"]
    g0 = av.INICIO_TESTE - DIA0_5B
    linhas = {}
    for lf in PRAZOS_FORNECEDOR:
        n = lf + 1
        ysum = np.lib.stride_tricks.sliding_window_view(y[:, g0:], n, axis=1).sum(axis=2)
        prev = n * np.maximum(p[:, g0: g0 + ysum.shape[1]], 0)
        ok = np.isfinite(ysum) & np.isfinite(prev)
        linhas[f"L_f = {lf}"] = {"Períodos": int(ok.sum()),
                                 "Viés médio": (prev[ok].sum() - ysum[ok].sum()) / ysum[ok].sum()}
    return pd.DataFrame(linhas).T.rename_axis("Prazo do fornecedor")


# -----------------------------------------------------------------------------
# Execucao da 5B
# -----------------------------------------------------------------------------
def figuras_5b_salvas():
    """Refaz so as figuras da 5B a partir de tbr_resultados_completos.csv."""
    longa = pd.read_csv(PASTA_TAB / "tbr_resultados_completos.csv")
    cols = [c for c in longa.columns if c not in ("L_f", "L_t", "theta", "Nível")
            and not c.startswith("Custo")]
    res = {(int(lf), None if lt == 0 else int(lt), th): g.set_index("Nível")[cols]
           for (lf, lt, th), g in longa.groupby(["L_f", "L_t", "theta"])}
    res = {(lf, lt, next(t for t in THETAS if np.isclose(t, th))): r
           for (lf, lt, th), r in res.items()}
    figuras_5b(res)


def main_5b():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--so-figuras" in sys.argv:
        return figuras_5b_salvas()
    inicio = time.time()
    print("=" * 70)
    print("ETAPA 5B — TRANSBORDO SOB REPOSICAO RESTRITA")
    print("=" * 70)
    itens, lojas, d, fech, S0, datas = carregar()
    dd = carregar_5b(itens, lojas)
    conferir_alvo_lf0(dd)
    prev0 = np.full(d.shape, np.nan)
    g0 = av.INICIO_TESTE - DIA0_5B
    prev0[dd["gi"], dd["gj"], :] = dd["prev_teste"]["comb"][:, g0:]

    # Motor novo com L_f = 0 tem de reproduzir o motor da Etapa 5
    base0, f0, _ = simular_rede(S0, d, fech, THETA_BASE)
    for lt in (None, 1, 2):
        a, _, _ = simular_rede(S0, d, fech, THETA_BASE, lt, falta_base=f0)
        b, _, _ = simular_rede_lf(S0, S0, prev0, d, fech, THETA_BASE, 0, lt, falta_base=f0)
        for k in INDICADORES:
            if k != "dias_cobre_alvo":
                assert np.allclose(a[k], b[k], atol=1e-9), (lt, k)
    print("      motor com L_f = 0 reproduz a Etapa 5 (sem transbordo, L_t = 1 e L_t = 2)")

    # Viés da aproximacao
    vies = vies_aproximacao(dd)
    av.salvar_tabela(vies, "tbr_vies_aproximacao", "Viés da aproximação do horizonte: "
                     "($L_f$ + 1) × previsão de 1 dia contra a demanda acumulada", ORIGEM,
                     formatos={"Períodos": "int", "Viés médio": "pct"},
                     nota=(f"{BASE} Períodos de $L_f$ + 1 dias no teste com todos os dias "
                           "válidos. Viés: soma das previsões escalonadas menos a soma da "
                           "demanda, sobre a soma da demanda (positivo = superestima)."))

    # VERIFICACAO OBRIGATORIA: cobertura da linha de base com prazo, sem transbordo
    print("\n[5B-1] Verificacao obrigatoria: cobertura sem transbordo, theta = 0,10...")
    alvos = {(0, v): S0 for v in VERSOES}
    modelos = {(0, v): alvos_modelo(dd, 0, v) for v in VERSOES}
    cob = {}
    for lf in PRAZOS_FORNECEDOR:
        for v in VERSOES:
            if lf > 0:
                alvos[(lf, v)], modelos[(lf, v)] = alvo_combinado(dd, lf, v, S0.shape)
            tot, _, _ = simular_rede_lf(alvos[(lf, v)], S0, prev0, d, fech, THETA_BASE, lf)
            r = resumir(tot)
            cob[(lf, v)] = r
            print(f"      L_f = {lf}, {v:<14}: cobertura 80/90/95% = "
                  + " / ".join(f"{r.loc[av.pct(t), 'Cobertura']:.1%}" for t in av.NIVEIS))
    ref = cob[(0, "incondicional")]
    linhas = {}
    for lf in PRAZOS_FORNECEDOR:
        for v, rotv in VERSOES.items():
            if lf == 0 and v == "dia_semana":
                continue
            r = cob[(lf, v)]
            linhas[(f"L_f = {lf}", "Etapa 4 (prazo zero)" if lf == 0 else rotv)] = {
                **{f"Cobertura {av.pct(t)}%": r.loc[av.pct(t), "Cobertura"] for t in av.NIVEIS},
                **{f"Atendimento {av.pct(t)}%": r.loc[av.pct(t), "Atendimento"]
                   for t in av.NIVEIS}}
    tcob = pd.DataFrame(linhas).T
    tcob.index.names = ["Prazo do fornecedor", "Versão do alvo"]
    av.salvar_tabela(
        tcob, "tbr_cobertura_base", "Verificação obrigatória: cobertura da linha de base "
        "com prazo do fornecedor, sem transbordo (θ = 0,10)", ORIGEM, largura_total=True,
        formatos={c: "pct" for c in tcob.columns},
        nota=(f"{BASE} Cobertura: \\% dos loja-dias sem falta. Referência: prazo zero "
              "(Etapa 4). Versão incondicional: margem pelo quantil de todos os erros "
              "acumulados da janela; por dia da semana: só os períodos que começam no mesmo "
              "dia da semana (cerca de 25 por janela)."))
    dist = {v: np.mean([abs(cob[(lf, v)].loc[av.pct(t), "Cobertura"]
                            - ref.loc[av.pct(t), "Cobertura"])
                        for lf in PRAZOS_FORNECEDOR[1:] for t in av.NIVEIS]) for v in VERSOES}
    escolhida = min(dist, key=dist.get)
    print(f"      distancia media a prazo zero: "
          + ", ".join(f"{v} {x * 100:.2f} p.p." for v, x in dist.items())
          + f" -> escolhida: {escolhida}")

    # Decomposicao da cobertura (decisao do usuario em 26/10): ciclo com o quantil
    # continuo, ciclo com a decisao arredondada e diaria na simulacao
    linhas = {}
    for lf in PRAZOS_FORNECEDOR:
        cc = coberturas_ciclo(dd, lf, modelos[(lf, escolhida)])
        for medida, chave in (("Ciclo, quantil contínuo", "continuo"),
                              ("Ciclo, decisão arredondada", "arredondado")):
            linhas[(f"L_f = {lf}", medida)] = {f"{k}%": cc[k][chave] for k in cc}
        linhas[(f"L_f = {lf}", "Diária, simulação")] = {
            f"{av.pct(t)}%": cob[(lf, escolhida)].loc[av.pct(t), "Cobertura"] for t in av.NIVEIS}
        # Mesma simulacao sem perecibilidade: isola o efeito da perda dentro do
        # horizonte, que o alvo nao desconta
        r0 = resumir(simular_rede_lf(alvos[(lf, escolhida)], S0, prev0, d, fech, 0.0, lf)[0])
        linhas[(f"L_f = {lf}", "Diária, simulação com θ = 0")] = {
            f"{av.pct(t)}%": r0.loc[av.pct(t), "Cobertura"] for t in av.NIVEIS}
    tdec = pd.DataFrame(linhas).T
    tdec.index.names = ["Prazo do fornecedor", "Medida"]
    av.salvar_tabela(
        tdec, "tbr_cobertura_decomposicao", "Decomposição da cobertura da linha de base: "
        "ciclo (quantil contínuo e decisão arredondada) e diária (θ = 0,10)", ORIGEM,
        formatos={c: "pct" for c in tdec.columns},
        nota=(f"{BASE} Ciclo: \\% das janelas de $L_f$ + 1 dias do teste (inteiras) em que a "
              "demanda acumulada não excedeu o alvo, contínuo ou arredondado (independe da "
              "simulação). Diária: \\% dos loja-dias sem falta na simulação sem transbordo "
              "(θ = 0,10). Diária com θ = 0: a mesma simulação sem perecibilidade, que isola o "
              "efeito da perda dentro do horizonte, não descontada no alvo. O ideal da "
              "cobertura de ciclo com o quantil contínuo é o nível nominal."))
    print("\n" + tdec.to_string(float_format=lambda v: f"{v:.1%}"))

    # Portao: cobertura diaria contra o NIVEL NOMINAL (decisao do usuario em 26/10)
    pior = min(cob[(lf, escolhida)].loc[av.pct(t), "Cobertura"] - t
               for lf in PRAZOS_FORNECEDOR[1:] for t in av.NIVEIS)
    if pior < -TOLERANCIA_COBERTURA:
        print(f"\n*** VERIFICACAO FALHOU: cobertura ate {pior * 100:.1f} p.p. abaixo do "
              "nivel nominal. Comparacoes com transbordo NAO foram rodadas. ***")
        (PASTA_TAB / "tbr_resumo_execucao.txt").write_text(
            f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | VERIFICACAO FALHOU "
            f"(pior desvio {pior * 100:.1f} p.p.); versao {escolhida}\n", encoding="utf-8")
        return
    print(f"      verificacao OK contra o nivel nominal (pior desvio {pior * 100:+.1f} p.p.)")
    if "--so-verificacao" in sys.argv:
        return
    conferir_conservacao({lf: alvos[(lf, escolhida)] for lf in PRAZOS_FORNECEDOR},
                         S0, prev0, d, fech)

    # Rodadas completas com a versao escolhida
    print("\n[5B-2] Simulando L_f x L_t x theta...")
    res, tots, origem = {}, {}, {}
    for lf in PRAZOS_FORNECEDOR:
        S = alvos[(lf, escolhida)]
        for th in THETAS:
            tot0, falta0, _ = simular_rede_lf(S, S0, prev0, d, fech, th, lf)
            res[(lf, None, th)], tots[(lf, None, th)] = resumir(tot0), tot0
            for lt in PRAZOS_TRANSBORDO:
                tot, _, o = simular_rede_lf(S, S0, prev0, d, fech, th, lf, lt, falta_base=falta0)
                res[(lf, lt, th)], tots[(lf, lt, th)], origem[(lf, lt, th)] = resumir(tot), tot, o
            del falta0
        print(f"      L_f = {lf}: ok")
    # Sensibilidade: descartando os primeiros L_f + 7 dias (theta base)
    res_aq = {}
    for lf in PRAZOS_FORNECEDOR:
        S = alvos[(lf, escolhida)]
        for lt in (None,) + PRAZOS_TRANSBORDO:
            f0 = (simular_rede_lf(S, S0, prev0, d, fech, THETA_BASE, lf)[1]
                  if lt is not None else None)
            tot, _, _ = simular_rede_lf(S, S0, prev0, d, fech, THETA_BASE, lf, lt,
                                        falta_base=f0, inicio_medicao=lf + 7)
            res_aq[(lf, lt)] = resumir(tot)
    tabelas_5b(res, tots, origem, res_aq, itens, escolhida, dist)
    figuras_5b(res)
    (PASTA_TAB / "tbr_resumo_execucao.txt").write_text(
        f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: "
        f"{(time.time() - inicio) / 60:.1f} min\n"
        f"Itens: {len(itens):,} | lojas: {len(lojas)} | dias-serie: {int((~np.isnan(d)).sum()):,}\n"
        f"L_f: {PRAZOS_FORNECEDOR} | L_t: {PRAZOS_TRANSBORDO} | thetas: {THETAS} | "
        f"versao do alvo: {escolhida} (distancia media a prazo zero: "
        + ", ".join(f"{v} {x * 100:.2f} p.p." for v, x in dist.items()) + ")\n",
        encoding="utf-8")
    print("      resultados/tabelas/tbr_resumo_execucao.txt")


def equilibrio(rs, rc, cr, nivel):
    """c_t de equilibrio (unidades de c_h) no nivel dado."""
    a, b = rs.loc[nivel], rc.loc[nivel]
    ganho = custo(a, cr, 0.0) - custo(b, cr, 0.0)
    return ganho / b["Volume enviado"] if b["Volume enviado"] > 0 else np.nan


def tabelas_5b(res, tots, origem, res_aq, itens, escolhida, dist):
    print("\n[5B-3] Tabelas...")
    nota_c = ("Custo por série-dia = perda (loja + trânsito) + $c_r$ × faltas + $c_t$ × volume, "
              "$c_h$ = 1.")
    base5b = (f"{BASE} Prazo do fornecedor $L_f$: pedido do fim do dia t chega em t+1+$L_f$; "
              f"alvo de $L_f$+1 dias (versão {VERSOES[escolhida].lower()}).")
    longa = []
    for (lf, lt, th), r in res.items():
        x = r.reset_index()
        x.insert(0, "theta", th)
        x.insert(0, "L_t", lt if lt else 0)
        x.insert(0, "L_f", lf)
        longa.append(x)
    longa = pd.concat(longa, ignore_index=True)
    for tau, cr in av.CUSTO_FALTA.items():
        for ct in CUSTOS_TRANSBORDO:
            longa[f"Custo 1:{cr}, c_t = {ct:.2f}"] = custo(longa, cr, ct)
    longa.to_csv(PASTA_TAB / "tbr_resultados_completos.csv", index=False, encoding="utf-8-sig")

    # c_t de equilibrio por L_f e L_t (theta base): no newsvendor e no otimo sem transbordo
    lin, lin_conf, num = {}, {}, []
    for tau, cr in av.CUSTO_FALTA.items():
        for lf in PRAZOS_FORNECEDOR:
            linha = {}
            for th in THETAS:
                sem = res[(lf, None, th)]
                n_ot = otimo(sem, cr, 0.0)[0]
                for lt in PRAZOS_TRANSBORDO:
                    com = res[(lf, lt, th)]
                    e_nv = equilibrio(sem, com, cr, av.pct(tau))
                    e_ot = equilibrio(sem, com, cr, n_ot)
                    num.append({"razao": rot_razao(tau), "L_f": lf, "L_t": lt, "theta": th,
                                "nivel_otimo_sem": n_ot, "ct_eq_newsvendor": e_nv,
                                "ct_eq_otimo": e_ot})
                    if th == THETA_BASE:
                        linha[f"Newsvendor, L_t = {lt}"] = e_nv
                        linha[f"Ótimo, L_t = {lt}"] = e_ot
                    lin_conf.setdefault((rot_razao(tau), f"L_f = {lf}", f"L_t = {lt}"), {})[
                        rot_theta(th)] = e_ot
            lin[(rot_razao(tau), f"L_f = {lf}")] = linha
    t = pd.DataFrame(lin).T
    t.index.names = ["Razão (nível)", "Prazo do fornecedor"]
    av.salvar_tabela(t, "tbr_ct_equilibrio", "Custo de transbordo de equilíbrio por prazo do "
                     "fornecedor e do transbordo (θ = 0,10)", ORIGEM,
                     nota=(f"{base5b} $c_t$ de equilíbrio = ganho de custo com $c_t$ = 0 ÷ "
                           "volume enviado, em unidades de $c_h$, no nível do newsvendor ou no "
                           "nível ótimo sem transbordo do mesmo $L_f$. Positivo: o transbordo "
                           "compensa até esse custo unitário. " + nota_c))
    pd.DataFrame(num).to_csv(PASTA_TAB / "tbr_ct_equilibrio_numerico.csv", index=False,
                             encoding="utf-8-sig")
    t = pd.DataFrame(lin_conf).T
    t.index.names = ["Razão (nível)", "Prazo do fornecedor", "Prazo do transbordo"]
    av.salvar_tabela(t, "tbr_ct_equilibrio_theta", "Confirmação nos demais θ: $c_t$ de "
                     "equilíbrio no nível ótimo sem transbordo", ORIGEM,
                     nota=f"{base5b} Unidades de $c_h$. " + nota_c)

    # Indicadores (theta base, nivel do newsvendor)
    lin = {}
    for tau, cr in av.CUSTO_FALTA.items():
        for lf in PRAZOS_FORNECEDOR:
            for lt in (None,) + PRAZOS_TRANSBORDO:
                r = res[(lf, lt, THETA_BASE)].loc[av.pct(tau)]
                lin[(rot_razao(tau), f"L_f = {lf}", nome_cen(lt))] = {
                    "Perda em loja": r["Perda em loja"], "Perda em trânsito": r["Perda em trânsito"],
                    "Faltas": r["Faltas"], "Atendimento": r["Atendimento"],
                    "Cobertura": r["Cobertura"], "Volume enviado": r["Volume enviado"],
                    "Loja-dias com envio": r["Loja-dias com envio"],
                    "Custo (c_t = 0)": custo(r, cr, 0.0)}
    t = pd.DataFrame(lin).T
    t.index.names = ["Razão (nível)", "Prazo do fornecedor", "Cenário"]
    av.salvar_tabela(t, "tbr_indicadores", "Indicadores com e sem transbordo por prazo do "
                     "fornecedor (θ = 0,10, nível do newsvendor)", ORIGEM, largura_total=True,
                     formatos={"Atendimento": "pct", "Cobertura": "pct",
                               "Loja-dias com envio": "pct"},
                     nota=f"{base5b} Perdas, faltas e volume em unidades por série-dia. " + nota_c)

    # Nivel otimo (c_t = 0): sem transbordo e com transbordo, por L_f
    lin, num = {}, []
    for tau, cr in av.CUSTO_FALTA.items():
        for lf in PRAZOS_FORNECEDOR:
            for lt in (None,) + PRAZOS_TRANSBORDO:
                lin_th = {}
                for th in THETAS:
                    for ct in ([0.0] if lt is None else CUSTOS_TRANSBORDO):
                        n, c = otimo(res[(lf, lt, th)], cr, ct)
                        num.append({"razao": rot_razao(tau), "L_f": lf, "L_t": lt or 0,
                                    "theta": th, "c_t": ct, "nivel_otimo": n, "custo_otimo": c})
                    n, _ = otimo(res[(lf, lt, th)], cr, 0.0)
                    lin_th[rot_theta(th)] = f"{n}%" + ("*" if n == av.pct(NIVEIS[-1]) else "")
                lin[(rot_razao(tau), f"L_f = {lf}", nome_cen(lt))] = lin_th
    t = pd.DataFrame(lin).T
    t.index.names = ["Razão (nível)", "Prazo do fornecedor", "Cenário"]
    av.salvar_tabela(t, "tbr_nivel_otimo", "Nível de serviço ótimo por prazo do fornecedor, "
                     "com e sem transbordo ($c_t$ = 0)", ORIGEM,
                     nota=f"{base5b} Varredura de 70\\% a 99\\%; * = no limite. " + nota_c)
    pd.DataFrame(num).to_csv(PASTA_TAB / "tbr_nivel_otimo_numerico.csv", index=False,
                             encoding="utf-8-sig")

    # Fronteira: perda no mesmo atendimento (theta base)
    lin = {}
    for lf in PRAZOS_FORNECEDOR:
        for lt in (None,) + PRAZOS_TRANSBORDO:
            r = res[(lf, lt, THETA_BASE)].sort_values("Atendimento")
            lin[(f"L_f = {lf}", nome_cen(lt))] = {
                f"Atendimento {av.pct(a)}%": np.interp(a, r["Atendimento"],
                                                       r["Perda % do recebido"],
                                                       left=np.nan, right=np.nan)
                for a in ATENDIMENTOS_FRONTEIRA}
    t = pd.DataFrame(lin).T
    t.index.names = ["Prazo do fornecedor", "Cenário"]
    av.salvar_tabela(t, "tbr_fronteira", "Perda por validade para o mesmo atendimento, por "
                     "prazo do fornecedor (θ = 0,10)", ORIGEM,
                     formatos={c: "pct" for c in t.columns},
                     nota=(f"{base5b} Perda (\\% do recebido do fornecedor) interpolada no "
                           "atendimento indicado, ao longo da varredura de níveis."))

    # Diferencas com IC (custo, perda, faltas), theta base, newsvendor, c_t = 0
    lin, num = {}, []
    item_serie = np.repeat(itens, 4)
    for tau, cr in av.CUSTO_FALTA.items():
        k = NIVEIS.index(tau)
        for lf in PRAZOS_FORNECEDOR:
            for lt in PRAZOS_TRANSBORDO:
                ts, tc = tots[(lf, None, THETA_BASE)], tots[(lf, lt, THETA_BASE)]
                n = ts["dias"][k].ravel()
                linha = {}
                for ind in ("Custo", "Perda total", "Faltas"):
                    def val(tt):
                        perda = tt["perda_loja"][k] + tt["perda_transito"][k]
                        return {"Custo": perda + cr * tt["falta"][k], "Perda total": perda,
                                "Faltas": tt["falta"][k]}[ind].ravel()
                    vs, vc = val(ts), val(tc)
                    v = av.diferenca_ic(vc - vs, item_serie, vs.sum() / n.sum(), n=n)
                    linha[ind] = av.texto_ic(v)
                    num.append({"razao": rot_razao(tau), "L_f": lf, "L_t": lt, "indicador": ind,
                                "dif_%": v[0], "ic_inf": v[1], "ic_sup": v[2]})
                lin[(rot_razao(tau), f"L_f = {lf}", f"L_t = {lt}")] = linha
    t = pd.DataFrame(lin).T
    t.index.names = ["Razão (nível)", "Prazo do fornecedor", "Prazo do transbordo"]
    av.salvar_tabela(t, "tbr_diferencas", "Efeito do transbordo por prazo do fornecedor, "
                     "com IC de 95\\% (θ = 0,10)", ORIGEM,
                     nota=(f"{base5b} Nível do newsvendor; $c_t$ = 0. Diferença \\% com "
                           "transbordo em relação a sem transbordo no mesmo $L_f$ (negativo = "
                           "reduz), IC 95\\% com o item como unidade."))
    pd.DataFrame(num).to_csv(PASTA_TAB / "tbr_diferencas_numerico.csv", index=False,
                             encoding="utf-8-sig")

    # Servico na origem (theta base, newsvendor)
    lin = {}
    for tau in av.CUSTO_FALTA:
        k = NIVEIS.index(tau)
        for lf in PRAZOS_FORNECEDOR:
            for lt in PRAZOS_TRANSBORDO:
                o = origem[(lf, lt, THETA_BASE)]
                lin[(rot_razao(tau), f"L_f = {lf}", f"L_t = {lt}")] = {
                    "Atendimento na origem, com": 1 - o["falta_com"][k] / o["demanda"][k],
                    "Atendimento na origem, sem": 1 - o["falta_sem"][k] / o["demanda"][k],
                    "Atendimento na rede, com": res[(lf, lt, THETA_BASE)].loc[av.pct(tau), "Atendimento"],
                    "Atendimento na rede, sem": res[(lf, None, THETA_BASE)].loc[av.pct(tau), "Atendimento"]}
    t = pd.DataFrame(lin).T
    t.index.names = ["Razão (nível)", "Prazo do fornecedor", "Prazo do transbordo"]
    av.salvar_tabela(t, "tbr_servico", "Atendimento na origem (dia seguinte ao envio) e na "
                     "rede, por prazo do fornecedor (θ = 0,10)", ORIGEM,
                     formatos={c: "pct" for c in t.columns},
                     nota=f"{base5b} Nível do newsvendor; \"sem\": mesmas loja-dias sem transbordo.")

    # Sensibilidade ao estado inicial: descartando os primeiros L_f + 7 dias
    lin = {}
    for tau, cr in av.CUSTO_FALTA.items():
        for lf in PRAZOS_FORNECEDOR:
            linha = {}
            for lt in PRAZOS_TRANSBORDO:
                linha[f"Completo, L_t = {lt}"] = equilibrio(res[(lf, None, THETA_BASE)],
                                                          res[(lf, lt, THETA_BASE)], cr, av.pct(tau))
                linha[f"Sem os L_f+7 dias iniciais, L_t = {lt}"] = \
                    equilibrio(res_aq[(lf, None)], res_aq[(lf, lt)], cr, av.pct(tau))
            lin[(rot_razao(tau), f"L_f = {lf}")] = linha
    t = pd.DataFrame(lin).T
    t.index.names = ["Razão (nível)", "Prazo do fornecedor"]
    av.salvar_tabela(t, "tbr_aquecimento", "Sensibilidade ao estado inicial: $c_t$ de "
                     "equilíbrio com e sem os primeiros $L_f$ + 7 dias (θ = 0,10)", ORIGEM,
                     nota=f"{base5b} Nível do newsvendor; unidades de $c_h$.")


def figuras_5b(res):
    print("\n[5B-4] Figuras...")
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (tau, cr) in zip(eixos, av.CUSTO_FALTA.items()):
        for lt in PRAZOS_TRANSBORDO:
            ys = [equilibrio(res[(lf, None, THETA_BASE)], res[(lf, lt, THETA_BASE)], cr,
                             otimo(res[(lf, None, THETA_BASE)], cr, 0.0)[0])
                  for lf in PRAZOS_FORNECEDOR]
            ax.plot(PRAZOS_FORNECEDOR, ys, color=CORES_LT[lt], linewidth=2, marker="o",
                    markersize=6, label=f"Transbordo, $L_t$ = {lt}")
        ax.axhline(0, color=COR_TEXTO_SEC, linewidth=1)
        ax.set_xticks(PRAZOS_FORNECEDOR)
        ax.set_xlabel("Prazo do fornecedor $L_f$")
        ax.set_title(f"Razão 1:{cr}", fontsize=10, loc="left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: num_pt(v, 2)))
        estilo_eixos(ax)
    eixos[0].set_ylabel("$c_t$ de equilíbrio (unidades de $c_h$)")
    eixos[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("$c_t$ de equilíbrio por prazo do fornecedor $L_f$, em dias além da entrega "
                 "noturna (θ = 0,10; nível ótimo sem transbordo)", fontsize=10, x=0.01, ha="left")
    salvar_figura(fig, "fig_tbr_equilibrio.png")

    fig, eixos = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    for ax, lf in zip(eixos, PRAZOS_FORNECEDOR):
        for lt in (None,) + PRAZOS_TRANSBORDO:
            r = res[(lf, lt, THETA_BASE)]
            ax.plot(r["Atendimento"] * 100, r["Perda % do recebido"] * 100,
                    color=CORES["Sem transbordo"] if lt is None else CORES_LT[lt],
                    linewidth=2, label=nome_cen(lt).replace("L_t", "$L_t$"))
        ax.set_title(f"$L_f$ = {lf}", fontsize=10, loc="left")
        ax.set_xlabel("Atendimento")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        estilo_eixos(ax)
    eixos[0].set_ylabel("Perda por validade\n(% do recebido)")
    eixos[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Fronteira desperdício × serviço por prazo do fornecedor (θ = 0,10)",
                 fontsize=11, x=0.01, ha="left")
    salvar_figura(fig, "fig_tbr_fronteira.png")


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    if "--etapa-5b" in sys.argv:
        return main_5b()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inicio = time.time()
    print("=" * 70)
    print("ETAPA 5 — TRANSBORDO LATERAL PREVENTIVO")
    print("=" * 70)
    conferencia_manual()
    itens, lojas, d, fech, S, datas = carregar()
    g, w = aproximacao(S, d, fech, datas)
    res, tots, origem = rodar(S, d, fech)
    ot = tabelas(res, tots, origem, itens)
    figuras(res)
    (PASTA_TAB / "tb_resumo_execucao.txt").write_text(
        f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: "
        f"{(time.time() - inicio) / 60:.1f} min\n"
        f"Itens: {len(itens):,} | lojas: {len(lojas)} | series: {len(itens) * len(lojas):,} | "
        f"dias-serie: {int((~np.isnan(d)).sum()):,}\n"
        f"Cenarios: {', '.join(CENARIOS)} x {len(NIVEIS)} niveis x {len(THETAS)} thetas; "
        f"c_t: {CUSTOS_TRANSBORDO}\n", encoding="utf-8")
    print("      resultados/tabelas/tb_resumo_execucao.txt")

    pd.set_option("display.width", 250)
    f = lambda v: f"{v:.4f}"
    print("\nAPROXIMACAO\n" + g.to_string(float_format=f))
    print("\n" + w.to_string(float_format=f))
    for th in (THETA_BASE,):
        for nome in CENARIOS:
            print(f"\n{nome}, theta {th}\n" + res[(nome, th)].loc[[80, 90, 95, 98, 99]]
                  .to_string(float_format=f))


if __name__ == "__main__":
    main()
