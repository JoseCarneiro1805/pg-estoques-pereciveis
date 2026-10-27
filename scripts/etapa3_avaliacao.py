"""
===============================================================================
ETAPA 3 — Avaliacao comum a todos os modelos de previsao
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

POR QUE ESTE ARQUIVO EXISTE
---------------------------
Holt-Winters, SARIMA, XGBoost e a combinacao sao avaliados pelas MESMAS
funcoes, no MESMO periodo e com os MESMOS custos. Se cada modelo tivesse sua
propria conta, uma diferenca de resultado poderia vir da conta, e nao do
modelo. Este arquivo nao e executado sozinho: os scripts dos modelos o usam.

O QUE ESTA AQUI
---------------
- Periodo de teste (90 dias) e de verificacao (ultimos 28 dias)
- Niveis de servico (80%, 90%, 95%) e razoes de custo (1:4, 1:9, 1:19)
- Regra de decisao: estoque = quantil arredondado para o inteiro mais proximo,
  nunca negativo ("para cima" fica como analise de sensibilidade)
- Metricas: MAE, RMSE, perda pinball, custo, cobertura e atendimento
- Classificacao Syntetos-Boylan do padrao de demanda de cada serie
- Gravacao das tabelas em CSV e LaTeX (booktabs) para o Overleaf

CONVENCAO DE COLUNAS DAS PREVISOES
----------------------------------
vendas         demanda real do dia
prev           previsao pontual (media esperada)
q_<metodo>_80  quantil previsto de 80% (idem _90, _95), valor continuo
===============================================================================
"""

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import PASTA_TAB

# -----------------------------------------------------------------------------
# PERIODO DE TESTE — decisao aprovada em 21/10/2026 (justificativa na monografia)
# -----------------------------------------------------------------------------
INICIO_TESTE = 1824          # d_1824 = 26/01/2016 (treino vai ate d_1823)
FIM_TESTE = 1913             # d_1913 = 24/04/2016, ultimo dia da base
INICIO_VERIF_28 = 1886       # d_1886 = 28/03/2016: ultimos 28 dias (convencao M5)

# -----------------------------------------------------------------------------
# NIVEIS DE SERVICO E CUSTOS
#
# Custo de sobra c_h = 1 (unidade de referencia). Custo de falta c_r tal que
# tau = c_r / (c_r + c_h), a formula do newsvendor:
#   1:4  -> tau = 4/5   = 80%
#   1:9  -> tau = 9/10  = 90%
#   1:19 -> tau = 19/20 = 95%
# -----------------------------------------------------------------------------
NIVEIS = (0.80, 0.90, 0.95)
CUSTO_SOBRA = 1
CUSTO_FALTA = {0.80: 4, 0.90: 9, 0.95: 19}
for _tau, _cr in CUSTO_FALTA.items():          # garante a coerencia acima
    assert abs(_cr / (_cr + CUSTO_SOBRA) - _tau) < 1e-12

# Limites da classificacao Syntetos-Boylan (Syntetos, Boylan & Croston, 2005)
LIMITE_ADI = 1.32            # intervalo medio entre vendas
LIMITE_CV2 = 0.49            # variabilidade do tamanho das vendas


def pct(tau):
    """0.80 -> 80 (usado nos nomes de colunas)."""
    return int(round(tau * 100))


# -----------------------------------------------------------------------------
# CONSTRUCAO DOS QUANTIS — mesma regra para todos os modelos
#
# empirico (ADOTADO): previsao + quantil dos erros de 1 passo dos ultimos
#                     180 dias (erro = venda - previsao; dias de loja fechada
#                     nao entram)
# normal (comparacao): previsao + z * raiz da media dos erros ao quadrado
# -----------------------------------------------------------------------------
JANELA_ERROS = 180
Z = {tau: NormalDist().inv_cdf(tau) for tau in NIVEIS}     # 0,84 / 1,28 / 1,64
METODOS = {"Empírico": "emp", "Normal": "norm"}

# Os quantis empiricos sao gerados de 70% a 99% (passo de 1 ponto) para a
# varredura do nivel de servico da Etapa 4; 80, 90 e 95 estao incluidos e sao
# os usados na avaliacao. A normal so e gerada nos tres niveis de avaliacao.
NIVEIS_QUANTIS = tuple(k / 100 for k in range(70, 100))
assert set(NIVEIS) <= set(NIVEIS_QUANTIS)


def colunas_quantis():
    """Nomes das colunas de quantis, na ordem devolvida por quantis()."""
    return ([f"q_emp_{pct(t)}" for t in NIVEIS_QUANTIS]
            + [f"q_norm_{pct(t)}" for t in NIVEIS])


def quantis(prev, erros):
    """Quantis de um dia: empiricos de 70% a 99% e normais de 80, 90 e 95%,
    na ordem de colunas_quantis(). erros: os erros validos da janela. Sem
    erros, o quantil e a propria previsao."""
    if len(erros) == 0:
        erros = np.zeros(1)
    rms = np.sqrt(np.mean(erros ** 2))
    emp = list(prev + np.quantile(erros, NIVEIS_QUANTIS))
    norm = [prev + Z[tau] * rms for tau in NIVEIS]
    return emp + norm


def quantis_janela(prev, erros, alvo, janela=JANELA_ERROS):
    """Mesma regra de quantis(), de uma vez para varios dias de uma serie.
    prev, erros: um valor por dia consecutivo (erro = NaN quando o dia nao
    conta). alvo: posicoes dos dias a prever. O dia p usa os erros validos de
    p-janela ate p-1. Devolve duas matrizes (dias x niveis): empirico (niveis
    de NIVEIS_QUANTIS) e normal (niveis de NIVEIS)."""
    w = sliding_window_view(np.concatenate([np.full(janela, np.nan), erros]),
                            janela)[alvo]                  # linha p = erros[p-janela:p]
    w = np.where(np.isnan(w).all(axis=1)[:, None], 0.0, w)   # sem erros -> zero
    emp = prev[alvo, None] + np.nanquantile(w, NIVEIS_QUANTIS, axis=1).T
    rms = np.sqrt(np.nanmean(w ** 2, axis=1))
    norm = prev[alvo, None] + rms[:, None] * np.array([Z[t] for t in NIVEIS])
    return emp, norm


# -----------------------------------------------------------------------------
# REGRA DE DECISAO E PERDAS
# -----------------------------------------------------------------------------
# Regra de arredondamento do estoque — decisao de 21/10/2026 (justificativa na monografia)
#   "proximo" (ADOTADA): inteiro mais proximo, a correcao de continuidade
#             padrao ao discretizar uma variavel continua. O quantil continuo
#             ja esta calibrado; arredondar sempre para cima acrescenta em
#             media meia unidade e faz a cobertura passar do nivel pedido.
#   "cima"    (regra original): mantida como analise de sensibilidade.
# A mudanca foi identificada DEPOIS de observar o teste; isso e declarado no texto.
REGRA_ARREDONDAMENTO = "proximo"
TEXTO_REGRA = {"proximo": "arredondado para o inteiro mais próximo",
               "cima": "arredondado para cima"}


def decisao(q, regra=REGRA_ARREDONDAMENTO):
    """Quantidade a estocar: o quantil arredondado para inteiro (nao existe
    meia unidade) e nunca negativo. Meio exato (ex.: 1,5) sobe para 2. A
    pequena folga na regra "cima" evita que 2,0000000001 vire 3 por
    imprecisao numerica."""
    q = np.asarray(q, float)
    if regra == "proximo":
        estoque = np.floor(q + 0.5)
    elif regra == "cima":
        estoque = np.ceil(q - 1e-9)
    else:
        raise ValueError(f"regra de arredondamento desconhecida: {regra}")
    return np.maximum(estoque, 0)


def pinball(d, q, tau):
    """Perda pinball do quantil q no nivel tau."""
    return np.where(d >= q, tau * (d - q), (1 - tau) * (q - d))


# -----------------------------------------------------------------------------
# METRICAS DE UM METODO SOBRE UM CONJUNTO DE DIAS-SERIE
# -----------------------------------------------------------------------------
def metricas(df, metodo, regra=REGRA_ARREDONDAMENTO):
    """Calcula todas as metricas de um metodo de quantis.

    - MAE e RMSE: da previsao PONTUAL (nao dependem do nivel de servico).
    - Pinball: do quantil continuo (qualidade da previsao probabilistica).
    - Custo, sobra, falta, cobertura e atendimento: da DECISAO de estoque
      (quantil arredondado pela regra), isto e, da consequencia operacional.
      Sobra e falta estao em unidades por serie-dia.
    - Cobertura: % dos dias em que a demanda coube no estoque (sem falta).
    - Cobertura do quantil: % dos dias com demanda <= quantil continuo, antes
      de arredondar. Avalia o modelo; a cobertura acima avalia a decisao.
    - Atendimento: % da demanda total que foi atendida (fill rate).
    """
    d = df["vendas"].to_numpy(float)
    prev = np.maximum(df["prev"].to_numpy(float), 0)     # demanda nunca e negativa
    r = {
        "Dias-série": len(d),
        "Demanda média": d.mean(),
        "MAE": np.abs(d - prev).mean(),
        "RMSE": np.sqrt(((d - prev) ** 2).mean()),
    }
    for tau in NIVEIS:
        k, cr = pct(tau), CUSTO_FALTA[tau]
        q = np.maximum(df[f"q_{metodo}_{k}"].to_numpy(float), 0)
        estoque = decisao(q, regra)
        sobra = np.maximum(estoque - d, 0)
        falta = np.maximum(d - estoque, 0)
        r[f"Pinball {k}%"] = pinball(d, q, tau).mean()
        r[f"Custo 1:{cr}"] = (CUSTO_SOBRA * sobra + cr * falta).mean()
        r[f"Sobra {k}%"] = sobra.mean()
        r[f"Falta {k}%"] = falta.mean()
        r[f"Cobertura {k}%"] = (d <= estoque).mean()
        r[f"Cobertura do quantil {k}%"] = (d <= q).mean()
        r[f"Atendimento {k}%"] = np.minimum(d, estoque).sum() / d.sum()
    return pd.Series(r)


def perdas(df, metodo, regra=REGRA_ARREDONDAMENTO):
    """Perdas linha a linha (por serie-dia), para testar diferencas entre
    modelos: erro absoluto, pinball e custo em cada nivel. A media de cada
    coluna reproduz a metrica correspondente de metricas()."""
    d = df["vendas"].to_numpy(float)
    prev = np.maximum(df["prev"].to_numpy(float), 0)
    # Erro quadratico: sua media e o MSE; testar o MSE equivale a testar o RMSE
    r = {"MAE": np.abs(d - prev), "Erro quadrático": (d - prev) ** 2}
    for tau in NIVEIS:
        k, cr = pct(tau), CUSTO_FALTA[tau]
        q = np.maximum(df[f"q_{metodo}_{k}"].to_numpy(float), 0)
        estoque = decisao(q, regra)
        r[f"Pinball {k}%"] = pinball(d, q, tau)
        r[f"Custo 1:{cr}"] = (CUSTO_SOBRA * np.maximum(estoque - d, 0)
                              + cr * np.maximum(d - estoque, 0))
    return pd.DataFrame(r, index=df.index)


# -----------------------------------------------------------------------------
# DIFERENCA ENTRE DOIS MODELOS COM INTERVALO DE CONFIANCA
#
# Metodo unico, usado em TODAS as tabelas de diferencas (comp_diferencas,
# *_empirico_vs_normal, sim_diferencas, sim_diferencas_otimo):
#   1. diferenca de custo (ou erro) entre os dois modelos em cada serie-dia
#   2. media dessa diferenca dentro de cada ITEM (as 4 lojas e todos os dias
#      juntos): as lojas de um mesmo item nao sao independentes entre si
#   3. media e erro-padrao entre os 1.437 itens; IC 95% = media +- 1,96 EP
#   4. tudo dividido pelo custo medio por serie-dia do modelo de referencia
# -----------------------------------------------------------------------------
def item_de(ids):
    """Item de cada serie: FOODS_1_001_CA_1_validation -> FOODS_1_001."""
    return pd.Series(np.asarray(ids)).str.rsplit("_", n=3).str[0].to_numpy()


def diferenca_ic(dif, item, ref, n=None, z=1.96):
    """Diferenca % (modelo A - modelo B) com IC de 95%, item como unidade.
    dif: diferenca por observacao — por serie-dia, ou o total de uma serie
         (nesse caso, n = numero de series-dia de cada observacao)
    item: item de cada observacao; ref: custo medio por serie-dia de B.
    Devolve (diferenca %, limite inferior %, limite superior %)."""
    dif = np.asarray(dif, float)
    n = np.ones_like(dif) if n is None else np.asarray(n, float)
    soma = pd.DataFrame({"d": dif, "n": n, "item": item}).groupby("item")[["d", "n"]].sum()
    por_item = soma["d"] / soma["n"]                 # media por serie-dia no item
    m, ep = por_item.mean(), por_item.std() / np.sqrt(len(por_item))
    return tuple(100 * v / ref for v in (m, m - z * ep, m + z * ep))


def texto_ic(v, casas=1):
    """(m, inf, sup) -> '+1,7 [+1,2; +2,2]' (casas = numero de casas decimais)."""
    f = f"+.{casas}f"
    return f"{v[0]:{f}} [{v[1]:{f}}; {v[2]:{f}}]".replace(".", ",")


def metricas_por(df, metodos, grupo=None, regra=REGRA_ARREDONDAMENTO):
    """Tabela de metricas com (grupo, metodo) nas linhas.
    metodos: dict {rotulo exibido: sufixo das colunas}, ex. {"Empírico": "emp"}."""
    partes = {}
    grupos = [(None, df)] if grupo is None else df.groupby(grupo, sort=True)
    for chave, g in grupos:
        for rotulo, metodo in metodos.items():
            partes[rotulo if chave is None else (chave, rotulo)] = metricas(g, metodo, regra)
    return pd.DataFrame(partes).T


def colunas(prefixo):
    """Nomes das tres colunas de um tipo de metrica, na ordem dos niveis."""
    if prefixo == "Custo":
        return [f"Custo 1:{CUSTO_FALTA[t]}" for t in NIVEIS]
    return [f"{prefixo} {pct(t)}%" for t in NIVEIS]


# -----------------------------------------------------------------------------
# PADRAO DE DEMANDA — classificacao Syntetos-Boylan
# -----------------------------------------------------------------------------
def classificar_sbc(base):
    """Classe de demanda de cada serie, calculada SO com dados de treino
    (antes do teste), em dias disponiveis e de loja aberta.
    ADI = dias observados / dias com venda; CV2 = (desvio / media)^2 do
    tamanho das vendas positivas."""
    tr = base[(base["dia"] < INICIO_TESTE) & base["disponivel"] & ~base["loja_fechada"]]
    pos = tr[tr["vendas"] > 0].groupby("id")["vendas"]
    adi = tr.groupby("id").size() / pos.size()
    cv2 = ((pos.std() / pos.mean()) ** 2).fillna(0)      # 1 so venda: sem variacao
    adi, cv2 = adi.align(cv2, join="left")
    classe = pd.Series(np.select(
        [(adi < LIMITE_ADI) & (cv2 < LIMITE_CV2), adi < LIMITE_ADI, cv2 < LIMITE_CV2],
        ["Suave", "Errática", "Intermitente"], "Irregular"), index=adi.index)
    classe[adi.isna()] = "Sem venda no treino"
    return classe.reindex(base["id"].unique(), fill_value="Sem venda no treino")


# -----------------------------------------------------------------------------
# TABELAS DE RESULTADO
# -----------------------------------------------------------------------------
def tabelas_modelo(aval, prefixo, modelo, origem, regra=REGRA_ARREDONDAMENTO,
                   sufixo="", metodos=None):
    """Gera as tabelas de um modelo (arquivos <prefixo>_*) para uma regra de
    arredondamento do estoque. Regra adotada: arquivos sem sufixo. Regra antiga ("cima"): sufixo
    _sensibilidade. As previsoes e os quantis sao os mesmos nas duas."""
    metodos = metodos or METODOS
    arred = TEXTO_REGRA[regra]
    print(f"\n      Tabelas de {modelo} (estoque = quantil {arred}):")
    nome = lambda n: f"{prefixo}_{n}{sufixo}"
    titulo = lambda t: t if not sufixo else f"{t} (sensibilidade: estoque {arred})"

    ger = metricas_por(aval, metodos, regra=regra)
    custo, cob, atd = colunas("Custo"), colunas("Cobertura"), colunas("Atendimento")
    periodo = "teste de 90 dias (26/01 a 24/04/2016), 5.748 séries"
    nota_custo = ("custo por série-dia com custo de sobra = 1; razão 1:4, 1:9 e 1:19 "
                  "(sobra:falta) corresponde aos níveis de serviço de 80\\%, 90\\% e 95\\%. "
                  f"Estoque = quantil {arred}.")

    # 1. Metricas gerais
    t1 = ger[["MAE", "RMSE"] + colunas("Pinball") + custo].rename_axis("Quantis")
    salvar_tabela(
        t1, nome("metricas"), titulo(f"{modelo} — métricas de acurácia e de custo"),
        origem,
        nota=(f"{periodo}. MAE e RMSE avaliam a previsão pontual e são iguais nos "
              "dois métodos de quantis. Pinball avalia o quantil previsto; o custo "
              "avalia a decisão de estoque. " + nota_custo))

    # 2. Calibracao: quantil antes de arredondar, decisao arredondada, atendimento
    cobq = colunas("Cobertura do quantil")
    t2 = ger[cobq + cob + atd].rename_axis("Quantis")
    t2.columns = ([f"Quantil {pct(t)}%" for t in NIVEIS]
                  + [f"Decisão {pct(t)}%" for t in NIVEIS]
                  + [f"Atend. {pct(t)}%" for t in NIVEIS])
    salvar_tabela(
        t2, nome("calibracao"), titulo(f"{modelo} — calibração dos quantis"), origem,
        formatos={c: "pct" for c in t2.columns},
        nota=(f"{periodo}. Quantil: \\% dos dias com demanda até o quantil previsto, "
              "antes de arredondar (avalia o modelo). Decisão: \\% dos dias sem falta, "
              f"com o estoque formado pelo quantil {arred} (avalia a decisão). O ideal "
              "é igual ao nível de serviço. Atend.: \\% da demanda total atendida."))

    # 3. Empirico x normal no custo, com decomposicao em sobra e falta e
    #    intervalo de confianca da diferenca (mesmo metodo das outras tabelas)
    linhas, numerico = {}, []
    e, n = ger.loc["Empírico"], ger.loc["Normal"]
    pe, pn = perdas(aval, "emp", regra), perdas(aval, "norm", regra)
    item = item_de(aval["id"])
    for tau, c in zip(NIVEIS, custo):
        k = pct(tau)
        ic = diferenca_ic(pn[c].to_numpy() - pe[c].to_numpy(), item, pe[c].mean())
        rot = f"{k}% ({c.split()[1]})"
        linhas[rot] = {
            "Custo empírico": e[c], "Custo normal": n[c],
            "Diferença [IC 95%]": texto_ic(ic),
            "Sobra empírico": e[f"Sobra {k}%"], "Sobra normal": n[f"Sobra {k}%"],
            "Falta empírico": e[f"Falta {k}%"], "Falta normal": n[f"Falta {k}%"],
        }
        numerico.append({"Nível (razão)": rot, "Diferença %": ic[0],
                         "IC95% inferior": ic[1], "IC95% superior": ic[2]})
    t3 = pd.DataFrame(linhas).T.rename_axis("Nível (razão)")
    salvar_tabela(
        t3, nome("empirico_vs_normal"),
        titulo(f"{modelo} — quantis empíricos versus normais"), origem,
        nota=(f"{periodo}. Diferença: custo normal em relação ao empírico, em \\% "
              "(positivo = normal mais caro), com intervalo de 95\\% de confiança "
              "entre colchetes, item como unidade (1.437 itens). Sobra e falta em "
              "unidades por série-dia. " + nota_custo))
    pd.DataFrame(numerico).to_csv(PASTA_TAB / f"{nome('empirico_vs_normal_numerico')}.csv",
                                  index=False, encoding="utf-8-sig")

    # 3b. Mesma comparacao por padrao de demanda e por loja (onde a diferenca aparece)
    def comparar(grupo, ordem=None):
        m = metricas_por(aval, metodos, grupo=grupo, regra=regra)
        emp, nor = m.xs("Empírico", level=1), m.xs("Normal", level=1)
        t = pd.DataFrame({
            "Séries": aval.groupby(grupo)["id"].nunique(),
            **{f"Dif. {c}": (nor[c] / emp[c] - 1) * 100 for c in custo},
            "Cob. quantil 95% emp.": emp["Cobertura do quantil 95%"],
            "Cob. quantil 95% norm.": nor["Cobertura do quantil 95%"],
        })
        return (t if ordem is None else t.reindex(ordem).dropna(how="all")), m

    por_cls_t, por_cls = comparar(
        "classe", ["Suave", "Errática", "Intermitente", "Irregular", "Sem venda no treino"])
    por_loja_t, _ = comparar("store_id")
    t3b = pd.concat({"Padrão de demanda": por_cls_t, "Loja": por_loja_t}
                    ).rename_axis(["Recorte", "Grupo"])
    salvar_tabela(
        t3b, nome("empirico_vs_normal_por_grupo"),
        titulo(f"{modelo} — quantis normais versus empíricos, por padrão de demanda "
               "e por loja"), origem,
        formatos={"Séries": "int", **{c: "sinal" for c in t3b.columns if c.startswith("Dif")},
                  **{c: "pct" for c in t3b.columns if c.startswith("Cob")}},
        nota=(f"{periodo}. Classes de Syntetos-Boylan calculadas no treino. Dif.: custo "
              "normal em relação ao empírico (positivo = normal mais caro). Cob. quantil "
              "95\\%: \\% dos dias com demanda até o quantil de 95\\% (ideal: 95\\%). "
              + nota_custo))

    # 4. Verificacao nos ultimos 28 dias
    ult = metricas_por(aval[aval["dia"] >= INICIO_VERIF_28], metodos, regra=regra)
    sel = ["MAE", "RMSE"] + custo + cob
    t4 = pd.concat({"90 dias": ger[sel], "Últimos 28 dias": ult[sel]}
                   ).rename_axis(["Período", "Quantis"])
    salvar_tabela(
        t4, nome("verificacao_28d"),
        titulo(f"{modelo} — teste completo versus últimos 28 dias"), origem,
        formatos={c: "pct" for c in cob},
        nota=("90 dias: 26/01 a 24/04/2016; últimos 28 dias: 28/03 a 24/04/2016 "
              "(convenção da competição M5), subconjunto do teste completo. " + nota_custo))

    # 5. Por loja
    loja = metricas_por(aval, metodos, grupo="store_id", regra=regra)
    loja["MAE / demanda"] = loja["MAE"] / loja["Demanda média"]
    for c in custo:
        loja[f"{c} / demanda"] = loja[c] / loja["Demanda média"]
    t5 = loja[["Demanda média", "MAE", "MAE / demanda"] + custo + cob
              ].rename_axis(["Loja", "Quantis"])
    salvar_tabela(
        t5, nome("por_loja"), titulo(f"{modelo} — resultados por loja"), origem,
        formatos={"Demanda média": 2, **{c: "pct" for c in cob}},
        nota=(f"{periodo}. MAE / demanda: erro relativo ao volume da loja, para "
              "comparar lojas de tamanhos diferentes. " + nota_custo))

    # CSV completo (todas as metricas, por loja e por classe) para consulta
    pd.concat({"Geral": ger.set_axis(pd.MultiIndex.from_product([["—"], ger.index])),
               "Loja": loja.drop(columns=[c for c in loja if c.endswith("/ demanda")]),
               "Classe": por_cls}).to_csv(PASTA_TAB / f"{nome('metricas_completas')}.csv",
                                          encoding="utf-8-sig")
    print(f"      resultados/tabelas/{nome('metricas_completas')}.csv")
    return ger, ult, loja, t3, t3b


def imprimir(ger, ult, loja, t3, t3b):
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"
    print("\nGERAL\n" + ger.T.to_string(float_format=fmt))
    print("\nULTIMOS 28 DIAS\n" + ult.T.to_string(float_format=fmt))
    print("\nPOR LOJA\n" + loja.T.to_string(float_format=fmt))
    print("\nEMPIRICO x NORMAL\n" + t3.to_string(float_format=fmt))
    print("\nPOR CLASSE E POR LOJA\n" + t3b.to_string(float_format=fmt))


# -----------------------------------------------------------------------------
# GRAVACAO DAS TABELAS (CSV + LaTeX)
# -----------------------------------------------------------------------------
def _esc(texto):
    return str(texto).replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def _fmt(valor, formato):
    """Formata um numero no padrao brasileiro para o LaTeX.
    formato: inteiro = casas decimais | "pct" = fracao em % | "sinal" = % com
    sinal (valor ja em %) | "int" = inteiro com separador de milhar."""
    if isinstance(valor, str):
        return _esc(valor)
    if pd.isna(valor):
        return "--"
    if formato == "int":
        return f"{valor:,.0f}".replace(",", ".")
    if formato == "pct":
        return f"{valor * 100:.1f}".replace(".", ",") + "\\%"
    if formato == "sinal":
        return f"{valor:+.1f}".replace(".", ",") + "\\%"
    return f"{valor:.{formato}f}".replace(".", ",")


def salvar_tabela(tab, nome, legenda, origem, nota, formatos=None, casas=3,
                  largura_total=False):
    """Salva a tabela em resultados/tabelas/<nome>.csv e <nome>.tex.
    O indice (uma ou mais colunas) vira as colunas de rotulo da esquerda.
    largura_total: reduz a tabela para caber na largura do texto (tabelas
    largas; requer \\usepackage{graphicx})."""
    PASTA_TAB.mkdir(parents=True, exist_ok=True)
    tab.to_csv(PASTA_TAB / f"{nome}.csv", encoding="utf-8-sig")

    formatos = formatos or {}
    rotulos = tab.index.to_frame(index=False).astype(str)
    nomes_rot = [n or "" for n in tab.index.names]
    pacotes = "\\usepackage{booktabs}" + (" e \\usepackage{graphicx}" if largura_total else "")
    linhas = [
        f"% Gerado por scripts/{origem} — requer {pacotes}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{legenda}}}",
        f"\\label{{tab:{nome}}}",
        *(["\\resizebox{\\textwidth}{!}{%"] if largura_total else []),
        f"\\begin{{tabular}}{{{'l' * len(nomes_rot)}{'r' * tab.shape[1]}}}",
        "\\toprule",
        " & ".join(_esc(c) for c in nomes_rot + list(tab.columns)) + " \\\\",
        "\\midrule",
    ]
    anterior = None
    for rot, valores in zip(rotulos.itertuples(index=False), tab.itertuples(index=False)):
        rot = list(rot)
        primeiro = rot[0]
        if len(rot) > 1:                      # agrupa linhas com o mesmo 1o rotulo
            if primeiro == anterior:
                rot[0] = ""
            elif anterior is not None:
                linhas.append("\\addlinespace")
        anterior = primeiro
        celulas = [_esc(r) for r in rot]
        celulas += [_fmt(v, formatos.get(c, casas)) for c, v in zip(tab.columns, valores)]
        linhas.append(" & ".join(celulas) + " \\\\")
    linhas += [
        "\\bottomrule",
        "\\end{tabular}",
        *(["}"] if largura_total else []),
        "\\par\\smallskip",
        f"{{\\footnotesize Nota: {nota}}}",
        "\\end{table}",
    ]
    # Simbolos que o pdflatex so aceita em modo matematico (o sinal de menos
    # tipografico U+2212 nem existe no pdflatex sem configuracao extra)
    texto = "\n".join(linhas) + "\n"
    texto = (texto.replace("θ", "$\\theta$").replace("·", "$\\cdot$")
             .replace("Δ", "$\\Delta$").replace("−", "$-$"))
    (PASTA_TAB / f"{nome}.tex").write_text(texto, encoding="utf-8")
    print(f"      resultados/tabelas/{nome}.(csv|tex)")
