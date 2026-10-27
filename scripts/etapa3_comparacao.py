"""
===============================================================================
ETAPA 3 — Comparacao dos modelos e combinacao simples
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE SCRIPT FAZ
---------------------
1. Junta as previsoes do Holt-Winters e do XGBoost (mesmas series e dias)
2. Monta a COMBINACAO SIMPLES: media dos quantis dos dois modelos, nivel a
   nivel (e media das previsoes pontuais). Cada modelo tem seus valores
   negativos levados a zero antes da media, porque demanda nao e negativa.
3. Avalia a combinacao com as mesmas tabelas dos modelos individuais
4. Compara os tres lado a lado e verifica se a ORDEM dos modelos muda entre
   o criterio de acuracia (MAE, RMSE) e o de custo — ponto central do TCC
5. Mede se as diferencas sao reais: intervalo de 95% de confianca da
   diferenca de cada criterio, usando o ITEM como unidade (as 4 lojas de um
   item nao sao independentes)

Todos os quantis aqui sao os EMPIRICOS (metodo adotado). Estoque = quantil
arredondado para o inteiro mais proximo.

PRE-REQUISITO: rodar antes etapa3_holt_winters.py e etapa3_xgboost.py
SAIDAS: dados/previsoes_combinacao.parquet
        resultados/tabelas/comb_*.csv/.tex  (tabelas da combinacao)
        resultados/tabelas/comp_*.csv/.tex  (comparacao entre modelos), inclusive
          comp_composicao_quantil / comp_composicao_diferenca (previsao x margem)
          e comp_ordenacao (posicao dos modelos com empates estatisticos)
        resultados/tabelas/comp_resumo_execucao.txt
===============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import PASTA, PASTA_TAB
import etapa3_avaliacao as av

ORIGEM = "etapa3_comparacao.py"
MODELOS = {"Holt-Winters": "previsoes_holt_winters.parquet",
           "XGBoost": "previsoes_xgboost.parquet"}
COMBINACAO = "Combinação"
CHAVES = ["id", "store_id", "dept_id", "dia", "date", "vendas", "loja_fechada", "classe"]
COLS_Q = ["prev"] + av.colunas_quantis()   # quantis de 70% a 99% e normais
SO_EMPIRICO = {"Empírico": "emp"}


# -----------------------------------------------------------------------------
# PREVISOES DOS MODELOS E COMBINACAO
# -----------------------------------------------------------------------------
def carregar():
    print("\n[1/4] Lendo as previsoes dos modelos...")
    prev = {}
    for nome, arq in MODELOS.items():
        df = pd.read_parquet(PASTA / arq, columns=CHAVES + COLS_Q)
        prev[nome] = df.sort_values(["id", "dia"], ignore_index=True)
        print(f"      {nome}: {len(df):,} dias-série")
    a, b = prev.values()
    assert a[["id", "dia"]].equals(b[["id", "dia"]]), "modelos com series-dias diferentes"

    comb = a[CHAVES].copy()
    for c in COLS_Q:
        comb[c] = np.mean([np.maximum(df[c].to_numpy(float), 0) for df in prev.values()],
                          axis=0)
    comb.to_parquet(PASTA / "previsoes_combinacao.parquet", index=False)
    print("      combinacao salva em: dados/previsoes_combinacao.parquet")
    prev[COMBINACAO] = comb
    return {n: df[~df["loja_fechada"]] for n, df in prev.items()}


# -----------------------------------------------------------------------------
# TABELAS COMPARATIVAS
# -----------------------------------------------------------------------------
def por_modelo(aval, grupo=None, filtro=None, ordem=None):
    """Metricas (quantis empiricos) com o modelo nas linhas; com grupo, as
    linhas sao (grupo, modelo), na ordem dada (ou alfabetica)."""
    partes = {}
    for nome, df in aval.items():
        df = df if filtro is None else df[filtro(df)]
        m = av.metricas_por(df, SO_EMPIRICO, grupo=grupo)
        partes[nome] = m.droplevel(-1) if grupo is not None else m.iloc[0]
    if grupo is None:
        return pd.DataFrame(partes).T.rename_axis("Modelo")
    t = pd.concat(partes).swaplevel()                     # (grupo, modelo)
    grupos = ordem or sorted(t.index.get_level_values(0).unique())
    idx = pd.MultiIndex.from_product([grupos, list(aval)])
    return t.reindex(idx).dropna(how="all")


def intervalo(aval, a, b, colunas):
    """Diferenca % de a em relacao a b em cada criterio, com IC de 95% (item
    como unidade). Negativo = a melhor que b (menor erro ou custo)."""
    pa, pb = av.perdas(aval[a], "emp"), av.perdas(aval[b], "emp")
    item = av.item_de(aval[a]["id"])
    return {c: av.diferenca_ic(pa[c].to_numpy() - pb[c].to_numpy(), item, pb[c].mean())
            for c in colunas}


def tabelas(aval):
    print("\n[3/4] Tabelas comparativas...")
    custo, pin = av.colunas("Custo"), av.colunas("Pinball")
    acur = ["MAE", "RMSE"]
    periodo = ("teste de 90 dias (26/01 a 24/04/2016), 5.748 séries, quantis "
               "empíricos, estoque = quantil arredondado para o inteiro mais próximo")
    nota_comb = ("Combinação: média simples dos quantis (e das previsões pontuais) "
                 "do Holt-Winters e do XGBoost.")

    # B. Lado a lado
    ger = por_modelo(aval)
    t1 = ger[acur + pin + custo]
    av.salvar_tabela(
        t1, "comp_metricas", "Comparação dos modelos — acurácia e custo", ORIGEM,
        nota=(f"{periodo}. {nota_comb} MAE e RMSE: previsão pontual; pinball: quantil; "
              "custo por série-dia com custo de sobra = 1 nas razões 1:4, 1:9 e 1:19 "
              "(níveis de 80\\%, 90\\% e 95\\%). Menor é melhor."))

    # Ordem dos modelos em cada criterio (1 = melhor)
    ordem = t1.rank(method="min").astype(int)
    av.salvar_tabela(
        ordem, "comp_ordem", "Ordem dos modelos em cada critério (1 = melhor)", ORIGEM,
        formatos={c: "int" for c in ordem.columns},
        nota=(f"{periodo}. Posição de cada modelo em cada critério, a partir da "
              "tabela de métricas. Se a ordem nas colunas de acurácia (MAE, RMSE) "
              "difere da ordem nas colunas de custo, o critério muda a escolha do modelo."))

    # Calibracao
    cobq, cob = av.colunas("Cobertura do quantil"), av.colunas("Cobertura")
    t2 = ger[cobq + cob]
    t2.columns = ([f"Quantil {av.pct(t)}%" for t in av.NIVEIS]
                  + [f"Decisão {av.pct(t)}%" for t in av.NIVEIS])
    av.salvar_tabela(
        t2, "comp_calibracao", "Comparação dos modelos — calibração", ORIGEM,
        formatos={c: "pct" for c in t2.columns},
        nota=(f"{periodo}. Quantil: \\% dos dias com demanda até o quantil previsto "
              "(avalia o modelo). Decisão: \\% dos dias sem falta (avalia o estoque). "
              "O ideal é igual ao nível de serviço."))

    # Diferencas com intervalo de confianca
    crit = ["MAE"] + pin + custo
    pares = [("XGBoost", "Holt-Winters"), (COMBINACAO, "Holt-Winters"),
             (COMBINACAO, "XGBoost")]
    longa, texto = [], {}
    for a, b in pares:
        lin = intervalo(aval, a, b, crit)
        rot = f"{a} vs {b}"
        # Uma casa, como as demais tabelas de diferenca; os valores exatos ficam
        # em comp_diferencas_numerico.csv
        texto[rot] = {c: av.texto_ic(v) for c, v in lin.items()}
        longa += [{"Comparação": rot, "Critério": c, "Diferença %": m,
                   "IC95% inferior": lo, "IC95% superior": hi}
                  for c, (m, lo, hi) in lin.items()]
    t3 = pd.DataFrame(texto).T.rename_axis("Comparação")
    av.salvar_tabela(
        t3, "comp_diferencas", "Diferenças entre modelos, com intervalo de 95\\% de confiança",
        ORIGEM,
        nota=(f"{periodo}. Diferença \\% do primeiro modelo em relação ao segundo "
              "(negativo = primeiro melhor) e intervalo de 95\\% de confiança entre "
              "colchetes, com o item como unidade (1.437 itens, cada um nas 4 lojas). "
              "Intervalo que não contém zero indica diferença distinguível do acaso."))
    pd.DataFrame(longa).to_csv(PASTA_TAB / "comp_diferencas_numerico.csv",
                               index=False, encoding="utf-8-sig")

    # Por loja e por classe de demanda
    sel = ["MAE"] + custo
    ordem_cls = ["Suave", "Errática", "Intermitente", "Irregular", "Sem venda no treino"]
    for grupo, nome, rot, ordem_g in (
            ("store_id", "comp_por_loja", "Loja", None),
            ("classe", "comp_por_classe", "Padrão de demanda", ordem_cls)):
        t = por_modelo(aval, grupo=grupo, ordem=ordem_g)[sel].rename_axis([rot, "Modelo"])
        av.salvar_tabela(
            t, nome, f"Comparação dos modelos — por {rot.lower()}", ORIGEM,
            nota=f"{periodo}. {nota_comb}")

    # Ultimos 28 dias
    ult = por_modelo(aval, filtro=lambda df: df["dia"] >= av.INICIO_VERIF_28)
    t4 = pd.concat({"90 dias": ger[acur + custo], "Últimos 28 dias": ult[acur + custo]},
                   names=["Período"])
    av.salvar_tabela(
        t4, "comp_verificacao_28d", "Comparação dos modelos — 90 dias versus últimos 28",
        ORIGEM, nota=f"{periodo}. Últimos 28 dias: 28/03 a 24/04/2016.")
    return t1, ordem, t2, t3, ult


# -----------------------------------------------------------------------------
# EXECUCAO
def registrar_tempo(inicio, aval):
    """Registra duracao e tamanho da execucao, como nos scripts dos modelos."""
    df = next(iter(aval.values()))
    texto = (f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: "
             f"{(time.time() - inicio) / 60:.1f} min\n"
             f"Modelos: {', '.join(aval)} | series: {df['id'].nunique():,} | "
             f"dias-serie avaliados por modelo: {len(df):,}\n")
    (PASTA_TAB / "comp_resumo_execucao.txt").write_text(texto, encoding="utf-8")
    print("      resultados/tabelas/comp_resumo_execucao.txt")


# -----------------------------------------------------------------------------
# COMPOSICAO DO QUANTIL: previsao pontual + margem de incerteza
# -----------------------------------------------------------------------------
def composicao(aval):
    """Quantil = previsao pontual + margem (quantil dos erros de cada modelo).
    Participacoes calculadas como razao das medias (media da previsao / media
    do quantil), e nao como media das razoes: em series intermitentes o
    quantil de um dia fica perto de zero e a razao diaria explode. Usa o
    quantil continuo, antes de arredondar, para que previsao + margem = quantil."""
    linhas, dif = {}, {}
    h, x = aval["Holt-Winters"], aval["XGBoost"]
    for tau in av.NIVEIS:
        k = av.pct(tau)
        col = f"q_emp_{k}"
        for nome, df in aval.items():
            p, q = df["prev"].to_numpy(float), df[col].to_numpy(float)
            linhas[(f"{k}%", nome)] = {
                "Previsão média": p.mean(), "Margem média": (q - p).mean(),
                "Quantil médio": q.mean(), "Previsão / quantil": p.mean() / q.mean(),
                "Margem / quantil": (q - p).mean() / q.mean()}
        dp = x["prev"].to_numpy(float) - h["prev"].to_numpy(float)
        dq = x[col].to_numpy(float) - h[col].to_numpy(float)
        dm = dq - dp
        margem_hw = (h[col] - h["prev"]).mean()
        dif[f"{k}%"] = {
            "Δ previsão": dp.mean(), "Δ margem": dm.mean(), "Δ quantil": dq.mean(),
            "Δ margem / margem HW": dm.mean() / margem_hw,
            "|Δ previsão|": np.abs(dp).mean(), "|Δ margem|": np.abs(dm).mean(),
            "|Δ quantil|": np.abs(dq).mean()}
    t = pd.DataFrame(linhas).T
    t.index.names = ["Nível", "Modelo"]
    nota = ("teste de 90 dias (26/01 a 24/04/2016), 5.748 séries, quantis empíricos. "
            "Quantil = previsão pontual + margem (quantil dos erros dos últimos 180 dias "
            "de cada modelo), antes de arredondar, em unidades por série-dia. "
            "Participações = média da parcela / média do quantil.")
    av.salvar_tabela(t, "comp_composicao_quantil", "Composição do quantil: previsão "
                     "pontual e margem de incerteza", ORIGEM,
                     formatos={"Previsão / quantil": "pct", "Margem / quantil": "pct"},
                     nota=nota)
    t2 = pd.DataFrame(dif).T.rename_axis("Nível")
    av.salvar_tabela(t2, "comp_composicao_diferenca", "De onde vem a diferença entre os "
                     "quantis do XGBoost e do Holt-Winters", ORIGEM,
                     formatos={"Δ margem / margem HW": "pct"},
                     nota=(nota + " Δ: XGBoost − Holt-Winters; média das diferenças "
                           "diárias (Δ) e média das diferenças em valor absoluto (|Δ|)."))
    return t, t2


# -----------------------------------------------------------------------------
# ORDENACAO DOS MODELOS COM EMPATES ESTATISTICOS
# -----------------------------------------------------------------------------
def ordenacao(aval):
    """Posicao de cada modelo em cada criterio = 1 + numero de modelos
    significativamente melhores que ele (IC 95% da diferenca exclui zero,
    metodo de av.diferenca_ic). Modelos na mesma posicao = empate."""
    criterios = {"MAE": "MAE", "RMSE": "Erro quadrático",
                 **{c: c for c in av.colunas("Pinball")},
                 **{c: c for c in av.colunas("Custo")}}
    nomes = list(aval)
    perd = {n: av.perdas(df, "emp") for n, df in aval.items()}
    item = av.item_de(aval[nomes[0]]["id"])
    texto, longa = {n: {} for n in nomes}, []
    for rot, col in criterios.items():
        piores = {n: 0 for n in nomes}           # n. de modelos melhores que n
        for i, a in enumerate(nomes):
            for b in nomes[i + 1:]:
                v = av.diferenca_ic(perd[a][col].to_numpy() - perd[b][col].to_numpy(),
                                    item, perd[b][col].mean())
                if v[2] < 0:
                    piores[b] += 1               # a melhor que b
                elif v[1] > 0:
                    piores[a] += 1               # b melhor que a
                longa.append({"Critério": rot, "Comparação": f"{a} vs {b}",
                              "Diferença %": v[0], "IC95% inferior": v[1],
                              "IC95% superior": v[2]})
        pos = {n: 1 + piores[n] for n in nomes}
        for n in nomes:
            empate = sum(pos[m] == pos[n] for m in nomes) > 1
            texto[n][rot] = f"{pos[n]}º" + (" (empate)" if empate else "")
    t = pd.DataFrame(texto).T.rename_axis("Modelo")
    av.salvar_tabela(
        t, "comp_ordenacao", "Posição dos modelos em cada critério, com empates "
        "estatísticos", ORIGEM,
        nota=("teste de 90 dias (26/01 a 24/04/2016), 5.748 séries, quantis empíricos, "
              "estoque = quantil arredondado para o inteiro mais próximo. Posição = 1 + "
              "número de modelos significativamente melhores (intervalo de 95\\% de "
              "confiança da diferença, item como unidade, sem conter zero). Modelos "
              "na mesma posição estão empatados. RMSE: teste feito sobre o erro "
              "quadrático médio, que ordena os modelos como o RMSE."))
    pd.DataFrame(longa).to_csv(PASTA_TAB / "comp_ordenacao_numerico.csv", index=False,
                               encoding="utf-8-sig")
    return t


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inicio = time.time()
    print("=" * 70)
    print("ETAPA 3 — COMPARACAO DOS MODELOS E COMBINACAO")
    print("=" * 70)
    aval = carregar()

    print("\n[2/4] Tabelas da combinacao (mesmo formato dos modelos)...")
    av.tabelas_modelo(aval[COMBINACAO], "comb", "Combinação HW + XGBoost", ORIGEM)
    av.tabelas_modelo(aval[COMBINACAO], "comb", "Combinação HW + XGBoost", ORIGEM,
                      regra="cima", sufixo="_sensibilidade")

    t1, ordem, t2, t3, ult = tabelas(aval)
    comp, comp_dif = composicao(aval)
    ordenado = ordenacao(aval)
    registrar_tempo(inicio, aval)
    pd.set_option("display.width", 200)
    print("\nCOMPOSICAO DO QUANTIL\n" + comp.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n" + comp_dif.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nORDENACAO\n" + ordenado.to_string())
    print("\n[4/4] Resultados")
    print("\nMETRICAS\n" + t1.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nORDEM\n" + ordem.to_string())
    print("\nCALIBRACAO\n" + t2.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nDIFERENCAS (IC 95%)\n" + t3.to_string())
    print("\nULTIMOS 28 DIAS\n" + ult[["MAE", "RMSE"] + av.colunas("Custo")].to_string(
        float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
