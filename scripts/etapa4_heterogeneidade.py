"""
===============================================================================
ETAPA 4 — Teste de heterogeneidade da taxa de perecibilidade (theta)
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

A PERGUNTA
----------
A Etapa 4 aplica o mesmo theta as 5.748 series. Os departamentos FOODS_1,
FOODS_2 e FOODS_3 vem anonimizados no M5, entao nao se sabe qual e mais
perecivel. Este script verifica se a simplificacao do theta uniforme governa
o resultado, repetindo a varredura da Etapa 4 (sem transbordo, prazo zero)
com theta diferente por departamento:

  H1: FOODS_1 = 0,05 | FOODS_2 = 0,10 | FOODS_3 = 0,15
  H2: FOODS_1 = 0,15 | FOODS_2 = 0,10 | FOODS_3 = 0,05
  H3: FOODS_1 = 0,20 | FOODS_2 = 0,10 | FOODS_3 = 0,05

O QUE E MEDIDO (combinacao, salvo no item 4)
--------------------------------------------
1. Nivel otimo de cada departamento x previsto pela formula
   tau* = c_r / (c_r + theta * c_h) com o theta do proprio departamento.
2. Nivel otimo UNICO para a rede e perda em relacao a diferenciar o nivel
   por departamento (cada um no seu otimo).
3. Nivel unico x formula com o theta medio ponderado pela demanda do teste.
4. Ordem dos tres modelos (HW, XGBoost, combinacao) em cada cenario, no nivel
   do newsvendor e no nivel otimo unico de cada modelo, com IC de 95%
   (item como unidade, mesma funcao das demais tabelas).

Mesmas regras da Etapa 4 (scripts/etapa4_simulacao.py); so muda o theta, que
passa a ser um valor por serie.

PRE-REQUISITO: etapa4_simulacao.py rodado (usa as mesmas previsoes)
SAIDAS: resultados/tabelas/het_*.csv e .tex, het_resumo_execucao.txt
===============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import PASTA_TAB, num_pt
import etapa3_avaliacao as av
import etapa4_simulacao as sim

ORIGEM = "etapa4_heterogeneidade.py"
DEPARTAMENTOS = ("FOODS_1", "FOODS_2", "FOODS_3")
CENARIOS = {"H1": (0.05, 0.10, 0.15), "H2": (0.15, 0.10, 0.05), "H3": (0.20, 0.10, 0.05)}
MODELOS = list(sim.MODELOS)
COMBINACAO = "Combinação"
BASE = ("simulação de 90 dias (26/01 a 24/04/2016), 5.748 séries, sem transbordo, "
        "reposição diária até o quantil empírico arredondado para o inteiro mais próximo; "
        "θ diferente por departamento. Custo por série-dia = perda por validade + $c_r$ × "
        "faltas, $c_h$ = 1.")


def rot_nivel(n):
    return f"{n}%" + ("*" if n == av.pct(sim.NIVEIS[-1]) else "")


def custo_por_nivel(tot, cr, mascara):
    """Custo por serie-dia em cada nivel, somando so as series da mascara."""
    perda = tot["perdido"][:, mascara].sum(axis=1)
    falta = tot["falta"][:, mascara].sum(axis=1)
    dias = tot["dias"][:, mascara].sum(axis=1)
    return (perda + cr * falta) / dias, perda + cr * falta, dias


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inicio = time.time()
    print("=" * 70)
    print("ETAPA 4 — HETEROGENEIDADE DE THETA POR DEPARTAMENTO")
    print("=" * 70)
    ids, item, d, fechado, alvos = sim.carregar()
    dep = pd.Series(ids).str[:7].to_numpy()
    assert set(dep) == set(DEPARTAMENTOS)
    volume = {g: np.nansum(d[dep == g]) for g in DEPARTAMENTOS}
    peso = {g: v / sum(volume.values()) for g, v in volume.items()}
    niveis = np.array([av.pct(t) for t in sim.NIVEIS])

    # Conferencia: theta uniforme por serie (0,10) reproduz a Etapa 4
    e4 = pd.read_csv(PASTA_TAB / "sim_resultados_completos.csv")
    maior = 0.0
    for m in MODELOS:
        r = sim.resumir(*sim.simular(alvos[m], d, fechado, np.full(len(ids), 0.10)))
        ref = e4[(e4["Modelo"] == m) & np.isclose(e4["theta"], 0.10)].set_index("Nível")
        for cr in av.CUSTO_FALTA.values():
            maior = max(maior, float(np.abs(r[f"Custo 1:{cr}"] - ref[f"Custo 1:{cr}"]).max()))
    assert maior < 1e-9, f"theta por serie nao reproduz a Etapa 4 (dif. {maior:.2e})"
    print(f"      theta uniforme por serie reproduz a Etapa 4 (maior diferenca {maior:.1e})")

    print("\n[2/3] Simulando 3 cenarios x 3 modelos...")
    tots = {}
    for cen, thetas in CENARIOS.items():
        theta_serie = np.select([dep == g for g in DEPARTAMENTOS], thetas)
        for m in MODELOS:
            tots[(cen, m)], _ = sim.simular(alvos[m], d, fechado, theta_serie)
        print(f"      {cen}: ok")

    print("\n[3/3] Tabelas...")
    # Cenarios: theta por departamento, peso de volume e theta medio ponderado
    t = pd.DataFrame({cen: {**{f"θ {g}": th for g, th in zip(DEPARTAMENTOS, ths)},
                            "θ médio ponderado": sum(peso[g] * th
                                                     for g, th in zip(DEPARTAMENTOS, ths))}
                      for cen, ths in CENARIOS.items()}).T.rename_axis("Cenário")
    pesos = pd.DataFrame({"Séries": {g: int((dep == g).sum()) for g in DEPARTAMENTOS},
                          "Participação na demanda": peso}).rename_axis("Departamento")
    av.salvar_tabela(t, "het_cenarios", "Cenários de θ heterogêneo por departamento", ORIGEM,
                     nota=(f"{BASE} θ médio ponderado pela participação de cada "
                           "departamento na demanda do período de teste."))
    av.salvar_tabela(pesos, "het_pesos", "Departamentos: séries e participação na demanda",
                     ORIGEM, formatos={"Séries": "int", "Participação na demanda": "pct"},
                     nota=f"{BASE} Demanda total do período de teste (dias disponíveis e de loja aberta).")

    # 1 a 3: combinacao
    lin_dep, lin_uni, num = {}, {}, []
    for cen, ths in CENARIOS.items():
        tot = tots[(cen, COMBINACAO)]
        theta_med = sum(peso[g] * th for g, th in zip(DEPARTAMENTOS, ths))
        for tau, cr in av.CUSTO_FALTA.items():
            soma_dif, dias_tot = 0.0, 0.0
            for g, th in zip(DEPARTAMENTOS, ths):
                c, soma, dias = custo_por_nivel(tot, cr, dep == g)
                k = int(np.argmin(c))
                soma_dif += soma[k]
                dias_tot += dias[k]
                prev = 100 * cr / (cr + th)
                lin_dep[(cen, sim.rotulo_razao(tau), g)] = {
                    "θ": th, "Previsto": num_pt(prev, 1) + "%", "Encontrado": rot_nivel(niveis[k])}
                num.append({"cenario": cen, "razao": sim.rotulo_razao(tau), "departamento": g,
                            "theta": th, "previsto": prev, "encontrado": int(niveis[k]),
                            "custo_otimo": c[k]})
            c_rede, _, _ = custo_por_nivel(tot, cr, np.ones(len(ids), bool))
            k = int(np.argmin(c_rede))
            dif = soma_dif / dias_tot
            prev = 100 * cr / (cr + theta_med)
            lin_uni[(cen, sim.rotulo_razao(tau))] = {
                "θ médio": theta_med, "Previsto com θ médio": num_pt(prev, 1) + "%",
                "Ótimo único": rot_nivel(niveis[k]), "Custo, nível único": c_rede[k],
                "Custo, níveis por departamento": dif,
                "Perda do nível único": (c_rede[k] / dif - 1) * 100}
    t = pd.DataFrame(lin_dep).T
    t.index.names = ["Cenário", "Razão (nível)", "Departamento"]
    av.salvar_tabela(t, "het_nivel_departamento", "Nível ótimo por departamento: previsto "
                     "pela fórmula com o θ do departamento × encontrado", ORIGEM,
                     formatos={"θ": 2},
                     nota=(f"{BASE} Combinação. Previsto: $c_r / (c_r + \\theta c_h)$ com o θ do "
                           "departamento. Encontrado: nível que minimiza o custo do "
                           "departamento na varredura de 70\\% a 99\\%; * = no limite."))
    pd.DataFrame(num).to_csv(PASTA_TAB / "het_nivel_departamento_numerico.csv", index=False,
                             encoding="utf-8-sig")
    t = pd.DataFrame(lin_uni).T
    t.index.names = ["Cenário", "Razão (nível)"]
    av.salvar_tabela(t, "het_nivel_unico", "Nível ótimo único para a rede × níveis "
                     "diferenciados por departamento", ORIGEM, largura_total=True,
                     formatos={"θ médio": 3, "Perda do nível único": "sinal"},
                     nota=(f"{BASE} Combinação. Ótimo único: um só nível para todas as séries. "
                           "Níveis por departamento: cada departamento no seu ótimo. Perda: "
                           "custo do nível único em relação ao diferenciado. Previsto com θ "
                           "médio: fórmula com o θ ponderado pela demanda."))

    # 4: ordem dos modelos (newsvendor e otimo unico de cada modelo), com IC
    lin_o, lin_d, num_d = {}, {}, []
    todos = np.ones(len(ids), bool)
    pares = [("XGBoost", "Holt-Winters"), ("Combinação", "Holt-Winters"),
             ("Combinação", "XGBoost")]
    for cen in CENARIOS:
        for tau, cr in av.CUSTO_FALTA.items():
            cust, nivel = {}, {}
            for m in MODELOS:
                c, _, _ = custo_por_nivel(tots[(cen, m)], cr, todos)
                kn, ko = sim.NIVEIS.index(tau), int(np.argmin(c))
                cust[m] = (c[kn], c[ko])
                nivel[m] = {"newsvendor": kn, "otimo": ko}
            rn = pd.Series({m: v[0] for m, v in cust.items()}).rank(method="min").astype(int)
            ro = pd.Series({m: v[1] for m, v in cust.items()}).rank(method="min").astype(int)
            for m in MODELOS:
                lin_o[(cen, sim.rotulo_razao(tau), m)] = {
                    "Custo, newsvendor": cust[m][0], "Ordem, newsvendor": rn[m],
                    "Nível ótimo": rot_nivel(niveis[nivel[m]["otimo"]]),
                    "Custo, ótimo": cust[m][1], "Ordem, ótimo": ro[m]}
            for a, b in pares:
                linha = {}
                for ref in ("newsvendor", "otimo"):
                    ta, tb = tots[(cen, a)], tots[(cen, b)]
                    ka, kb = nivel[a][ref], nivel[b][ref]
                    ca = ta["perdido"][ka] + cr * ta["falta"][ka]
                    cb = tb["perdido"][kb] + cr * tb["falta"][kb]
                    v = av.diferenca_ic(ca - cb, item, cb.sum() / tb["dias"][kb].sum(),
                                        n=tb["dias"][kb])
                    linha["Newsvendor" if ref == "newsvendor" else "Nível ótimo"] = av.texto_ic(v)
                    num_d.append({"cenario": cen, "razao": sim.rotulo_razao(tau),
                                  "comparacao": f"{a} vs {b}", "referencia": ref,
                                  "dif_%": v[0], "ic_inf": v[1], "ic_sup": v[2]})
                lin_d[(cen, sim.rotulo_razao(tau), f"{a} vs {b}")] = linha
    t = pd.DataFrame(lin_o).T
    t.index.names = ["Cenário", "Razão (nível)", "Modelo"]
    av.salvar_tabela(t, "het_ordem_modelos", "Ordem dos modelos de previsão com θ "
                     "heterogêneo", ORIGEM,
                     formatos={"Ordem, newsvendor": "int", "Ordem, ótimo": "int"},
                     nota=(f"{BASE} Ordem: 1 = menor custo, pela estimativa pontual. Ótimo: "
                           "nível único de cada modelo na varredura de 70\\% a 99\\%."))
    t = pd.DataFrame(lin_d).T
    t.index.names = ["Cenário", "Razão (nível)", "Comparação"]
    av.salvar_tabela(t, "het_diferencas", "Diferenças de custo entre modelos com θ "
                     "heterogêneo, com IC de 95\\%", ORIGEM,
                     nota=(f"{BASE} Diferença \\% do primeiro modelo em relação ao segundo "
                           "(negativo = primeiro mais barato), intervalo de 95\\% entre "
                           "colchetes, item como unidade. Nível ótimo: cada modelo no seu."))
    pd.DataFrame(num_d).to_csv(PASTA_TAB / "het_diferencas_numerico.csv", index=False,
                               encoding="utf-8-sig")

    (PASTA_TAB / "het_resumo_execucao.txt").write_text(
        f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: "
        f"{(time.time() - inicio) / 60:.1f} min\n"
        f"Series: {len(ids):,} | cenarios: {', '.join(CENARIOS)} | modelos: {', '.join(MODELOS)}\n",
        encoding="utf-8")
    print("      resultados/tabelas/het_resumo_execucao.txt")
    pd.set_option("display.width", 220)
    for nome in ("het_cenarios", "het_pesos", "het_nivel_departamento", "het_nivel_unico",
                 "het_ordem_modelos", "het_diferencas"):
        print(f"\n{nome.upper()}\n" + pd.read_csv(PASTA_TAB / f"{nome}.csv").to_string(index=False))


if __name__ == "__main__":
    main()
