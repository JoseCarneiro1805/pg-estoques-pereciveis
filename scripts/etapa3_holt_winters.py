"""
===============================================================================
ETAPA 3 — Holt-Winters com quantis empiricos (validacao com origem movel)
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE SCRIPT FAZ
---------------------
1. Para cada uma das 5.748 series, preve a demanda de cada dia do teste
   (d_1824 a d_1913, 90 dias) usando so os dias anteriores — origem movel
2. Constroi os quantis de 80%, 90% e 95% por dois metodos:
     empirico (ADOTADO): previsao + quantil dos erros dos ultimos 180 dias
     normal (comparacao): previsao + z * raiz da media dos erros ao quadrado
3. Avalia pelo modulo comum (scripts/etapa3_avaliacao.py) e salva as tabelas

O MODELO
--------
Holt-Winters aditivo com sazonalidade semanal (m = 7), sem tendencia:
    nivel:   l(t) = alfa * (y(t) - s(t-7)) + (1 - alfa) * l(t-1)
    sazonal: s(t) = gama * (y(t) - l(t-1)) + (1 - gama) * s(t-7)
    previsao para o dia seguinte: l(t) + s(t+1-7)
Sazonalidade anual nao entra: e limitacao declarada (justificativa na monografia).

ORIGEM MOVEL — COMO O MODELO AVANCA SEM VER O FUTURO
----------------------------------------------------
- A cada 7 dias (13 origens), os parametros alfa e gama sao reestimados com
  todo o historico ate a vespera da origem.
- Dentro da semana, os parametros ficam fixos, mas o ESTADO (nivel e efeitos
  dos dias da semana) e atualizado todo dia com a venda do dia anterior.
  Assim, a previsao do dia t sempre usa as vendas ate t-1, e nada depois.
- Os erros usados nos quantis do dia t sao os de dias anteriores a t.

CASOS ESPECIAIS
---------------
- Serie com menos de 28 dias de historico na origem (itens recem-lancados):
  usa a media historica como previsao e os proprios valores observados como
  distribuicao. Fica marcada como "media_historica" e e contada no resumo.
- Dias de loja fechada: mantidos no ajuste (sao poucos Natais), mas excluidos
  da janela de erros, porque zero com a loja fechada nao e erro de demanda.

DECISAO DE ESTOQUE
------------------
Estoque = quantil arredondado para o inteiro mais proximo (regra adotada).
A regra original, arredondar para cima, e reportada como sensibilidade nos
arquivos com sufixo _sensibilidade. As previsoes sao as mesmas nas duas.

COMO USAR
---------
    python scripts/etapa3_holt_winters.py               (todas as series)
    python scripts/etapa3_holt_winters.py --amostra 40  (teste rapido, nao salva)
    python scripts/etapa3_holt_winters.py --so-tabelas  (refaz so as tabelas)

SAIDAS
------
dados/previsoes_holt_winters.parquet       previsoes e quantis por serie-dia
resultados/tabelas/hw_*.csv e .tex         tabelas para o Overleaf (regra adotada)
resultados/tabelas/hw_*_sensibilidade.*    as mesmas, arredondando para cima
resultados/tabelas/hw_resumo_execucao.txt
dados/previsoes_pontuais_holt_winters.npz  previsoes de 1 passo por origem (Etapa 5B)
===============================================================================
"""

import os
# Cada processo paralelo usa 1 nucleo; sem isto, as bibliotecas numericas
# disputariam os nucleos entre si e o conjunto ficaria mais lento.
# Precisa vir antes de importar numpy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import PASTA, PASTA_TAB, ARQUIVO_SAIDA, LOJAS
import etapa3_avaliacao as av

# -----------------------------------------------------------------------------
# CONFIGURACAO — decisoes aprovadas em 21/10/2026 (justificativa na monografia)
# -----------------------------------------------------------------------------
PASSO_REESTIMACAO = 7        # dias entre reestimacoes dos parametros
JANELA_ERROS = av.JANELA_ERROS   # 180 dias de erros para os quantis (modulo comum)
HIST_MINIMO = 28             # historico minimo para ajustar o Holt-Winters
ESPEC = dict(trend=None, seasonal="add", seasonal_periods=7)
ARQ_PREVISOES = PASTA / "previsoes_holt_winters.parquet"
ARQ_PONTUAIS = PASTA / "previsoes_pontuais_holt_winters.npz"   # usado na Etapa 5B
ORIGEM = "etapa3_holt_winters.py"
ORIGENS = list(range(av.INICIO_TESTE, av.FIM_TESTE + 1, PASSO_REESTIMACAO))


# -----------------------------------------------------------------------------
# AJUSTE DO MODELO EM UMA ORIGEM
# -----------------------------------------------------------------------------
def ajustar(treino, estendida):
    """Estima alfa e gama com o treino e roda a recursao, com esses parametros
    fixos, ate o fim da semana. Devolve as previsoes de 1 passo de todos os
    dias (cada uma usa so os dias anteriores) e os parametros; None se falhar."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ajuste = ExponentialSmoothing(
                treino, **ESPEC, initialization_method="estimated").fit()
            p = ajuste.params
            semana = ExponentialSmoothing(
                estendida, **ESPEC, initialization_method="known",
                initial_level=p["initial_level"],
                initial_seasonal=p["initial_seasons"],
            ).fit(smoothing_level=p["smoothing_level"],
                  smoothing_seasonal=p["smoothing_seasonal"], optimized=False)
        except Exception:
            return None
    prev = np.asarray(semana.fittedvalues, float)
    if not np.all(np.isfinite(prev)):
        return None
    # Conferencia: no trecho de treino, a recursao estendida tem de reproduzir
    # exatamente o ajuste original (garante que nada foi alterado no caminho)
    if not np.allclose(prev[:len(treino)], ajuste.fittedvalues, atol=1e-6):
        raise RuntimeError("Recursao estendida difere do ajuste original")
    return prev, (p["smoothing_level"], p["smoothing_seasonal"])


# -----------------------------------------------------------------------------
# PREVISAO DE UMA SERIE EM TODO O TESTE (roda em paralelo)
# -----------------------------------------------------------------------------
def prever_serie(tarefa):
    """tarefa = (id, primeiro dia disponivel, vendas, loja fechada).
    Os vetores comecam no primeiro dia disponivel e vao ate d_1913."""
    sid, ini, y, fechado = tarefa
    linhas, parametros, falhas = [], [], 0
    # Para a Etapa 5B: previsoes de 1 passo de t0-180 a t1-1 em cada origem (as
    # mesmas que geram os erros dos quantis) e se a origem usou media historica
    janelas = np.full((len(ORIGENS), JANELA_ERROS + PASSO_REESTIMACAO), np.nan, np.float32)
    usou_media = np.zeros(len(ORIGENS), bool)

    for k, t0 in enumerate(ORIGENS):
        t1 = min(t0 + PASSO_REESTIMACAO, av.FIM_TESTE + 1)    # semana: t0 .. t1-1
        dias = [t for t in range(t0, t1) if t >= ini]
        if not dias:
            continue
        n_treino = t0 - ini

        resultado = None
        if n_treino >= HIST_MINIMO:
            resultado = ajustar(y[:n_treino], y[:t1 - ini])
            falhas += resultado is None

        if resultado is not None:
            prev, par = resultado
            parametros.append(par)
            dj = np.arange(t0 - JANELA_ERROS, t1)
            ok = dj >= ini
            janelas[k, :len(dj)][ok] = prev[dj[ok] - ini]
            erros = y[:t1 - ini] - prev
            for t in dias:
                p = t - ini
                j = slice(max(0, p - JANELA_ERROS), p)          # so dias < t
                linhas.append([t, prev[p], 0, *av.quantis(prev[p], erros[j][~fechado[j]])])
        else:
            # Pouco historico: media historica; distribuicao = valores observados
            usou_media[k] = True
            for t in dias:
                p = t - ini
                h = y[:p][~fechado[:p]]
                media = h.mean() if len(h) else 0.0
                linhas.append([t, media, 1, *av.quantis(media, h - media)])

    return sid, np.array(linhas, float), parametros, falhas, janelas, usou_media


# -----------------------------------------------------------------------------
# DADOS
# -----------------------------------------------------------------------------
def carregar():
    print("\n[1/4] Lendo a base analitica...")
    base = pd.read_parquet(PASTA / ARQUIVO_SAIDA, columns=[
        "id", "store_id", "dept_id", "dia", "date", "vendas", "disponivel",
        "loja_fechada"])
    classe = av.classificar_sbc(base)
    base = base[base["disponivel"]].sort_values(["id", "dia"], ignore_index=True)

    tarefas = []
    for sid, g in base.groupby("id", sort=False):
        dia = g["dia"].to_numpy()
        # Etapa 1 verificou: dias disponiveis formam um bloco continuo ate o fim
        assert (np.diff(dia) == 1).all() and dia[-1] == av.FIM_TESTE, sid
        tarefas.append((sid, int(dia[0]), g["vendas"].to_numpy(float),
                        g["loja_fechada"].to_numpy(bool)))
    print(f"      series: {len(tarefas):,}")
    return base, classe, tarefas


# -----------------------------------------------------------------------------
# EXECUCAO EM PARALELO
# -----------------------------------------------------------------------------
def rodar(tarefas, nucleos):
    print(f"\n[2/4] Holt-Winters com origem movel ({nucleos} processos em paralelo)...")
    colunas = ["dia", "prev", "media_historica"] + av.colunas_quantis()
    partes, parametros, falhas, ids, janelas, medias = [], [], 0, [], [], []
    with ProcessPoolExecutor(max_workers=nucleos) as ex:
        for sid, linhas, par, f, jan, um in tqdm(ex.map(prever_serie, tarefas, chunksize=8),
                                                 total=len(tarefas), unit="série"):
            df = pd.DataFrame(linhas, columns=colunas)
            df.insert(0, "id", sid)
            partes.append(df)
            parametros += par
            falhas += f
            ids.append(sid)
            janelas.append(jan)
            medias.append(um)
    prev = pd.concat(partes, ignore_index=True)
    prev["dia"] = prev["dia"].astype(int)
    prev["media_historica"] = prev["media_historica"].astype(bool)
    pontuais = dict(ids=np.array(ids), origens=np.array(ORIGENS), janelas=np.stack(janelas),
                    usou_media=np.stack(medias))
    return prev, np.array(parametros), falhas, pontuais


# -----------------------------------------------------------------------------
# GERACAO DAS PREVISOES
# -----------------------------------------------------------------------------
def gerar_previsoes(args):
    """Roda o Holt-Winters e devolve as previsoes junto com a demanda real.
    No modo --amostra, imprime as metricas e devolve None (nada e salvo)."""
    inicio = time.time()
    base, classe, tarefas = carregar()
    if args.amostra:
        rng = np.random.default_rng(0)
        tarefas = [tarefas[i] for i in rng.choice(len(tarefas), args.amostra, replace=False)]
        print(f"      MODO TESTE: {len(tarefas)} series, nada sera salvo")

    prev, parametros, falhas, pontuais = rodar(tarefas, args.nucleos)
    duracao = time.time() - inicio

    print("\n[3/4] Juntando previsoes e demanda real...")
    teste = base.loc[base["dia"] >= av.INICIO_TESTE,
                     ["id", "store_id", "dept_id", "dia", "date", "vendas", "loja_fechada"]]
    prev = teste.merge(prev, on=["id", "dia"], how="inner", validate="1:1")
    prev["classe"] = prev["id"].map(classe)
    esperado = len(teste[teste["id"].isin([t[0] for t in tarefas])])
    assert len(prev) == esperado, (len(prev), esperado)

    resumo = [
        f"Execucao: {time.strftime('%d/%m/%Y %H:%M')} | duracao: {duracao / 60:.1f} min "
        f"| processos: {args.nucleos}",
        f"Series: {prev['id'].nunique():,} | dias-serie previstos: {len(prev):,} "
        f"| avaliados (loja aberta): {int((~prev['loja_fechada']).sum()):,}",
        f"Origens de reestimacao: {len(range(av.INICIO_TESTE, av.FIM_TESTE + 1, PASSO_REESTIMACAO))}"
        f" (a cada {PASSO_REESTIMACAO} dias) | janela de erros: {JANELA_ERROS} dias",
        f"Ajustes que falharam (substituidos pela media historica): {falhas}",
        f"Dias-serie com media historica (pouco historico ou falha): "
        f"{int(prev['media_historica'].sum()):,} em "
        f"{prev.loc[prev['media_historica'], 'id'].nunique()} series",
        f"Parametros estimados ({len(parametros):,} ajustes): alfa mediano "
        f"{np.median(parametros[:, 0]):.3f} (p10-p90 {np.percentile(parametros[:, 0], 10):.3f}"
        f"-{np.percentile(parametros[:, 0], 90):.3f}) | gama mediano "
        f"{np.median(parametros[:, 1]):.3f} (p10-p90 {np.percentile(parametros[:, 1], 10):.3f}"
        f"-{np.percentile(parametros[:, 1], 90):.3f})",
    ]
    print("\n" + "\n".join("      " + r for r in resumo))

    if args.amostra:
        ger = av.metricas_por(prev[~prev["loja_fechada"]], av.METODOS)
        print("\n" + ger.T.to_string(float_format=lambda v: f"{v:.4f}"))
        return None

    prev.to_parquet(ARQ_PREVISOES, index=False)
    np.savez_compressed(ARQ_PONTUAIS, **pontuais)
    print(f"      previsoes pontuais por origem (Etapa 5B): dados/{ARQ_PONTUAIS.name}")
    print(f"      previsoes salvas em: dados/{ARQ_PREVISOES.name}")
    PASTA_TAB.mkdir(parents=True, exist_ok=True)
    (PASTA_TAB / "hw_resumo_execucao.txt").write_text("\n".join(resumo) + "\n",
                                                      encoding="utf-8")
    print("      resultados/tabelas/hw_resumo_execucao.txt")
    return prev


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", type=int, default=0,
                        help="roda so N series sorteadas, sem salvar (teste rapido)")
    parser.add_argument("--so-tabelas", action="store_true",
                        help="nao reestima: refaz so as tabelas a partir das previsoes salvas")
    parser.add_argument("--nucleos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    print("=" * 70)
    print("ETAPA 3 — HOLT-WINTERS COM QUANTIS EMPIRICOS (ORIGEM MOVEL)")
    print("=" * 70)

    if args.so_tabelas:
        prev = pd.read_parquet(ARQ_PREVISOES)
        print(f"\n      previsoes lidas de: dados/{ARQ_PREVISOES.name}")
    else:
        prev = gerar_previsoes(args)
        if prev is None:
            return

    aval = prev[~prev["loja_fechada"]]
    resultados = av.tabelas_modelo(aval, "hw", "Holt-Winters", ORIGEM)   # regra adotada
    av.tabelas_modelo(aval, "hw", "Holt-Winters", ORIGEM,
                      regra="cima", sufixo="_sensibilidade")             # regra antiga
    av.imprimir(*resultados)


if __name__ == "__main__":
    main()
