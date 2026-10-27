"""
===============================================================================
ETAPA 1 — Verificacao empirica do criterio de disponibilidade
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

PREMISSA TESTADA
----------------
A ausencia de registro em sell_prices.csv indica que o produto NAO estava
disponivel na loja naquele periodo. Se a premissa vale, os zeros de venda
nesses dias nao sao demanda nula, e sim produto fora da prateleira.

AS QUATRO VERIFICACOES
----------------------
1. Existem vendas positivas em dias sem preco? (se sim, a premissa falha)
2. Os dias sem preco formam um bloco continuo no inicio de cada serie?
3. Como se distribui a data do primeiro registro de preco?
4. Quantos dias separam o primeiro preco da primeira venda positiva?

OBSERVACAO SOBRE A GRANULARIDADE
--------------------------------
O preco no M5 e SEMANAL (chave wm_yr_wk; cada semana comeca no sabado).
Portanto a disponibilidade e medida em blocos de 7 dias: um item lancado
numa quarta-feira aparece como "com preco" desde o sabado anterior.

SAIDAS
------
- resultados/tabelas/verificacao_disponibilidade.txt
- resultados/figuras/disponibilidade_series_tempo.png (300 dpi)
===============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # gera arquivos sem abrir janela
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Reaproveita a leitura, o recorte e a juntada da Etapa 1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_preparar_dados import (
    PASTA, carregar_arquivos, aplicar_recorte, formato_longo, juntar_covariaveis,
)

RAIZ = Path(__file__).resolve().parent.parent
ARQ_TXT = RAIZ / "resultados" / "tabelas" / "verificacao_disponibilidade.txt"
ARQ_FIG = RAIZ / "resultados" / "figuras" / "disponibilidade_series_tempo.png"

# Limite de tolerancia combinado: acima disto o criterio e invalidado
LIMITE_VENDA_SEM_PRECO = 0.005

relatorio = []          # linhas do relatorio (vao para a tela e para o .txt)


def escreve(texto=""):
    print(texto)
    relatorio.append(texto)


# -----------------------------------------------------------------------------
# Monta matrizes serie x dia (5.748 x 1.913) de vendas e de disponibilidade
# -----------------------------------------------------------------------------
def montar_matrizes(base: pd.DataFrame):
    base = base[["id", "d", "date", "vendas", "sell_price"]].copy()
    base["dia"] = base["d"].str[2:].astype(int)            # d_1 -> 1
    base = base.sort_values(["id", "dia"])

    ids = base["id"].unique()
    datas = pd.to_datetime(
        base.drop_duplicates("dia").sort_values("dia")["date"].values
    )
    n_series, n_dias = len(ids), len(datas)

    vendas = base["vendas"].to_numpy().reshape(n_series, n_dias)
    com_preco = base["sell_price"].notna().to_numpy().reshape(n_series, n_dias)
    return base, ids, datas, vendas, com_preco


# -----------------------------------------------------------------------------
# Verificacao 1 — vendas positivas sem registro de preco
# -----------------------------------------------------------------------------
def verificacao_1(base):
    escreve("\n" + "-" * 70)
    escreve("VERIFICACAO 1 — Vendas positivas em dias SEM registro de preco")
    escreve("-" * 70)

    total = len(base)
    sem_preco = base["sell_price"].isna()
    positivas = base["vendas"] > 0
    conflito = sem_preco & positivas
    n = int(conflito.sum())

    escreve(f"Linhas totais (serie x dia) ........: {total:>12,}")
    escreve(f"Linhas sem registro de preco .......: {int(sem_preco.sum()):>12,}"
            f"  ({sem_preco.mean():.2%})")
    escreve(f"Linhas com venda > 0 ...............: {int(positivas.sum()):>12,}")
    escreve(f"Venda > 0 E sem preco ..............: {n:>12,}"
            f"  ({n / total:.4%} das linhas)")
    escreve(f"Unidades vendidas em dias sem preco : "
            f"{int(base.loc[conflito, 'vendas'].sum()):>12,}")

    if n > 0:
        escreve("\nExemplos:")
        escreve(base.loc[conflito, ["id", "date", "vendas"]].head(10).to_string())

    status = "APROVADA" if n / total <= LIMITE_VENDA_SEM_PRECO else "REPROVADA"
    escreve(f"\nResultado: {status} (limite de tolerancia: "
            f"{LIMITE_VENDA_SEM_PRECO:.1%} das linhas)")
    return n / total


# -----------------------------------------------------------------------------
# Verificacao 2 — padrao dos dias sem preco dentro de cada serie
# -----------------------------------------------------------------------------
def verificacao_2(com_preco):
    escreve("\n" + "-" * 70)
    escreve("VERIFICACAO 2 — Os dias sem preco formam bloco continuo no inicio?")
    escreve("-" * 70)

    n_series, n_dias = com_preco.shape
    n_disp = com_preco.sum(axis=1)
    nunca = n_disp == 0

    # primeiro e ultimo dia com preco (indices 0..n_dias-1)
    primeiro = np.argmax(com_preco, axis=1)
    ultimo = n_dias - 1 - np.argmax(com_preco[:, ::-1], axis=1)

    completa = n_disp == n_dias
    lacuna_meio = (~nunca) & (n_disp < ultimo - primeiro + 1)
    lacuna_fim = (~nunca) & (ultimo < n_dias - 1)
    so_inicio = (~nunca) & (primeiro > 0) & ~lacuna_meio & ~lacuna_fim
    outro = ~(completa | so_inicio | nunca)

    # numero de interrupcoes (blocos sem preco depois do primeiro preco)
    trans = np.diff(com_preco.astype(np.int8), axis=1)
    n_interrupcoes = (trans == -1).sum(axis=1)

    def linha(rotulo, mascara):
        k = int(mascara.sum())
        escreve(f"  {rotulo:<48}: {k:>6,}  ({k / n_series:6.2%})")

    escreve(f"Total de series: {n_series:,}\n")
    escreve("Classificacao (mutuamente exclusiva):")
    linha("Preco em todos os dias (sem lacuna)", completa)
    linha("Sem preco APENAS num bloco continuo no inicio", so_inicio)
    linha("Padrao diferente (ver detalhe abaixo)", outro)
    linha("Nunca tem preco", nunca)

    escreve("\nDetalhe do 'padrao diferente' (pode haver sobreposicao):")
    linha("Preco interrompido no meio da serie", lacuna_meio)
    linha("Preco some no final da serie (descontinuado?)", lacuna_fim)

    if lacuna_meio.any():
        dias_lacuna = (ultimo - primeiro + 1 - n_disp)[lacuna_meio]
        escreve("\nSeries com interrupcao no meio:")
        escreve(f"  numero de interrupcoes por serie: "
                f"min {n_interrupcoes[lacuna_meio].min()}, "
                f"mediana {np.median(n_interrupcoes[lacuna_meio]):.0f}, "
                f"max {n_interrupcoes[lacuna_meio].max()}")
        escreve(f"  dias sem preco no meio, por serie: "
                f"min {dias_lacuna.min()}, mediana {np.median(dias_lacuna):.0f}, "
                f"max {dias_lacuna.max()}")
        escreve(f"  total de dias-serie em lacunas internas: {dias_lacuna.sum():,}")

    return primeiro, nunca, lacuna_meio


# -----------------------------------------------------------------------------
# Verificacao 3 — distribuicao da data do primeiro registro de preco
# -----------------------------------------------------------------------------
def verificacao_3(primeiro, nunca, datas):
    escreve("\n" + "-" * 70)
    escreve("VERIFICACAO 3 — Data do primeiro registro de preco")
    escreve("-" * 70)

    validos = ~nunca
    n = int(validos.sum())
    datas_ini = pd.Series(datas[primeiro[validos]])

    no_dia1 = int((primeiro[validos] == 0).sum())
    escreve(f"Primeiro dia da base: {datas[0].date()}")
    escreve(f"Series com preco desde o primeiro dia: {no_dia1:,} ({no_dia1 / n:.2%})")
    escreve(f"Series que entram depois ..............: {n - no_dia1:,} "
            f"({(n - no_dia1) / n:.2%})")

    escreve("\nSeries que entram depois, por ano do primeiro preco:")
    depois = datas_ini[primeiro[validos] > 0]
    tabela = depois.dt.year.value_counts().sort_index()
    acumulado = no_dia1
    escreve(f"  {'ano':<6}{'series':>8}{'% do total':>12}{'disponiveis ao fim do ano':>28}")
    for ano, k in tabela.items():
        acumulado += k
        escreve(f"  {ano:<6}{k:>8,}{k / n:>12.2%}{acumulado:>28,}")


# -----------------------------------------------------------------------------
# Verificacao 4 — primeiro preco x primeira venda positiva
# -----------------------------------------------------------------------------
def verificacao_4(vendas, primeiro, nunca):
    escreve("\n" + "-" * 70)
    escreve("VERIFICACAO 4 — Dias entre o primeiro preco e a primeira venda > 0")
    escreve("-" * 70)

    positiva = vendas > 0
    nunca_vendeu = ~positiva.any(axis=1)
    primeira_venda = np.argmax(positiva, axis=1)

    ok = ~nunca & ~nunca_vendeu
    diferenca = primeira_venda[ok] - primeiro[ok]
    escreve(f"Series que nunca registraram venda: {int(nunca_vendeu.sum()):,}")
    escreve(f"Series analisadas ................: {int(ok.sum()):,}")
    escreve(f"Diferenca negativa (venda antes do preco): {int((diferenca < 0).sum()):,}")

    def resumo(rotulo, dif):
        escreve(f"\n{rotulo} (n = {len(dif):,})")
        escreve(f"  diferenca zero ....: {int((dif == 0).sum()):>6,} ({(dif == 0).mean():.2%})")
        escreve(f"  ate 6 dias ........: {int((dif <= 6).sum()):>6,} ({(dif <= 6).mean():.2%})"
                "   <- dentro da mesma semana de preco")
        escreve(f"  mediana ...........: {np.median(dif):>6.0f} dias")
        escreve(f"  media .............: {dif.mean():>6.1f} dias")
        q = np.percentile(dif, [75, 90, 99])
        escreve(f"  percentis 75/90/99 : {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} dias")
        escreve(f"  maximo ............: {dif.max():>6,} dias")

    resumo("Todas as series", diferenca)

    # Para quem ja tinha preco no dia 1, o "primeiro preco" e so o inicio da
    # base (dado censurado a esquerda). O teste mais limpo e com os lancamentos.
    lancadas = primeiro[ok] > 0
    resumo("Apenas series lancadas DEPOIS do inicio da base", diferenca[lancadas])
    resumo("Apenas series com preco desde o dia 1", diferenca[~lancadas])


# -----------------------------------------------------------------------------
# Figura — numero de series disponiveis ao longo do tempo
# -----------------------------------------------------------------------------
def figura_disponibilidade(com_preco, datas):
    n_series = com_preco.shape[0]
    disponiveis = com_preco.sum(axis=0)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(datas, disponiveis, color="#2a78d6", linewidth=2)
    ax.axhline(n_series, color="#52514e", linewidth=1, linestyle="--")
    ax.text(datas[len(datas) // 2], n_series, f"total do recorte: {n_series:,}".replace(",", "."),
            va="bottom", ha="center", fontsize=9, color="#52514e")

    ax.set_ylim(0, n_series * 1.08)
    ax.set_ylabel("Séries produto-loja disponíveis")
    ax.set_xlabel("Data")
    ax.set_title("Séries com registro de preço (produto disponível) ao longo do tempo\n"
                 "FOODS — lojas da Califórnia", fontsize=11, loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    )
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    fig.tight_layout()

    ARQ_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ARQ_FIG, dpi=300)
    plt.close(fig)
    escreve(f"\nFigura salva em: {ARQ_FIG.relative_to(RAIZ)}")
    escreve(f"  disponiveis no primeiro dia: {disponiveis[0]:,} | "
            f"no ultimo dia: {disponiveis[-1]:,}")


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    vendas_largo, calendario, precos = carregar_arquivos(PASTA)
    vendas_largo = aplicar_recorte(vendas_largo)
    base = juntar_covariaveis(formato_longo(vendas_largo), calendario, precos)

    base, ids, datas, vendas, com_preco = montar_matrizes(base)

    escreve("=" * 70)
    escreve("VERIFICACAO EMPIRICA — AUSENCIA DE PRECO = PRODUTO INDISPONIVEL")
    escreve("Recorte: FOODS, lojas CA_1 a CA_4, 1.913 dias (d_1 a d_1913)")
    escreve("=" * 70)

    taxa = verificacao_1(base)
    primeiro, nunca, _ = verificacao_2(com_preco)
    verificacao_3(primeiro, nunca, datas)
    verificacao_4(vendas, primeiro, nunca)
    figura_disponibilidade(com_preco, datas)

    ARQ_TXT.parent.mkdir(parents=True, exist_ok=True)
    ARQ_TXT.write_text("\n".join(relatorio) + "\n", encoding="utf-8")
    print(f"Relatorio salvo em: {ARQ_TXT.relative_to(RAIZ)}")

    if taxa > LIMITE_VENDA_SEM_PRECO:
        print("\nATENCAO: verificacao 1 reprovada — nao aplicar o ajuste.")


if __name__ == "__main__":
    main()
