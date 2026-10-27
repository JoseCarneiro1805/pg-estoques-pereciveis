"""
===============================================================================
ETAPA 1 + 2 — Preparacao da base analitica
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE SCRIPT FAZ
---------------------
1. Le os tres arquivos do M5 (vendas, calendario, precos)
2. Filtra a categoria FOODS e as quatro lojas da California
3. Transforma o formato "largo" (uma coluna por dia) em "longo" (uma linha por
   produto-loja-dia), que e o formato que os modelos de previsao precisam
4. Junta as informacoes de calendario (data, evento, SNAP) e de preco, e marca
   se o produto estava DISPONIVEL naquele dia naquela loja
5. Imprime as estatisticas descritivas do Capitulo 4 (bruta x ajustada)
6. Gera as figuras descritivas e a tabela LaTeX, e salva a base pronta

COLUNA "disponivel" — LEIA ANTES DE MODELAR
-------------------------------------------
disponivel = True  quando existe registro de preco (sell_prices.csv) para o
                   item, na loja, naquela semana.
disponivel = False quando nao existe. Nesses dias a venda e sempre zero, mas
                   esse zero NAO e demanda nula: o produto nao estava a venda.

As linhas indisponiveis sao MANTIDAS na base (para rastreabilidade), mas
DEVEM SER EXCLUIDAS na modelagem, na simulacao de estoque e no transbordo:

    base = base[base["disponivel"]]

Fundamentacao: scripts/etapa1_verificar_disponibilidade.py
(0 vendas em dias sem preco; os dias sem preco formam sempre um bloco continuo
no inicio da serie). Ressalva: pela documentacao do M5, o preco so e
registrado a partir da semana da primeira venda, entao o criterio coincide,
por construcao, com "a partir da primeira venda" arredondado para a semana.

OUTRAS COLUNAS DE CONTROLE (nenhuma linha e removida)
-----------------------------------------------------
pos_reforma_ca2 (0/1): 1 na loja CA_2 a partir de 26/05/2015 (reforma/
                   ampliacao; quebra estrutural). 0 nas demais lojas sempre.
                   Entra como covariavel nos modelos.
loja_fechada (bool): True em todo 25/12 (Natal) e em qualquer loja-dia em que
                   todas as series venderam zero. O modelo aprende o dia; a
                   simulacao de estoque NAO repoe para dia de loja fechada.

COMO USAR
---------
- Os tres CSVs do M5 devem estar na pasta dados/
- Rode. Ao final, sera criado o arquivo dados/base_analitica.parquet

ONDE ISTO ENTRA NO TCC
----------------------
Os numeros impressos ao final alimentam a secao 4.1 (Caracterizacao
Exploratoria dos Dados).
===============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # gera arquivos sem abrir janela
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# -----------------------------------------------------------------------------
# CONFIGURACAO — ajuste apenas esta parte se precisar
# -----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "dados"                     # pasta com os CSVs
PASTA_FIG = RAIZ / "resultados" / "figuras"
PASTA_TAB = RAIZ / "resultados" / "tabelas"
CATEGORIA = "FOODS"                # categoria do recorte
LOJAS = ["CA_1", "CA_2", "CA_3", "CA_4"]   # as quatro lojas da California
ARQUIVO_SAIDA = "base_analitica.parquet"
LOJA_REFORMA = "CA_2"                      # loja com quebra estrutural
DATA_REFORMA = pd.Timestamp("2015-05-26")  # d_1579, inicio do periodo pos-reforma

# Dias da semana em portugues, na ordem de segunda a domingo
DIAS_PT = {
    "Monday": "Segunda", "Tuesday": "Terça", "Wednesday": "Quarta",
    "Thursday": "Quinta", "Friday": "Sexta", "Saturday": "Sábado",
    "Sunday": "Domingo",
}

# Cores das figuras (paleta de referencia, validada para daltonismo)
COR_PRINCIPAL = "#2a78d6"      # azul — serie principal
COR_DESTAQUE = "#eb6834"       # laranja — destaque (ex.: massa em zero)
COR_TEXTO_SEC = "#52514e"      # cinza — textos e linhas de referencia
COR_GRADE = "#e5e4df"          # cinza claro — grade


# -----------------------------------------------------------------------------
# PASSO 1 — Ler os tres arquivos
# -----------------------------------------------------------------------------
def carregar_arquivos(pasta: Path):
    print("\n[1/6] Lendo os arquivos do M5...")

    vendas = pd.read_csv(pasta / "sales_train_validation.csv")
    calendario = pd.read_csv(pasta / "calendar.csv")
    precos = pd.read_csv(pasta / "sell_prices.csv")

    print(f"      vendas .......: {vendas.shape[0]:>7,} series")
    print(f"      calendario ...: {calendario.shape[0]:>7,} dias")
    print(f"      precos .......: {precos.shape[0]:>7,} registros")
    return vendas, calendario, precos


# -----------------------------------------------------------------------------
# PASSO 2 — Aplicar o recorte do projeto (FOODS + California)
# -----------------------------------------------------------------------------
def aplicar_recorte(vendas: pd.DataFrame):
    print("\n[2/6] Aplicando o recorte do projeto...")

    antes = len(vendas)
    vendas = vendas[
        (vendas["cat_id"] == CATEGORIA) & (vendas["store_id"].isin(LOJAS))
    ].copy()

    print(f"      antes ........: {antes:>7,} series (base inteira)")
    print(f"      depois .......: {len(vendas):>7,} series ({CATEGORIA} / California)")
    return vendas


# -----------------------------------------------------------------------------
# PASSO 3 — Converter de formato largo para longo
#
# O M5 vem com uma coluna por dia (d_1, d_2, ... d_1913). Precisamos de uma
# linha por produto-loja-dia. E isso que o pandas chama de "melt".
# -----------------------------------------------------------------------------
def formato_longo(vendas: pd.DataFrame):
    print("\n[3/6] Convertendo para formato longo...")

    colunas_id = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    colunas_dia = [c for c in vendas.columns if c.startswith("d_")]

    longo = vendas.melt(
        id_vars=colunas_id,
        value_vars=colunas_dia,
        var_name="d",
        value_name="vendas",
    )

    print(f"      dias por serie: {len(colunas_dia):>7,}")
    print(f"      linhas geradas: {len(longo):>7,}")
    return longo


# -----------------------------------------------------------------------------
# PASSO 4 — Juntar calendario e precos; marcar disponibilidade
# -----------------------------------------------------------------------------
def juntar_covariaveis(longo, calendario, precos):
    print("\n[4/6] Juntando calendario e precos...")

    # --- calendario: traz data, evento e a flag SNAP ---
    cal = calendario[[
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "snap_CA",
    ]].copy()

    base = longo.merge(cal, on="d", how="left")
    base["date"] = pd.to_datetime(base["date"])
    base["dia"] = base["d"].str[2:].astype(int)     # d_1 -> 1 (indice numerico)

    # Renomeia a coluna SNAP: como so usamos California, fica so "snap"
    base = base.rename(columns={"snap_CA": "snap"})

    # Cria uma variavel simples indicando se houve algum evento no dia
    base["tem_evento"] = base["event_name_1"].notna().astype(int)

    # --- precos: a chave e loja + item + semana ---
    base = base.merge(
        precos, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )

    # --- disponibilidade: existe registro de preco para item-loja-semana ---
    # Linhas com disponivel=False ficam na base, mas devem ser EXCLUIDAS na
    # modelagem (ver cabecalho do script).
    base["disponivel"] = base["sell_price"].notna()

    print(f"      linhas disponiveis ....: {base['disponivel'].mean():.1%}")
    print(f"      linhas indisponiveis ..: {(~base['disponivel']).mean():.1%}"
          "  (mantidas; excluir na modelagem)")

    return base


# -----------------------------------------------------------------------------
# PASSO 4b — Marcar eventos operacionais (reforma da CA_2 e loja fechada)
# -----------------------------------------------------------------------------
def marcar_eventos_operacionais(base):
    print("\n[4b] Marcando eventos operacionais...")

    # --- Reforma da CA_2 ---
    # Quebra estrutural em 26/05/2015 (d_1579): ~220 itens estreiam, vendas da
    # loja sobem ~80% e itens antigos +34% a/a (scripts/etapa1_investigacoes.py).
    # Tratamento: variavel binaria, SEM cortar dados. Demais lojas = 0 sempre.
    base["pos_reforma_ca2"] = (
        (base["store_id"] == LOJA_REFORMA) & (base["date"] >= DATA_REFORMA)
    ).astype(int)
    n = base.loc[base["pos_reforma_ca2"] == 1, "date"].nunique()
    print(f"      pos_reforma_ca2 = 1 ...: {LOJA_REFORMA} a partir de "
          f"{DATA_REFORMA.date()} ({n} dias)")

    # --- Loja fechada ---
    # (a) todo 25/12, por regra: as lojas Walmart fecham no Natal;
    # (b) qualquer loja-dia em que TODAS as series disponiveis venderam zero.
    # As linhas NAO sao removidas: o modelo aprende o dia, e a simulacao de
    # estoque nao deve repor para um dia de loja fechada.
    disp = base[base["disponivel"]]
    total_loja_dia = disp.groupby(["store_id", "date"])["vendas"].sum()
    zerados = total_loja_dia[total_loja_dia == 0].index

    natal = (base["date"].dt.month == 12) & (base["date"].dt.day == 25)
    tudo_zero = pd.MultiIndex.from_frame(base[["store_id", "date"]]).isin(zerados)
    base["loja_fechada"] = natal | tudo_zero

    # Contabilidade dos loja-dias marcados como fechados
    e_natal = [d.month == 12 and d.day == 25 for _, d in total_loja_dia.index]
    natal_loja_dia = total_loja_dia[e_natal]
    resid = natal_loja_dia[natal_loja_dia > 0]
    fora_natal = [(l, d) for l, d in zerados if not (d.month == 12 and d.day == 25)]
    fechados = base.loc[base["loja_fechada"], ["store_id", "date"]].drop_duplicates()

    print(f"      loja-dias marcados como fechados: {len(fechados)}")
    print(f"        Natais (5 datas x {len(LOJAS)} lojas) ....: {len(natal_loja_dia)}")
    print(f"          com venda total zero ..........: {int((natal_loja_dia == 0).sum())}")
    print(f"          com venda residual (1-7 un.) ..: {len(resid)}"
          "  (marcados por regra do Natal)")
    print(f"        fora do Natal com venda total zero: {len(fora_natal)}")
    for l, d in fora_natal:
        print(f"          {l} {d.date()}")
    if len(resid):
        print("      vendas residuais em 25/12: "
              + ", ".join(f"{l} {d.year}: {v:.0f} un." for (l, d), v in resid.items()))
    print(f"      linhas marcadas (serie x dia): {int(base['loja_fechada'].sum()):,} | "
          f"destas, disponiveis: {int((base['loja_fechada'] & base['disponivel']).sum()):,}")
    return base


# -----------------------------------------------------------------------------
# PASSO 5 — Estatisticas para a secao 4.1 do TCC (bruta x ajustada)
# -----------------------------------------------------------------------------
def medidas(b: pd.DataFrame):
    """Calcula todas as estatisticas descritivas de um recorte da base.
    Devolve uma lista de (rotulo, valor) na ordem de apresentacao."""
    v = b["vendas"]
    lin = [
        ("Linhas (série × dia)", len(b)),
        ("Proporção de zeros", (v == 0).mean()),
        ("Demanda média", v.mean()),
        ("Demanda mediana", v.median()),
        ("Desvio-padrão", v.std()),
        ("Máximo", v.max()),
    ]
    por_dia = b.groupby("weekday")["vendas"].mean()
    for dia_en, dia_pt in DIAS_PT.items():
        lin.append((f"Média — {dia_pt}", por_dia[dia_en]))
    com = v[b["snap"] == 1].mean()
    sem = v[b["snap"] == 0].mean()
    lin += [
        ("Média — dias com SNAP", com),
        ("Média — dias sem SNAP", sem),
        ("Efeito SNAP (%)", (com / sem - 1) * 100),
    ]
    for loja, valor in b.groupby("store_id")["vendas"].mean().items():
        lin.append((f"Média — loja {loja}", valor))
    return lin


def formatar(rotulo, x, decimal="."):
    if rotulo.startswith("Linhas") or rotulo == "Máximo":
        s = f"{x:,.0f}"
        return s.replace(",", ".") if decimal == "," else s
    if rotulo == "Proporção de zeros":
        s = f"{x:.1%}"
    elif rotulo.startswith("Efeito"):
        s = f"{x:+.1f}%"
    else:
        s = f"{x:.2f}"
    return s.replace(".", ",") if decimal == "," else s


def estatisticas(base: pd.DataFrame):
    print("\n[5/6] Estatisticas descritivas")
    print("=" * 70)

    # Tres recortes, do mais amplo ao adotado:
    #   bruta    : todas as linhas
    #   disp     : so dias disponiveis (criterio A), inclui lojas fechadas
    #   demanda  : dias disponiveis E loja aberta  <- ADOTADO no texto
    # Zero em loja fechada nao e demanda nula, e ausencia de operacao; por
    # isso a taxa de intermitencia da DEMANDA usa o recorte "demanda".
    disp = base[base["disponivel"]]
    demanda = disp[~disp["loja_fechada"]]

    n_series = base["id"].nunique()
    print(f"Series produto-loja .........: {n_series:,}")
    print(f"Dias de observacao ..........: {base['d'].nunique():,}")
    print(f"Periodo .....................: {base['date'].min().date()} a "
          f"{base['date'].max().date()}")
    print(f"Lojas .......................: {', '.join(sorted(base.store_id.unique()))}")
    print(f"Itens distintos .............: {base['item_id'].nunique():,}")
    print(f"Dias-serie disponiveis ......: {len(disp):,} de {len(base):,} "
          f"({len(disp) / len(base):.1%})")
    print(f"  menos loja fechada ........: {len(demanda):,} "
          f"(excluidos {len(disp) - len(demanda):,})")

    # --- INTERMITENCIA: proporcao de zeros por dois criterios ---
    # Criterio A (ADOTADO): conta a serie a partir do 1o registro de preco.
    # Criterio B (artigo de referencia): a partir da 1a venda positiva.
    primeira_venda = base.loc[base["vendas"] > 0].groupby("id")["dia"].min()
    apos_venda = base["dia"] >= base["id"].map(primeira_venda)
    aberta = ~base["loja_fechada"]

    def zeros(mascara):
        return (base.loc[mascara, "vendas"] == 0).mean()

    todas = pd.Series(True, index=base.index)
    z = {
        "bruta": (zeros(todas), zeros(aberta)),
        "A": (zeros(base["disponivel"]), zeros(base["disponivel"] & aberta)),
        "B": (zeros(apos_venda), zeros(apos_venda & aberta)),
    }

    print("\nProporcao de zeros:                        com fechamento  sem fechamento")
    print(f"  bruta, todos os dias (NAO usar) ........: {z['bruta'][0]:>12.2%}"
          f"{z['bruta'][1]:>16.2%}")
    print(f"  criterio A — desde o 1o preco ..........: {z['A'][0]:>12.2%}"
          f"{z['A'][1]:>16.2%}  <- ADOTADO")
    print(f"  criterio B — desde a 1a venda (artigo) .: {z['B'][0]:>12.2%}"
          f"{z['B'][1]:>16.2%}")
    print(f"  diferenca A - B (sem fechamento) .......: "
          f"{(z['A'][1] - z['B'][1]) * 100:+.2f} p.p.")
    print(f"  efeito de excluir fechamentos (crit. A) : "
          f"{(z['A'][1] - z['A'][0]) * 100:+.2f} p.p.")

    print("\n  por loja (sem fechamento)   bruta    crit. A   crit. B")
    for loja in LOJAS:
        m = (base["store_id"] == loja) & aberta
        print(f"  {loja:<24}{zeros(m):>9.1%}"
              f"{zeros(m & base['disponivel']):>10.1%}"
              f"{zeros(m & apos_venda):>10.1%}")

    # --- Tabela lado a lado: bruta x disponiveis x disponiveis sem fechamento ---
    colunas = {
        "Bruta (todos os dias)": medidas(base),
        "Disponíveis (com fechamento)": medidas(disp),
        "Disponíveis, loja aberta (adotada)": medidas(demanda),
    }
    rotulos = [r for r, _ in next(iter(colunas.values()))]
    tabela = pd.DataFrame({"Estatística": rotulos})
    for nome, lin in colunas.items():
        tabela[nome] = [v for _, v in lin]

    print("\nDescritivas — tres recortes:")
    print(f"  {'':<28}{'bruta':>12}{'disponiveis':>13}{'disp.+aberta':>14}")
    for _, r in tabela.iterrows():
        rot = r["Estatística"]
        print(f"  {rot:<28}" + f"{formatar(rot, r.iloc[1]):>12}"
              f"{formatar(rot, r.iloc[2]):>13}{formatar(rot, r.iloc[3]):>14}")
    print("=" * 70)

    salvar_tabelas(tabela, z)
    return demanda


def salvar_tabelas(tabela, z):
    """Salva a tabela descritiva em CSV e em LaTeX (booktabs) para o Overleaf."""
    PASTA_TAB.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(PASTA_TAB / "estatisticas_descritivas.csv",
                  index=False, encoding="utf-8-sig")

    # Tabela de intermitencia (criterios x fechamento), tambem em CSV
    pd.DataFrame(
        {"Com fechamento": [v[0] for v in z.values()],
         "Sem fechamento": [v[1] for v in z.values()]},
        index=["Bruta (todos os dias)", "Critério A — desde o 1º preço (adotado)",
               "Critério B — desde a 1ª venda"],
    ).rename_axis("Proporção de zeros").to_csv(
        PASTA_TAB / "proporcao_zeros.csv", encoding="utf-8-sig")

    linhas = [
        "% Gerado por scripts/etapa1_preparar_dados.py — requer \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Estatísticas descritivas da demanda diária — FOODS, lojas da Califórnia}",
        "\\label{tab:descritivas}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Estatística & Bruta & Dias & Dias disponíveis \\\\",
        " & (todos os dias) & disponíveis & e loja aberta \\\\",
        "\\midrule",
    ]
    for _, r in tabela.iterrows():
        rot = r["Estatística"]
        vals = [formatar(rot, r.iloc[k], ",").replace("%", "\\%") for k in (1, 2, 3)]
        if rot in ("Média — Segunda", "Média — dias com SNAP", "Média — loja CA_1"):
            linhas.append("\\addlinespace")
        linhas.append(f"{rot.replace('_', chr(92) + '_')} & " + " & ".join(vals) + " \\\\")
    pct = lambda x: f"{x:.1%}".replace(".", ",").replace("%", "\\%")
    linhas += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\smallskip",
        "{\\footnotesize Nota: dias disponíveis são os que têm registro de preço "
        "(produto à venda na loja); loja aberta exclui os dias de fechamento (25/12). "
        "A última coluna é a adotada no texto. Proporção de zeros, sem fechamento: "
        f"critério A (desde o primeiro preço) {pct(z['A'][1])}; critério B (desde a "
        f"primeira venda) {pct(z['B'][1])}; sem ajuste {pct(z['bruta'][1])}.}}",
        "\\end{table}",
    ]
    (PASTA_TAB / "estatisticas_descritivas.tex").write_text(
        "\n".join(linhas) + "\n", encoding="utf-8")
    print("Tabelas salvas em: resultados/tabelas/estatisticas_descritivas.(csv|tex)"
          " e proporcao_zeros.csv")


# -----------------------------------------------------------------------------
# PASSO 6 — Figuras descritivas (dias disponiveis e loja aberta)
# -----------------------------------------------------------------------------
def num_pt(x, casas=0):
    """Formata numero no padrao brasileiro (1.234,5)."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def estilo_eixos(ax, grade="y"):
    """Estilo comum: grade discreta, sem bordas superior/direita."""
    ax.grid(axis=grade, color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")


def salvar_figura(fig, nome):
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PASTA_FIG / nome, dpi=300)
    plt.close(fig)
    print(f"      resultados/figuras/{nome}")


def figuras(disp: pd.DataFrame):
    """disp: dias disponiveis e loja aberta (recorte adotado)."""
    print("\n[6/6] Gerando figuras (dias disponiveis e loja aberta)...")
    fmt_milhar = mticker.FuncFormatter(lambda v, _: num_pt(v))
    fmt_2casas = mticker.FuncFormatter(lambda v, _: num_pt(v, 1))

    # --- 1. Vendas diarias agregadas ao longo do tempo ---
    diario = disp.groupby("date")["vendas"].sum()
    media_movel = diario.rolling(28, center=True).mean()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(diario.index, diario.values, color=COR_PRINCIPAL, linewidth=0.5,
            alpha=0.45, label="Total diário")
    ax.plot(media_movel.index, media_movel.values, color=COR_PRINCIPAL,
            linewidth=2, label="Média móvel de 28 dias")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Unidades vendidas por dia")
    ax.set_xlabel("Data")
    ax.set_title("Vendas diárias agregadas — FOODS, lojas da Califórnia",
                 fontsize=11, loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(fmt_milhar)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    estilo_eixos(ax)
    salvar_figura(fig, "fig_vendas_diarias.png")

    # --- 2. Media de vendas por dia da semana ---
    por_dia = disp.groupby("weekday")["vendas"].mean().reindex(DIAS_PT.keys())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    barras = ax.bar(list(DIAS_PT.values()), por_dia.values, color=COR_PRINCIPAL,
                    width=0.7, edgecolor="white", linewidth=1.5)
    ax.bar_label(barras, labels=[num_pt(v, 2) for v in por_dia.values],
                 padding=3, fontsize=9, color=COR_TEXTO_SEC)
    ax.set_ylabel("Vendas médias por série-dia")
    ax.set_title("Média de vendas por dia da semana (dias disponíveis, loja aberta)",
                 fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(fmt_2casas)
    ax.set_ylim(0, por_dia.max() * 1.15)
    estilo_eixos(ax)
    salvar_figura(fig, "fig_media_dia_semana.png")

    # --- 3. Histograma da demanda diaria, destacando a massa em zero ---
    LIMITE = 20
    cont = disp["vendas"].clip(upper=LIMITE).value_counts(normalize=True).sort_index()
    rotulos = [str(int(k)) for k in cont.index[:-1]] + [f"{LIMITE}+"]
    cores = [COR_DESTAQUE] + [COR_PRINCIPAL] * (len(cont) - 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(rotulos, cont.values * 100, color=cores, width=0.8,
           edgecolor="white", linewidth=1)
    ax.annotate(f"venda zero: {num_pt(cont.iloc[0] * 100, 1)}% dos dias",
                xy=(0, cont.iloc[0] * 100), xytext=(2.2, cont.iloc[0] * 100 * 0.92),
                fontsize=9, color=COR_TEXTO_SEC,
                arrowprops=dict(arrowstyle="-", color=COR_TEXTO_SEC, linewidth=0.8))
    ax.set_xlabel(f"Unidades vendidas no dia (valores ≥ {LIMITE} agrupados)")
    ax.set_ylabel("% dos dias-série")
    ax.set_title("Distribuição da demanda diária por série (dias disponíveis, loja aberta)",
                 fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    estilo_eixos(ax)
    salvar_figura(fig, "fig_histograma_demanda.png")

    # --- 4. Vendas em dias com e sem SNAP ---
    com = disp.loc[disp["snap"] == 1, "vendas"].mean()
    sem = disp.loc[disp["snap"] == 0, "vendas"].mean()
    fig, ax = plt.subplots(figsize=(5, 3.8))
    barras = ax.bar(["Dias sem SNAP", "Dias com SNAP"], [sem, com],
                    color=COR_PRINCIPAL, width=0.55, edgecolor="white", linewidth=1.5)
    ax.bar_label(barras, labels=[num_pt(sem, 2), num_pt(com, 2)], padding=3,
                 fontsize=9, color=COR_TEXTO_SEC)
    ax.set_ylabel("Vendas médias por série-dia")
    ax.set_title(f"Efeito do SNAP: {(com / sem - 1) * 100:+.1f}%".replace(".", ","),
                 fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(fmt_2casas)
    ax.set_ylim(0, com * 1.15)
    estilo_eixos(ax)
    salvar_figura(fig, "fig_snap.png")


# -----------------------------------------------------------------------------
# EXECUCAO
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ETAPA 1 — PREPARACAO DA BASE ANALITICA")
    print("=" * 70)

    vendas, calendario, precos = carregar_arquivos(PASTA)
    vendas = aplicar_recorte(vendas)
    longo = formato_longo(vendas)
    base = juntar_covariaveis(longo, calendario, precos)
    base = marcar_eventos_operacionais(base)
    demanda = estatisticas(base)
    figuras(demanda)

    base.to_parquet(PASTA / ARQUIVO_SAIDA, index=False)
    print(f"\nBase salva em: dados/{ARQUIVO_SAIDA}")
    print("Use este arquivo nas proximas etapas, filtrando disponivel == True.\n")

    return base


if __name__ == "__main__":
    base = main()
