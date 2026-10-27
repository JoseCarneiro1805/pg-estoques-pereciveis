"""
===============================================================================
ETAPA 2 — Engenharia de variaveis para os modelos de aprendizado de maquina
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE ARQUIVO FAZ
----------------------
Monta, para cada serie-dia, as variaveis explicativas do XGBoost (e de outros
modelos globais que vierem depois). Nao e executado sozinho: o script do
modelo o usa. As variaveis sao calculadas na hora (leva menos de 1 minuto).

REGRA DE OURO — NADA DO FUTURO
------------------------------
No dia t, as variaveis de VENDA usam somente vendas ate t-1. Isso vale para
defasagens, medias moveis, dias desde a ultima venda etc. Como todas as
variaveis de venda olham apenas para tras, a mesma tabela serve para qualquer
origem da validacao: o que muda de uma origem para outra e so quais linhas
entram no treino.

Calendario, eventos, SNAP e preco do proprio dia t sao CONHECIDOS DE ANTEMAO
(o calendario e publico e o preco e definido pela propria rede), por isso
podem entrar com o valor do dia t.

DIAS QUE NAO SAO DEMANDA
------------------------
Dias sem preco (produto fora de venda) e dias de loja fechada viram "ausentes"
nas variaveis de venda: um zero nesses dias nao e demanda nula e nao deve
puxar as medias para baixo.

COMO FUNCIONA
-------------
A base e uma grade completa (5.748 series x 1.913 dias). As contas sao feitas
nessa grade, com uma linha por serie e uma coluna por dia, o que e muito mais
rapido do que calcular serie a serie.
===============================================================================
"""

import warnings

import numpy as np
import pandas as pd

# Defasagens de venda (em dias). 364 = mesmo dia da semana, um ano antes.
DEFASAGENS = (1, 2, 3, 7, 14, 28, 364)
JANELAS_MEDIA = (7, 28, 91)

# Familia de cada variavel (usada para resumir a importancia)
FAMILIAS = {
    "Defasagens de venda": [f"venda_lag_{k}" for k in DEFASAGENS],
    "Médias móveis e histórico": (
        [f"media_{w}d" for w in JANELAS_MEDIA]
        + ["desvio_28d", "zeros_28d", "media_mesmo_dia_4s", "dias_desde_venda",
           "idade_serie"]),
    "Preço": ["preco", "preco_var_semana", "preco_rel_historico", "preco_rel_departamento"],
    "Calendário": ["dia_semana", "mes", "dia_mes", "semana_ano"],
    "Eventos": ["tem_evento", "tipo_evento", "nome_evento", "evento_amanha"],
    "SNAP": ["snap"],
    "Reforma CA_2 e fechamento": ["pos_reforma_ca2", "loja_fechada"],
    "Identificadores": ["loja", "departamento", "item"],
}
VARIAVEIS = [v for vs in FAMILIAS.values() for v in vs]
CATEGORICAS = ["tipo_evento", "nome_evento", "loja", "departamento", "item"]


# -----------------------------------------------------------------------------
# Operacoes na grade (linha = serie, coluna = dia)
# -----------------------------------------------------------------------------
def _desloca(m, k):
    """Valor de k dias antes (as primeiras k colunas ficam ausentes)."""
    out = np.full_like(m, np.nan)
    out[:, k:] = m[:, :-k]
    return out


def _acumulado(m):
    """Somas acumuladas ignorando ausentes, com uma coluna de zeros na frente:
    s[:, t] = soma dos dias 0..t-1."""
    zeros = np.zeros((m.shape[0], 1), dtype=np.float64)
    return np.hstack([zeros, np.nancumsum(m, axis=1, dtype=np.float64)])


def _janela(m, w):
    """Soma e contagem de valores presentes nos w dias ANTERIORES (t-w .. t-1)."""
    s, c = _acumulado(m), _acumulado(~np.isnan(m))
    t = np.arange(m.shape[1])
    ini = np.maximum(t - w, 0)
    return s[:, t] - s[:, ini], c[:, t] - c[:, ini]


def _media_janela(m, w):
    soma, n = _janela(m, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, soma / n, np.nan)


# -----------------------------------------------------------------------------
# Montagem
# -----------------------------------------------------------------------------
class Grade:
    """Variaveis na forma de grade + utilitario para extrair linhas."""

    def __init__(self, ids, dias, por_serie, por_dia, fixas):
        self.ids, self.dias = ids, dias        # rotulos das linhas e colunas
        self.por_serie = por_serie             # {nome: matriz serie x dia}
        self.por_dia = por_dia                 # {nome: vetor por dia}
        self.fixas = fixas                     # {nome: vetor por serie}

    def tabela(self, mascara):
        """DataFrame com as variaveis das celulas onde mascara (serie x dia) e
        verdadeira, na ordem serie -> dia."""
        lin, col = np.nonzero(mascara)
        dados = {"id": self.ids[lin], "dia": self.dias[col]}
        for nome in VARIAVEIS:
            if nome in self.por_serie:
                dados[nome] = self.por_serie[nome][lin, col]
            elif nome in self.por_dia:
                dados[nome] = self.por_dia[nome][col]
            else:
                dados[nome] = self.fixas[nome][lin]
        return pd.DataFrame(dados)


def montar_variaveis(base):
    """base: a base analitica completa (todas as linhas, inclusive
    indisponiveis). Devolve uma Grade com todas as variaveis."""
    base = base.sort_values(["id", "dia"], ignore_index=True)
    ids = base["id"].unique()
    dias = np.sort(base["dia"].unique())
    n, T = len(ids), len(dias)
    assert len(base) == n * T, "a base precisa ser uma grade completa serie x dia"
    grade = lambda col, tipo=np.float32: base[col].to_numpy(tipo).reshape(n, T)

    disp = grade("disponivel", bool)
    fechada = grade("loja_fechada", bool)
    # Venda como demanda observada: ausente se sem preco ou loja fechada
    v = np.where(disp & ~fechada, grade("vendas", np.float64), np.nan)

    f = {}
    # --- Defasagens ---
    for k in DEFASAGENS:
        f[f"venda_lag_{k}"] = _desloca(v, k)

    # --- Medias moveis e historico (sempre ate t-1) ---
    for w in JANELAS_MEDIA:
        f[f"media_{w}d"] = _media_janela(v, w)
    media2 = _media_janela(v ** 2, 28)
    f["desvio_28d"] = np.sqrt(np.maximum(media2 - f["media_28d"] ** 2, 0))
    f["zeros_28d"] = _media_janela(np.where(np.isnan(v), np.nan, (v == 0).astype(float)), 28)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # janela toda ausente
        f["media_mesmo_dia_4s"] = np.nanmean(
            np.stack([_desloca(v, k) for k in (7, 14, 21, 28)]), axis=0)
    t = np.arange(T, dtype=np.float64)
    ultima = np.maximum.accumulate(np.where(v > 0, t, -1), axis=1)   # ate t
    ultima = _desloca(ultima, 1)                                      # ate t-1
    f["dias_desde_venda"] = np.where(ultima >= 0, t - ultima, np.nan)
    primeiro = np.argmax(disp, axis=1)[:, None]
    f["idade_serie"] = np.where(disp, t - primeiro, np.nan)

    # --- Preco (conhecido no proprio dia) ---
    p = np.where(disp, grade("sell_price", np.float64), np.nan)
    f["preco"] = p
    with np.errstate(invalid="ignore", divide="ignore"):
        f["preco_var_semana"] = p / _desloca(p, 7) - 1
        s, c = _acumulado(p), _acumulado(~np.isnan(p))
        f["preco_rel_historico"] = p / (s[:, 1:] / c[:, 1:])          # media ate t
        loja_dep = (base.groupby("id", sort=False)[["store_id", "dept_id"]].first()
                    .reindex(ids).astype(str).agg("|".join, axis=1).to_numpy())
        rel = np.full_like(p, np.nan)
        for g in np.unique(loja_dep):
            linhas = loja_dep == g
            rel[linhas] = p[linhas] / np.nanmean(p[linhas], axis=0)
        f["preco_rel_departamento"] = rel

    # --- Controles da Etapa 1 ---
    f["pos_reforma_ca2"] = grade("pos_reforma_ca2")
    f["loja_fechada"] = fechada.astype(np.float32)

    por_serie = {k: m.astype(np.float32) for k, m in f.items()}

    # --- Calendario e eventos (iguais para todas as series no mesmo dia) ---
    cal = base.drop_duplicates("dia").set_index("dia").loc[dias]
    evento = cal["tem_evento"].to_numpy(np.float32)
    por_dia = {
        "dia_semana": cal["wday"].to_numpy(np.float32),           # 1 = sabado (M5)
        "mes": cal["month"].to_numpy(np.float32),
        "dia_mes": cal["date"].dt.day.to_numpy(np.float32),
        "semana_ano": cal["date"].dt.isocalendar().week.to_numpy(np.float32),
        "tem_evento": evento,
        "tipo_evento": pd.Categorical(cal["event_type_1"].fillna("nenhum")),
        "nome_evento": pd.Categorical(cal["event_name_1"].fillna("nenhum")),
        "evento_amanha": np.append(evento[1:], 0).astype(np.float32),
        "snap": cal["snap"].to_numpy(np.float32),
    }

    # --- Identificadores (fixos por serie) ---
    ident = base.groupby("id", sort=False)[["store_id", "dept_id", "item_id"]].first().reindex(ids)
    fixas = {
        "loja": pd.Categorical(ident["store_id"]),
        "departamento": pd.Categorical(ident["dept_id"]),
        "item": pd.Categorical(ident["item_id"]),
    }
    return Grade(ids, dias, por_serie, por_dia, fixas), disp, fechada, grade("vendas")
