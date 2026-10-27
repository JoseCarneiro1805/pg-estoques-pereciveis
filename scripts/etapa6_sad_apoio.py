"""
===============================================================================
ETAPA 6 — Arquivo de apoio e motor de trajetoria da aba de OPERACAO do SAD
Projeto de Graduacao — Gestao de estoques pereciveis no varejo alimentar
===============================================================================

O QUE ESTE SCRIPT FAZ
---------------------
A aba de Operacao do sad.py mostra, para uma loja e uma data do periodo de
teste (26/01 a 24/04/2016), o estoque de cada item na abertura, a previsao, o
alvo de reposicao, o pedido ao fornecedor e os transbordos sugeridos. Esses
numeros vem da MESMA simulacao da Etapa 5B (modelo: combinacao), num unico
cenario (nivel de servico, theta, L_f, L_t), reproduzida dia a dia ate a data.

1. Arquivo de apoio (python scripts/etapa6_sad_apoio.py):
   dados/sad_operacao.npz com, na grade item x loja x dia do teste,
   - demanda observada, dias fechados e previsao pontual da combinacao;
   - alvo de reposicao arredondado, niveis de 70% a 99%, para L_f = 0 a 3.
     L_f = 0: quantis empiricos da combinacao (previsoes_combinacao.parquet),
     como na Etapa 4. L_f = 1 a 3: alvo de L_f+1 dias da Etapa 5B (versao
     incondicional), reconstruido pelas MESMAS funcoes de etapa5_transbordo.py
     a partir de dados/previsoes_pontuais_*.npz.
   Nenhuma tabela ou figura das etapas anteriores e regravada.

2. Motor de trajetoria (funcao trajetoria): o mesmo motor de
   simular_rede_lf (etapa5_transbordo.py), para um nivel so, guardando o
   estado de cada dia em vez de so os totais. Importado pelo sad.py.

CONFERENCIAS (rodam ao gerar o arquivo)
---------------------------------------
- Totais do motor = totais de simular_rede_lf, L_f = 0 a 3, sem transbordo e
  L_t = 1 e 2, theta = 0,10, niveis de 80/90/95/99%.
- Perda, faltas e volume = tbr_resultados_completos.csv (Etapa 5B) nos mesmos
  cenarios: prova de que os alvos reconstruidos sao os da 5B.
- Decomposicao do transbordo em pares origem -> destino soma o volume enviado.

PRE-REQUISITO: etapa3_comparacao.py e os modelos (previsoes salvas em dados/)
SAIDA: dados/sad_operacao.npz (nao versionado, como os demais arquivos de dados)
===============================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

ARQ_APOIO = Path(__file__).resolve().parent.parent / "dados" / "sad_operacao.npz"
REGISTRO = ("perda", "abertura", "caminho", "pedido", "recebido", "estoque", "vendido",
            "falta", "chegada", "envio", "recebe")


# -----------------------------------------------------------------------------
# MOTOR (um nivel; mesma sequencia de eventos de simular_rede_lf)
# -----------------------------------------------------------------------------
def trajetoria(S, S1, prev0, d, fech, theta, prazo_f, prazo=None, tipo=np.float32):
    """Estado diario de cada serie num cenario. S: alvo do horizonte de L_f+1
    dias (itens x lojas x dias); S1: alvo de 1 dia (estado inicial com L_f > 0);
    prev0: previsao diaria; d: demanda (NaN antes de a serie existir);
    prazo: noites do transbordo (None = sem transbordo).
    Registros por dia t:
      perda     perda da noite (theta x sobra da vespera)
      abertura  estoque em maos na abertura (apos a perda e as chegadas do dia)
      caminho   pedidos e transbordos que chegam em t+1..t+L_f (0 se L_f = 0)
      pedido    pedido ao fornecedor feito para o dia t
      recebido  entrega do fornecedor no dia t
      estoque   estoque para a venda do dia
      envio     transbordo que SAI da loja ao fim do dia t (antes da perda)
      recebe    transbordo destinado a loja ao fim do dia t (antes da perda em transito)"""
    I, J, D = S.shape
    ativa = ~np.isnan(d)
    aberta = ativa & ~fech
    primeiro = ativa & ~np.concatenate([np.zeros((I, J, 1), bool), ativa[:, :, :-1]], axis=2)
    sobra = np.zeros((I, J))
    pend_f, pend_t = {}, {}                          # dia de chegada -> quantidade
    reg = {k: np.zeros((I, J, D), tipo) for k in REGISTRO}
    fator = (1 - theta) ** prazo if prazo else 1.0
    soma = lambda dic, dias: sum((dic.get(k, 0.0) for k in dias), 0.0)
    for t in range(D):
        a = aberta[:, :, t]
        perdido = theta * sobra
        chega_f, chega_t = pend_f.pop(t, 0.0), pend_t.pop(t, 0.0)
        if prazo_f > 0 and primeiro[:, :, t].any():           # estado inicial (decisao 7 da 5B)
            p0 = primeiro[:, :, t]
            chega_f = chega_f + np.where(p0, S1[:, :, t], 0)
            for k in range(1, prazo_f + 1):
                if t + k <= D - 1:
                    pend_f[t + k] = pend_f.get(t + k, 0.0) + np.where(
                        p0, np.maximum(np.nan_to_num(prev0[:, :, t]), 0), 0)
        disponivel = sobra - perdido + chega_f + chega_t
        if prazo_f == 0:                                       # chega antes da abertura
            caminho = 0.0
            pedido = np.where(a, np.maximum(S[:, :, t] - disponivel, 0), 0)
            recebido = pedido
            estoque = disponivel + pedido
        else:                                                  # chega em t + L_f
            horizonte = range(t + 1, t + prazo_f + 1)
            # mesma ordem de soma da 5B: o teste "excedente == 0" do transbordo e
            # sensivel a diferencas de arredondamento
            posicao = disponivel + soma(pend_f, horizonte) + soma(pend_t, horizonte)
            caminho = posicao - disponivel
            pedido = np.where(a, np.maximum(S[:, :, t] - posicao, 0), 0)
            if t + prazo_f <= D - 1:
                pend_f[t + prazo_f] = pend_f.get(t + prazo_f, 0.0) + pedido
            recebido = chega_f
            estoque = disponivel
        dem = np.where(a, np.nan_to_num(d[:, :, t]), 0)
        vendido = np.minimum(estoque, dem)
        falta = dem - vendido
        sobra = estoque - vendido
        for k, v in (("perda", perdido), ("abertura", disponivel), ("caminho", caminho),
                     ("pedido", pedido), ("recebido", recebido), ("estoque", estoque),
                     ("vendido", vendido), ("falta", falta), ("chegada", chega_t)):
            reg[k][:, :, t] = v
        if prazo and t + prazo <= D - 1:                       # transbordo ao fim do dia
            alvo = S[:, :, t + 1]
            horizonte = range(t + 1, t + 2 + prazo_f)
            pipe_f = soma(pend_f, horizonte)
            excedente = np.where(ativa[:, :, t] & ativa[:, :, t + 1],
                                 np.minimum(sobra, np.maximum(sobra + pipe_f - alvo, 0)), 0)
            projetado = (1 - theta) * sobra + pipe_f + soma(pend_t, horizonte)
            precisa = np.where(aberta[:, :, t + 1] & (excedente == 0),
                               np.maximum(alvo - projetado, 0), 0)
            E = excedente.sum(axis=1, keepdims=True)
            F = precisa.sum(axis=1, keepdims=True)
            volume = np.minimum(E, F)
            with np.errstate(invalid="ignore", divide="ignore"):
                envio = np.where(E > 0, excedente * volume / E, 0)
                recebe = np.where(F > 0, precisa * volume / F, 0)
            sobra = sobra - envio
            pend_t[t + prazo] = pend_t.get(t + prazo, 0.0) + recebe * fator
            reg["envio"][:, :, t] = envio
            reg["recebe"][:, :, t] = recebe
    return reg


def pares_transbordo(envio, recebe, minimo=1e-9):
    """Decompoe o rateio proporcional de um dia (itens x lojas) em pares
    origem -> destino: fluxo(i -> j) = envio_i x recebe_j / volume do item.
    Devolve (item, origem, destino, quantidade enviada)."""
    volume = envio.sum(axis=1)
    pares = []
    for it in np.flatnonzero(volume > minimo):
        fluxo = np.outer(envio[it], recebe[it]) / volume[it]
        for o, dest in zip(*np.nonzero(fluxo > minimo)):
            pares.append((it, o, dest, fluxo[o, dest]))
    return pares


# -----------------------------------------------------------------------------
# ARQUIVO DE APOIO E CONFERENCIAS
# -----------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import etapa3_avaliacao as av
    import etapa5_transbordo as e5
    from etapa1_preparar_dados import PASTA_TAB

    inicio = time.time()
    print("=" * 70)
    print("ETAPA 6 — ARQUIVO DE APOIO DA ABA DE OPERACAO DO SAD")
    print("=" * 70)
    itens, lojas, d, fech, S0, datas = e5.carregar()
    dd = e5.carregar_5b(itens, lojas)
    prev0 = np.full(d.shape, np.nan)
    prev0[dd["gi"], dd["gj"], :] = dd["prev_teste"]["comb"][:, av.INICIO_TESTE - e5.DIA0_5B:]
    alvos = {0: S0}
    for lf in e5.PRAZOS_FORNECEDOR[1:]:
        print(f"      alvo de {lf + 1} dias (L_f = {lf})...")
        alvos[lf], _ = e5.alvo_combinado(dd, lf, "incondicional", S0.shape)
    maior = max(float(a.max()) for a in alvos.values())
    assert maior < np.iinfo(np.uint16).max and all(np.allclose(a, np.round(a)) for a in alvos.values())

    print("\n[conferencias] motor x simular_rede_lf x tbr_resultados_completos.csv (theta = 0,10)")
    tbr = pd.read_csv(PASTA_TAB / "tbr_resultados_completos.csv")
    th = e5.THETA_BASE
    dif_motor, dif_tbr = 0.0, 0.0
    for lf, S in alvos.items():
        for lt in (None,) + e5.PRAZOS_TRANSBORDO:
            for nivel in (80, 90, 95, 99):
                k = e5.NIVEIS.index(nivel / 100)
                r = trajetoria(S[k], S0[k], prev0, d, fech, th, lf, lt, tipo=np.float64)
                ref, _, _ = e5.simular_rede_lf(S[k:k + 1], S0[k:k + 1], prev0, d, fech, th, lf,
                                               lt, falta_base=np.zeros((1,) + d.shape, np.float32))
                fator = (1 - th) ** lt if lt else 1.0
                meu = {"perda_loja": r["perda"].sum(2), "fornecedor": r["recebido"].sum(2),
                       "chegada": r["chegada"].sum(2), "vendido": r["vendido"].sum(2),
                       "falta": r["falta"].sum(2), "enviado": r["envio"].sum(2),
                       "perda_transito": r["recebe"].sum(2) * (1 - fator),
                       "estoque": np.where(~np.isnan(d) & ~fech, r["estoque"], 0).sum(2)}
                for c, v in meu.items():
                    dif = float(np.abs(v - ref[c][0]).max())
                    if dif > 1e-9:
                        print(f"      DIFERENCA L_f = {lf}, L_t = {lt}, nivel {nivel}, {c}: {dif:.2e}")
                    dif_motor = max(dif_motor, dif)
                dias = (~np.isnan(d) & ~fech).sum()
                linha = tbr[(tbr["L_f"] == lf) & (tbr["L_t"] == (lt or 0))
                            & np.isclose(tbr["theta"], th) & (tbr["Nível"] == nivel)].iloc[0]
                for c, v in (("Perda total", meu["perda_loja"].sum() + meu["perda_transito"].sum()),
                             ("Faltas", meu["falta"].sum()), ("Volume enviado", meu["enviado"].sum())):
                    dif_tbr = max(dif_tbr, abs(v / dias - linha[c]))
                if lt and nivel == 90:                      # pares somam o volume enviado
                    soma_pares = sum(sum(q for *_, q in pares_transbordo(r["envio"][:, :, t],
                                                                         r["recebe"][:, :, t]))
                                     for t in range(d.shape[2]))
                    assert abs(soma_pares - r["envio"].sum()) < 1e-6, (lf, lt, soma_pares)
        print(f"      L_f = {lf}: ok")
    assert dif_motor < 1e-9, f"motor nao reproduz simular_rede_lf (dif. {dif_motor:.1e})"
    assert dif_tbr < 1e-9, f"motor nao reproduz a tabela da 5B (dif. {dif_tbr:.1e})"
    print(f"      motor = simular_rede_lf (maior dif. {dif_motor:.1e}); "
          f"= tbr_resultados_completos.csv (maior dif. {dif_tbr:.1e}); pares de transbordo ok")

    np.savez_compressed(
        ARQ_APOIO, itens=itens.astype(str), lojas=lojas.astype(str),
        datas=pd.DatetimeIndex(datas).strftime("%Y-%m-%d").to_numpy(str),
        niveis=np.array([av.pct(t) for t in e5.NIVEIS]),
        demanda=d.astype(np.float32), fechada=fech, previsao=prev0.astype(np.float32),
        **{f"alvo_lf{lf}": S.astype(np.uint16) for lf, S in alvos.items()})
    print(f"\n      {ARQ_APOIO.relative_to(ARQ_APOIO.parent.parent)} "
          f"({ARQ_APOIO.stat().st_size / 1e6:.1f} MB) | {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
