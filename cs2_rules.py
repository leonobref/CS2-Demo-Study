"""
cs2_rules.py
============
Mineração de regras de associação (Apriori / FP-Growth) — TCC 2.

Transação ("cesta"): um round visto pela perspectiva de UM lado.
Cada round gera duas cestas (T e CT). Os itens são variáveis discretizadas
do contexto pré-round e da abertura do round; o item de resultado
(resultado=vitoria / resultado=derrota) é o consequente de interesse.

Exemplo de regra:
  {lado=T, compra=full_buy, adversario=force_buy, opening=ganhou}
      → {resultado=vitoria}   suporte 0.04 | confiança 0.91 | lift 1.82

Leitura das métricas (Agrawal & Srikant, 1994):
  suporte   → fração das cestas que contêm antecedente + consequente
  confiança → P(consequente | antecedente)
  lift      → confiança / P(consequente); > 1 indica associação positiva

Controle de falsos positivos: com centenas de regras testadas, algumas terão
lift alto por acaso. Cada regra recebe um teste exato de Fisher (unilateral)
e o p-valor é corrigido por Benjamini-Hochberg (coluna p_ajustado).
Ressalva: as duas cestas de um mesmo round não são independentes
(o que é vitória para um lado é derrota para o outro); use --lado T ou
--lado CT para análises em que essa dependência importe.

Saídas (pasta --output):
  transacoes_itens.csv     itens de cada cesta (auditoria da discretização)
  regras_vitoria.csv       regras com consequente resultado=vitoria
  regras_derrota.csv       regras com consequente resultado=derrota
  fig_regras_top.png       melhores regras de vitória por lift

Uso:
  python cs2_rules.py cs2.db --output results/regras
  python cs2_rules.py --csv datasets/model_rounds.csv --algoritmo fpgrowth --min-support 0.03
"""

import argparse
import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import fisher_exact

from cs2_model_data import build_round_dataset, build_side_dataset

log = logging.getLogger("cs2_rules")

OUTCOME_ITEMS = {"resultado=vitoria", "resultado=derrota"}


# =============================================================================
# DISCRETIZAÇÃO → ITENS
# =============================================================================

def _bin(value, edges, labels):
    if pd.isna(value):
        return None
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def side_to_items(r: pd.Series, econ_gap: float) -> list:
    """
    Converte uma linha do dataset por lado em itens 'variavel=valor'.
    Os cortes são fixos e documentados aqui para poderem ser citados no texto.
    """
    items = [
        f"lado={r['side']}",
        f"mapa={str(r['map_name']).replace('de_', '')}",
        f"fase={r['phase']}",
        f"compra={r['buy_team']}",
        f"adversario={r['buy_opp']}",
        "resultado=vitoria" if r["win"] == 1 else "resultado=derrota",
    ]
    if r["is_pistol"] == 1:
        items.append("pistol=sim")

    # Vantagem econômica: diferença de equipamento total entre as equipes
    ed = r["equip_diff"]
    if pd.notna(ed):
        items.append("economia=vantagem" if ed > econ_gap
                     else "economia=desvantagem" if ed < -econ_gap
                     else "economia=equilibrio")

    items.append("awp=sim" if r["awps"] and r["awps"] > 0 else "awp=nao")

    sd = r["score_diff"]
    if pd.notna(sd):
        items.append("placar=frente" if sd >= 3 else "placar=atras" if sd <= -3
                     else "placar=parelho")

    if pd.notna(r["won_prev"]):
        items.append("round_anterior=venceu" if r["won_prev"] == 1
                     else "round_anterior=perdeu")
    if pd.notna(r["opening_won"]):
        items.append("opening=ganhou" if r["opening_won"] == 1 else "opening=perdeu")

    # Utilidade antes do opening kill
    items.append(f"smokes_pre={_bin(r['smokes_pre'], [0, 2], ['0', '1-2', '3+'])}")
    items.append(f"flashes_pre={_bin(r['flashes_pre'], [0, 2], ['0', '1-2', '3+'])}")
    items.append("molotov_pre=sim" if r["molotovs_pre"] > 0 else "molotov_pre=nao")
    if pd.notna(r["util_pre"]) and pd.notna(r["util_pre_opp"]):
        du = r["util_pre"] - r["util_pre_opp"]
        items.append("utilidade_vs_adv=mais" if du >= 2
                     else "utilidade_vs_adv=menos" if du <= -2
                     else "utilidade_vs_adv=similar")
    return [i for i in items if i and not i.endswith("=None")]


# =============================================================================
# MINERAÇÃO
# =============================================================================

def mine_rules(transactions: list, algorithm: str, min_support: float,
               min_confidence: float, min_lift: float, max_len: int,
               min_improvement: float = 0.02) -> pd.DataFrame:
    te = TransactionEncoder()
    basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    finder = apriori if algorithm == "apriori" else fpgrowth
    freq = finder(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    log.info(f"  {len(freq):,} itemsets frequentes ({algorithm}, suporte ≥ {min_support})")
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    # Só interessam regras do tipo  contexto → resultado
    rules = rules[rules["consequents"].apply(lambda c: len(c) == 1 and next(iter(c)) in OUTCOME_ITEMS)]
    rules = rules[rules["antecedents"].apply(lambda a: not (a & OUTCOME_ITEMS))]
    # Regra só com o lado no antecedente é a taxa de vitória do lado — não é padrão
    rules = rules[rules["antecedents"].apply(lambda a: not all(i.startswith("lado=") for i in a))]
    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return rules

    rules = _prune_redundant(rules, min_improvement)
    rules = _fisher_bh(rules, basket)
    rules["n_cestas"] = (rules["support"] * len(basket)).round().astype(int)
    rules["antecedente"] = rules["antecedents"].apply(lambda s: " & ".join(sorted(s)))
    rules["consequente"] = rules["consequents"].apply(lambda s: next(iter(s)))
    cols = ["antecedente", "consequente", "n_cestas", "support", "confidence", "lift",
            "leverage", "conviction", "p_valor", "p_ajustado", "n_itens"]
    out = rules[cols].rename(columns={"support": "suporte", "confidence": "confianca",
                                      "leverage": "alavancagem", "conviction": "conviccao"})
    return out.sort_values(["consequente", "lift"], ascending=[True, False]).reset_index(drop=True)


def _prune_redundant(rules: pd.DataFrame, min_improvement: float) -> pd.DataFrame:
    """
    Remove a regra A∪B → Y quando existe uma regra mais simples A → Y cuja
    confiança fica a menos de `min_improvement` da dela: o item extra não
    acrescenta poder explicativo (critério de melhoria mínima, Bayardo et al., 1999).
    """
    rules = rules.copy()
    rules["n_itens"] = rules["antecedents"].apply(len)
    rules = rules.sort_values("n_itens")
    kept = []
    best = {}  # consequente → lista de (antecedente, confiança)
    for _, r in rules.iterrows():
        cons = next(iter(r["consequents"]))
        redundant = any(a < r["antecedents"] and c >= r["confidence"] - min_improvement
                        for a, c in best.get(cons, []))
        if not redundant:
            kept.append(r)
            best.setdefault(cons, []).append((r["antecedents"], r["confidence"]))
    return pd.DataFrame(kept)


def _fisher_bh(rules: pd.DataFrame, basket: pd.DataFrame) -> pd.DataFrame:
    pvals = []
    for _, r in rules.iterrows():
        a = basket[list(r["antecedents"])].all(axis=1).values
        y = basket[next(iter(r["consequents"]))].values
        table = [[np.sum(a & y), np.sum(a & ~y)], [np.sum(~a & y), np.sum(~a & ~y)]]
        pvals.append(fisher_exact(table, alternative="greater")[1])
    rules = rules.copy()
    rules["p_valor"] = pvals
    rules["p_ajustado"] = _benjamini_hochberg(np.array(pvals))
    return rules


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# =============================================================================
# FIGURA
# =============================================================================

def plot_top(rules: pd.DataFrame, out: Path, top: int = 12):
    r = rules[(rules["consequente"] == "resultado=vitoria") & (rules["p_ajustado"] < 0.05)]
    if r.empty:
        log.warning("  Nenhuma regra de vitória significativa para plotar")
        return
    r = r.nlargest(top, "lift").iloc[::-1]
    labels = ["\n".join(textwrap.wrap(a.replace(" & ", "  ·  "), 60)) for a in r["antecedente"]]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(r) + 1.5))
    ax.barh(range(len(r)), r["lift"], color="#3A6EA5", height=0.6)
    for i, (l, c, s) in enumerate(zip(r["lift"], r["confianca"], r["n_cestas"])):
        ax.text(l + 0.02, i, f"conf. {c:.0%} · n={s}", va="center", fontsize=8)
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(1, color="#666", lw=1, ls="--")
    ax.set_xlabel("Lift (regras → resultado=vitoria, p ajustado < 0,05)")
    ax.set_xlim(0.9, r["lift"].max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_regras_top.png", dpi=200)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Regras de associação contexto → resultado (TCC 2)")
    ap.add_argument("db", nargs="?", help="Banco SQLite (ex.: cs2.db)")
    ap.add_argument("--csv", help="Usar model_rounds.csv já gerado em vez do banco")
    ap.add_argument("--output", "-o", default="results/regras")
    ap.add_argument("--algoritmo", choices=["apriori", "fpgrowth"], default="apriori")
    ap.add_argument("--lado", choices=["ambos", "T", "CT"], default="ambos")
    ap.add_argument("--min-support", type=float, default=0.02)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--min-lift", type=float, default=1.1)
    ap.add_argument("--max-len", type=int, default=4,
                    help="Tamanho máximo do itemset (antecedente + consequente)")
    ap.add_argument("--min-improvement", type=float, default=0.02,
                    help="Ganho mínimo de confiança sobre regras mais simples")
    ap.add_argument("--econ-gap", type=float, default=5000,
                    help="Diferença de equipamento ($) para considerar vantagem/desvantagem")
    args = ap.parse_args()

    if not args.db and not args.csv:
        ap.error("informe o banco ou --csv")
    rounds = pd.read_csv(args.csv) if args.csv else build_round_dataset(args.db)
    sides = build_side_dataset(rounds)
    if args.lado != "ambos":
        sides = sides[sides["side"] == args.lado]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    transactions = [side_to_items(r, args.econ_gap) for _, r in sides.iterrows()]
    pd.DataFrame({"match_id": sides["match_id"].values, "round_num": sides["round_num"].values,
                  "lado": sides["side"].values,
                  "itens": [" | ".join(t) for t in transactions]}
                 ).to_csv(out / "transacoes_itens.csv", index=False)
    log.info(f"🧺 {len(transactions):,} cestas | lado={args.lado}")

    rules = mine_rules(transactions, args.algoritmo, args.min_support,
                       args.min_confidence, args.min_lift, args.max_len,
                       args.min_improvement)
    if rules.empty:
        log.warning("Nenhuma regra encontrada — tente reduzir --min-support ou --min-lift")
        return

    for cons, name in [("resultado=vitoria", "regras_vitoria.csv"),
                       ("resultado=derrota", "regras_derrota.csv")]:
        rules[rules["consequente"] == cons].to_csv(out / name, index=False)
    plot_top(rules, out)

    sig = rules[rules["p_ajustado"] < 0.05]
    print("\n" + "=" * 70)
    print(f"{len(rules)} regras não redundantes | {len(sig)} com p ajustado < 0,05")
    print("=" * 70)
    pd.set_option("display.width", 160, "display.max_colwidth", 90)
    show = ["antecedente", "n_cestas", "confianca", "lift", "p_ajustado"]
    for cons in ["resultado=vitoria", "resultado=derrota"]:
        print(f"\nTop 8 → {cons}")
        print(sig[sig["consequente"] == cons].nlargest(8, "lift")[show].round(3).to_string(index=False))
    print(f"\n💾 Resultados em {out}/")


if __name__ == "__main__":
    main()
