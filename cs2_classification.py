"""
cs2_classification.py
=====================
Modelos de classificação do resultado do round (TCC 2 — Modelagem).

Pergunta analítica: quais variáveis contextuais explicam / predizem a vitória
do lado T em um round?

Dois cenários de variáveis (definidos em cs2_model_data.py):
  pre   → só informação disponível ao fim do freeze time (economia, placar,
          mapa, round anterior). Responde: "o que a decisão de compra e o
          contexto já dizem sobre o round?"
  full  → pre + eventos até o opening kill (lado do opening kill, tempo até
          ele, utilidade usada antes dele). Responde: "quanto a abertura
          do round muda o quadro?"

Modelos: Dummy (linha de base), Regressão Logística, Árvore de Decisão,
Random Forest, XGBoost.

Validação: StratifiedGroupKFold — todos os rounds de uma mesma série
(MD3/MD5) ficam no mesmo fold, para que o modelo não "aprenda" a partida
que está sendo avaliado. Agrupar por mapa (--group match) também é possível.

Saídas (pasta --output):
  metricas_resumo.csv            média ± desvio por modelo e cenário
  metricas_por_fold.csv          valores de cada fold
  importancia_permutacao.csv     queda de ROC-AUC ao embaralhar cada variável
                                 (calculada nos folds de TESTE)
  coeficientes_logistica.csv     coeficientes padronizados (cenário full e pre)
  arvore_<cenario>.txt           regras da árvore de decisão treinada em todo o dataset
  fig_metricas.png, fig_importancia_<modelo>_<cenario>.png

Uso:
  python cs2_classification.py cs2.db --output results/classificacao
  python cs2_classification.py --csv datasets/model_rounds.csv --output results/classificacao
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from cs2_model_data import (IN_ROUND_FEATURES, PRE_ROUND_CATEGORICAL,
                            PRE_ROUND_FEATURES, build_round_dataset)

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost é opcional
    XGBClassifier = None

log = logging.getLogger("cs2_classification")
SEED = 42
TARGET = "t_win"

SCENARIOS = {
    "pre": (PRE_ROUND_FEATURES, PRE_ROUND_CATEGORICAL),
    "full": (PRE_ROUND_FEATURES + IN_ROUND_FEATURES, PRE_ROUND_CATEGORICAL),
}


# =============================================================================
# MODELOS
# =============================================================================

def make_models() -> dict:
    """Hiperparâmetros conservadores (amostra de ~2 mil rounds)."""
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistica": LogisticRegression(max_iter=2000, C=1.0),
        "arvore": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                max_features="sqrt", n_jobs=-1,
                                                random_state=SEED),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          eval_metric="logloss", random_state=SEED,
                                          n_jobs=-1)
    else:
        log.warning("xgboost não instalado — modelo ignorado (pip install xgboost)")
    return models


def make_pipeline(model, num_cols, cat_cols, scale: bool = True) -> Pipeline:
    # Escalonamento só importa para a logística; a árvore de leitura usa scale=False
    # para que os limiares fiquem nas unidades originais (dinheiro, contagens, segundos)
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         cat_cols),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", pre), ("model", model)])


# =============================================================================
# AVALIAÇÃO
# =============================================================================

def _metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    out = {
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, proba),
    }
    out["roc_auc"] = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan
    return out


def evaluate(df: pd.DataFrame, n_splits: int, group_col: str, n_repeats_perm: int):
    groups = df[group_col]
    n_groups = groups.nunique()
    if n_groups < n_splits:
        log.warning(f"Só {n_groups} grupos — reduzindo para {n_groups} folds")
        n_splits = n_groups
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y = df[TARGET].values

    fold_rows, perm_rows = [], []
    for scen, (num, cat) in SCENARIOS.items():
        num = [c for c in num if c in df.columns]
        cat = [c for c in cat if c in df.columns]
        X = df[num + cat]
        for name, model in make_models().items():
            for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
                pipe = make_pipeline(model, num, cat)
                pipe.fit(X.iloc[tr], y[tr])
                proba = pipe.predict_proba(X.iloc[te])[:, 1]
                fold_rows.append({"cenario": scen, "modelo": name, "fold": fold,
                                  "n_teste": len(te), **_metrics(y[te], proba)})

                if name != "dummy" and n_repeats_perm > 0:
                    pi = permutation_importance(pipe, X.iloc[te], y[te], scoring="roc_auc",
                                                n_repeats=n_repeats_perm, random_state=SEED,
                                                n_jobs=-1)
                    for col, m in zip(X.columns, pi.importances_mean):
                        perm_rows.append({"cenario": scen, "modelo": name, "fold": fold,
                                          "variavel": col, "queda_auc": m})
            log.info(f"  ✔ {scen:<4} | {name}")

    folds = pd.DataFrame(fold_rows)
    metric_cols = ["acuracia", "precisao", "recall", "f1", "roc_auc", "brier"]
    summary = folds.groupby(["cenario", "modelo"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().sort_values(["cenario", "roc_auc_mean"],
                                                ascending=[True, False])

    perm = pd.DataFrame(perm_rows)
    if not perm.empty:
        perm = (perm.groupby(["cenario", "modelo", "variavel"])["queda_auc"]
                    .agg(["mean", "std"]).reset_index()
                    .rename(columns={"mean": "queda_auc_media", "std": "queda_auc_desvio"})
                    .sort_values(["cenario", "modelo", "queda_auc_media"],
                                 ascending=[True, True, False]))
    return folds, summary, perm, n_splits


def interpret_full_fit(df: pd.DataFrame, out: Path):
    """Ajusta logística e árvore em todo o dataset, só para leitura (não para avaliação)."""
    coef_rows = []
    for scen, (num, cat) in SCENARIOS.items():
        num = [c for c in num if c in df.columns]
        cat = [c for c in cat if c in df.columns]
        X, y = df[num + cat], df[TARGET].values

        lr = make_pipeline(LogisticRegression(max_iter=2000), num, cat).fit(X, y)
        names = lr.named_steps["prep"].get_feature_names_out()
        for n, c in zip(names, lr.named_steps["model"].coef_[0]):
            coef_rows.append({"cenario": scen, "variavel": n, "coef_padronizado": c,
                              "odds_ratio_por_dp": float(np.exp(c))})

        tree = make_pipeline(DecisionTreeClassifier(max_depth=4, min_samples_leaf=30,
                                                    random_state=SEED), num, cat,
                             scale=False).fit(X, y)
        txt = export_text(tree.named_steps["model"],
                          feature_names=list(tree.named_steps["prep"].get_feature_names_out()),
                          show_weights=True, decimals=2)
        (out / f"arvore_{scen}.txt").write_text(
            "# Árvore de decisão (max_depth=4) ajustada em todo o dataset.\n"
            "# Limiares nas unidades originais (equipamento em $, contagens, segundos).\n"
            "# Variáveis categóricas aparecem como indicadores 0/1 (ex.: map_name_de_nuke).\n"
            "# weights = [rounds em que CT venceu, rounds em que T venceu]\n\n"
            + txt, encoding="utf-8")

    coefs = pd.DataFrame(coef_rows)
    coefs["abs"] = coefs["coef_padronizado"].abs()
    coefs = coefs.sort_values(["cenario", "abs"], ascending=[True, False]).drop(columns="abs")
    coefs.to_csv(out / "coeficientes_logistica.csv", index=False)


# =============================================================================
# FIGURAS
# =============================================================================

BAR = "#3A6EA5"
BASE = "#9AA5B1"


def plot_metrics(summary: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scen in zip(axes, ["pre", "full"]):
        s = summary[summary["cenario"] == scen].sort_values("roc_auc_mean")
        colors = [BASE if m == "dummy" else BAR for m in s["modelo"]]
        ax.barh(s["modelo"], s["roc_auc_mean"], xerr=s["roc_auc_std"], color=colors,
                height=0.6, error_kw={"elinewidth": 1, "ecolor": "#444"})
        for i, v in enumerate(s["roc_auc_mean"]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        ax.axvline(0.5, color="#666", lw=1, ls="--")
        ax.set_xlim(0.4, 1.0)
        ax.set_title(f"Cenário {scen}", fontsize=11)
        ax.set_xlabel("ROC-AUC (média ± desvio entre folds)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Predição da vitória do lado T por round", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig_metricas.png", dpi=200)
    plt.close(fig)


def plot_importance(perm: pd.DataFrame, out: Path, top: int = 15):
    for (scen, model), p in perm.groupby(["cenario", "modelo"]):
        p = p.nlargest(top, "queda_auc_media").iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(p) + 1.2))
        ax.barh(p["variavel"], p["queda_auc_media"], xerr=p["queda_auc_desvio"],
                color=BAR, height=0.6, error_kw={"elinewidth": 1, "ecolor": "#444"})
        ax.axvline(0, color="#666", lw=1)
        ax.set_xlabel("Queda média de ROC-AUC ao embaralhar a variável (folds de teste)")
        ax.set_title(f"Importância por permutação — {model} ({scen})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"fig_importancia_{model}_{scen}.png", dpi=200)
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Classificação do resultado do round (TCC 2)")
    ap.add_argument("db", nargs="?", help="Banco SQLite (ex.: cs2.db)")
    ap.add_argument("--csv", help="Usar model_rounds.csv já gerado em vez do banco")
    ap.add_argument("--output", "-o", default="results/classificacao")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--group", choices=["series", "match"], default="series",
                    help="Unidade de agrupamento da validação cruzada")
    ap.add_argument("--perm-repeats", type=int, default=10,
                    help="Repetições da importância por permutação (0 desliga)")
    args = ap.parse_args()

    if not args.db and not args.csv:
        ap.error("informe o banco ou --csv")
    df = pd.read_csv(args.csv) if args.csv else build_round_dataset(args.db)
    group_col = "series" if args.group == "series" else "match_id"

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"🎯 {len(df):,} rounds | {df[group_col].nunique()} grupos ({args.group}) | "
             f"{args.folds} folds | vitória T = {df[TARGET].mean():.3f}")
    folds, summary, perm, k = evaluate(df, args.folds, group_col, args.perm_repeats)

    folds.to_csv(out / "metricas_por_fold.csv", index=False)
    summary.to_csv(out / "metricas_resumo.csv", index=False)
    if not perm.empty:
        perm.to_csv(out / "importancia_permutacao.csv", index=False)
        plot_importance(perm, out)
    interpret_full_fit(df, out)
    plot_metrics(summary, out)

    pd.set_option("display.width", 140)
    print("\n" + "=" * 70)
    print(f"RESUMO — {k} folds agrupados por {args.group}")
    print("=" * 70)
    cols = ["cenario", "modelo", "acuracia_mean", "f1_mean", "roc_auc_mean",
            "roc_auc_std", "brier_mean"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\n💾 Resultados em {out}/")


if __name__ == "__main__":
    main()
