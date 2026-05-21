"""
cs2_eda.py
==========
4.4 — Análise Exploratória de Dados (EDA)

Visualizações e estatísticas cobertas:
  - Estatísticas descritivas (média, mediana, desvio padrão, quartis)
  - Correlação de Pearson e Spearman
  - Histogramas de distribuição
  - Boxplots por grupo (time, buy_type, round)
  - Scatter plots (ex: equip_value vs win_rate)
  - Heatmap de posicionamento sobreposto ao mapa
  - Heatmap de correlação entre variáveis
  - Distribuição de granadas por tipo e área
  - Vantagem econômica por round

Uso:
    python cs2_eda.py cs2.db
    python cs2_eda.py cs2.db --match-id 1 --output plots/
"""

import argparse
import logging
import sqlite3
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
log = logging.getLogger("cs2_eda")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# =============================================================================
# TEMA GLOBAL
# =============================================================================

PALETTE = {"T": "#E8543A", "CT": "#4A90D9", "neutral": "#7C8FA6"}
T_COLOR  = PALETTE["T"]
CT_COLOR = PALETTE["CT"]

def _setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi":       150,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "sans-serif",
    })

_setup_style()

# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _save(fig, output_dir: Path, name: str):
    if output_dir:
        path = output_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        log.info(f"  💾 {path}")
    plt.show()
    plt.close(fig)


def load_db(db_path: str, match_id: int = None) -> dict:
    """Carrega dados do banco, opcionalmente filtrado por match_id."""
    con = sqlite3.connect(db_path)
    filt = f" WHERE match_id = {match_id}" if match_id else ""

    data = {}
    for t in ["matches","players","rounds","kills","grenades","economy","player_ticks"]:
        try:
            data[t] = pd.read_sql(f"SELECT * FROM {t}{filt}", con)
        except Exception:
            data[t] = pd.DataFrame()
    con.close()
    return data


# =============================================================================
# 1. ESTATÍSTICAS DESCRITIVAS
# =============================================================================

def descriptive_stats(data: dict) -> pd.DataFrame:
    """
    Gera tabela de estatísticas descritivas para as principais variáveis numéricas.
    Retorna DataFrame e imprime no terminal.
    """
    log.info("📐 Estatísticas descritivas...")

    kills = data.get("kills", pd.DataFrame())
    econ  = data.get("economy", pd.DataFrame())
    ticks = data.get("player_ticks", pd.DataFrame())

    frames = []

    # Kills
    if not kills.empty:
        for col in ["distance", "headshot"]:
            if col in kills.columns:
                s = kills[col].dropna()
                frames.append({
                    "variável":    f"kills.{col}",
                    "n":           len(s),
                    "média":       round(s.mean(), 3),
                    "mediana":     round(s.median(), 3),
                    "desvio_pad":  round(s.std(), 3),
                    "q25":         round(s.quantile(0.25), 3),
                    "q75":         round(s.quantile(0.75), 3),
                    "min":         round(s.min(), 3),
                    "max":         round(s.max(), 3),
                })

    # Economia
    if not econ.empty:
        for col in ["equip_value", "money_start", "money_spent"]:
            if col in econ.columns:
                s = econ[col].dropna()
                frames.append({
                    "variável":    f"economy.{col}",
                    "n":           len(s),
                    "média":       round(s.mean(), 1),
                    "mediana":     round(s.median(), 1),
                    "desvio_pad":  round(s.std(), 1),
                    "q25":         round(s.quantile(0.25), 1),
                    "q75":         round(s.quantile(0.75), 1),
                    "min":         round(s.min(), 1),
                    "max":         round(s.max(), 1),
                })

    # Ticks
    if not ticks.empty:
        for col in ["hp", "speed"]:
            if col in ticks.columns:
                s = ticks[col].dropna()
                frames.append({
                    "variável":    f"ticks.{col}",
                    "n":           len(s),
                    "média":       round(s.mean(), 2),
                    "mediana":     round(s.median(), 2),
                    "desvio_pad":  round(s.std(), 2),
                    "q25":         round(s.quantile(0.25), 2),
                    "q75":         round(s.quantile(0.75), 2),
                    "min":         round(s.min(), 2),
                    "max":         round(s.max(), 2),
                })

    df = pd.DataFrame(frames)
    if not df.empty:
        print("\n" + df.to_string(index=False))
    return df


# =============================================================================
# 2. HISTOGRAMAS
# =============================================================================

def plot_distributions(data: dict, output_dir: Path = None):
    """Histogramas das principais variáveis numéricas."""
    log.info("📊 Gerando histogramas...")

    kills = data.get("kills", pd.DataFrame())
    econ  = data.get("economy", pd.DataFrame())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Distribuição das Variáveis Principais", fontsize=14, fontweight="bold", y=1.01)

    # 1. Distância de kills
    ax = axes[0, 0]
    if not kills.empty and "distance" in kills.columns:
        d = kills["distance"].dropna()
        ax.hist(d, bins=40, color="#4A90D9", edgecolor="white", linewidth=0.5)
        ax.axvline(d.mean(),   color="red",    linestyle="--", linewidth=1.2, label=f"Média {d.mean():.0f}")
        ax.axvline(d.median(), color="orange", linestyle=":",  linewidth=1.2, label=f"Mediana {d.median():.0f}")
        ax.set_xlabel("Distância da kill (unidades)")
        ax.set_ylabel("Frequência")
        ax.set_title("Distância das eliminações")
        ax.legend(fontsize=8)

    # 2. Headshot rate por round
    ax = axes[0, 1]
    if not kills.empty and "headshot" in kills.columns and "round_num" in kills.columns:
        hs_round = kills.groupby("round_num")["headshot"].mean()
        ax.bar(hs_round.index, hs_round.values, color="#E8543A", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Round")
        ax.set_ylabel("Taxa de headshot")
        ax.set_title("Headshot rate por round")

    # 3. Equipamento por time
    ax = axes[0, 2]
    if not econ.empty and "equip_value" in econ.columns and "team_label" in econ.columns:
        for team, color in [("CT", CT_COLOR), ("T", T_COLOR)]:
            vals = econ[econ["team_label"] == team]["equip_value"].dropna()
            ax.hist(vals, bins=30, color=color, alpha=0.6, label=team, edgecolor="white")
        ax.set_xlabel("Valor do equipamento (R$)")
        ax.set_ylabel("Frequência")
        ax.set_title("Equipamento por time")
        ax.legend()

    # 4. Kills por round
    ax = axes[1, 0]
    if not kills.empty and "round_num" in kills.columns:
        kpr = kills.groupby("round_num").size()
        ax.hist(kpr.values, bins=15, color="#6C8EBF", edgecolor="white", linewidth=0.5)
        ax.axvline(kpr.mean(), color="red", linestyle="--", linewidth=1.2,
                   label=f"Média {kpr.mean():.1f}")
        ax.set_xlabel("Kills por round")
        ax.set_ylabel("Frequência de rounds")
        ax.set_title("Distribuição de kills/round")
        ax.legend(fontsize=8)

    # 5. Tipo de compra por time
    ax = axes[1, 1]
    if not econ.empty and "buy_type" in econ.columns and "team_label" in econ.columns:
        buy_dist = econ.groupby(["team_label", "buy_type"]).size().unstack(fill_value=0)
        buy_dist.plot(kind="bar", ax=ax, color=["#7C8FA6","#E8543A","#4A90D9","#5BA85C"],
                      edgecolor="white", width=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Rounds")
        ax.set_title("Tipo de compra por time")
        ax.legend(fontsize=8, title="Tipo")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # 6. Duração dos rounds
    ax = axes[1, 2]
    rounds = data.get("rounds", pd.DataFrame())
    if not rounds.empty and "tick_start" in rounds.columns and "tick_end" in rounds.columns:
        dur = ((rounds["tick_end"] - rounds["tick_start"]) / 64).dropna()
        ax.hist(dur, bins=20, color="#5BA85C", edgecolor="white", linewidth=0.5)
        ax.axvline(dur.mean(), color="red", linestyle="--", linewidth=1.2,
                   label=f"Média {dur.mean():.0f}s")
        ax.set_xlabel("Duração (segundos)")
        ax.set_ylabel("Frequência")
        ax.set_title("Duração dos rounds")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, output_dir, "01_histogramas")


# =============================================================================
# 3. BOXPLOTS
# =============================================================================

def plot_boxplots(data: dict, output_dir: Path = None):
    """Boxplots comparativos por grupo."""
    log.info("📦 Gerando boxplots...")

    kills = data.get("kills", pd.DataFrame())
    econ  = data.get("economy", pd.DataFrame())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Boxplots Comparativos", fontsize=14, fontweight="bold")

    # 1. Distância de kill por time atacante
    ax = axes[0]
    if not kills.empty and "distance" in kills.columns and "attacker_team" in kills.columns:
        df_plot = kills[kills["attacker_team"].isin(["T","CT"])].dropna(subset=["distance"])
        order = ["T", "CT"]
        sns.boxplot(data=df_plot, x="attacker_team", y="distance",
                    order=order, palette=PALETTE, ax=ax,
                    width=0.5, fliersize=2, linewidth=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Distância (unidades)")
        ax.set_title("Distância das kills por time")

        # Teste t entre T e CT
        t_dist  = df_plot[df_plot["attacker_team"]=="T"]["distance"]
        ct_dist = df_plot[df_plot["attacker_team"]=="CT"]["distance"]
        if len(t_dist) > 1 and len(ct_dist) > 1:
            _, pval = stats.ttest_ind(t_dist, ct_dist, nan_policy="omit")
            ax.set_title(f"Distância por time\n(p={pval:.3f}{'*' if pval<0.05 else ''})")

    # 2. Equip_value por buy_type
    ax = axes[1]
    if not econ.empty and "equip_value" in econ.columns and "buy_type" in econ.columns:
        order = ["eco", "force_buy", "half_buy", "full_buy"]
        colors = ["#E87D3A", "#E8C13A", "#4A90D9", "#5BA85C"]
        sns.boxplot(data=econ, x="buy_type", y="equip_value",
                    order=order, palette=dict(zip(order, colors)),
                    ax=ax, width=0.5, fliersize=2, linewidth=0.8)
        ax.set_xlabel("Tipo de compra")
        ax.set_ylabel("Valor do equipamento (R$)")
        ax.set_title("Equipamento por tipo de compra")
        ax.set_xticklabels(["Eco","Force","Half","Full"], rotation=0)

    # 3. HP médio por time ao longo dos rounds
    ax = axes[2]
    ticks = data.get("player_ticks", pd.DataFrame())
    if not ticks.empty and "hp" in ticks.columns and "team_label" in ticks.columns and "round_num" in ticks.columns:
        sample = ticks[ticks["alive"] == 1].dropna(subset=["hp","team_label","round_num"])
        sample = sample[sample["team_label"].isin(["T","CT"])]
        # Agrega por round para não explodir o boxplot
        hp_round = sample.groupby(["round_num","team_label"])["hp"].mean().reset_index()
        sns.boxplot(data=hp_round, x="team_label", y="hp",
                    palette=PALETTE, ax=ax, width=0.5, fliersize=2, linewidth=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("HP médio por round")
        ax.set_title("HP médio (jogadores vivos) por time")

    plt.tight_layout()
    _save(fig, output_dir, "02_boxplots")


# =============================================================================
# 4. CORRELAÇÕES
# =============================================================================

def plot_correlation(data: dict, output_dir: Path = None):
    """Heatmap de correlação de Pearson e Spearman entre variáveis econômicas/performance."""
    log.info("🔗 Calculando correlações...")

    kills = data.get("kills", pd.DataFrame())
    econ  = data.get("economy", pd.DataFrame())
    rounds= data.get("rounds", pd.DataFrame())

    # Monta dataset por round com variáveis mistas
    if kills.empty or econ.empty or rounds.empty:
        log.warning("  Dados insuficientes para correlação.")
        return

    kpr = kills.groupby(["match_id","round_num"]).agg(
        kills_total = ("tick", "count"),
        hs_total    = ("headshot", "sum"),
        hs_rate     = ("headshot", "mean"),
        avg_dist    = ("distance", "mean"),
        trades      = ("is_trade_kill", "sum"),
    ).reset_index()

    econ_ct = econ[econ["team_label"]=="CT"].groupby(["match_id","round_num"]).agg(
        ct_equip = ("equip_value","sum"),
        ct_rifles= ("has_rifle","sum"),
        ct_awp   = ("has_awp","sum"),
    ).reset_index()

    econ_t = econ[econ["team_label"]=="T"].groupby(["match_id","round_num"]).agg(
        t_equip  = ("equip_value","sum"),
        t_rifles = ("has_rifle","sum"),
        t_awp    = ("has_awp","sum"),
    ).reset_index()

    merged = kpr.merge(econ_ct, on=["match_id","round_num"], how="left") \
               .merge(econ_t,  on=["match_id","round_num"], how="left") \
               .merge(rounds[["match_id","round_num","winner_team"]], on=["match_id","round_num"], how="left")

    merged["ct_wins"]    = (merged["winner_team"] == "CT").astype(int)
    merged["eco_adv_ct"] = merged["ct_equip"].fillna(0) - merged["t_equip"].fillna(0)

    numeric_cols = ["kills_total","hs_rate","avg_dist","trades",
                    "ct_equip","t_equip","eco_adv_ct","ct_rifles","t_rifles","ct_wins"]
    df_corr = merged[numeric_cols].dropna()

    if df_corr.empty or len(df_corr) < 5:
        log.warning("  Poucos dados para correlação significativa.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Análise de Correlação", fontsize=14, fontweight="bold")

    labels = {
        "kills_total": "Kills/round", "hs_rate": "HS rate",
        "avg_dist": "Dist. média", "trades": "Trade kills",
        "ct_equip": "Equip CT", "t_equip": "Equip T",
        "eco_adv_ct": "Vant. Eco CT", "ct_rifles": "Rifles CT",
        "t_rifles": "Rifles T", "ct_wins": "Vitória CT",
    }

    for ax, method in zip(axes, ["pearson", "spearman"]):
        corr = df_corr.rename(columns=labels).corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, ax=ax,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    square=True, linewidths=0.3,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"Correlação de {method.capitalize()}", fontsize=12)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    _save(fig, output_dir, "03_correlacao")

    # Log correlações mais fortes com ct_wins
    with pd.option_context("display.float_format", "{:.3f}".format):
        top = df_corr.rename(columns=labels).corr("spearman")["Vitória CT"] \
                     .drop("Vitória CT").abs().sort_values(ascending=False)
        log.info(f"  Top correlações (Spearman) com Vitória CT:\n{top.to_string()}")


# =============================================================================
# 5. SCATTER PLOTS
# =============================================================================

def plot_scatter(data: dict, output_dir: Path = None):
    """Scatter plots de relações chave."""
    log.info("🔵 Gerando scatter plots...")

    kills = data.get("kills", pd.DataFrame())
    econ  = data.get("economy", pd.DataFrame())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Scatter Plots", fontsize=14, fontweight="bold")

    # 1. Equip CT vs Equip T (colorido por vencedor)
    ax = axes[0]
    rounds = data.get("rounds", pd.DataFrame())
    if not econ.empty and not rounds.empty:
        ct = econ[econ["team_label"]=="CT"].groupby(["match_id","round_num"])["equip_value"].sum().reset_index(name="ct_equip")
        t  = econ[econ["team_label"]=="T"].groupby(["match_id","round_num"])["equip_value"].sum().reset_index(name="t_equip")
        merged = ct.merge(t, on=["match_id","round_num"]) \
                   .merge(rounds[["match_id","round_num","winner_team"]], on=["match_id","round_num"])

        merged = merged.dropna(subset=["winner_team"])
        colors = merged["winner_team"].map({"CT": CT_COLOR, "T": T_COLOR})
        ax.scatter(merged["t_equip"], merged["ct_equip"],
                   c=colors, alpha=0.6, s=40, edgecolors="white", linewidths=0.4)
        ax.plot([0, merged[["t_equip","ct_equip"]].max().max()],
                [0, merged[["t_equip","ct_equip"]].max().max()],
                color="gray", linestyle="--", linewidth=0.8, label="Paridade")
        ax.set_xlabel("Equipamento T (R$)")
        ax.set_ylabel("Equipamento CT (R$)")
        ax.set_title("Equip CT vs T por round")
        patches = [mpatches.Patch(color=CT_COLOR, label="CT vence"),
                   mpatches.Patch(color=T_COLOR,  label="T vence")]
        ax.legend(handles=patches, fontsize=8)

    # 2. Distância vs Headshot (por time)
    ax = axes[1]
    if not kills.empty and "distance" in kills.columns and "headshot" in kills.columns:
        sample = kills[kills["attacker_team"].isin(["T","CT"])].dropna(subset=["distance","headshot"])
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        colors = sample["attacker_team"].map(PALETTE)
        jitter = np.random.uniform(-0.02, 0.02, len(sample))
        ax.scatter(sample["distance"], sample["headshot"] + jitter,
                   c=colors, alpha=0.3, s=15, edgecolors="none")
        ax.set_xlabel("Distância (unidades)")
        ax.set_ylabel("Headshot (1=sim, 0=não)")
        ax.set_title("Distância vs Headshot")
        patches = [mpatches.Patch(color=PALETTE["CT"], label="CT"),
                   mpatches.Patch(color=PALETTE["T"],  label="T")]
        ax.legend(handles=patches, fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Não", "Sim"])

    plt.tight_layout()
    _save(fig, output_dir, "04_scatter")


# =============================================================================
# 6. GRANADAS
# =============================================================================

def plot_grenades(data: dict, output_dir: Path = None):
    """Distribuição de granadas por tipo, time e efetividade."""
    log.info("💣 Gerando análise de granadas...")

    grenades = data.get("grenades", pd.DataFrame())
    if grenades.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Análise de Granadas", fontsize=14, fontweight="bold")

    grenade_colors = {
        "Flashbang": "#F5C242", "HEGrenade": "#E8543A",
        "SmokeGrenade": "#7C8FA6", "Molotov": "#D4541B",
        "Inferno": "#D4541B", "Decoy": "#5BA85C",
    }

    # 1. Contagem por tipo
    ax = axes[0]
    tipo_ct = grenades["grenade_type"].value_counts()
    colors  = [grenade_colors.get(t, "#999") for t in tipo_ct.index]
    bars = ax.barh(tipo_ct.index, tipo_ct.values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, tipo_ct.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9)
    ax.set_xlabel("Quantidade")
    ax.set_title("Granadas por tipo")

    # 2. Granadas por time
    ax = axes[1]
    if "thrower_team" in grenades.columns:
        grp = grenades.groupby(["thrower_team","grenade_type"]).size().unstack(fill_value=0)
        grp = grp.reindex(columns=[c for c in grenade_colors if c in grp.columns])
        colors_stacked = [grenade_colors.get(c,"#999") for c in grp.columns]
        grp.plot(kind="bar", ax=ax, color=colors_stacked,
                 edgecolor="white", linewidth=0.3, width=0.6)
        ax.set_xlabel("Time")
        ax.set_ylabel("Quantidade")
        ax.set_title("Granadas por time e tipo")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(fontsize=7, title="Tipo", loc="upper right")

    # 3. Granadas por round (tendência)
    ax = axes[2]
    if "round_num" in grenades.columns:
        grn_round = grenades.groupby(["round_num","grenade_type"]).size().unstack(fill_value=0)
        for col, color in grenade_colors.items():
            if col in grn_round.columns:
                ax.plot(grn_round.index, grn_round[col],
                        label=col, color=color, linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Round")
        ax.set_ylabel("Quantidade")
        ax.set_title("Granadas por round")
        ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, output_dir, "05_granadas")


# =============================================================================
# 7. HEATMAP DE POSICIONAMENTO
# =============================================================================

def plot_positioning_heatmap(data: dict, team: str = "CT",
                             round_num: int = None, output_dir: Path = None):
    """
    Heatmap de posicionamento 2D sobreposto ao mapa.
    Separa por time, opcionalmente filtra por round.
    """
    log.info(f"🗺  Gerando heatmap de posicionamento ({team})...")

    ticks = data.get("player_ticks", pd.DataFrame())
    if ticks.empty or "pos_x" not in ticks.columns:
        log.warning("  Sem dados de posicionamento.")
        return

    df = ticks.copy()
    df = df[df["team_label"] == team]
    df = df[df["alive"] == 1]

    if round_num:
        df = df[df["round_num"] == round_num]

    df = df.dropna(subset=["pos_x","pos_y"])
    if df.empty:
        log.warning("  Nenhum dado após filtros.")
        return

    # Amostra para performance
    if len(df) > 200_000:
        df = df.sample(200_000, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_suf = f" — Round {round_num}" if round_num else ""
    fig.suptitle(f"Heatmap de Posicionamento — Time {team}{title_suf}",
                 fontsize=13, fontweight="bold")

    color = CT_COLOR if team == "CT" else T_COLOR

    for ax, stat in zip(axes, ["count", "mean"]):
        h = ax.hist2d(df["pos_x"], df["pos_y"],
                      bins=60,
                      cmap="hot_r" if team=="T" else "Blues",
                      density=(stat=="mean"))
        fig.colorbar(h[3], ax=ax,
                     label="Densidade" if stat=="mean" else "Contagem de ticks")
        ax.set_xlabel("Posição X")
        ax.set_ylabel("Posição Y")
        ax.set_title("Densidade relativa" if stat=="mean" else "Frequência absoluta")
        ax.set_aspect("equal")

    plt.tight_layout()
    fname = f"06_heatmap_{team.lower()}" + (f"_r{round_num}" if round_num else "")
    _save(fig, output_dir, fname)


# =============================================================================
# 8. ECONOMIA POR ROUND
# =============================================================================

def plot_economy_timeline(data: dict, output_dir: Path = None):
    """Evolução econômica de CT e T ao longo dos rounds."""
    log.info("💰 Gerando timeline econômica...")

    econ   = data.get("economy", pd.DataFrame())
    rounds = data.get("rounds", pd.DataFrame())
    if econ.empty or rounds.empty:
        return

    econ_round = econ.groupby(["round_num","team_label"])["equip_value"].sum().unstack(fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Evolução Econômica por Round", fontsize=13, fontweight="bold")

    # 1. Equip value absoluto
    ax = axes[0]
    if "CT" in econ_round.columns:
        ax.fill_between(econ_round.index, econ_round["CT"], alpha=0.3, color=CT_COLOR)
        ax.plot(econ_round.index, econ_round["CT"], color=CT_COLOR, linewidth=2, label="CT")
    if "T" in econ_round.columns:
        ax.fill_between(econ_round.index, econ_round["T"], alpha=0.3, color=T_COLOR)
        ax.plot(econ_round.index, econ_round["T"], color=T_COLOR, linewidth=2, label="T")
    ax.set_ylabel("Equipamento total (R$)")
    ax.set_title("Valor de equipamento por time")
    ax.legend()

    # Marca troca de lado (round 13 em partidas de 24)
    max_round = econ_round.index.max()
    if max_round >= 13:
        ax.axvline(12.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(12.7, ax.get_ylim()[1]*0.95, "Troca de lado", fontsize=8, color="gray")

    # 2. Vantagem relativa CT - T
    ax = axes[1]
    if "CT" in econ_round.columns and "T" in econ_round.columns:
        adv = econ_round["CT"] - econ_round["T"]
        colors = [CT_COLOR if v >= 0 else T_COLOR for v in adv]
        ax.bar(adv.index, adv.values, color=colors, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Vantagem CT (R$)")
        ax.set_title("Vantagem econômica relativa (CT − T)")
        ax.set_xlabel("Round")

    # Overlay com vencedor
    for _, r in rounds.iterrows():
        rn = r.get("round_num")
        winner = r.get("winner_team")
        if rn and winner:
            axes[0].axvspan(rn-0.5, rn+0.5,
                            alpha=0.07,
                            color=CT_COLOR if winner=="CT" else T_COLOR)

    plt.tight_layout()
    _save(fig, output_dir, "07_economia_timeline")


# =============================================================================
# 9. WIN RATE POR CONTEXTO
# =============================================================================

def plot_win_rates(data: dict, output_dir: Path = None):
    """Win rate por opening kill, clutch, buy_type e eco_advantage."""
    log.info("🏆 Gerando win rates...")

    kills  = data.get("kills", pd.DataFrame())
    rounds = data.get("rounds", pd.DataFrame())
    econ   = data.get("economy", pd.DataFrame())

    if rounds.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Win Rate por Contexto", fontsize=13, fontweight="bold")

    # 1. Opening kill → win rate
    ax = axes[0]
    if not kills.empty and "round_num" in kills.columns:
        opening = kills.sort_values("tick").groupby(["match_id","round_num"]).first() \
                       .reset_index()[["match_id","round_num","attacker_team"]]
        opening = opening.merge(rounds[["match_id","round_num","winner_team"]],
                                on=["match_id","round_num"])
        opening["won_opening"] = (opening["attacker_team"] == opening["winner_team"]).astype(int)
        wr = opening.groupby("attacker_team")["won_opening"].mean()
        bars = ax.bar(wr.index, wr.values * 100,
                      color=[CT_COLOR if t=="CT" else T_COLOR for t in wr.index],
                      edgecolor="white", width=0.4)
        for bar, val in zip(bars, wr.values*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{val:.0f}%", ha="center", fontsize=10, fontweight="500")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Win rate (%)")
        ax.set_title("Win rate após opening kill")

    # 2. Win rate por buy_type
    ax = axes[1]
    if not econ.empty:
        merged = econ.merge(rounds[["match_id","round_num","winner_team"]],
                            on=["match_id","round_num"], how="left")
        merged["won"] = (merged["team_label"] == merged["winner_team"]).astype(int)
        wr_buy = merged.groupby("buy_type")["won"].mean() * 100
        order  = ["eco","force_buy","half_buy","full_buy"]
        wr_buy = wr_buy.reindex([o for o in order if o in wr_buy.index])
        colors = ["#E87D3A","#E8C13A","#4A90D9","#5BA85C"]
        bars   = ax.bar(range(len(wr_buy)), wr_buy.values,
                        color=colors[:len(wr_buy)], edgecolor="white", width=0.5)
        for bar, val in zip(bars, wr_buy.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{val:.0f}%", ha="center", fontsize=10, fontweight="500")
        ax.set_xticks(range(len(wr_buy)))
        ax.set_xticklabels(["Eco","Force","Half","Full"], rotation=0)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Win rate (%)")
        ax.set_title("Win rate por tipo de compra")

    # 3. Win rate por win_reason
    ax = axes[2]
    if "win_reason_label" in rounds.columns:
        wr = rounds.dropna(subset=["winner_team"]).groupby(["win_reason_label","winner_team"]).size().unstack(fill_value=0)
        if not wr.empty and not wr.select_dtypes("number").empty:
            wr.plot(kind="barh", ax=ax,
                    color=[CT_COLOR, T_COLOR], edgecolor="white", linewidth=0.3)
            ax.set_xlabel("Rounds")
            ax.set_title("Rounds por tipo de vitória")
            ax.legend(title="Time", fontsize=8)
        else:
            ax.set_title("Rounds por tipo de vitória\n(sem dados de winner_team)")
            ax.axis("off")

    plt.tight_layout()
    _save(fig, output_dir, "08_win_rates")


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

def run_eda(db_path: str, match_id: int = None, output_dir: str = None):
    """Executa toda a EDA e salva os plots."""
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    log.info(f"🔍 Iniciando EDA: {db_path}" + (f" | match_id={match_id}" if match_id else ""))
    data = load_db(db_path, match_id)

    descriptive_stats(data)
    plot_distributions(data, out)
    plot_boxplots(data, out)
    plot_correlation(data, out)
    plot_scatter(data, out)
    plot_grenades(data, out)
    plot_positioning_heatmap(data, team="CT", output_dir=out)
    plot_positioning_heatmap(data, team="T",  output_dir=out)
    plot_economy_timeline(data, out)
    plot_win_rates(data, out)

    log.info("✅ EDA concluída.")
    if out:
        plots = list(out.glob("*.png"))
        log.info(f"  📁 {len(plots)} plots salvos em: {out}")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="CS2 EDA — Análise Exploratória")
    ap.add_argument("db",                           help="Caminho para cs2.db")
    ap.add_argument("--match-id",  type=int,        help="Filtrar por match_id")
    ap.add_argument("--output",    default="plots", help="Diretório de saída dos plots")
    args = ap.parse_args()

    run_eda(args.db, match_id=args.match_id, output_dir=args.output)


if __name__ == "__main__":
    main()
