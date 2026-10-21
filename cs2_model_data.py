"""
cs2_model_data.py
=================
Construção dos datasets de modelagem do TCC 2 a partir do banco SQLite
gerado pelo cs2_pipeline.py (tabelas matches, rounds, economy, kills, grenades).

Unidade de análise: o ROUND.

Dois formatos de saída:

  build_round_dataset()  → 1 linha por round, alvo t_win (o lado T venceu?)
                           Usado pelos modelos de classificação.

  build_side_dataset()   → 2 linhas por round (perspectiva T e perspectiva CT),
                           alvo win. Usado pelas regras de associação, em que
                           cada (round, lado) vira uma "cesta".

Grupos de variáveis
-------------------
  PRÉ-ROUND (conhecidas no fim do freeze time — sem vazamento do resultado):
    mapa, número do round, round de pistol, metade (1ª/2ª/prorrogação),
    placar de cada equipe ANTES do round, resultado do round anterior,
    valor de equipamento por lado, rifles, AWPs, colete, capacete, kits.

  INTRA-ROUND (ocorrem durante o round, mas antes do primeiro contato decisivo):
    lado que fez o opening kill, tempo até o opening kill, arma do opening kill,
    granadas usadas por cada lado ANTES do opening kill.

  Nunca entram como feature: total de kills do round, sobreviventes, motivo
  de vitória — esses praticamente definem o resultado (vazamento).

Decisões de implementação (ver notas técnicas do projeto)
----------------------------------------------------------
  * O demoparser2 não popula attacker_steam_id. O lado de quem matou é
    derivado do lado da VÍTIMA, obtido na tabela economy (round a round,
    o que respeita a troca de lado no intervalo). attacker_team é usado
    apenas quando estiver preenchido.
  * Kills: round_num > 0, armas de faca excluídas, lados T/CT apenas.
  * As colunas t_score/ct_score da tabela rounds acumulam vitórias por LADO,
    não por equipe. O placar por equipe é reconstruído aqui identificando
    cada equipe pelo seu elenco (steam_ids) na tabela economy.
  * economy.money_start contém round_start_equip_value (equipamento trazido
    do round anterior), não o dinheiro em caixa. Por isso a feature
    econômica principal é equip_value.

Uso:
  python cs2_model_data.py cs2.db --output datasets/
"""

import argparse
import logging
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("cs2_model_data")

TICKRATE = 64
SIDES = ("T", "CT")
KNIFE_PATTERN = r"knife|bayonet|karambit|dagger"

# Mesmos limiares do _classify_buy() do cs2_extractors.py, aplicados à média
# de equipamento por jogador da equipe.
BUY_THRESHOLDS = [(1000, "eco"), (2700, "force_buy"), (4000, "half_buy")]

GRENADE_COLS = {
    "SmokeGrenade": "smokes",
    "Flashbang": "flashes",
    "HEGrenade": "hes",
    "Molotov": "molotovs",
}


# =============================================================================
# LEITURA
# =============================================================================

def _read(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def load_tables(db_path: str) -> dict:
    """Carrega as tabelas necessárias (player_ticks não é lida — pesada demais)."""
    conn = sqlite3.connect(db_path)
    try:
        t = {
            "matches": _read(conn, "SELECT id AS match_id, demo_file, map_name FROM matches"),
            "rounds": _read(conn, """
                SELECT match_id, round_num, tick_start, tick_freeze_end, tick_end,
                       winner_team, win_reason_label
                FROM rounds
                WHERE round_num > 0 AND winner_team IN ('T', 'CT')
            """),
            "economy": _read(conn, """
                SELECT match_id, round_num, steam_id, team_label,
                       money_start, equip_value, buy_type,
                       has_rifle, has_awp, has_armor, has_helmet, has_defuser
                FROM economy
                WHERE round_num > 0 AND team_label IN ('T', 'CT')
            """),
            "kills": _read(conn, """
                SELECT match_id, round_num, tick, attacker_team, victim_team,
                       victim_steam_id, weapon
                FROM kills
                WHERE round_num > 0
            """),
            "grenades": _read(conn, """
                SELECT match_id, round_num, tick, thrower_steam_id, thrower_team,
                       grenade_type
                FROM grenades
                WHERE round_num > 0
            """),
        }
    finally:
        conn.close()

    for name, df in t.items():
        log.info(f"  {name:<9}: {len(df):>8,} linhas")
    return t


# =============================================================================
# AUXILIARES
# =============================================================================

def series_key(demo_file: str) -> str:
    """
    Chave da série (MD3/MD5) a partir do nome da demo.
    'faze-vs-vitality-m1-nuke-p1.dem' → 'faze-vs-vitality'
    Se o padrão '-m<n>' não existir, devolve o nome sem extensão
    (cada mapa vira seu próprio grupo — comportamento conservador).
    """
    stem = Path(str(demo_file)).stem.lower()
    key = re.sub(r"-m\d+.*$", "", stem)
    return key or stem


def classify_team_buy(mean_equip: float) -> str:
    if pd.isna(mean_equip):
        return "unknown"
    for limit, label in BUY_THRESHOLDS:
        if mean_equip < limit:
            return label
    return "full_buy"


def round_phase(round_num: int) -> str:
    """CS2 usa MR12: 1–12 primeira metade, 13–24 segunda, 25+ prorrogação."""
    if round_num <= 12:
        return "1a_metade"
    if round_num <= 24:
        return "2a_metade"
    return "prorrogacao"


def _side_lookup(economy: pd.DataFrame) -> pd.DataFrame:
    """(match_id, round_num, steam_id) → lado naquele round."""
    return (economy[["match_id", "round_num", "steam_id", "team_label"]]
            .dropna(subset=["steam_id"])
            .drop_duplicates(["match_id", "round_num", "steam_id"]))


# =============================================================================
# 1. ECONOMIA POR LADO (pré-round)
# =============================================================================

def economy_features(economy: pd.DataFrame) -> pd.DataFrame:
    g = economy.groupby(["match_id", "round_num", "team_label"])
    agg = g.agg(
        equip_total=("equip_value", "sum"),
        equip_mean=("equip_value", "mean"),
        equip_inicio_total=("money_start", "sum"),
        rifles=("has_rifle", "sum"),
        awps=("has_awp", "sum"),
        armors=("has_armor", "sum"),
        helmets=("has_helmet", "sum"),
        defusers=("has_defuser", "sum"),
        n_players=("steam_id", "nunique"),
    ).reset_index()
    agg["buy_team"] = agg["equip_mean"].apply(classify_team_buy)

    wide = agg.pivot(index=["match_id", "round_num"], columns="team_label")
    wide.columns = [f"{side.lower()}_{col}" for col, side in wide.columns]
    wide = wide.reset_index()

    for col in ["equip_total", "rifles", "awps"]:
        if f"t_{col}" in wide and f"ct_{col}" in wide:
            wide[f"diff_{col}"] = wide[f"t_{col}"] - wide[f"ct_{col}"]
    return wide


# =============================================================================
# 2. PLACAR POR EQUIPE (reconstruído pelo elenco)
# =============================================================================

def team_context(rounds: pd.DataFrame, economy: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica as duas equipes de cada mapa pelo elenco e calcula, para cada
    round, o placar ANTES do round e o resultado do round anterior,
    na perspectiva do lado T e do lado CT.
    """
    rows = []
    for match_id, r_match in rounds.groupby("match_id"):
        e_match = economy[economy["match_id"] == match_id]
        if e_match.empty:
            continue
        first_round = e_match["round_num"].min()
        roster_a = set(e_match[(e_match["round_num"] == first_round)
                               & (e_match["team_label"] == "T")]["steam_id"])
        if not roster_a:
            continue

        score = {"A": 0, "B": 0}
        last_winner = None
        for _, r in r_match.sort_values("round_num").iterrows():
            t_players = set(e_match[(e_match["round_num"] == r["round_num"])
                                    & (e_match["team_label"] == "T")]["steam_id"])
            if not t_players:
                continue
            # Equipe que está de T neste round = maior sobreposição de elenco
            overlap_a = len(t_players & roster_a)
            t_team = "A" if overlap_a >= len(t_players) / 2 else "B"
            ct_team = "B" if t_team == "A" else "A"

            rows.append({
                "match_id": match_id,
                "round_num": r["round_num"],
                "t_score_before": score[t_team],
                "ct_score_before": score[ct_team],
                "t_won_prev": (np.nan if last_winner is None
                               else int(last_winner == t_team)),
            })
            winner = t_team if r["winner_team"] == "T" else ct_team
            score[winner] += 1
            last_winner = winner

    ctx = pd.DataFrame(rows)
    if not ctx.empty:
        ctx["score_diff"] = ctx["t_score_before"] - ctx["ct_score_before"]
    return ctx


# =============================================================================
# 3. OPENING KILL E UTILIDADE ANTES DO PRIMEIRO CONTATO (intra-round)
# =============================================================================

def _kills_with_sides(kills: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    k = kills[~kills["weapon"].fillna("").str.lower().str.contains(KNIFE_PATTERN)].copy()

    k = k.merge(sides.rename(columns={"steam_id": "victim_steam_id",
                                      "team_label": "victim_side"}),
                on=["match_id", "round_num", "victim_steam_id"], how="left")
    # Fallback: victim_team, se a coluna tiver sido preenchida
    k["victim_side"] = k["victim_side"].fillna(
        k["victim_team"].where(k["victim_team"].isin(SIDES)))

    # Lado de quem matou: attacker_team quando existir; senão o oposto da vítima
    opposite = k["victim_side"].map({"T": "CT", "CT": "T"})
    k["killer_side"] = k["attacker_team"].where(k["attacker_team"].isin(SIDES), opposite)
    k = k.dropna(subset=["victim_side", "killer_side"])
    k["teamkill"] = (k["killer_side"] == k["victim_side"]).astype(int)
    return k


def opening_features(kills: pd.DataFrame, grenades: pd.DataFrame,
                     rounds: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    k = _kills_with_sides(kills, sides)
    k = k[k["teamkill"] == 0].sort_values("tick")
    first = k.groupby(["match_id", "round_num"]).head(1)

    op = first[["match_id", "round_num", "tick", "killer_side", "weapon"]].rename(
        columns={"tick": "opening_tick", "killer_side": "opening_side",
                 "weapon": "opening_weapon"})
    op = rounds[["match_id", "round_num", "tick_freeze_end", "tick_end"]].merge(
        op, on=["match_id", "round_num"], how="left")

    op["opening_t"] = (op["opening_side"] == "T").astype(float)
    op.loc[op["opening_side"].isna(), "opening_t"] = np.nan
    op["opening_awp"] = op["opening_weapon"].fillna("").str.lower().str.contains("awp").astype(int)
    op["time_to_opening_s"] = (op["opening_tick"] - op["tick_freeze_end"]) / TICKRATE

    # --- Granadas lançadas antes do opening kill (ou no round todo, se não houve kill)
    g = grenades.copy()
    g = g.merge(sides.rename(columns={"steam_id": "thrower_steam_id",
                                      "team_label": "thrower_side"}),
                on=["match_id", "round_num", "thrower_steam_id"], how="left")
    g["thrower_side"] = g["thrower_side"].fillna(
        g["thrower_team"].where(g["thrower_team"].isin(SIDES)))
    g = g[g["grenade_type"].isin(GRENADE_COLS) & g["thrower_side"].isin(SIDES)]

    g = g.merge(op[["match_id", "round_num", "tick_freeze_end", "opening_tick", "tick_end"]],
                on=["match_id", "round_num"], how="inner")
    limit = g["opening_tick"].fillna(g["tick_end"])
    g = g[(g["tick"] <= limit)]
    # Granadas lançadas antes do freeze_end não existem no jogo; o filtro protege
    # contra atribuição de round imprecisa
    g = g[g["tick_freeze_end"].isna() | (g["tick"] >= g["tick_freeze_end"])]

    g["gcol"] = g["grenade_type"].map(GRENADE_COLS)
    counts = (g.groupby(["match_id", "round_num", "thrower_side", "gcol"])
                .size().unstack(["thrower_side", "gcol"], fill_value=0))
    counts.columns = [f"{side.lower()}_{gcol}_pre" for side, gcol in counts.columns]
    counts = counts.reset_index()

    out = op.drop(columns=["tick_freeze_end", "tick_end", "opening_tick",
                           "opening_weapon"]).merge(counts, on=["match_id", "round_num"],
                                                    how="left")
    for side in ("t", "ct"):
        for gcol in GRENADE_COLS.values():
            c = f"{side}_{gcol}_pre"
            if c not in out:
                out[c] = 0
            out[c] = out[c].fillna(0).astype(int)
        out[f"{side}_util_pre"] = out[[f"{side}_{gc}_pre" for gc in GRENADE_COLS.values()]].sum(axis=1)
    out["diff_util_pre"] = out["t_util_pre"] - out["ct_util_pre"]
    return out


# =============================================================================
# 4. DATASET FINAL — 1 LINHA POR ROUND
# =============================================================================

PRE_ROUND_FEATURES = [
    "round_num", "is_pistol",
    "t_score_before", "ct_score_before", "t_won_prev",
    "t_equip_total", "ct_equip_total",
    "t_rifles", "ct_rifles",
    "t_awps", "ct_awps",
    "t_armors", "ct_armors", "t_helmets", "ct_helmets", "ct_defusers",
]
PRE_ROUND_CATEGORICAL = ["map_name", "phase", "t_buy_team", "ct_buy_team"]

IN_ROUND_FEATURES = [
    "opening_t", "opening_awp", "time_to_opening_s",
    "t_smokes_pre", "t_flashes_pre", "t_hes_pre", "t_molotovs_pre",
    "ct_smokes_pre", "ct_flashes_pre", "ct_hes_pre", "ct_molotovs_pre",
    "t_util_pre", "ct_util_pre",
]

# Diferenças T − CT: combinações lineares exatas de colunas já presentes nas
# listas acima. Ficam no CSV (úteis para EDA e regras de associação), mas
# não entram nos modelos, para não tornar instáveis os coeficientes da
# logística nem dividir artificialmente a importância por permutação.
DIFF_COLUMNS = ["score_diff", "diff_equip_total", "diff_rifles", "diff_util_pre"]


def build_round_dataset(db_path: str) -> pd.DataFrame:
    log.info(f"📂 Lendo {db_path}")
    t = load_tables(db_path)
    rounds, economy = t["rounds"], t["economy"]
    sides = _side_lookup(economy)

    df = rounds.merge(t["matches"], on="match_id", how="left")
    df["series"] = df["demo_file"].apply(series_key)
    df["t_win"] = (df["winner_team"] == "T").astype(int)
    df["is_pistol"] = df["round_num"].isin([1, 13]).astype(int)
    df["phase"] = df["round_num"].apply(round_phase)

    df = df.merge(economy_features(economy), on=["match_id", "round_num"], how="left")
    df = df.merge(team_context(rounds, economy), on=["match_id", "round_num"], how="left")
    df = df.merge(opening_features(t["kills"], t["grenades"], rounds, sides),
                  on=["match_id", "round_num"], how="left")

    # Rounds sem snapshot econômico de algum dos lados não servem ao modelo
    before = len(df)
    df = df.dropna(subset=["t_equip_total", "ct_equip_total"])
    if len(df) < before:
        log.warning(f"  ⚠ {before - len(df)} rounds sem economia de um dos lados foram descartados")

    # Controle de qualidade: lados com número de jogadores diferente de 5
    odd = df[(df["t_n_players"] != 5) | (df["ct_n_players"] != 5)]
    if len(odd):
        log.warning(f"  ⚠ {len(odd)} rounds com contagem de jogadores ≠ 5 em algum lado "
                    f"(verificar coaches/desconexões)")

    keep = (["match_id", "series", "map_name", "round_num", "phase", "winner_team",
             "win_reason_label", "t_win", "t_buy_team", "ct_buy_team", "opening_side"]
            + [c for c in PRE_ROUND_FEATURES + IN_ROUND_FEATURES + DIFF_COLUMNS
               if c not in ("round_num",)])
    keep = list(dict.fromkeys(c for c in keep if c in df.columns))
    df = df[keep].sort_values(["match_id", "round_num"]).reset_index(drop=True)

    log.info(f"✅ Dataset por round: {len(df):,} rounds | {df['match_id'].nunique()} mapas | "
             f"{df['series'].nunique()} séries | taxa de vitória T = {df['t_win'].mean():.3f}")
    return df


# =============================================================================
# 5. DATASET POR LADO — 2 LINHAS POR ROUND (regras de associação)
# =============================================================================

def build_side_dataset(round_df: pd.DataFrame) -> pd.DataFrame:
    """Reescreve cada round nas perspectivas T e CT ("minha equipe" x "adversário")."""
    parts = []
    for side, opp in (("T", "CT"), ("CT", "T")):
        s, o = side.lower(), opp.lower()
        p = pd.DataFrame({
            "match_id": round_df["match_id"],
            "series": round_df["series"],
            "map_name": round_df["map_name"],
            "round_num": round_df["round_num"],
            "phase": round_df["phase"],
            "is_pistol": round_df["is_pistol"],
            "side": side,
            "win": (round_df["winner_team"] == side).astype(int),
            "buy_team": round_df[f"{s}_buy_team"],
            "buy_opp": round_df[f"{o}_buy_team"],
            "equip_diff": round_df[f"{s}_equip_total"] - round_df[f"{o}_equip_total"],
            "awps": round_df[f"{s}_awps"],
            "score_diff": (round_df["score_diff"] if side == "T" else -round_df["score_diff"]),
            "won_prev": (round_df["t_won_prev"] if side == "T" else 1 - round_df["t_won_prev"]),
            "opening_won": np.where(round_df["opening_side"].isna(), np.nan,
                                    (round_df["opening_side"] == side).astype(float)),
            "smokes_pre": round_df[f"{s}_smokes_pre"],
            "flashes_pre": round_df[f"{s}_flashes_pre"],
            "molotovs_pre": round_df[f"{s}_molotovs_pre"],
            "util_pre": round_df[f"{s}_util_pre"],
            "util_pre_opp": round_df[f"{o}_util_pre"],
        })
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Gera os datasets de modelagem (TCC 2)")
    ap.add_argument("db", help="Caminho do banco SQLite (ex.: cs2.db)")
    ap.add_argument("--output", "-o", default="datasets", help="Pasta de saída")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rounds = build_round_dataset(args.db)
    rounds.to_csv(out / "model_rounds.csv", index=False)
    build_side_dataset(rounds).to_csv(out / "model_sides.csv", index=False)
    log.info(f"💾 Salvo em {out}/model_rounds.csv e {out}/model_sides.csv")


if __name__ == "__main__":
    main()
