"""
cs2_preparation.py
==================
4.3 — Preparação e Estruturação dos Dados

Etapas:
  1. Limpeza       → valores ausentes, duplicatas, inconsistências
  2. Transformação → normalização, encoding de categorias
  3. Enriquecimento → features derivadas:
       - KDR (kill/death ratio) por jogador por partida
       - Taxa de sucesso de granadas por tipo e área do mapa
       - Vantagem econômica relativa entre equipes por round
       - Sequências de ações por rodada (multi-kill, clutch)
       - KAST aproximado (Kill/Assist/Survive/Trade por round)

Uso:
    from cs2_preparation import prepare_all
    datasets = prepare_all("cs2.db")

    # ou individual
    from cs2_preparation import load_raw, clean, build_features
    raw  = load_raw("cs2.db")
    clean_data = clean(raw)
    final = build_features(clean_data)
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("cs2_prep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# =============================================================================
# 1. CARREGAMENTO
# =============================================================================

def load_raw(db_path: str) -> dict:
    """Carrega tabelas do banco em DataFrames. player_ticks é excluído — use _ticks_sample() para amostras."""
    db_path = str(db_path)
    log.info(f"📂 Carregando dados de: {db_path}")
    con = sqlite3.connect(db_path)

    tables = {
        "matches":  "SELECT * FROM matches",
        "players":  "SELECT * FROM players",
        "rounds":   "SELECT * FROM rounds",
        "kills":    "SELECT * FROM kills",
        "grenades": "SELECT * FROM grenades",
        "economy":  "SELECT * FROM economy",
    }

    data = {}
    for name, query in tables.items():
        try:
            df = pd.read_sql(query, con)
            log.info(f"  ✅ {name:<12} {len(df):>10,} linhas | {len(df.columns)} colunas")
            data[name] = df
        except Exception as e:
            log.warning(f"  ⚠ {name}: {e}")
            data[name] = pd.DataFrame()

    con.close()
    return data


def _ticks_sample(db_path: str, cols: list, match_id: int = None, n: int = 500_000) -> pd.DataFrame:
    """Sample ~n alive rows from player_ticks via rowid-modulo — no full-table sort."""
    filt = f" AND match_id = {match_id}" if match_id else ""
    col_str = ", ".join(cols)
    con = sqlite3.connect(db_path)
    try:
        total = con.execute(
            f"SELECT COUNT(*) FROM player_ticks WHERE 1=1{filt}"
        ).fetchone()[0]
        if total == 0:
            return pd.DataFrame(columns=cols)
        if total <= n:
            df = pd.read_sql(
                f"SELECT {col_str} FROM player_ticks WHERE alive = 1{filt}", con
            )
        else:
            step = max(1, total // n)
            df = pd.read_sql(
                f"SELECT {col_str} FROM player_ticks"
                f" WHERE rowid % {step} = 0 AND alive = 1{filt}"
                f" LIMIT {n}",
                con,
            )
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    return df


# =============================================================================
# 2. LIMPEZA
# =============================================================================

def _report_missing(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Loga colunas com valores ausentes."""
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        pct = (miss / len(df) * 100).round(1)
        for col in miss.index:
            log.debug(f"  [{label}] {col}: {miss[col]} nulos ({pct[col]}%)")
    return df


def clean_kills(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    original = len(df)

    # Remove duplicatas exatas
    df = df.drop_duplicates(subset=["match_id", "tick", "victim_steam_id"])

    # Remove kills sem vítima identificada
    df = df[df["victim_steam_id"].notna()]

    # Garante tipos corretos
    for col in ["headshot", "penetrated", "noscope", "thrusmoke"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Distância: outliers absurdos (> 8192 unidades = fora do mapa) → NaN
    if "distance" in df.columns:
        df.loc[df["distance"] > 8192, "distance"] = np.nan

    # Team label consistente
    for col in ["attacker_team", "victim_team"]:
        if col in df.columns:
            df[col] = df[col].str.upper().where(df[col].isin(["T","CT"]), other=np.nan)

    log.info(f"  kills:    {original:,} → {len(df):,} ({original - len(df)} removidas)")
    return _report_missing(df, "kills")


def clean_grenades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    original = len(df)

    df = df.drop_duplicates(subset=["match_id", "tick", "thrower_steam_id", "grenade_type"])
    df = df[df["grenade_type"].notna()]

    # Normaliza nome do tipo
    type_map = {
        "flashbang": "Flashbang", "hegrenade": "HEGrenade",
        "smokegrenade": "SmokeGrenade", "molotov": "Molotov",
        "inferno": "Inferno", "decoy": "Decoy",
    }
    df["grenade_type"] = df["grenade_type"].apply(
        lambda x: type_map.get(str(x).lower(), x) if pd.notna(x) else x
    )

    log.info(f"  grenades: {original:,} → {len(df):,} ({original - len(df)} removidas)")
    return df


def clean_economy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    original = len(df)

    df = df.drop_duplicates(subset=["match_id", "steam_id", "round_num"])

    # Valores impossíveis de dinheiro (CS2 máx = 16000)
    for col in ["money_start", "equip_value", "money_spent"]:
        if col in df.columns:
            df.loc[df[col] > 16000, col] = np.nan
            df.loc[df[col] < 0,    col] = np.nan

    log.info(f"  economy:  {original:,} → {len(df):,} ({original - len(df)} removidas)")
    return df


def clean_rounds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    original = len(df)

    df = df.drop_duplicates(subset=["match_id", "round_num"])

    # Remove rounds sem resultado definido (warmup, etc.)
    df = df[df["winner_team"].isin(["T", "CT"])]

    log.info(f"  rounds:   {original:,} → {len(df):,} ({original - len(df)} removidas)")
    return df


def clean_ticks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    original = len(df)

    # Remove duplicatas
    df = df.drop_duplicates(subset=["match_id", "steam_id", "tick"])

    # HP impossível
    if "hp" in df.columns:
        df.loc[(df["hp"] < 0) | (df["hp"] > 100), "hp"] = np.nan

    # Posição fora dos limites do mapa (CS2 coord range ~±8192)
    for col in ["pos_x", "pos_y", "pos_z"]:
        if col in df.columns:
            df.loc[df[col].abs() > 16384, col] = np.nan

    log.info(f"  ticks:    {original:,} → {len(df):,} ({original - len(df)} removidas)")
    return df


def clean(data: dict) -> dict:
    """Aplica limpeza em todas as tabelas. player_ticks não é incluído — amostrado em build_features()."""
    log.info("🧹 Iniciando limpeza...")
    return {
        "matches":  data["matches"].copy(),
        "players":  data["players"].copy(),
        "rounds":   clean_rounds(data["rounds"].copy()),
        "kills":    clean_kills(data["kills"].copy()),
        "grenades": clean_grenades(data["grenades"].copy()),
        "economy":  clean_economy(data["economy"].copy()),
    }


# =============================================================================
# 3. FEATURES DERIVADAS
# =============================================================================

def feature_kdr(kills: pd.DataFrame) -> pd.DataFrame:
    """
    KDR (Kill/Death Ratio) por jogador por partida.
    Também calcula: kills, deaths, assists, headshot_rate, avg_distance.
    """
    if kills.empty:
        return pd.DataFrame()

    k = kills.groupby(["match_id", "attacker_steam_id"]).agg(
        kills        = ("tick", "count"),
        headshots    = ("headshot", "sum"),
        wallbangs    = ("penetrated", "sum"),
        noscopes     = ("noscope", "sum"),
        smoke_kills  = ("thrusmoke", "sum"),
        trade_kills  = ("is_trade_kill", "sum"),
        avg_distance = ("distance", "mean"),
    ).reset_index().rename(columns={"attacker_steam_id": "steam_id"})

    d = kills.groupby(["match_id", "victim_steam_id"]).agg(
        deaths = ("tick", "count")
    ).reset_index().rename(columns={"victim_steam_id": "steam_id"})

    df = k.merge(d, on=["match_id", "steam_id"], how="outer").fillna(0)
    df["kills"]  = df["kills"].astype(int)
    df["deaths"] = df["deaths"].astype(int)

    # KDR: evita divisão por zero
    df["kdr"] = (df["kills"] / df["deaths"].replace(0, 1)).round(3)

    # Headshot rate
    df["hs_rate"] = (df["headshots"] / df["kills"].replace(0, 1)).round(3)

    log.info(f"  ✅ feature_kdr: {len(df)} linhas")
    return df


def feature_grenade_success(grenades: pd.DataFrame, kills: pd.DataFrame) -> pd.DataFrame:
    """
    Taxa de sucesso de granadas por tipo.
    Métrica: proporção de granadas que precederam uma kill no mesmo round (janela 5s / 320 ticks).
    Usa merge vetorizado em vez de iterrows.
    """
    if grenades.empty:
        return pd.DataFrame()

    WINDOW = 320

    g = grenades.reset_index().rename(columns={"index": "_gidx"})
    g["_tick_end"] = g["tick"] + WINDOW

    if not kills.empty:
        k = kills[["match_id", "round_num", "tick"]].rename(columns={"tick": "_kill_tick"})
        m = g.merge(k, on=["match_id", "round_num"], how="left")
        m["_eff"] = (
            (m["_kill_tick"] >= m["tick"]) & (m["_kill_tick"] <= m["_tick_end"])
        ).astype(int)
        eff_map = m.groupby("_gidx")["_eff"].max().fillna(0).astype(int)
        g["_effective"] = g["_gidx"].map(eff_map).fillna(0).astype(int)
    else:
        g["_effective"] = 0

    result = g.groupby(["match_id", "grenade_type"]).agg(
        total     = ("_gidx",      "count"),
        effective = ("_effective", "sum"),
    ).reset_index()
    result["success_rate"] = (result["effective"] / result["total"].replace(0, 1)).round(3)

    log.info(f"  ✅ feature_grenade_success: {len(result)} linhas")
    return result


def feature_economy_advantage(economy: pd.DataFrame) -> pd.DataFrame:
    """
    Vantagem econômica relativa entre equipes por round.
    eco_advantage = equip_value_CT - equip_value_T (positivo = CT mais rica)
    """
    if economy.empty:
        return pd.DataFrame()

    team_econ = economy.groupby(["match_id", "round_num", "team_label"]).agg(
        total_equip  = ("equip_value",  "sum"),
        avg_equip    = ("equip_value",  "mean"),
        total_money  = ("money_start",  "sum"),
        n_full_buy   = ("buy_type",     lambda x: (x == "full_buy").sum()),
        n_eco        = ("buy_type",     lambda x: (x == "eco").sum()),
        n_force      = ("buy_type",     lambda x: (x == "force_buy").sum()),
        n_awp        = ("has_awp",      "sum"),
        n_rifle      = ("has_rifle",    "sum"),
    ).reset_index()

    ct = team_econ[team_econ["team_label"] == "CT"].add_suffix("_ct").rename(
        columns={"match_id_ct": "match_id", "round_num_ct": "round_num"})
    t  = team_econ[team_econ["team_label"] == "T"].add_suffix("_t").rename(
        columns={"match_id_t": "match_id", "round_num_t": "round_num"})

    df = ct.merge(t, on=["match_id", "round_num"], how="outer")
    df["eco_advantage"] = (
        df.get("total_equip_ct", 0).fillna(0) -
        df.get("total_equip_t",  0).fillna(0)
    ).round(0).astype(int)

    # Classificação da situação econômica
    def classify_situation(row):
        ct = row.get("total_equip_ct", 0) or 0
        t  = row.get("total_equip_t", 0)  or 0
        diff = abs(ct - t)
        if diff < 2000:
            return "equilibrado"
        elif ct > t:
            return "vantagem_ct"
        else:
            return "vantagem_t"

    df["econ_situation"] = df.apply(classify_situation, axis=1)

    log.info(f"  ✅ feature_economy_advantage: {len(df)} linhas")
    return df


def feature_round_sequences(kills: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Sequências de ações por rodada:
      - multi_kill_max: maior sequência de kills consecutivas de um jogador
      - opening_kill_team: qual time fez o primeiro abate
      - clutch_situation: se algum jogador ficou 1vN
      - round_duration_ticks: duração efetiva do round
    """
    if kills.empty or rounds.empty:
        return pd.DataFrame()

    rows = []
    for _, r in rounds.iterrows():
        match_id  = r["match_id"]
        round_num = r["round_num"]

        rk = kills[
            (kills["match_id"]  == match_id) &
            (kills["round_num"] == round_num)
        ].sort_values("tick")

        # Opening kill
        opening_team = None
        if not rk.empty:
            opening_team = rk.iloc[0].get("attacker_team")

        # Multi-kill máximo por jogador
        multi_kill_max = 0
        if not rk.empty:
            ks_per_player = rk.groupby("attacker_steam_id").size()
            multi_kill_max = int(ks_per_player.max()) if not ks_per_player.empty else 0

        # Clutch: lone player (1vN, N≥2) who wins the round.
        # clutch_vs = N at the moment the 1vN starts (0 = no successful clutch).
        # A 1vN state that occurs mid-round but the lone player loses is NOT a clutch.
        winner = r.get("winner_team")
        clutch_vs = 0
        if not rk.empty and winner in ("T", "CT"):
            t_alive  = 5
            ct_alive = 5
            for _, kill in rk.iterrows():
                if kill.get("victim_team") == "T":
                    t_alive  = max(0, t_alive  - 1)
                elif kill.get("victim_team") == "CT":
                    ct_alive = max(0, ct_alive - 1)
                # Detect 1vN and check if the lone player's team wins
                if t_alive == 1 and ct_alive >= 2 and winner == "T":
                    clutch_vs = ct_alive
                    break
                if ct_alive == 1 and t_alive >= 2 and winner == "CT":
                    clutch_vs = t_alive
                    break

        # Duração do round
        duration = None
        ts = r.get("tick_start")
        te = r.get("tick_end")
        if pd.notna(ts) and pd.notna(te):
            duration = int(te) - int(ts)

        rows.append({
            "match_id":           match_id,
            "round_num":          round_num,
            "winner_team":        r.get("winner_team"),
            "win_reason_label":   r.get("win_reason_label"),
            "opening_kill_team":  opening_team,
            "opening_wins_round": int(opening_team == r.get("winner_team")) if opening_team else None,
            "kill_count":         len(rk),
            "multi_kill_max":     multi_kill_max,
            "clutch_vs":          clutch_vs,
            "duration_ticks":     duration,
            "duration_secs":      round(duration / 64, 1) if duration else None,
        })

    df = pd.DataFrame(rows)
    log.info(f"  ✅ feature_round_sequences: {len(df)} linhas")
    return df


def _compute_trade_kills(kills: pd.DataFrame, trade_window: int = 320) -> pd.DataFrame:
    """
    Adiciona is_trade_kill: 1 quando a kill ocorre dentro de `trade_window` ticks
    após qualquer outra morte no mesmo (match, round). Usa numpy por grupo — sem iterrows.
    """
    if kills.empty or "tick" not in kills.columns or "round_num" not in kills.columns:
        kills["is_trade_kill"] = 0
        return kills

    df = kills.sort_values(["match_id", "round_num", "tick"]).reset_index(drop=True)
    is_trade = np.zeros(len(df), dtype=np.int8)

    for _, grp in df.groupby(["match_id", "round_num"], sort=False):
        ticks = grp["tick"].values
        idx   = grp.index.values
        for i in range(1, len(ticks)):
            if np.any(ticks[i] - ticks[:i] <= trade_window):
                is_trade[idx[i]] = 1

    df["is_trade_kill"] = is_trade
    return df


def feature_kast(kills: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """
    KAST aproximado por jogador por partida — vetorizado.
    K = fez ao menos 1 kill no round
    A = fez ao menos 1 assist no round
    S = não morreu no round
    T = foi trade-killed (colega vingou a morte em até 5s)
    """
    if kills.empty or rounds.empty:
        return pd.DataFrame()

    rounds_long = rounds[["match_id", "round_num"]].drop_duplicates()

    # All players seen in this match (via kills or deaths)
    players_per_match = pd.concat([
        kills[["match_id", "attacker_steam_id"]].rename(columns={"attacker_steam_id": "steam_id"}),
        kills[["match_id", "victim_steam_id"]].rename(columns={"victim_steam_id": "steam_id"}),
    ]).drop_duplicates().dropna(subset=["steam_id"])

    # Cross product: every player × every round of their match
    pr = players_per_match.merge(rounds_long, on="match_id", how="left")

    key = ["match_id", "round_num", "steam_id"]

    K = (kills.groupby(["match_id", "round_num", "attacker_steam_id"]).size()
              .gt(0).astype(int).reset_index(name="K")
              .rename(columns={"attacker_steam_id": "steam_id"}))

    D = (kills.groupby(["match_id", "round_num", "victim_steam_id"]).size()
              .gt(0).astype(int).reset_index(name="D")
              .rename(columns={"victim_steam_id": "steam_id"}))

    A = pd.DataFrame(columns=key + ["A"])
    if "assister_steam_id" in kills.columns:
        A = (kills.dropna(subset=["assister_steam_id"])
                  .groupby(["match_id", "round_num", "assister_steam_id"]).size()
                  .gt(0).astype(int).reset_index(name="A")
                  .rename(columns={"assister_steam_id": "steam_id"}))

    T = pd.DataFrame(columns=key + ["T"])
    if "is_trade_kill" in kills.columns:
        T = (kills[kills["is_trade_kill"] == 1]
                  .groupby(["match_id", "round_num", "victim_steam_id"]).size()
                  .gt(0).astype(int).reset_index(name="T")
                  .rename(columns={"victim_steam_id": "steam_id"}))

    df = pr
    for flag_df, col in [(K, "K"), (D, "D"), (A, "A"), (T, "T")]:
        if not flag_df.empty:
            df = df.merge(flag_df[key + [col]], on=key, how="left")
        else:
            df[col] = np.nan

    # S = survived (no D entry for this player/round)
    df["S"] = df["D"].isna().astype(int)
    for col in ["K", "A", "T", "D"]:
        df[col] = df[col].fillna(0).astype(int)

    df["kast_round"] = ((df["K"] + df["A"] + df["S"] + df["T"]) > 0).astype(int)

    rounds_per_match = (rounds.groupby("match_id")["round_num"]
                              .count().reset_index(name="rounds_total"))

    result = (df.groupby(["match_id", "steam_id"])
                .agg(kast_rounds=("kast_round", "sum"))
                .reset_index()
                .merge(rounds_per_match, on="match_id", how="left"))

    result["kast_pct"] = (
        result["kast_rounds"] / result["rounds_total"].replace(0, 1) * 100
    ).round(1)

    log.info(f"  ✅ feature_kast: {len(result)} linhas")
    return result


def feature_positioning_zones(ticks: pd.DataFrame, n_zones: int = 10) -> pd.DataFrame:
    """
    Divide o mapa em zonas por quantil (grid adaptativo) e calcula
    frequência de ocupação por time, zona e round.
    Útil para heatmaps de posicionamento.
    """
    if ticks.empty or "pos_x" not in ticks.columns:
        return pd.DataFrame()

    df = ticks.dropna(subset=["pos_x", "pos_y"]).copy()

    # Cria zonas por quantil (grid adaptativo ao mapa)
    df["zone_x"] = pd.qcut(df["pos_x"], q=n_zones, labels=False, duplicates="drop")
    df["zone_y"] = pd.qcut(df["pos_y"], q=n_zones, labels=False, duplicates="drop")
    df["zone"]   = df["zone_x"].astype(str) + "_" + df["zone_y"].astype(str)

    zone_stats = df.groupby(["match_id", "team_label", "zone"]).agg(
        tick_count   = ("tick", "count"),
        avg_hp       = ("hp",   "mean"),
        alive_ticks  = ("alive","sum"),
    ).reset_index()

    # Frequência relativa por time
    total_per_team = zone_stats.groupby(["match_id","team_label"])["tick_count"].transform("sum")
    zone_stats["zone_freq"] = (zone_stats["tick_count"] / total_per_team).round(4)

    log.info(f"  ✅ feature_positioning_zones: {len(zone_stats)} zonas")
    return zone_stats


def _enrich_kills_teams(kills: pd.DataFrame, economy: pd.DataFrame) -> pd.DataFrame:
    """
    Fills attacker_team and victim_team from the economy table when the kills
    table has them NULL (the extractor does not write those columns).
    """
    if kills.empty or economy.empty:
        return kills

    team_map = (economy[economy["team_label"].isin(["T", "CT"])]
                [["match_id", "round_num", "steam_id", "team_label"]]
                .drop_duplicates(subset=["match_id", "round_num", "steam_id"]))

    kills = kills.merge(
        team_map.rename(columns={"steam_id": "attacker_steam_id", "team_label": "_at"}),
        on=["match_id", "round_num", "attacker_steam_id"], how="left",
    )
    kills["attacker_team"] = kills["attacker_team"].fillna(kills.pop("_at"))

    kills = kills.merge(
        team_map.rename(columns={"steam_id": "victim_steam_id", "team_label": "_vt"}),
        on=["match_id", "round_num", "victim_steam_id"], how="left",
    )
    kills["victim_team"] = kills["victim_team"].fillna(kills.pop("_vt"))

    return kills


def build_features(data: dict, db_path: str = None) -> dict:
    """Constrói todas as features derivadas e retorna dict com DataFrames."""
    log.info("⚙️  Construindo features derivadas...")

    kills    = data.get("kills",    pd.DataFrame())
    grenades = data.get("grenades", pd.DataFrame())
    economy  = data.get("economy",  pd.DataFrame())
    rounds   = data.get("rounds",   pd.DataFrame())

    kills = _compute_trade_kills(kills)
    kills = _enrich_kills_teams(kills, economy)

    # Sample ticks from DB for positioning zones — never load the full table
    ticks = pd.DataFrame()
    if db_path:
        log.info("  🔀 Amostrando player_ticks para zonas de posicionamento...")
        ticks = _ticks_sample(
            db_path,
            cols=["match_id", "steam_id", "team_label", "tick", "round_num", "pos_x", "pos_y", "hp", "alive"],
        )
        ticks = clean_ticks(ticks)

    features = {
        **data,
        "feat_kdr":           feature_kdr(kills),
        "feat_grenade":       feature_grenade_success(grenades, kills),
        "feat_economy_adv":   feature_economy_advantage(economy),
        "feat_round_seq":     feature_round_sequences(kills, rounds),
        "feat_kast":          feature_kast(kills, rounds),
        "feat_zones":         feature_positioning_zones(ticks),
    }

    # Dataset mestre por jogador/partida (join das features principais)
    kdr  = features["feat_kdr"]
    kast = features["feat_kast"]
    if not kdr.empty and not kast.empty:
        player_master = kdr.merge(kast, on=["match_id", "steam_id"], how="outer")
        # Adiciona nome do jogador
        if not data["players"].empty:
            player_master = player_master.merge(
                data["players"][["steam_id", "last_known_name"]],
                on="steam_id", how="left"
            )
        features["feat_player_master"] = player_master
        log.info(f"  ✅ feat_player_master: {len(player_master)} linhas")

    return features


# =============================================================================
# 4. NORMALIZAÇÃO
# =============================================================================

def normalize(df: pd.DataFrame, cols: list, method: str = "minmax") -> pd.DataFrame:
    """
    Normaliza colunas numéricas.
    method: 'minmax' (0-1) | 'zscore' (média 0, desvio 1)
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if method == "minmax":
            mn, mx = s.min(), s.max()
            df[f"{col}_norm"] = ((df[col] - mn) / (mx - mn)).round(4) if mx != mn else 0.0
        elif method == "zscore":
            df[f"{col}_norm"] = stats.zscore(df[col].fillna(df[col].median())).round(4)
    return df


# =============================================================================
# 5. ORQUESTRADOR PRINCIPAL
# =============================================================================

FEAT_TABLES = [
    "feat_kdr", "feat_grenade", "feat_economy_adv",
    "feat_round_seq", "feat_kast", "feat_zones", "feat_player_master",
]


def save_features(features: dict, db_path: str) -> None:
    """Persiste todos os DataFrames de feature de volta no banco SQLite."""
    log.info(f"💾 Salvando features em: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        for name in FEAT_TABLES:
            df = features.get(name)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            df.to_sql(name, con, if_exists="replace", index=False)
            log.info(f"  ✅ {name:<28} {len(df):>8,} linhas")
    finally:
        con.close()


def prepare_all(db_path: str, normalize_features: bool = True) -> dict:
    """
    Executa pipeline completo de preparação:
      load → clean → features → (normalização opcional) → salva no SQLite

    Retorna dict com todos os DataFrames prontos para EDA e modelagem.
    """
    raw      = load_raw(db_path)
    cleaned  = clean(raw)
    features = build_features(cleaned, db_path=db_path)

    if normalize_features and "feat_player_master" in features:
        pm = features["feat_player_master"]
        num_cols = ["kills", "deaths", "kdr", "hs_rate", "kast_pct", "avg_distance"]
        features["feat_player_master"] = normalize(pm, num_cols, method="minmax")

    save_features(features, db_path)
    log.info("✅ Preparação concluída.")
    _print_summary(features)
    return features


def _print_summary(features: dict):
    print("\n" + "="*55)
    print("📊 DATASETS PRONTOS")
    print("="*55)
    for name, df in features.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"  {name:<28} {len(df):>8,} linhas | {len(df.columns):>2} cols")
    print("="*55)


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "cs2.db"
    datasets = prepare_all(db)
