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
    """Carrega todas as tabelas do banco em DataFrames."""
    db_path = str(db_path)
    log.info(f"📂 Carregando dados de: {db_path}")
    con = sqlite3.connect(db_path)

    tables = {
        "matches":      "SELECT * FROM matches",
        "players":      "SELECT * FROM players",
        "rounds":       "SELECT * FROM rounds",
        "kills":        "SELECT * FROM kills",
        "grenades":     "SELECT * FROM grenades",
        "economy":      "SELECT * FROM economy",
        "ticks":        "SELECT * FROM player_ticks",
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
    for col in ["headshot", "penetrated", "noscope", "thrusmoke", "is_trade_kill"]:
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
    """Aplica limpeza em todas as tabelas."""
    log.info("🧹 Iniciando limpeza...")
    return {
        "matches":  data["matches"].copy(),
        "players":  data["players"].copy(),
        "rounds":   clean_rounds(data["rounds"].copy()),
        "kills":    clean_kills(data["kills"].copy()),
        "grenades": clean_grenades(data["grenades"].copy()),
        "economy":  clean_economy(data["economy"].copy()),
        "ticks":    clean_ticks(data["ticks"].copy()),
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
    """
    if grenades.empty:
        return pd.DataFrame()

    WINDOW = 320  # ticks (~5 segundos a 64hz)

    stats_rows = []
    for (match_id, grenade_type), grp in grenades.groupby(["match_id", "grenade_type"]):
        total = len(grp)
        effective = 0

        if not kills.empty:
            match_kills = kills[(kills["match_id"] == match_id)]
            for _, g in grp.iterrows():
                nearby_kill = match_kills[
                    (match_kills["round_num"] == g.get("round_num", -1)) &
                    (match_kills["tick"] >= g["tick"]) &
                    (match_kills["tick"] <= g["tick"] + WINDOW)
                ]
                if not nearby_kill.empty:
                    effective += 1

        stats_rows.append({
            "match_id":      match_id,
            "grenade_type":  grenade_type,
            "total":         total,
            "effective":     effective,
            "success_rate":  round(effective / total, 3) if total > 0 else 0.0,
        })

    df = pd.DataFrame(stats_rows)
    log.info(f"  ✅ feature_grenade_success: {len(df)} linhas")
    return df


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

        # Clutch: 1 jogador de um time vs 2+ do outro
        clutch = 0
        if not rk.empty:
            # Conta sobreviventes por time a cada kill
            t_alive  = 5
            ct_alive = 5
            for _, kill in rk.iterrows():
                if kill.get("victim_team") == "T":
                    t_alive  = max(0, t_alive  - 1)
                elif kill.get("victim_team") == "CT":
                    ct_alive = max(0, ct_alive - 1)
                if t_alive == 1 and ct_alive >= 2:
                    clutch = 1
                    break
                if ct_alive == 1 and t_alive >= 2:
                    clutch = 1
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
            "clutch_situation":   clutch,
            "duration_ticks":     duration,
            "duration_secs":      round(duration / 64, 1) if duration else None,
        })

    df = pd.DataFrame(rows)
    log.info(f"  ✅ feature_round_sequences: {len(df)} linhas")
    return df


def feature_kast(kills: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """
    KAST aproximado por jogador por partida.
    KAST = % de rounds onde o jogador fez Kill, Assist, Survived ou Trade.

    K = fez ao menos 1 kill no round
    A = fez ao menos 1 assist no round
    S = não morreu no round
    T = foi trade-killed (colega vingou a morte em até 5s)
    """
    if kills.empty or rounds.empty:
        return pd.DataFrame()

    all_players = pd.concat([
        kills[["match_id", "attacker_steam_id"]].rename(columns={"attacker_steam_id": "steam_id"}),
        kills[["match_id", "victim_steam_id"]].rename(columns={"victim_steam_id": "steam_id"}),
    ]).drop_duplicates().dropna()

    results = []
    for (match_id, steam_id), _ in all_players.groupby(["match_id", "steam_id"]):
        match_rounds = rounds[rounds["match_id"] == match_id]["round_num"].tolist()
        if not match_rounds:
            continue

        kast_rounds = 0
        for rnd in match_rounds:
            rk = kills[(kills["match_id"] == match_id) & (kills["round_num"] == rnd)]

            K = int(steam_id in rk["attacker_steam_id"].values)
            A = int("assister_steam_id" in rk.columns and steam_id in rk["assister_steam_id"].values)
            S = int(steam_id not in rk["victim_steam_id"].values)
            T = int(rk[rk["victim_steam_id"] == steam_id]["is_trade_kill"].sum() > 0
                    if not rk.empty and "is_trade_kill" in rk.columns else 0)

            if K or A or S or T:
                kast_rounds += 1

        results.append({
            "match_id": match_id,
            "steam_id": steam_id,
            "rounds_total": len(match_rounds),
            "kast_rounds":  kast_rounds,
            "kast_pct":     round(kast_rounds / len(match_rounds) * 100, 1),
        })

    df = pd.DataFrame(results)
    log.info(f"  ✅ feature_kast: {len(df)} linhas")
    return df


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


def build_features(data: dict) -> dict:
    """Constrói todas as features derivadas e retorna dict com DataFrames."""
    log.info("⚙️  Construindo features derivadas...")

    kills    = data.get("kills",    pd.DataFrame())
    grenades = data.get("grenades", pd.DataFrame())
    economy  = data.get("economy",  pd.DataFrame())
    rounds   = data.get("rounds",   pd.DataFrame())
    ticks    = data.get("ticks",    pd.DataFrame())

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

def prepare_all(db_path: str, normalize_features: bool = True) -> dict:
    """
    Executa pipeline completo de preparação:
      load → clean → features → (normalização opcional)

    Retorna dict com todos os DataFrames prontos para EDA e modelagem.
    """
    raw      = load_raw(db_path)
    cleaned  = clean(raw)
    features = build_features(cleaned)

    if normalize_features and "feat_player_master" in features:
        pm = features["feat_player_master"]
        num_cols = ["kills", "deaths", "kdr", "hs_rate", "kast_pct", "avg_distance"]
        features["feat_player_master"] = normalize(pm, num_cols, method="minmax")

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
