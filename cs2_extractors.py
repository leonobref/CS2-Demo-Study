"""
cs2_extractors.py
=================
Módulo de extração de eventos de demos CS2 via demoparser2.

Cada função recebe um DemoParser já instanciado e retorna
um DataFrame pronto para inserção no banco de dados.

Eventos cobertos:
  - extract_kills()       → eliminações com contexto tático
  - extract_grenades()    → granadas (trajetória + detonação)
  - extract_rounds()      → resultado de cada round
  - extract_economy()     → snapshot econômico por jogador por round
  - extract_ticks()       → posicionamento por tick (tabela pesada)
"""

import math
import time
import logging
from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

log = logging.getLogger("cs2_extractors")

# =============================================================================
# CONSTANTES
# =============================================================================

# Código de vitória do CS2 → label legível
WIN_REASON_MAP = {
    1:  "ct_killed_all",
    7:  "bomb_defused",
    8:  "bomb_exploded",
    9:  "t_killed_all",
    12: "ct_rescued_hostage",
    13: "t_prevented_rescue",
    17: "round_draw",
}

# Tipos de granada normalizados
GRENADE_TYPE_MAP = {
    "flashbang_detonate":       "Flashbang",
    "hegrenade_detonate":       "HEGrenade",
    "smokegrenade_detonate":    "SmokeGrenade",
    "molotov_detonate":         "Molotov",
    "inferno_startburn":        "Inferno",
    "decoy_started":            "Decoy",
}

# Rifles (para classificar buy_type)
RIFLES = {
    "weapon_ak47", "weapon_m4a1", "weapon_m4a1_silencer",
    "weapon_awp",  "weapon_sg556", "weapon_aug",
    "weapon_famas","weapon_galil", "weapon_ssg08",
}

PLAYER_PROPS = [
    "X", "Y", "Z", "yaw", "pitch",
    "health", "armor_value", "is_alive", "team_num",
    "active_weapon", "has_defuser", "has_helmet",
    "flash_duration", "ducking", "in_crouch",
    "velocity_X", "velocity_Y", "velocity_Z",
    "is_scoped", "spotted", "balance",
]

ECONOMY_PROPS = [
    "balance", "current_equip_value",
    "round_start_equip_value", "has_helmet",
    "has_defuser", "active_weapon", "team_num",
]

# =============================================================================
# UTILITÁRIOS
# =============================================================================

def _safe_event(parser: DemoParser, event: str, player: list = None, other: list = None) -> pd.DataFrame:
    """parse_event com fallback silencioso."""
    try:
        kw = {}
        if player:
            kw["player"] = player
        if other:
            kw["other"] = other
        df = parser.parse_event(event, **kw)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as e:
        log.debug(f"Evento '{event}' não disponível: {e}")
        return pd.DataFrame()


def _safe_ticks(parser: DemoParser, props: list) -> pd.DataFrame:
    """parse_ticks removendo props inválidas automaticamente."""
    available = props.copy()
    while available:
        try:
            return parser.parse_ticks(available)
        except Exception as e:
            msg = str(e).lower()
            removed = False
            for p in available[:]:
                if p.lower() in msg:
                    log.debug(f"Prop '{p}' removida: {e}")
                    available.remove(p)
                    removed = True
                    break
            if not removed:
                if len(available) > 3:
                    available.pop()
                else:
                    raise
    raise RuntimeError("Nenhuma prop válida.")


def _dist3d(x1, y1, z1, x2, y2, z2) -> float:
    try:
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    except Exception:
        return None


def _classify_buy(equip_value: float) -> str:
    """Classifica o tipo de compra pelo valor do equipamento."""
    if pd.isna(equip_value) or equip_value < 0:
        return "unknown"
    if equip_value < 1000:
        return "eco"
    elif equip_value < 2700:
        return "force_buy"
    elif equip_value < 4000:
        return "half_buy"
    else:
        return "full_buy"


def _assign_rounds(df: pd.DataFrame, rounds_df: pd.DataFrame, tick_col: str = "tick") -> pd.DataFrame:
    """Associa cada linha ao round correspondente pelo tick."""
    df = df.copy()
    df["round_num"] = -1
    if rounds_df.empty or tick_col not in df.columns:
        return df
    for _, r in rounds_df.iterrows():
        ts, te, rn = r["tick_start"], r["tick_end"], r["round_num"]
        if pd.isna(ts) or pd.isna(te) or pd.isna(rn):
            continue
        mask = (df[tick_col] >= ts) & (df[tick_col] <= te)
        df.loc[mask, "round_num"] = int(rn)
    return df


# =============================================================================
# 1. ROUNDS
# =============================================================================

def extract_rounds(parser: DemoParser) -> pd.DataFrame:
    """
    Extrai resultado de cada round.

    Colunas retornadas:
        round_num, tick_start, tick_freeze_end, tick_end,
        winner_team, win_reason, win_reason_label, t_score, ct_score
    """
    log.info("  → Extraindo rounds...")

    starts    = _safe_event(parser, "round_start")
    ends      = _safe_event(parser, "round_officially_ended")
    freeze    = _safe_event(parser, "round_freeze_end")
    round_end = _safe_event(parser, "round_end",
                    other=["winner", "reason", "legacy", "player_count"])

    if starts.empty or ends.empty:
        log.warning("  ⚠ Eventos de round não encontrados.")
        return pd.DataFrame()

    n = min(len(starts), len(ends))
    rows = []
    t_score = ct_score = 0

    for i in range(n):
        raw_start = starts.iloc[i]["tick"]
        raw_end   = ends.iloc[i]["tick"]
        if pd.isna(raw_start) or pd.isna(raw_end):
            continue
        tick_start = int(raw_start)
        tick_end   = int(raw_end)

        # Freeze end deste round
        tick_freeze = None
        if not freeze.empty:
            candidates = freeze[(freeze["tick"] >= tick_start) & (freeze["tick"] <= tick_end)]
            if not candidates.empty:
                raw_freeze = candidates.iloc[0]["tick"]
                tick_freeze = int(raw_freeze) if pd.notna(raw_freeze) else None

        # Resultado do round
        winner_team = win_reason_raw = win_reason_label = None
        if not round_end.empty:
            candidates = round_end[
                (round_end["tick"] >= tick_start) & (round_end["tick"] <= tick_end)
            ]
            if not candidates.empty:
                row_re = candidates.iloc[0]

                reason_val = row_re.get("reason", None)
                if reason_val is not None and pd.notna(reason_val):
                    try:
                        win_reason_raw   = int(reason_val)
                        win_reason_label = WIN_REASON_MAP.get(win_reason_raw, f"reason_{win_reason_raw}")
                    except (ValueError, TypeError):
                        win_reason_raw   = None
                        win_reason_label = str(reason_val)

                winner_raw = row_re.get("winner", None)
                if winner_raw is not None and pd.notna(winner_raw):
                    w = str(winner_raw).strip()
                    if w in ("T", "2"):
                        winner_team = "T"
                        t_score += 1
                    elif w in ("CT", "3"):
                        winner_team = "CT"
                        ct_score += 1

        rows.append({
            "round_num":        i + 1,
            "tick_start":       tick_start,
            "tick_freeze_end":  tick_freeze,
            "tick_end":         tick_end,
            "winner_team":      winner_team,
            "win_reason":       win_reason_raw,
            "win_reason_label": win_reason_label,
            "t_score":          t_score,
            "ct_score":         ct_score,
        })

    df = pd.DataFrame(rows)
    log.info(f"  ✅ {len(df)} rounds extraídos")
    return df


# =============================================================================
# 2. KILLS
# =============================================================================

def extract_kills(parser: DemoParser, rounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai todas as eliminações com contexto tático.

    Colunas retornadas:
        tick, round_num,
        attacker_steam_id, attacker_name, attacker_team, attacker_x/y/z,
        victim_steam_id, victim_name, victim_team, victim_x/y/z,
        assister_steam_id, assister_name,
        weapon, headshot, penetrated, noscope, thrusmoke, distance,
        is_trade_kill, bomb_planted
    """
    log.info("  → Extraindo kills...")

    KILL_PLAYER = ["X", "Y", "Z", "team_num", "health", "has_helmet", "armor_value"]
    KILL_OTHER  = ["weapon", "headshot", "penetrated", "noscope", "thrusmoke",
                   "assistedflash", "assistblind"]

    df = _safe_event(parser, "player_death", player=KILL_PLAYER, other=KILL_OTHER)

    if df.empty:
        log.warning("  ⚠ Nenhum evento player_death encontrado.")
        return pd.DataFrame()

    # Renomear colunas geradas pelo demoparser2
    rename = {}
    for col in df.columns:
        if col.endswith("_X") and "attacker" not in col.lower() and "victim" not in col.lower():
            # posições podem vir como X, Y, Z por jogador
            pass

    # Fix: steam_id pode vir como 'attacker_steamid' ou 'user_steamid'
    for src, dst in [
        ("attacker_steamid",  "attacker_steam_id"),
        ("user_steamid",      "victim_steam_id"),
        ("assister_steamid",  "assister_steam_id"),
    ]:
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # demoparser2 retorna colunas prefixadas com "attacker_" e "user_" (vítima)
    col_map = {}
    for col in df.columns:
        if col.startswith("user_"):
            col_map[col] = col.replace("user_", "victim_", 1)
        elif col.startswith("attacker_"):
            pass  # já está correto
        elif col.startswith("assister_"):
            pass

    df = df.rename(columns=col_map)

    # Colunas de posição (podem ter sufixos diferentes dependendo da versão)
    for prefix, axis in [("attacker", ["X","Y","Z"]), ("victim", ["X","Y","Z"])]:
        for ax in axis:
            src = f"{prefix}_{ax}"
            dst = f"{prefix}_{ax.lower()}"
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

    # Distância 3D attacker → victim
    if all(c in df.columns for c in ["attacker_X","attacker_Y","attacker_Z",
                                      "victim_X","victim_Y","victim_Z"]):
        df["distance"] = df.apply(lambda r: _dist3d(
            r.get("attacker_X"), r.get("attacker_Y"), r.get("attacker_Z"),
            r.get("victim_X"),   r.get("victim_Y"),   r.get("victim_Z"),
        ), axis=1)
    else:
        df["distance"] = None

    # Associa rounds
    df = _assign_rounds(df, rounds_df)

    # Trade kill: morte dentro de 5 segundos (~320 ticks a 64hz) após outra morte
    # no mesmo round
    if "tick" in df.columns:
        df = df.sort_values("tick").reset_index(drop=True)
        trade_window = 320
        df["is_trade_kill"] = 0
        for idx, row in df.iterrows():
            if idx == 0:
                continue
            prev = df[
                (df.index < idx) &
                (df["round_num"] == row["round_num"]) &
                (df["tick"] >= row["tick"] - trade_window)
            ]
            if not prev.empty:
                df.at[idx, "is_trade_kill"] = 1

    # Bomb planted (simplificado)
    df["bomb_planted"] = 0

    # Normaliza tipos
    for col in ["headshot", "penetrated", "noscope", "thrusmoke"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    log.info(f"  ✅ {len(df)} kills extraídas")
    return df


# =============================================================================
# 3. GRENADES
# =============================================================================

def extract_grenades(parser: DemoParser, rounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai uso de granadas: flashbangs, HEs, smokes, molotovs, decoys.

    Colunas retornadas:
        tick, round_num,
        thrower_steam_id, thrower_name, thrower_team,
        grenade_type, det_x, det_y, det_z,
        duration_ticks, players_flashed
    """
    log.info("  → Extraindo granadas...")

    GRENADE_EVENTS = [
        "flashbang_detonate",
        "hegrenade_detonate",
        "smokegrenade_detonate",
        "molotov_detonate",
        "inferno_startburn",
        "decoy_started",
    ]

    THROWER_PROPS = ["X", "Y", "Z", "team_num"]

    all_dfs = []

    for event_name in GRENADE_EVENTS:
        df_ev = _safe_event(parser, event_name, player=THROWER_PROPS)
        if df_ev.empty:
            continue

        df_ev["grenade_type"] = GRENADE_TYPE_MAP.get(event_name, event_name)

        # Posição de detonação (pode vir como x/y/z ou X/Y/Z dependendo da versão)
        for ax_up, ax_lo in [("X","det_x"), ("Y","det_y"), ("Z","det_z")]:
            if ax_up in df_ev.columns:
                df_ev[ax_lo] = df_ev[ax_up]
            elif f"user_{ax_up}" in df_ev.columns:
                df_ev[ax_lo] = df_ev[f"user_{ax_up}"]

        # Renomear thrower
        for col in df_ev.columns:
            if col.startswith("user_steamid"):
                df_ev.rename(columns={col: "thrower_steam_id"}, inplace=True)
            elif col.startswith("user_name"):
                df_ev.rename(columns={col: "thrower_name"}, inplace=True)
            elif col.startswith("user_team_num"):
                df_ev.rename(columns={col: "thrower_team_num"}, inplace=True)

        if "thrower_steam_id" not in df_ev.columns and "attacker_steamid" in df_ev.columns:
            df_ev.rename(columns={"attacker_steamid": "thrower_steam_id"}, inplace=True)

        df_ev["duration_ticks"] = None
        df_ev["players_flashed"] = None

        all_dfs.append(df_ev)

    if not all_dfs:
        log.warning("  ⚠ Nenhum evento de granada encontrado.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Duração de smokes: ticks entre smokegrenade_detonate e smokegrenade_expired
    smoke_start = _safe_event(parser, "smokegrenade_detonate")
    smoke_end   = _safe_event(parser, "smokegrenade_expired")
    if not smoke_start.empty and not smoke_end.empty:
        smoke_mask = df["grenade_type"] == "SmokeGrenade"
        smoke_rows = df[smoke_mask].copy().reset_index()
        for i, (orig_idx, s_row) in enumerate(zip(smoke_rows["index"], smoke_rows.itertuples())):
            if i < len(smoke_end):
                dur = int(smoke_end.iloc[i]["tick"]) - int(smoke_start.iloc[i]["tick"])
                df.at[orig_idx, "duration_ticks"] = dur

    # Duração de infernos
    inf_start = _safe_event(parser, "inferno_startburn")
    inf_end   = _safe_event(parser, "inferno_extinguish")
    if not inf_start.empty and not inf_end.empty:
        inf_mask = df["grenade_type"] == "Inferno"
        inf_rows = df[inf_mask].copy().reset_index()
        for i, (orig_idx, _) in enumerate(zip(inf_rows["index"], inf_rows.itertuples())):
            if i < len(inf_end):
                dur = int(inf_end.iloc[i]["tick"]) - int(inf_start.iloc[i]["tick"])
                df.at[orig_idx, "duration_ticks"] = dur

    # Team label do lançador
    if "thrower_team_num" in df.columns:
        df["thrower_team"] = df["thrower_team_num"].map({2: "T", 3: "CT"})

    # Associa rounds
    df = _assign_rounds(df, rounds_df)

    log.info(f"  ✅ {len(df)} eventos de granada extraídos")
    return df


# =============================================================================
# 4. ECONOMY
# =============================================================================

def extract_economy(parser: DemoParser, rounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Snapshot econômico por jogador por round (início de cada round).

    Captura o estado financeiro logo após o freeze_end de cada round.
    Colunas retornadas:
        steam_id, round_num, team_label,
        money_start, equip_value, money_spent, buy_type,
        has_rifle, has_awp, has_armor, has_helmet, has_defuser
    """
    log.info("  → Extraindo economia...")

    if rounds_df.empty:
        log.warning("  ⚠ Sem rounds para extrair economia.")
        return pd.DataFrame()

    try:
        df_ticks = _safe_ticks(parser, ECONOMY_PROPS)
    except Exception as e:
        log.warning(f"  ⚠ Erro ao extrair props de economia: {e}")
        return pd.DataFrame()

    # Renomear colunas padrão
    rename = {
        "steamid":                  "steam_id",
        "name":                     "player_name",
        "team_num":                 "team",
        "balance":                  "money",
        "current_equip_value":      "equip_value",
        "round_start_equip_value":  "equip_start",
        "has_helmet":               "helmet",
        "has_defuser":              "defuser",
        "active_weapon":            "weapon",
    }
    df_ticks = df_ticks.rename(columns={k: v for k, v in rename.items() if k in df_ticks.columns})

    rows = []
    for _, r in rounds_df.iterrows():
        # Snapshot no tick de freeze_end (momento da primeira ação)
        tick_fe  = r.get("tick_freeze_end")
        tick_st  = r.get("tick_start")
        ref_tick = tick_fe if pd.notna(tick_fe) else tick_st
        if pd.isna(ref_tick):
            continue

        # Pega o tick mais próximo após freeze_end
        candidates = df_ticks[df_ticks["tick"] >= int(ref_tick)]
        if candidates.empty:
            continue

        snapshot_tick = candidates["tick"].min()
        snap = df_ticks[df_ticks["tick"] == snapshot_tick]

        for _, p in snap.iterrows():
            steam_id   = p.get("steam_id")
            money      = p.get("money")
            equip_val  = p.get("equip_value")
            equip_start= p.get("equip_start")
            weapon     = str(p.get("weapon", "")).lower()
            team       = p.get("team")
            team_label = {2: "T", 3: "CT"}.get(int(team), "Unknown") if pd.notna(team) else "Unknown"

            spent = None
            if pd.notna(money) and pd.notna(equip_start):
                spent = max(0, int(equip_start) - int(money))

            rows.append({
                "steam_id":     steam_id,
                "round_num":    int(r["round_num"]) if pd.notna(r["round_num"]) else None,
                "team_label":   team_label,
                "money_start":  int(equip_start) if pd.notna(equip_start) else None,
                "equip_value":  int(equip_val)   if pd.notna(equip_val)   else None,
                "money_spent":  spent,
                "buy_type":     _classify_buy(equip_val),
                "has_rifle":    int(weapon in RIFLES),
                "has_awp":      int("awp" in weapon),
                "has_armor":    int(int(p.get("armor", 0) or 0) > 0) if "armor" in p else None,
                "has_helmet":   int(bool(p.get("helmet", False))),
                "has_defuser":  int(bool(p.get("defuser", False))),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["steam_id", "round_num"])

    log.info(f"  ✅ {len(df)} snapshots econômicos extraídos")
    return df


# =============================================================================
# 5. PLAYER TICKS (posicionamento)
# =============================================================================

def extract_ticks(parser: DemoParser, rounds_df: pd.DataFrame,
                  tick_rate: int = 1) -> pd.DataFrame:
    """
    Posicionamento e estado de todos os jogadores por tick.

    tick_rate: 1=todos os ticks, 8=1 a cada 8 (recomendado para datasets grandes)
    """
    log.info(f"  → Extraindo ticks (tick_rate={tick_rate})...")

    t0 = time.time()
    df = _safe_ticks(parser, PLAYER_PROPS)

    # Renomear
    rename = {
        "steamid":      "steam_id",
        "name":         "player_name",
        "X":            "pos_x",
        "Y":            "pos_y",
        "Z":            "pos_z",
        "yaw":          "view_yaw",
        "pitch":        "view_pitch",
        "health":       "hp",
        "armor_value":  "armor",
        "is_alive":     "alive",
        "team_num":     "team",
        "active_weapon":"weapon",
        "has_defuser":  "defuser",
        "has_helmet":   "helmet",
        "flash_duration":"flash_duration",
        "ducking":      "crouching",
        "in_crouch":    "in_crouch",
        "velocity_X":   "vel_x",
        "velocity_Y":   "vel_y",
        "velocity_Z":   "vel_z",
        "is_scoped":    "scoped",
        "spotted":      "spotted",
        "balance":      "money",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Team label
    if "team" in df.columns:
        df["team_label"] = df["team"].map({2: "T", 3: "CT"}).fillna("Unknown")

    # Speed
    if "vel_x" in df.columns and "vel_y" in df.columns:
        df["speed"] = (df["vel_x"]**2 + df["vel_y"]**2) ** 0.5

    # Downsample
    if tick_rate > 1:
        unique_ticks = df["tick"].unique()
        sampled = unique_ticks[::tick_rate]
        df = df[df["tick"].isin(sampled)]

    # Associa rounds
    df = _assign_rounds(df, rounds_df)

    elapsed = time.time() - t0
    log.info(f"  ✅ {len(df):,} ticks extraídos em {elapsed:.1f}s")
    return df


# =============================================================================
# ORQUESTRADOR: extrai tudo de uma demo
# =============================================================================

def extract_all(demo_path: str, tick_rate: int = 1) -> dict:
    """
    Extrai todos os eventos de uma demo.

    Retorna dict com:
        header   → dict com metadados
        rounds   → DataFrame
        kills    → DataFrame
        grenades → DataFrame
        economy  → DataFrame
        ticks    → DataFrame
    """
    demo_path = str(demo_path)
    parser = DemoParser(demo_path)

    log.info(f"📁 {Path(demo_path).name}")

    # Cabeçalho
    try:
        header = parser.parse_header()
    except Exception:
        header = {}

    log.info(f"  🗺  Mapa: {header.get('map_name','?')} | "
             f"Servidor: {header.get('server_name','?')}")

    # Ordem importa: rounds primeiro (usado por kills, grenades, economy)
    rounds   = extract_rounds(parser)
    kills    = extract_kills(parser, rounds)
    grenades = extract_grenades(parser, rounds)
    economy  = extract_economy(parser, rounds)
    ticks    = extract_ticks(parser, rounds, tick_rate=tick_rate)

    return {
        "header":   header,
        "rounds":   rounds,
        "kills":    kills,
        "grenades": grenades,
        "economy":  economy,
        "ticks":    ticks,
    }
