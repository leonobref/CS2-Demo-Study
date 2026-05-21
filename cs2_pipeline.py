"""
cs2_pipeline.py
================
Pipeline completo CS2 Demo → Database.

Parseia demos, extrai todos os eventos e popula um banco
relacional (SQLite ou PostgreSQL) de forma incremental
e idempotente (demos já processadas são puladas).

Estrutura do banco:
  matches       → 1 linha por demo
  players       → jogadores únicos
  match_players → jogador ↔ partida
  rounds        → resultado de cada round
  kills         → eliminações
  grenades      → granadas
  economy       → snapshot econômico por jogador/round
  player_ticks  → posicionamento por tick (maior tabela)

Uso:
  # SQLite (zero configuração)
  python cs2_pipeline.py demos/ --db sqlite:///cs2.db

  # Pasta com tick-rate reduzido (recomendado para datasets grandes)
  python cs2_pipeline.py demos/ --db sqlite:///cs2.db --tick-rate 8

  # PostgreSQL com workers paralelos
  python cs2_pipeline.py demos/ \\
      --db postgresql://user:senha@localhost:5432/cs2 \\
      --workers 4 --tick-rate 4

  # Só um arquivo
  python cs2_pipeline.py partida.dem --db sqlite:///cs2.db
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column
from sqlalchemy import Integer, BigInteger, SmallInteger, Float, Boolean, Text
from tqdm import tqdm

# Módulo de extração (mesmo diretório)
sys.path.insert(0, str(Path(__file__).parent))
from cs2_extractors import extract_all

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cs2_pipeline")

# =============================================================================
# SCHEMA SQLAlchemy (portável SQLite + PostgreSQL)
# =============================================================================

metadata = MetaData()

t_matches = Table("matches", metadata,
    Column("id",            Integer,  primary_key=True, autoincrement=True),
    Column("demo_file",     Text,     nullable=False, unique=True),
    Column("map_name",      Text),
    Column("server_name",   Text),
    Column("processed_at",  Text),
    Column("tick_count",    Integer),
    Column("round_count",   Integer),
    Column("duration_secs", Float),
)

t_players = Table("players", metadata,
    Column("steam_id",        Text, primary_key=True),
    Column("last_known_name", Text),
    Column("first_seen_at",   Text),
    Column("last_seen_at",    Text),
)

t_match_players = Table("match_players", metadata,
    Column("match_id",   Integer, nullable=False),
    Column("steam_id",   Text,    nullable=False),
    Column("team_label", Text),
)

t_rounds = Table("rounds", metadata,
    Column("id",               Integer, primary_key=True, autoincrement=True),
    Column("match_id",         Integer, nullable=False),
    Column("round_num",        Integer),
    Column("tick_start",       Integer),
    Column("tick_freeze_end",  Integer),
    Column("tick_end",         Integer),
    Column("winner_team",      Text),
    Column("win_reason",       Integer),
    Column("win_reason_label", Text),
    Column("t_score",          Integer),
    Column("ct_score",         Integer),
)

t_kills = Table("kills", metadata,
    Column("id",                Integer, primary_key=True, autoincrement=True),
    Column("match_id",          Integer,    nullable=False),
    Column("round_id",          Integer),
    Column("tick",              Integer),
    Column("round_num",         Integer),
    Column("attacker_steam_id", Text),
    Column("attacker_name",     Text),
    Column("attacker_team",     Text),
    Column("attacker_x",        Float),
    Column("attacker_y",        Float),
    Column("attacker_z",        Float),
    Column("victim_steam_id",   Text),
    Column("victim_name",       Text),
    Column("victim_team",       Text),
    Column("victim_x",          Float),
    Column("victim_y",          Float),
    Column("victim_z",          Float),
    Column("assister_steam_id", Text),
    Column("assister_name",     Text),
    Column("weapon",            Text),
    Column("headshot",          Integer),
    Column("penetrated",        Integer),
    Column("noscope",           Integer),
    Column("thrusmoke",         Integer),
    Column("distance",          Float),
    Column("is_trade_kill",     Integer),
    Column("bomb_planted",      Integer),
)

t_grenades = Table("grenades", metadata,
    Column("id",                Integer, primary_key=True, autoincrement=True),
    Column("match_id",          Integer,    nullable=False),
    Column("round_id",          Integer),
    Column("tick",              Integer),
    Column("round_num",         Integer),
    Column("thrower_steam_id",  Text),
    Column("thrower_name",      Text),
    Column("thrower_team",      Text),
    Column("grenade_type",      Text),
    Column("det_x",             Float),
    Column("det_y",             Float),
    Column("det_z",             Float),
    Column("duration_ticks",    Integer),
    Column("players_flashed",   Integer),
)

t_economy = Table("economy", metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("match_id",     Integer,    nullable=False),
    Column("round_id",     Integer),
    Column("steam_id",     Text),
    Column("round_num",    Integer),
    Column("team_label",   Text),
    Column("money_start",  Integer),
    Column("equip_value",  Integer),
    Column("money_spent",  Integer),
    Column("buy_type",     Text),
    Column("has_rifle",    Integer),
    Column("has_awp",      Integer),
    Column("has_armor",    Integer),
    Column("has_helmet",   Integer),
    Column("has_defuser",  Integer),
)

t_player_ticks = Table("player_ticks", metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("match_id",      Integer,    nullable=False),
    Column("round_id",      Integer),
    Column("steam_id",      Text),
    Column("tick",          Integer),
    Column("round_num",     SmallInteger),
    Column("pos_x",         Float),
    Column("pos_y",         Float),
    Column("pos_z",         Float),
    Column("view_yaw",      Float),
    Column("view_pitch",    Float),
    Column("hp",            Integer),
    Column("armor",         Integer),
    Column("alive",         Integer),
    Column("team",          Integer),
    Column("team_label",    Text),
    Column("weapon",        Text),
    Column("helmet",        Integer),
    Column("defuser",       Integer),
    Column("vel_x",         Float),
    Column("vel_y",         Float),
    Column("vel_z",         Float),
    Column("speed",         Float),
    Column("crouching",     Integer),
    Column("in_crouch",     Integer),
    Column("scoped",        Integer),
    Column("spotted",       Integer),
    Column("flash_duration",Float),
    Column("money",         Integer),
)

ALL_TABLES = [
    t_matches, t_players, t_match_players,
    t_rounds, t_kills, t_grenades, t_economy, t_player_ticks,
]

INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_kills_match       ON kills         (match_id)",
    "CREATE INDEX IF NOT EXISTS idx_kills_round       ON kills         (match_id, round_num)",
    "CREATE INDEX IF NOT EXISTS idx_kills_attacker    ON kills         (attacker_steam_id)",
    "CREATE INDEX IF NOT EXISTS idx_kills_victim      ON kills         (victim_steam_id)",
    "CREATE INDEX IF NOT EXISTS idx_grenades_match    ON grenades      (match_id, round_num)",
    "CREATE INDEX IF NOT EXISTS idx_grenades_type     ON grenades      (match_id, grenade_type)",
    "CREATE INDEX IF NOT EXISTS idx_ticks_match       ON player_ticks  (match_id, steam_id)",
    "CREATE INDEX IF NOT EXISTS idx_ticks_round       ON player_ticks  (match_id, round_num)",
    "CREATE INDEX IF NOT EXISTS idx_ticks_tick        ON player_ticks  (match_id, tick)",
    "CREATE INDEX IF NOT EXISTS idx_econ_match        ON economy       (match_id, round_num)",
    "CREATE INDEX IF NOT EXISTS idx_econ_player       ON economy       (steam_id)",
    "CREATE INDEX IF NOT EXISTS idx_rounds_match      ON rounds        (match_id)",
]

# =============================================================================
# ENGINE
# =============================================================================

def build_engine(db_url: str):
    if db_url.startswith("sqlite"):
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        with engine.connect() as c:
            c.execute(text("PRAGMA journal_mode=WAL"))
            c.execute(text("PRAGMA synchronous=NORMAL"))
            c.execute(text("PRAGMA cache_size=-128000"))
            c.execute(text("PRAGMA temp_store=MEMORY"))
            c.commit()
    else:
        engine = create_engine(db_url, pool_size=5, max_overflow=10)
    return engine


def init_schema(engine):
    metadata.create_all(engine)
    with engine.connect() as c:
        for ddl in INDEX_DDL:
            c.execute(text(ddl))
        c.commit()
    log.info("✅ Schema e índices criados.")

# =============================================================================
# HELPERS DE INSERÇÃO
# =============================================================================

def _is_processed(engine, demo_file: str) -> bool:
    with engine.connect() as c:
        row = c.execute(
            text("SELECT id FROM matches WHERE demo_file = :f"),
            {"f": demo_file}
        ).fetchone()
    return row is not None


def _upsert_player(conn, steam_id, name, now):
    """Upsert idempotente de jogador."""
    conn.execute(text("""
        INSERT INTO players (steam_id, last_known_name, first_seen_at, last_seen_at)
        VALUES (:sid, :name, :now, :now)
        ON CONFLICT(steam_id) DO UPDATE SET
            last_known_name = excluded.last_known_name,
            last_seen_at    = excluded.last_seen_at
    """), {"sid": steam_id, "name": name, "now": now})


def _bulk_insert(engine, table, df: pd.DataFrame, chunk_size: int = 50_000, label: str = ""):
    """Insert em chunks com barra de progresso."""
    if df is None or df.empty:
        return 0

    # Filtra apenas colunas existentes na tabela (exclui id — gerado pelo banco)
    valid_cols = {c.name for c in table.columns} - {"id"}
    df = df[[c for c in df.columns if c in valid_cols]].copy()

    # Converte extensões de tipo do pandas (StringDtype, BooleanDtype, etc.)
    # e inteiros sem sinal para tipos que o SQLite/SQLAlchemy aceitam
    for col in df.columns:
        if hasattr(df[col].dtype, "numpy_dtype"):
            df[col] = df[col].astype(object)
        elif pd.api.types.is_unsigned_integer_dtype(df[col]):
            df[col] = df[col].astype("int64")
    df = df.where(pd.notnull(df), None)

    total = len(df)

    df.to_sql(
        table.name,
        engine,
        if_exists="append",
        index=False,
        chunksize=chunk_size,
    )

    return total


def _get_round_id_map(engine, match_id: int) -> dict:
    """Retorna {round_num: round.id} para o match."""
    with engine.connect() as c:
        rows = c.execute(
            text("SELECT id, round_num FROM rounds WHERE match_id = :mid"),
            {"mid": match_id}
        ).fetchall()
    return {r[1]: r[0] for r in rows}


def _attach_round_id(df: pd.DataFrame, round_id_map: dict) -> pd.DataFrame:
    if df is None or df.empty or "round_num" not in df.columns:
        return df
    df = df.copy()
    df["round_id"] = df["round_num"].map(round_id_map)
    return df

# =============================================================================
# INSERÇÃO DE UMA DEMO
# =============================================================================

def _insert_demo(engine, demo_path: str, tick_rate: int, chunk_size: int) -> dict:
    """
    Extrai e insere todos os dados de uma demo no banco.
    Retorna dict de status.
    """
    demo_file = Path(demo_path).name
    result    = {"demo": demo_file, "status": "ok", "rows": {}, "error": None}

    if _is_processed(engine, demo_file):
        log.info(f"⏭  Pulando (já processada): {demo_file}")
        result["status"] = "skipped"
        return result

    t_total = time.time()

    try:
        # ── 1. Extrair tudo ──────────────────────────────────────────────────
        data = extract_all(demo_path, tick_rate=tick_rate)

        header   = data["header"]
        rounds   = data["rounds"]
        kills    = data["kills"]
        grenades = data["grenades"]
        economy  = data["economy"]
        ticks    = data["ticks"]

        now = pd.Timestamp.utcnow().isoformat()

        # ── 2. Inserir match ─────────────────────────────────────────────────
        with engine.connect() as conn:
            res = conn.execute(t_matches.insert().values(
                demo_file    = demo_file,
                map_name     = header.get("map_name", "unknown"),
                server_name  = header.get("server_name", ""),
                processed_at = now,
                tick_count   = int(ticks["tick"].nunique()) if not ticks.empty and "tick" in ticks.columns else 0,
                round_count  = len(rounds),
                duration_secs= round(ticks["tick"].nunique() / 64, 1) if not ticks.empty and "tick" in ticks.columns else 0,
            ))
            match_id = res.inserted_primary_key[0]
            conn.commit()

        log.info(f"  🆔 match_id={match_id} | mapa={header.get('map_name','?')}")

        # ── 3. Jogadores únicos ──────────────────────────────────────────────
        all_players = {}
        for df_src, sid_col, name_col in [
            (ticks,    "steam_id", "player_name"),
            (kills,    "attacker_steam_id", "attacker_name"),
            (kills,    "victim_steam_id",   "victim_name"),
            (grenades, "thrower_steam_id",  "thrower_name"),
        ]:
            if df_src is None or df_src.empty:
                continue
            if sid_col in df_src.columns:
                sub = df_src[[sid_col] + ([name_col] if name_col in df_src.columns else [])].drop_duplicates(sid_col)
                for _, row in sub.iterrows():
                    sid = row[sid_col]
                    if pd.notna(sid) and sid not in all_players:
                        all_players[sid] = row.get(name_col, "")

        with engine.connect() as conn:
            for sid, name in all_players.items():
                _upsert_player(conn, sid, name, now)
            conn.commit()

        log.info(f"  👥 {len(all_players)} jogadores registrados")

        # ── 4. Match players ─────────────────────────────────────────────────
        if not ticks.empty and "steam_id" in ticks.columns:
            mp_cols = ["steam_id"] + (["team_label"] if "team_label" in ticks.columns else [])
            mp = ticks[mp_cols].drop_duplicates("steam_id").copy()
            mp["match_id"] = match_id
            with engine.connect() as conn:
                conn.execute(t_match_players.insert(), mp.to_dict("records"))
                conn.commit()

        # ── 5. Rounds ────────────────────────────────────────────────────────
        if not rounds.empty:
            rounds_to_insert = rounds.copy()
            rounds_to_insert["match_id"] = match_id
            _bulk_insert(engine, t_rounds, rounds_to_insert, label="rounds   ")
            result["rows"]["rounds"] = len(rounds)

        round_id_map = _get_round_id_map(engine, match_id)

        # ── 6. Kills ─────────────────────────────────────────────────────────
        if not kills.empty:
            kills_ins = kills.copy()
            kills_ins["match_id"] = match_id
            kills_ins = _attach_round_id(kills_ins, round_id_map)
            n = _bulk_insert(engine, t_kills, kills_ins, chunk_size, label="kills    ")
            result["rows"]["kills"] = n
            log.info(f"  🔫 {n:,} kills inseridas")

        # ── 7. Grenades ──────────────────────────────────────────────────────
        if not grenades.empty:
            gren_ins = grenades.copy()
            gren_ins["match_id"] = match_id
            gren_ins = _attach_round_id(gren_ins, round_id_map)
            n = _bulk_insert(engine, t_grenades, gren_ins, chunk_size, label="granadas ")
            result["rows"]["grenades"] = n
            log.info(f"  💣 {n:,} granadas inseridas")

        # ── 8. Economy ───────────────────────────────────────────────────────
        if not economy.empty:
            econ_ins = economy.copy()
            econ_ins["match_id"] = match_id
            econ_ins = _attach_round_id(econ_ins, round_id_map)
            n = _bulk_insert(engine, t_economy, econ_ins, chunk_size, label="economia ")
            result["rows"]["economy"] = n
            log.info(f"  💰 {n:,} snapshots de economia inseridos")

        # ── 9. Player ticks ──────────────────────────────────────────────────
        if not ticks.empty:
            ticks_ins = ticks.copy()
            ticks_ins["match_id"] = match_id
            ticks_ins = _attach_round_id(ticks_ins, round_id_map)
            n = _bulk_insert(engine, t_player_ticks, ticks_ins, chunk_size, label="ticks    ")
            result["rows"]["ticks"] = n
            log.info(f"  📍 {n:,} ticks inseridos")

        elapsed = time.time() - t_total
        log.info(f"✅ {demo_file} concluído em {elapsed:.1f}s")

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
        log.error(f"❌ Erro em {demo_file}: {e}", exc_info=True)

    return result


# =============================================================================
# PIPELINE PÚBLICO
# =============================================================================

def run_pipeline(
    input_path: str,
    db_url:     str,
    tick_rate:  int = 8,
    workers:    int = 1,
    chunk_size: int = 50_000,
):
    """
    Executa o pipeline completo em um arquivo ou diretório de demos.
    """
    engine = build_engine(db_url)
    init_schema(engine)

    # Coleta demos
    p = Path(input_path)
    demos = sorted(p.glob("**/*.dem")) if p.is_dir() else ([p] if p.suffix == ".dem" else [])

    if not demos:
        log.error("Nenhuma demo .dem encontrada.")
        sys.exit(1)

    log.info(f"🎯 {len(demos)} demo(s) | db={db_url} | tick_rate={tick_rate}")

    # SQLite não suporta escrita paralela
    if db_url.startswith("sqlite") and workers > 1:
        log.warning("SQLite não suporta escrita paralela → workers=1")
        workers = 1

    results = []

    if workers <= 1:
        for demo in demos:
            r = _insert_demo(engine, str(demo), tick_rate, chunk_size)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_insert_demo, engine, str(d), tick_rate, chunk_size): d
                for d in demos
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    # ── Resumo ───────────────────────────────────────────────────────────────
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors  = [r for r in results if r["status"] == "error"]

    print("\n" + "="*60)
    print("📊 RESUMO FINAL DO PIPELINE")
    print("="*60)
    print(f"  ✅ Processadas  : {len(ok)}")
    print(f"  ⏭  Puladas      : {len(skipped)}")
    print(f"  ❌ Erros        : {len(errors)}")

    if ok:
        for table in ["rounds", "kills", "grenades", "economy", "ticks"]:
            total = sum(r["rows"].get(table, 0) for r in ok)
            if total:
                print(f"  📦 {table:<10} : {total:>12,} linhas")

    if errors:
        print("\n  Demos com erro:")
        for r in errors:
            print(f"    • {r['demo']} → {r['error']}")

    print("="*60)


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="CS2 Full Pipeline: Demos → Banco de Dados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python cs2_pipeline.py demos/      --db sqlite:///cs2.db
  python cs2_pipeline.py demo.dem    --db sqlite:///cs2.db --tick-rate 8
  python cs2_pipeline.py demos/      --db postgresql://u:p@localhost/cs2 --workers 4
        """
    )
    ap.add_argument("input",                           help="Arquivo .dem ou diretório")
    ap.add_argument("--db",    default="sqlite:///cs2.db", help="URL do banco (SQLAlchemy)")
    ap.add_argument("--tick-rate",  "-t", type=int, default=8,      help="Downsample de ticks (1=todos, padrão=8)")
    ap.add_argument("--workers",    "-w", type=int, default=1,      help="Workers paralelos (só PostgreSQL)")
    ap.add_argument("--chunk-size", "-c", type=int, default=50_000, help="Linhas por batch INSERT")
    args = ap.parse_args()

    run_pipeline(
        input_path = args.input,
        db_url     = args.db,
        tick_rate  = args.tick_rate,
        workers    = args.workers,
        chunk_size = args.chunk_size,
    )


if __name__ == "__main__":
    main()
