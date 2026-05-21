-- =============================================================================
-- CS2 FULL PIPELINE — Schema Completo
-- Compatível com PostgreSQL e SQLite
-- =============================================================================
-- Tabelas:
--   matches          → metadados de cada demo processada
--   players          → jogadores únicos (por steam_id)
--   match_players    → participação de jogador em partida
--   rounds           → resultado e contexto de cada round
--   kills            → eliminações com contexto tático
--   grenades         → trajetória e detonações de granadas
--   player_ticks     → posicionamento por tick (tabela mais pesada)
--   economy          → snapshot econômico por jogador por round
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. MATCHES
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    demo_file       TEXT        NOT NULL UNIQUE,
    map_name        TEXT,
    server_name     TEXT,
    network_proto   INTEGER,
    processed_at    TEXT        DEFAULT (datetime('now')),
    tick_count      INTEGER,
    round_count     INTEGER,
    duration_secs   REAL
);

-- -----------------------------------------------------------------------------
-- 2. PLAYERS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    steam_id        TEXT        PRIMARY KEY,
    last_known_name TEXT,
    first_seen_at   TEXT        DEFAULT (datetime('now')),
    last_seen_at    TEXT        DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------------
-- 3. MATCH_PLAYERS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_players (
    match_id        INTEGER     NOT NULL REFERENCES matches(id)  ON DELETE CASCADE,
    steam_id        TEXT        NOT NULL REFERENCES players(steam_id),
    team_label      TEXT,
    PRIMARY KEY (match_id, steam_id)
);

-- -----------------------------------------------------------------------------
-- 4. ROUNDS — resultado e contexto de cada rodada
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rounds (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER     NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    round_num       INTEGER     NOT NULL,

    -- Ticks de referência
    tick_start      INTEGER,
    tick_freeze_end INTEGER,    -- fim da fase de compra
    tick_end        INTEGER,    -- round_officially_ended

    -- Resultado
    winner_team     TEXT,       -- 'T' | 'CT'
    win_reason      INTEGER,    -- código interno do CS2
    win_reason_label TEXT,      -- 'ct_killed_all', 'bomb_defused', etc.

    -- Contagem de rounds vencidos até aqui
    t_score         INTEGER,
    ct_score        INTEGER,

    UNIQUE (match_id, round_num)
);

-- -----------------------------------------------------------------------------
-- 5. KILLS — cada eliminação com contexto tático completo
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kills (
    id                  INTEGER     PRIMARY KEY AUTOINCREMENT,
    match_id            INTEGER     NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    round_id            INTEGER     REFERENCES rounds(id),
    tick                INTEGER     NOT NULL,
    round_num           INTEGER,

    -- Atacante
    attacker_steam_id   TEXT,
    attacker_name       TEXT,
    attacker_team       TEXT,
    attacker_x          REAL,
    attacker_y          REAL,
    attacker_z          REAL,

    -- Vítima
    victim_steam_id     TEXT,
    victim_name         TEXT,
    victim_team         TEXT,
    victim_x            REAL,
    victim_y            REAL,
    victim_z            REAL,
    victim_hp_before    INTEGER,    -- HP antes de morrer (nem sempre disponível)

    -- Assistente
    assister_steam_id   TEXT,
    assister_name       TEXT,

    -- Mecânica da morte
    weapon              TEXT,
    headshot            INTEGER,    -- 0/1
    penetrated          INTEGER,    -- 0/1 (wallbang)
    noscope             INTEGER,    -- 0/1
    thrusmoke           INTEGER,    -- 0/1 (smoke kill)
    distance            REAL,       -- distância 3D entre atacante e vítima

    -- Contexto
    is_trade_kill       INTEGER,    -- 0/1 calculado em pós-processamento
    bomb_planted        INTEGER     -- 0/1: bomba estava plantada no momento?
);

-- -----------------------------------------------------------------------------
-- 6. GRENADES — trajetória e detonação de cada granada
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS grenades (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER     NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    round_id        INTEGER     REFERENCES rounds(id),
    tick            INTEGER     NOT NULL,
    round_num       INTEGER,

    -- Quem lançou
    thrower_steam_id    TEXT,
    thrower_name        TEXT,
    thrower_team        TEXT,

    -- Tipo e posição de detonação
    grenade_type        TEXT,   -- 'Flashbang', 'HEGrenade', 'SmokeGrenade',
                                --  'Molotov', 'Decoy', 'Inferno'
    det_x               REAL,   -- posição X de detonação
    det_y               REAL,
    det_z               REAL,

    -- Para smokes e incendiários: duração / área
    duration_ticks      INTEGER,    -- quantos ticks ficou ativo
    -- Para flashes: quantos jogadores afetados (calculado em pós)
    players_flashed     INTEGER
);

-- -----------------------------------------------------------------------------
-- 7. PLAYER_TICKS — posicionamento e estado por tick (tabela mais pesada)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS player_ticks (
    id              INTEGER     PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER     NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    round_id        INTEGER     REFERENCES rounds(id),
    steam_id        TEXT        NOT NULL,
    tick            INTEGER     NOT NULL,
    round_num       SMALLINT,

    -- Posição 3D
    pos_x           REAL,
    pos_y           REAL,
    pos_z           REAL,

    -- Câmera
    view_yaw        REAL,
    view_pitch      REAL,

    -- Estado vital
    hp              INTEGER,
    armor           INTEGER,
    alive           INTEGER,    -- 0/1
    team            INTEGER,    -- 2=T, 3=CT
    team_label      TEXT,

    -- Equipamento
    weapon          TEXT,
    helmet          INTEGER,
    defuser         INTEGER,

    -- Movimento
    vel_x           REAL,
    vel_y           REAL,
    vel_z           REAL,
    speed           REAL,
    crouching       INTEGER,
    in_crouch       INTEGER,

    -- Situação tática
    scoped          INTEGER,
    spotted         INTEGER,
    flash_duration  REAL,

    -- Economia (snapshot do tick)
    money           INTEGER
);

-- -----------------------------------------------------------------------------
-- 8. ECONOMY — snapshot econômico por jogador por round
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS economy (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id                INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    round_id                INTEGER REFERENCES rounds(id),
    steam_id                TEXT    NOT NULL,
    round_num               INTEGER NOT NULL,
    team_label              TEXT,

    -- Dinheiro
    money_start             INTEGER,    -- dinheiro no início do round
    money_spent             INTEGER,    -- calculado: start - equip_value
    equip_value             INTEGER,    -- valor do equipamento comprado

    -- Tipo de compra
    buy_type                TEXT,       -- 'full_buy', 'eco', 'force_buy', 'half_buy'

    -- Equipamentos relevantes
    has_rifle               INTEGER,    -- 0/1 (AK/M4/AWP...)
    has_awp                 INTEGER,
    has_armor               INTEGER,
    has_helmet              INTEGER,
    has_defuser             INTEGER,    -- só CT

    UNIQUE (match_id, steam_id, round_num)
);

-- =============================================================================
-- ÍNDICES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_kills_match      ON kills        (match_id);
CREATE INDEX IF NOT EXISTS idx_kills_round      ON kills        (match_id, round_num);
CREATE INDEX IF NOT EXISTS idx_kills_attacker   ON kills        (attacker_steam_id);
CREATE INDEX IF NOT EXISTS idx_kills_victim     ON kills        (victim_steam_id);

CREATE INDEX IF NOT EXISTS idx_grenades_match   ON grenades     (match_id, round_num);
CREATE INDEX IF NOT EXISTS idx_grenades_type    ON grenades     (match_id, grenade_type);
CREATE INDEX IF NOT EXISTS idx_grenades_thrower ON grenades     (thrower_steam_id);

CREATE INDEX IF NOT EXISTS idx_ticks_match      ON player_ticks (match_id, steam_id);
CREATE INDEX IF NOT EXISTS idx_ticks_round      ON player_ticks (match_id, round_num);
CREATE INDEX IF NOT EXISTS idx_ticks_tick       ON player_ticks (match_id, tick);

CREATE INDEX IF NOT EXISTS idx_econ_match       ON economy      (match_id, round_num);
CREATE INDEX IF NOT EXISTS idx_econ_player      ON economy      (steam_id);

CREATE INDEX IF NOT EXISTS idx_rounds_match     ON rounds       (match_id);

-- =============================================================================
-- VIEWS ANALÍTICAS
-- =============================================================================

-- Kills enriquecidas com mapa e round
CREATE VIEW IF NOT EXISTS v_kills AS
SELECT
    k.*,
    m.map_name,
    m.demo_file,
    r.winner_team,
    r.win_reason_label
FROM kills k
JOIN matches m ON k.match_id = m.id
LEFT JOIN rounds r ON k.round_id = r.id;

-- Economia por partida com nomes de jogadores
CREATE VIEW IF NOT EXISTS v_economy AS
SELECT
    e.*,
    p.last_known_name AS player_name,
    m.map_name,
    m.demo_file
FROM economy e
JOIN players p ON e.steam_id = p.steam_id
JOIN matches m ON e.match_id = m.id;

-- Posicionamento enriquecido
CREATE VIEW IF NOT EXISTS v_player_ticks AS
SELECT
    pt.*,
    p.last_known_name AS player_name,
    m.map_name,
    m.demo_file
FROM player_ticks pt
JOIN players p ON pt.steam_id = p.steam_id
JOIN matches m ON pt.match_id = m.id;
