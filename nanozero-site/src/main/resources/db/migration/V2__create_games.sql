-- Parties jouées persistées (incrément 1 — cf. backend-persistance-parties.md §5).
CREATE TABLE games (
    id              BIGSERIAL    PRIMARY KEY,

    -- Identifiant public partageable (lien /play/?game=<share_id>). base62(10) ≈ 8,4e17 → non énumérable.
    share_id        VARCHAR(12)  NOT NULL,

    -- PGN complet (en-têtes neutres : White "Joueur" / Black "NanoZero gen-NNN"). TEXT, recharge directe par le front.
    pgn             TEXT         NOT NULL CHECK (length(pgn) <= 65536),

    -- Métadonnées de jeu (normalisées : requêtes par level/result).
    player_color    VARCHAR(1)   NOT NULL CHECK (player_color IN ('w','b')),
    level_id        VARCHAR(16)  NOT NULL CHECK (level_id IN ('chill','club','full')),
    result          VARCHAR(8)   NOT NULL CHECK (result IN ('1-0','0-1','1/2-1/2','*')),
    ply_count       SMALLINT     NOT NULL CHECK (ply_count >= 0),

    -- Version réseau (ex. 'gen-031'), imposée par le serveur (/api/site/info), pas champ libre.
    network_version VARCHAR(32)  NOT NULL,

    -- Diagnostic / hardware (anonyme, GROSSIER) — pour stats SQL & anomalies de force.
    device_class    VARCHAR(8)   CHECK (device_class IN ('mobile','desktop')),
    backend         VARCHAR(8)   CHECK (backend IN ('webgpu','wasm')),     -- ORT-Web : pas de 'cpu' (= wasm)
    effective_sims  SMALLINT     CHECK (effective_sims >= 0),              -- sims RÉELS (après cappedSims)
    client          JSONB,                                                -- {cores,memGb,browser,wasmThreads,avgThinkMs,appVersion}

    -- Auth différée (Lot C1) : nullable au MVP.
    pseudo          VARCHAR(64),

    -- Extensions futures (évals WDL/temps par coup) sans rebump de schéma.
    metadata        JSONB,

    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX games_share_id_uidx     ON games (share_id);
CREATE INDEX        games_pseudo_created_idx ON games (pseudo, created_at DESC) WHERE pseudo IS NOT NULL;
CREATE INDEX        games_created_at_idx     ON games (created_at DESC);
-- Diagnostic d'anomalies de force : « parties full où Nano perd, par backend/sims/device ».
CREATE INDEX        games_diag_idx           ON games (level_id, result, backend, device_class);
