"""Versioned PostgreSQL schema for distributed Expand.

Migrations are additive and run under a transaction-scoped advisory lock.  No SQLite
data is implicitly copied or transformed.
"""

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: r"""
CREATE TABLE IF NOT EXISTS schain_schema_migrations (
    version integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS schain_experiments (
    experiment_id text PRIMARY KEY,
    schema_version integer NOT NULL,
    manifest jsonb NOT NULL,
    manifest_hash text NOT NULL,
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('running','completed','failed','paused')),
    max_tier integer NOT NULL CHECK (max_tier >= 0),
    save_enabled boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS schain_workers (
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    worker_id text NOT NULL,
    session_id text NOT NULL,
    status text NOT NULL DEFAULT 'active'
        CHECK (status IN ('active','stopped','lost','failed')),
    git_commit text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    statistics jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    heartbeat_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    stopped_at timestamptz,
    PRIMARY KEY (experiment_id, worker_id)
);

CREATE TABLE IF NOT EXISTS schain_frontier (
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    entity_key text NOT NULL,
    entity_name text NOT NULL,
    entity_type text NOT NULL DEFAULT 'Company',
    tier integer NOT NULL CHECK (tier >= 0),
    expandable boolean NOT NULL DEFAULT true,
    discovered_from text,
    discovery_provenance jsonb NOT NULL DEFAULT '[]'::jsonb,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','leased','completed','failed','terminal')),
    priority double precision NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, entity_key)
);
CREATE INDEX IF NOT EXISTS schain_frontier_tier_status_idx
    ON schain_frontier(experiment_id, tier, status);

CREATE TABLE IF NOT EXISTS schain_tasks (
    task_id text PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    task_key text NOT NULL,
    entity_key text NOT NULL,
    entity_name text NOT NULL,
    tier integer NOT NULL CHECK (tier >= 0),
    task_type text NOT NULL DEFAULT 'expand_company',
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','leased','retry','completed','failed')),
    priority double precision NOT NULL DEFAULT 0,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    result jsonb,
    attempts integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL CHECK (max_attempts > 0),
    worker_id text,
    lease_token bigint NOT NULL DEFAULT 0,
    lease_expires_at timestamptz,
    available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_error text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    completed_at timestamptz,
    UNIQUE (experiment_id, task_key),
    FOREIGN KEY (experiment_id, entity_key)
        REFERENCES schain_frontier(experiment_id, entity_key) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS schain_tasks_claim_idx
    ON schain_tasks(experiment_id, status, tier, priority DESC, available_at, created_at);
CREATE INDEX IF NOT EXISTS schain_tasks_lease_idx
    ON schain_tasks(experiment_id, lease_expires_at) WHERE status='leased';

CREATE TABLE IF NOT EXISTS schain_retrieval_checkpoints (
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    entity_key text NOT NULL,
    tier integer NOT NULL,
    results jsonb NOT NULL,
    content_hash text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, entity_key, tier)
);

CREATE TABLE IF NOT EXISTS schain_documents (
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    document_id text NOT NULL,
    content_hash text NOT NULL,
    task_id text REFERENCES schain_tasks(task_id),
    origin_entity text,
    tier integer,
    stage text NOT NULL CHECK (stage IN ('retrieved','extracted','resolved','completed')),
    document jsonb NOT NULL,
    extraction jsonb,
    resolution jsonb,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, document_id),
    UNIQUE (experiment_id, content_hash)
);
CREATE INDEX IF NOT EXISTS schain_documents_task_idx
    ON schain_documents(experiment_id, task_id);

CREATE TABLE IF NOT EXISTS schain_facts (
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    fact_id text NOT NULL,
    document_id text NOT NULL,
    task_id text REFERENCES schain_tasks(task_id),
    tier integer,
    accepted boolean NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, fact_id, accepted),
    FOREIGN KEY (experiment_id, document_id)
        REFERENCES schain_documents(experiment_id, document_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS schain_facts_document_idx
    ON schain_facts(experiment_id, document_id, accepted);

CREATE TABLE IF NOT EXISTS schain_entity_aliases (
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    mention_key text NOT NULL,
    entity_type text NOT NULL,
    mention text NOT NULL,
    canonical_id text NOT NULL,
    canonical_name text NOT NULL,
    method text NOT NULL,
    evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, mention_key)
);
CREATE INDEX IF NOT EXISTS schain_alias_canonical_idx
    ON schain_entity_aliases(experiment_id, canonical_id);

CREATE TABLE IF NOT EXISTS schain_graph_outbox (
    outbox_id text PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    task_id text NOT NULL REFERENCES schain_tasks(task_id) ON DELETE CASCADE,
    document_id text NOT NULL,
    tier integer NOT NULL,
    payload jsonb NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','leased','retry','acknowledged','failed')),
    attempts integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 10,
    worker_id text,
    lease_token bigint NOT NULL DEFAULT 0,
    lease_expires_at timestamptz,
    available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_error text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    acknowledged_at timestamptz,
    UNIQUE (experiment_id, document_id)
);
CREATE INDEX IF NOT EXISTS schain_outbox_claim_idx
    ON schain_graph_outbox(experiment_id, status, tier, available_at, created_at);

CREATE TABLE IF NOT EXISTS schain_events (
    event_id bigserial PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES schain_experiments(experiment_id) ON DELETE CASCADE,
    worker_id text,
    task_id text,
    event_type text NOT NULL,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS schain_events_experiment_time_idx
    ON schain_events(experiment_id, created_at DESC);
""",
}
