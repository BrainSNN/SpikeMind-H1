CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT,
    abstract TEXT,
    url TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases_json TEXT,
    description TEXT,
    source TEXT,
    source_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    relation_id TEXT PRIMARY KEY,
    head_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    tail_entity_id TEXT NOT NULL,
    evidence_text TEXT,
    source TEXT,
    source_id TEXT,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS external_terms (
    term_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases_json TEXT,
    entity_type TEXT,
    description TEXT,
    created_at TEXT NOT NULL
);
