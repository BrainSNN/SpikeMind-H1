from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from depression_kg.config import SETTINGS
from depression_kg.utils import now_iso, stable_id


class SQLiteStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path or SETTINGS.sqlite_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = OFF")

    def initialize(self) -> None:
        with open(SETTINGS.schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def upsert_documents(self, docs: list[dict[str, Any]]) -> None:
        sql = """
        INSERT OR REPLACE INTO documents (
            doc_id, source, source_id, title, abstract, url, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                doc["doc_id"],
                doc["source"],
                doc["source_id"],
                doc.get("title", ""),
                doc.get("abstract", ""),
                doc.get("url", ""),
                json.dumps(doc.get("metadata", {}), ensure_ascii=False),
                now_iso(),
            )
            for doc in docs
        ]
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def upsert_entities(self, entities: list[dict[str, Any]]) -> dict[str, str]:
        sql = """
        INSERT OR REPLACE INTO entities (
            entity_id, canonical_name, entity_type, aliases_json, description, source, source_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        name_to_id: dict[str, str] = {}
        rows = []
        timestamp = now_iso()
        for entity in entities:
            entity_id = entity.get("canonical_id") or stable_id(entity["canonical_name"], entity["entity_type"], prefix="ent")
            name_to_id[entity["canonical_name"]] = entity_id
            rows.append(
                (
                    entity_id,
                    entity["canonical_name"],
                    entity["entity_type"],
                    json.dumps(entity.get("aliases", []), ensure_ascii=False),
                    entity.get("description", ""),
                    entity.get("source", "seed"),
                    entity.get("source_id", ""),
                    timestamp,
                    timestamp,
                )
            )
        self.conn.executemany(sql, rows)
        self.conn.commit()
        return name_to_id

    def insert_relations(self, relations: list[dict[str, Any]], name_to_id: dict[str, str]) -> None:
        sql = """
        INSERT OR REPLACE INTO relations (
            relation_id, head_entity_id, relation_type, tail_entity_id, evidence_text, source, source_id, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = []
        timestamp = now_iso()
        for rel in relations:
            head_id = name_to_id.get(rel["head_name"])
            tail_id = name_to_id.get(rel["tail_name"])
            if not head_id or not tail_id:
                continue
            relation_id = stable_id(head_id, rel["relation_type"], tail_id, rel.get("source_id", ""), prefix="rel")
            rows.append(
                (
                    relation_id,
                    head_id,
                    rel["relation_type"],
                    tail_id,
                    rel.get("evidence_text", ""),
                    rel.get("source", ""),
                    rel.get("source_id", ""),
                    rel.get("confidence", 0.0),
                    timestamp,
                )
            )
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def upsert_external_terms(self, terms: list[dict[str, Any]]) -> None:
        sql = """
        INSERT OR REPLACE INTO external_terms (
            term_id, source, external_id, name, aliases_json, entity_type, description, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = []
        timestamp = now_iso()
        for term in terms:
            term_id = stable_id(term["source"], term["id"], prefix="ext")
            rows.append(
                (
                    term_id,
                    term["source"],
                    term["id"],
                    term["name"],
                    json.dumps(term.get("aliases", []), ensure_ascii=False),
                    term.get("type", ""),
                    term.get("description", ""),
                    timestamp,
                )
            )
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def export_csvs(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for table in ["documents", "entities", "relations", "external_terms"]:
            df = pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
            df.to_csv(output_dir / f"{table}.csv", index=False)

    def close(self) -> None:
        self.conn.close()
