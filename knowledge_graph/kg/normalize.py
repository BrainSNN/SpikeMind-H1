from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from depression_kg.config import SETTINGS
from depression_kg.utils import normalize_text


class CanonicalMapper:
    def __init__(self, canonical_map_path: str | Path | None = None) -> None:
        path = Path(canonical_map_path or SETTINGS.canonical_map_path)
        self.map: dict[str, dict[str, str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_term = normalize_text(row["raw_term"])
                self.map[raw_term] = {
                    "canonical_id": row["canonical_id"],
                    "canonical_name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                }

    def normalize_entity_name(self, entity_name: str, entity_type: str) -> dict[str, str]:
        key = normalize_text(entity_name)
        if key in self.map:
            return self.map[key]
        return {
            "canonical_id": "",
            "canonical_name": entity_name,
            "entity_type": entity_type,
        }


def consolidate_entities(entity_hits: list[dict[str, Any]], mapper: CanonicalMapper) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, str], dict[str, Any]] = {}

    for hit in entity_hits:
        normalized = mapper.normalize_entity_name(hit["canonical_name"], hit["entity_type"])
        key = (normalized["canonical_name"], normalized["entity_type"])
        aliases = set([hit["matched_text"], hit["canonical_name"]])

        if key not in bucket:
            bucket[key] = {
                "canonical_id": normalized["canonical_id"],
                "canonical_name": normalized["canonical_name"],
                "entity_type": normalized["entity_type"],
                "aliases": aliases,
                "description": hit.get("description", ""),
            }
        else:
            bucket[key]["aliases"].update(aliases)

    for value in bucket.values():
        value["aliases"] = sorted(a for a in value["aliases"] if a)

    return list(bucket.values())
