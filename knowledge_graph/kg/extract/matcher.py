from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from depression_kg.config import SETTINGS
from depression_kg.utils import normalize_text


@dataclass
class SeedTerm:
    name: str
    entity_type: str
    aliases: list[str]
    description: str = ""


class EntityMatcher:
    def __init__(self, seed_terms_path: str | Path | None = None, external_terms: list[dict] | None = None) -> None:
        path = Path(seed_terms_path or SETTINGS.seed_terms_path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.seed_terms: list[SeedTerm] = [
            SeedTerm(
                name=item["name"],
                entity_type=item["type"],
                aliases=item.get("aliases", []),
                description=item.get("description", ""),
            )
            for item in raw
        ]
        # 添加外部术语
        if external_terms:
            for term in external_terms:
                self.seed_terms.append(
                    SeedTerm(
                        name=term["name"],
                        entity_type=term.get("type", "unknown"),
                        aliases=term.get("aliases", []),
                        description=term.get("description", ""),
                    )
                )

    def extract(self, text: str) -> list[dict]:
        normalized = normalize_text(text)
        hits: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for seed in self.seed_terms:
            candidates = [seed.name] + seed.aliases
            for candidate in candidates:
                candidate_norm = normalize_text(candidate)
                pattern = rf"(?<!\w){re.escape(candidate_norm)}(?!\w)"
                match = re.search(pattern, normalized)
                if match:
                    key = (seed.name, seed.entity_type)
                    if key not in seen:
                        seen.add(key)
                        hits.append(
                            {
                                "matched_text": candidate,
                                "canonical_name": seed.name,
                                "entity_type": seed.entity_type,
                                "description": seed.description,
                            }
                        )
        return hits

    def names_by_type(self, hits: Iterable[dict], entity_type: str) -> list[str]:
        return [x["canonical_name"] for x in hits if x["entity_type"] == entity_type]
