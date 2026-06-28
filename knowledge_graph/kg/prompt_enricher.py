from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from depression_kg.config import OUTPUT_DIR
from depression_kg.utils import normalize_text


# Common paraphrases that frequently appear in free-text prompts.
PHRASE_HINTS: dict[str, str] = {
    "lack of interest": "loss of interest",
    "poor sleep": "sleep disturbance",
    "sleep is poor": "sleep disturbance",
    "gamma-band abnormality": "eeg feature",
    "gamma band abnormality": "eeg feature",
    "gamma abnormality": "eeg feature",

    "dream a lot": "rem sleep",
    "dream a lot at night": "rem sleep",
    "many dreams": "rem sleep",
    "frequent dreaming": "rem sleep",
    "vivid dreams": "rem sleep",
    "dreaming a lot": "rem sleep",
    "wake up at night": "nocturnal awakening",
    "waking up during the night": "nocturnal awakening",
    "wake up many times": "sleep fragmentation",
    "frequent awakenings": "sleep fragmentation",
    "interrupted sleep": "sleep fragmentation",
    "broken sleep": "sleep fragmentation",
    "light sleep": "n1 sleep",
    "sleep lightly": "n1 sleep",
    "not sleeping deeply": "n3 sleep",
    "lack of deep sleep": "n3 sleep",
    "deep sleep is reduced": "n3 sleep",
    "sleep staging": "sleep staging",
    "sleep stage classification": "sleep staging"
}


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class RelationRecord:
    relation_id: str
    head_entity_id: str
    relation_type: str
    tail_entity_id: str
    evidence_text: str
    source: str
    source_id: str
    confidence: float


@dataclass(frozen=True)
class MatchRecord:
    entity: EntityRecord
    reasons: tuple[str, ...]


class PromptEnricher:
    def __init__(self, entities_csv: str | Path | None = None, relations_csv: str | Path | None = None) -> None:
        self.entities_csv = Path(entities_csv or (OUTPUT_DIR / "entities.csv"))
        self.relations_csv = Path(relations_csv or (OUTPUT_DIR / "relations.csv"))

        self.entities = self._load_entities(self.entities_csv)
        self.relations = self._load_relations(self.relations_csv)

        self.entity_by_id = {entity.entity_id: entity for entity in self.entities}
        self.alias_to_entity_ids = self._build_alias_index(self.entities)
        self.relations_by_entity_id = self._build_relations_index(self.relations)

    def enrich_prompt(self, prompt_text: str, top_k_relations: int = 10) -> str:
        matched_records = self.match_entities(prompt_text)
        matched_ids = [record.entity.entity_id for record in matched_records]
        expanded_anchor_ids = self._expand_anchor_ids(matched_ids)
        related_relations = self._collect_related_relations(matched_ids, expanded_anchor_ids, top_k_relations=top_k_relations)
        return self._compose_prompt(prompt_text, matched_records, related_relations)

    def match_entities(self, text: str) -> list[MatchRecord]:
        normalized_text = normalize_text(text)
        match_reasons: dict[str, set[str]] = {}

        for alias_norm, entity_ids in self.alias_to_entity_ids.items():
            if self._contains_phrase(normalized_text, alias_norm):
                for entity_id in entity_ids:
                    match_reasons.setdefault(entity_id, set()).add(f"exact:{alias_norm}")

        # Small phrase-hint layer for common paraphrases in free-text prompts.
        for phrase_norm, alias_norm in PHRASE_HINTS.items():
            if phrase_norm in normalized_text:
                for entity_id in self.alias_to_entity_ids.get(alias_norm, set()):
                    match_reasons.setdefault(entity_id, set()).add(f"hint:{phrase_norm}->{alias_norm}")

        # Soft match by token overlap to recover mild wording variation.
        for segment in self._split_segments(normalized_text):
            segment_tokens = set(self._tokenize(segment))
            if len(segment_tokens) < 2:
                continue
            for alias_norm, entity_ids in self.alias_to_entity_ids.items():
                alias_tokens = set(self._tokenize(alias_norm))
                if len(alias_tokens) < 2:
                    continue
                overlap_ratio = len(segment_tokens & alias_tokens) / len(alias_tokens)
                if overlap_ratio >= 0.6:
                    for entity_id in entity_ids:
                        match_reasons.setdefault(entity_id, set()).add(f"soft:{segment}~{alias_norm}")

        results: list[MatchRecord] = []
        for entity_id, reasons in sorted(match_reasons.items()):
            entity = self.entity_by_id.get(entity_id)
            if not entity:
                continue
            results.append(MatchRecord(entity=entity, reasons=tuple(sorted(reasons))))
        return results

    def _expand_anchor_ids(self, anchor_ids: Iterable[str]) -> set[str]:
        anchors = set(anchor_ids)
        expanded = set(anchors)
        if not anchors:
            return expanded

        # Add likely disease neighbors so we can pull treatment / scale context.
        for relation in self.relations:
            if relation.tail_entity_id in anchors and relation.relation_type in {"has_symptom", "has_risk_factor", "related_to", "measured_by"}:
                expanded.add(relation.head_entity_id)
            if relation.head_entity_id in anchors and relation.relation_type in {"has_symptom", "has_risk_factor", "related_to", "measured_by"}:
                expanded.add(relation.tail_entity_id)
        return expanded

    def _collect_related_relations(
        self,
        anchor_ids: list[str],
        expanded_anchor_ids: set[str],
        top_k_relations: int,
    ) -> list[RelationRecord]:
        if not expanded_anchor_ids:
            return []

        anchor_set = set(anchor_ids)
        candidates: dict[tuple[str, str, str], tuple[float, RelationRecord]] = {}
        for entity_id in expanded_anchor_ids:
            for relation in self.relations_by_entity_id.get(entity_id, []):
                score = relation.confidence
                if relation.head_entity_id in anchor_set or relation.tail_entity_id in anchor_set:
                    score += 0.30
                if relation.head_entity_id in expanded_anchor_ids and relation.tail_entity_id in expanded_anchor_ids:
                    score += 0.15
                if relation.relation_type in {"has_symptom", "related_to", "measured_by", "treats"}:
                    score += 0.05
                score += self._depression_focus_boost(relation)

                conceptual_key = (relation.head_entity_id, relation.relation_type, relation.tail_entity_id)
                existing = candidates.get(conceptual_key)
                if existing is None or score > existing[0]:
                    candidates[conceptual_key] = (score, relation)

        ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:top_k_relations]]

    def _compose_prompt(self, prompt_text: str, matches: list[MatchRecord], relations: list[RelationRecord]) -> str:
        lines: list[str] = []
        lines.append("You are given an input clinical text and curated depression KG context.")
        lines.append("")
        lines.append("Input text:")
        lines.append(prompt_text)
        lines.append("")

        if matches:
            lines.append("Detected entities from KG:")
            for record in matches:
                entity = record.entity
                desc = entity.description.strip() if entity.description else "No description available."
                lines.append(f"- {entity.canonical_name} [{entity.entity_type}]: {desc}")
            lines.append("")
        else:
            lines.append("Detected entities from KG:")
            lines.append("- None")
            lines.append("")

        if relations:
            lines.append("Related KG evidence:")
            for relation in relations:
                head = self.entity_by_id.get(relation.head_entity_id)
                tail = self.entity_by_id.get(relation.tail_entity_id)
                head_name = head.canonical_name if head else relation.head_entity_id
                tail_name = tail.canonical_name if tail else relation.tail_entity_id
                evidence = self._truncate_text(relation.evidence_text, 160)
                lines.append(
                    f"- {head_name} --{relation.relation_type}--> {tail_name} "
                    f"(confidence={relation.confidence:.2f}, source={relation.source}:{relation.source_id})"
                )
                if evidence:
                    lines.append(f"  Evidence: {evidence}")
            lines.append("")
        else:
            lines.append("Related KG evidence:")
            lines.append("- None")
            lines.append("")

        lines.append("Use the input text and KG evidence above to produce a richer, clinically grounded interpretation.")
        return "\n".join(lines).strip()

    @staticmethod
    def _load_entities(path: Path) -> list[EntityRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Entity file not found: {path}. Please run pipeline first.")

        records: list[EntityRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                aliases = PromptEnricher._parse_aliases(row.get("aliases_json", ""))
                canonical_name = (row.get("canonical_name") or "").strip()
                if canonical_name:
                    aliases.append(canonical_name)
                aliases = sorted({alias for alias in aliases if alias})

                entity_id = (row.get("entity_id") or "").strip()
                if not entity_id:
                    continue

                records.append(
                    EntityRecord(
                        entity_id=entity_id,
                        canonical_name=canonical_name,
                        entity_type=(row.get("entity_type") or "unknown").strip(),
                        aliases=tuple(aliases),
                        description=(row.get("description") or "").strip(),
                    )
                )
        return records

    @staticmethod
    def _load_relations(path: Path) -> list[RelationRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Relation file not found: {path}. Please run pipeline first.")

        records: list[RelationRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                relation_id = (row.get("relation_id") or "").strip()
                if not relation_id:
                    continue
                records.append(
                    RelationRecord(
                        relation_id=relation_id,
                        head_entity_id=(row.get("head_entity_id") or "").strip(),
                        relation_type=(row.get("relation_type") or "").strip(),
                        tail_entity_id=(row.get("tail_entity_id") or "").strip(),
                        evidence_text=(row.get("evidence_text") or "").strip(),
                        source=(row.get("source") or "").strip(),
                        source_id=(row.get("source_id") or "").strip(),
                        confidence=PromptEnricher._to_float(row.get("confidence", "0")),
                    )
                )
        return records

    @staticmethod
    def _build_alias_index(entities: list[EntityRecord]) -> dict[str, set[str]]:
        alias_to_entity_ids: dict[str, set[str]] = {}
        for entity in entities:
            for alias in entity.aliases:
                alias_norm = normalize_text(alias)
                if alias_norm:
                    alias_to_entity_ids.setdefault(alias_norm, set()).add(entity.entity_id)
        return alias_to_entity_ids

    @staticmethod
    def _build_relations_index(relations: list[RelationRecord]) -> dict[str, list[RelationRecord]]:
        index: dict[str, list[RelationRecord]] = {}
        for relation in relations:
            index.setdefault(relation.head_entity_id, []).append(relation)
            index.setdefault(relation.tail_entity_id, []).append(relation)
        return index

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        return re.search(pattern, text) is not None

    @staticmethod
    def _split_segments(text: str) -> list[str]:
        temp = re.sub(r"[\n;]", ",", text)
        raw_segments = re.split(r",|\.| and | with ", temp)
        return [normalize_text(seg) for seg in raw_segments if normalize_text(seg)]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9-]+", text.lower())

    @staticmethod
    def _parse_aliases(raw: str) -> list[str]:
        raw = (raw or "").strip()
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            pass

        return [raw]

    @staticmethod
    def _to_float(raw: str | float) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + "..."

    def _depression_focus_boost(self, relation: RelationRecord) -> float:
        head = self.entity_by_id.get(relation.head_entity_id)
        tail = self.entity_by_id.get(relation.tail_entity_id)
        combined = f"{head.canonical_name if head else ''} {tail.canonical_name if tail else ''}".lower()
        if "depress" in combined:
            return 0.10
        return 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich free-text prompt with related depression KG evidence.")
    parser.add_argument("--text", type=str, default="", help="Input prompt text. If omitted, interactive input is used.")
    parser.add_argument("--top-k-relations", type=int, default=10, help="Number of related relations to include.")
    parser.add_argument("--entities-csv", type=str, default=str(OUTPUT_DIR / "entities.csv"), help="Path to entities.csv")
    parser.add_argument("--relations-csv", type=str, default=str(OUTPUT_DIR / "relations.csv"), help="Path to relations.csv")
    return parser
