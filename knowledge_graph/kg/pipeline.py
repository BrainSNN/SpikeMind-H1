from __future__ import annotations

from collections import Counter
from typing import Any
import json

from depression_kg.clients.ctgov_client import ClinicalTrialsClient
from depression_kg.clients.mesh_client import MeSHClient
from depression_kg.clients.pubmed_client import PubMedClient
from depression_kg.config import OUTPUT_DIR, SETTINGS
from depression_kg.extract.matcher import EntityMatcher
from depression_kg.extract.relation_rules import RelationExtractor
from depression_kg.normalize import CanonicalMapper, consolidate_entities
from depression_kg.storage import SQLiteStore
from depression_kg.utils import dump_json


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_seed_catalog_entities(matcher: EntityMatcher, mapper: CanonicalMapper) -> list[dict[str, Any]]:
    seed_hits: list[dict[str, Any]] = []
    for seed in matcher.seed_terms:
        candidates = [seed.name] + seed.aliases
        for candidate in candidates:
            seed_hits.append(
                {
                    "matched_text": candidate,
                    "canonical_name": seed.name,
                    "entity_type": seed.entity_type,
                    "description": seed.description,
                }
            )
    return consolidate_entities(seed_hits, mapper)


def _merge_entities(*entity_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for entities in entity_lists:
        for entity in entities:
            key = (entity["canonical_name"], entity["entity_type"])
            aliases = set(entity.get("aliases", []))
            if key not in merged:
                merged[key] = {
                    "canonical_id": entity.get("canonical_id", ""),
                    "canonical_name": entity["canonical_name"],
                    "entity_type": entity["entity_type"],
                    "aliases": aliases,
                    "description": entity.get("description", ""),
                }
            else:
                merged[key]["aliases"].update(aliases)
                if not merged[key].get("description") and entity.get("description"):
                    merged[key]["description"] = entity["description"]
                if not merged[key].get("canonical_id") and entity.get("canonical_id"):
                    merged[key]["canonical_id"] = entity["canonical_id"]

    for value in merged.values():
        value["aliases"] = sorted(a for a in value["aliases"] if a)
    return list(merged.values())


def _pick_anchor(
    entity_name: str,
    entity_type: str,
    entities_by_type: dict[str, list[str]],
    entity_name_set: set[str],
) -> str | None:
    preferred_anchors: dict[str, list[str]] = {
        "disease": ["major depressive disorder", "depressive disorder"],
        "symptom": ["sadness", "anhedonia"],
        "risk_factor": ["psychological stress", "social isolation"],
        "treatment": ["psychotherapy", "cognitive behavioral therapy"],
        "drug": ["fluoxetine", "sertraline"],
        "scale": ["phq-9", "hamd"],
        "biomarker": ["serotonin", "cortisol"],
        "sleep_symptom": ["sleep disturbance", "insomnia symptoms"],
        "sleep_stage": ["rem sleep", "n2 sleep"],
        "sleep_pattern": ["sleep fragmentation", "sleep architecture disruption"],
        "eeg_feature": ["sleep spindle", "slow wave"],
        "emotion": ["depressed mood", "sadness"],
        "affect_dimension": ["negative affect", "valence"],
        "emotion_regulation": ["cognitive reappraisal", "mindfulness"],
        "task": ["sleep staging"],
    }

    for candidate in preferred_anchors.get(entity_type, []):
        if candidate in entity_name_set and candidate != entity_name:
            return candidate

    same_type = entities_by_type.get(entity_type, [])
    for candidate in same_type:
        if candidate != entity_name:
            return candidate

    fallback_order = ["major depressive disorder", "depressive disorder", "sadness"]
    for candidate in fallback_order:
        if candidate in entity_name_set and candidate != entity_name:
            return candidate

    for candidate in sorted(entity_name_set):
        if candidate != entity_name:
            return candidate
    return None


def _relation_type_for_entity(entity_type: str) -> str:
    mapping = {
        "disease": "related_to",
        "symptom": "associated_with",
        "risk_factor": "has_risk_factor",
        "treatment": "treats",
        "drug": "treats",
        "scale": "measured_by",
        "biomarker": "related_to",
        "sleep_symptom": "associated_with",
        "sleep_stage": "related_to",
        "sleep_pattern": "related_to",
        "eeg_feature": "characterized_by",
        "emotion": "associated_with",
        "affect_dimension": "has_dimension",
        "emotion_regulation": "modulates",
        "task": "related_to",
    }
    return mapping.get(entity_type, "related_to")


def _ensure_entity_connectivity(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    entity_name_set = {entity["canonical_name"] for entity in entities}
    entities_by_type: dict[str, list[str]] = {}
    for entity in entities:
        entities_by_type.setdefault(entity["entity_type"], []).append(entity["canonical_name"])

    connected: set[str] = set()
    for rel in relations:
        head = rel.get("head_name", "")
        tail = rel.get("tail_name", "")
        if head in entity_name_set:
            connected.add(head)
        if tail in entity_name_set:
            connected.add(tail)

    initially_connected = len(connected)
    isolated_entities = [e for e in entities if e["canonical_name"] not in connected]
    catalog_relations: list[dict[str, Any]] = []

    for entity in isolated_entities:
        head_name = entity["canonical_name"]
        entity_type = entity["entity_type"]
        tail_name = _pick_anchor(head_name, entity_type, entities_by_type, entity_name_set)
        if not tail_name:
            continue

        rel_type = _relation_type_for_entity(entity_type)
        source_id = f"catalog_link::{head_name}::{rel_type}::{tail_name}"
        catalog_relations.append(
            {
                "head_name": head_name,
                "relation_type": rel_type,
                "tail_name": tail_name,
                "evidence_text": "Catalog-generated relation to reduce isolated entities in KG visualization.",
                "source": "catalog",
                "source_id": source_id,
                "confidence": 0.55,
            }
        )
        connected.add(head_name)
        connected.add(tail_name)

    return relations + catalog_relations, initially_connected, len(connected)


def run_pipeline(pubmed_queries: list[str], ctgov_query: str, pubmed_retmax: int = 20, offline_fallback: bool = True) -> dict[str, Any]:
    pubmed_client = PubMedClient()
    ctgov_client = ClinicalTrialsClient()
    mesh_client = MeSHClient()

    fetch_errors: list[str] = []
    try:
        pubmed_docs = pubmed_client.fetch_by_queries(pubmed_queries, retmax=pubmed_retmax)
    except Exception as e:
        if not offline_fallback:
            raise
        fetch_errors.append(f"pubmed: {e}")
        pubmed_docs = _load_json(SETTINGS.sample_pubmed_path)

    try:
        ctgov_docs = ctgov_client.search(ctgov_query, page_size=20)
    except Exception as e:
        if not offline_fallback:
            raise
        fetch_errors.append(f"clinicaltrials: {e}")
        ctgov_docs = _load_json(SETTINGS.sample_ctgov_path)

    external_terms = []
    try:
        mesh_terms = mesh_client.search_terms("depressive disorder", retmax=10)
        external_terms.extend(mesh_terms)
    except Exception as e:
        fetch_errors.append(f"mesh: {e}")

    store = SQLiteStore()
    store.initialize()
    store.upsert_external_terms(external_terms)

    all_docs = pubmed_docs + ctgov_docs

    matcher = EntityMatcher(external_terms=external_terms)
    relation_extractor = RelationExtractor()
    mapper = CanonicalMapper()

    all_entity_hits: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []

    for doc in all_docs:
        text = ((doc.get("title") or "") + "\n" + (doc.get("abstract") or "")).strip()
        hits = matcher.extract(text)
        all_entity_hits.extend(hits)
        all_relations.extend(relation_extractor.extract(doc, hits))

    matched_entities = consolidate_entities(all_entity_hits, mapper)
    seed_catalog_entities = _build_seed_catalog_entities(matcher, mapper)
    all_entities = _merge_entities(seed_catalog_entities, matched_entities)
    all_relations, initially_connected, finally_connected = _ensure_entity_connectivity(all_entities, all_relations)

    store.upsert_documents(all_docs)
    name_to_id = store.upsert_entities(all_entities)
    store.insert_relations(all_relations, name_to_id)
    store.export_csvs(OUTPUT_DIR)
    store.close()

    entity_type_counts = Counter(entity["entity_type"] for entity in all_entities)
    relation_type_counts = Counter(rel["relation_type"] for rel in all_relations)

    summary = {
        "document_count": len(all_docs),
        "entity_count": len(all_entities),
        "matched_entity_count": len(matched_entities),
        "seed_catalog_entity_count": len(seed_catalog_entities),
        "relation_count": len(all_relations),
        "connected_entity_count_before_catalog_links": initially_connected,
        "connected_entity_count_after_catalog_links": finally_connected,
        "entity_connection_ratio": round((finally_connected / len(all_entities)) if all_entities else 0.0, 4),
        "entity_type_counts": dict(entity_type_counts),
        "relation_type_counts": dict(relation_type_counts),
        "fetch_errors": fetch_errors,
        "outputs": {
            "sqlite_db": str(OUTPUT_DIR / "depression_kg.db"),
            "documents_csv": str(OUTPUT_DIR / "documents.csv"),
            "entities_csv": str(OUTPUT_DIR / "entities.csv"),
            "relations_csv": str(OUTPUT_DIR / "relations.csv"),
        },
    }

    dump_json(summary, OUTPUT_DIR / "pipeline_summary.json")
    return summary
