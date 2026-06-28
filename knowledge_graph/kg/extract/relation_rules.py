from __future__ import annotations

from typing import Any


class RelationExtractor:
    def extract(self, doc: dict[str, Any], entity_hits: list[dict]) -> list[dict[str, Any]]:
        text = ((doc.get("title") or "") + " " + (doc.get("abstract") or "")).lower()
        names_by_type: dict[str, list[str]] = {}
        for hit in entity_hits:
            names_by_type.setdefault(hit["entity_type"], []).append(hit["canonical_name"])

        relations: list[dict[str, Any]] = []

        diseases = names_by_type.get("disease", [])
        symptoms = names_by_type.get("symptom", [])
        drugs = names_by_type.get("drug", [])
        treatments = names_by_type.get("treatment", [])
        scales = names_by_type.get("scale", [])
        risk_factors = names_by_type.get("risk_factor", [])
        biomarkers = names_by_type.get("biomarker", [])

        for disease in diseases:
            for symptom in symptoms:
                relations.append(self._rel(disease, "has_symptom", symptom, doc, 0.75))
            for risk in risk_factors:
                if any(word in text for word in ["risk", "associated", "predict", "factor"]):
                    relations.append(self._rel(disease, "has_risk_factor", risk, doc, 0.72))
            for scale in scales:
                relations.append(self._rel(disease, "measured_by", scale, doc, 0.70))
            for biomarker in biomarkers:
                if any(word in text for word in ["eeg", "biomarker", "marker", "signal"]):
                    relations.append(self._rel(disease, "related_to", biomarker, doc, 0.67))

        for drug in drugs:
            for disease in diseases:
                if any(word in text for word in ["treat", "therapy", "improve", "response", "antidepressant"]):
                    relations.append(self._rel(drug, "treats", disease, doc, 0.80))

        for treatment in treatments:
            for disease in diseases:
                if any(word in text for word in ["treat", "therapy", "intervention", "psychotherapy", "cbt"]):
                    relations.append(self._rel(treatment, "treats", disease, doc, 0.78))

        sleep_stages = names_by_type.get("sleep_stage", [])
        sleep_symptoms = names_by_type.get("sleep_symptom", [])
        sleep_patterns = names_by_type.get("sleep_pattern", [])
        eeg_features = names_by_type.get("eeg_feature", [])

        for stage in sleep_stages:
            for symptom in sleep_symptoms:
                relations.append(self._rel(stage, "associated_with", symptom, doc, 0.78))

            for feat in eeg_features:
                relations.append(self._rel(stage, "characterized_by", feat, doc, 0.82))

        for symptom in sleep_symptoms:
            for stage in sleep_stages:
                relations.append(self._rel(symptom, "suggests", stage, doc, 0.75))

        for pattern in sleep_patterns:
            for stage in sleep_stages:
                relations.append(self._rel(pattern, "related_to", stage, doc, 0.74))

        for disease in diseases:
            for stage in sleep_stages:
                relations.append(self._rel(disease, "related_to", stage, doc, 0.70))

        # Emotion-focused relations for affective understanding.
        emotions = names_by_type.get("emotion", [])
        affect_dimensions = names_by_type.get("affect_dimension", [])
        emotion_regulations = names_by_type.get("emotion_regulation", [])

        for disease in diseases:
            for emotion in emotions:
                relations.append(self._rel(disease, "associated_with", emotion, doc, 0.73))

        for symptom in symptoms:
            for emotion in emotions:
                relations.append(self._rel(symptom, "associated_with", emotion, doc, 0.71))

        for sleep_symptom in sleep_symptoms:
            for emotion in emotions:
                relations.append(self._rel(sleep_symptom, "associated_with", emotion, doc, 0.69))

        for emotion in emotions:
            for dim in affect_dimensions:
                relations.append(self._rel(emotion, "has_dimension", dim, doc, 0.68))
            for strategy in emotion_regulations:
                if any(word in text for word in ["regulation", "reappraisal", "suppression", "coping", "mindfulness"]):
                    relations.append(self._rel(strategy, "modulates", emotion, doc, 0.70))

        deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for rel in relations:
            key = (rel["head_name"], rel["relation_type"], rel["tail_name"], rel["source_id"])
            deduped[key] = rel
        return list(deduped.values())

    @staticmethod
    def _rel(head_name: str, relation_type: str, tail_name: str, doc: dict[str, Any], confidence: float) -> dict[str, Any]:
        return {
            "head_name": head_name,
            "relation_type": relation_type,
            "tail_name": tail_name,
            "evidence_text": (doc.get("abstract") or doc.get("title") or "")[:500],
            "source": doc.get("source", ""),
            "source_id": doc.get("source_id", ""),
            "confidence": confidence,
        }
