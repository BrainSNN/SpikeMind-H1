from __future__ import annotations

from typing import Any

import requests

from depression_kg.config import SETTINGS


class ClinicalTrialsClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.user_agent})

    def search(self, query: str, page_size: int = 20) -> list[dict[str, Any]]:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": query,
            "pageSize": page_size,
            "format": "json",
        }
        r = self.session.get(url, params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        data = r.json()

        studies = data.get("studies", [])
        return [self._normalize_new_study(s) for s in studies]

    def _normalize_legacy_study(self, study: dict[str, Any]) -> dict[str, Any]:
        nct_id = self._first(study.get("NCTId", []))
        title = self._first(study.get("BriefTitle", []))
        conditions = study.get("Condition", [])
        interventions = study.get("InterventionName", [])
        brief_summary = self._first(study.get("BriefSummary", []))
        url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""
        return {
            "doc_id": f"ctgov:{nct_id}",
            "source": "clinicaltrials",
            "source_id": nct_id,
            "title": title,
            "abstract": brief_summary,
            "url": url,
            "metadata": {
                "conditions": conditions,
                "interventions": interventions,
            },
        }

    def _normalize_new_study(self, study: dict[str, Any]) -> dict[str, Any]:
        identification = study.get("protocolSection", {}).get("identificationModule", {})
        conditions_mod = study.get("protocolSection", {}).get("conditionsModule", {})
        design_mod = study.get("protocolSection", {}).get("designModule", {})
        desc_mod = study.get("protocolSection", {}).get("descriptionModule", {})
        arms_mod = study.get("protocolSection", {}).get("armsInterventionsModule", {})

        nct_id = identification.get("nctId", "")
        title = identification.get("briefTitle", "")
        conditions = conditions_mod.get("conditions", []) or []
        interventions = []
        for item in arms_mod.get("interventions", []) or []:
            name = item.get("name")
            if name:
                interventions.append(name)
        brief_summary = desc_mod.get("briefSummary", "")
        model = design_mod.get("studyType", "")
        url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""

        return {
            "doc_id": f"ctgov:{nct_id}",
            "source": "clinicaltrials",
            "source_id": nct_id,
            "title": title,
            "abstract": brief_summary,
            "url": url,
            "metadata": {
                "conditions": conditions,
                "interventions": interventions,
                "study_type": model,
            },
        }

    @staticmethod
    def _first(values: list[str]) -> str:
        return values[0] if values else ""
