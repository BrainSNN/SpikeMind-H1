from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from depression_kg.config import SETTINGS


class MeSHClient:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.user_agent})

    def _common_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if SETTINGS.ncbi_email:
            params["email"] = SETTINGS.ncbi_email
        if SETTINGS.ncbi_api_key:
            params["api_key"] = SETTINGS.ncbi_api_key
        return params

    def search_terms(self, query: str, retmax: int = 20) -> list[dict[str, Any]]:
        params = {
            **self._common_params(),
            "db": "mesh",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
        }
        r = self.session.get(self.BASE_URL + "/esearch.fcgi", params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        return self._fetch_details(ids)

    def _fetch_details(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        params = {
            **self._common_params(),
            "db": "mesh",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        r = self.session.get(self.BASE_URL + "/efetch.fcgi", params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        return self._parse_mesh_xml(r.text)

    def _parse_mesh_xml(self, xml_text: str) -> list[dict[str, Any]]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []  # 返回空列表如果解析失败
        terms = []
        for record in root.findall(".//DescriptorRecord"):
            mesh_id = record.findtext(".//DescriptorUI", "")
            name = record.findtext(".//DescriptorName/String", "")
            if not name:
                continue
            aliases = []
            for term in record.findall(".//TermList/Term/String"):
                term_name = term.text
                if term_name and term_name != name:
                    aliases.append(term_name)
            entity_type = "disease" if "disorder" in name.lower() else "unknown"
            terms.append({
                "id": mesh_id,
                "name": name,
                "aliases": aliases,
                "type": entity_type,
                "source": "mesh",
                "description": f"MeSH term: {name}",
            })
        return terms