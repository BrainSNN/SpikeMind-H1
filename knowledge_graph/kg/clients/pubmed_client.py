from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from depression_kg.config import SETTINGS

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class PubMedClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SETTINGS.user_agent})

    def _common_params(self) -> dict[str, str]:
        params: dict[str, str] = {"tool": SETTINGS.ncbi_tool}
        if SETTINGS.ncbi_email:
            params["email"] = SETTINGS.ncbi_email
        if SETTINGS.ncbi_api_key:
            params["api_key"] = SETTINGS.ncbi_api_key
        return params

    def search(self, query: str, retmax: int = 20) -> list[str]:
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
        }
        r = self.session.get(BASE + "esearch.fcgi", params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_details(self, pmids: list[str]) -> str:
        if not pmids:
            return ""
        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        r = self.session.get(BASE + "efetch.fcgi", params=params, timeout=SETTINGS.request_timeout)
        r.raise_for_status()
        return r.text

    def parse_articles(self, xml_text: str) -> list[dict[str, Any]]:
        if not xml_text.strip():
            return []

        root = ET.fromstring(xml_text)
        articles: list[dict[str, Any]] = []

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", default="")
            title = article.findtext(".//ArticleTitle", default="")
            abstract_parts = article.findall(".//Abstract/AbstractText")
            abstract = " ".join("".join(node.itertext()) for node in abstract_parts if node is not None)

            article_ids = []
            for aid in article.findall(".//ArticleId"):
                id_type = aid.attrib.get("IdType", "")
                value = (aid.text or "").strip()
                if value:
                    article_ids.append({"id_type": id_type, "value": value})

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            articles.append(
                {
                    "doc_id": f"pubmed:{pmid}",
                    "source": "pubmed",
                    "source_id": pmid,
                    "title": title,
                    "abstract": abstract,
                    "url": url,
                    "metadata": {"article_ids": article_ids},
                }
            )
        return articles

    def fetch_by_queries(self, queries: list[str], retmax: int = 20, sleep_seconds: float = 0.34) -> list[dict[str, Any]]:
        all_articles: list[dict[str, Any]] = []

        for query in queries:
            pmids = self.search(query=query, retmax=retmax)
            xml_text = self.fetch_details(pmids)
            articles = self.parse_articles(xml_text)
            all_articles.extend(articles)
            time.sleep(sleep_seconds)

        return all_articles
