from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    ncbi_tool: str = os.getenv("NCBI_TOOL", "depression_kg_template")
    ncbi_email: str = os.getenv("NCBI_EMAIL", "")
    ncbi_api_key: str = os.getenv("NCBI_API_KEY", "")
    request_timeout: int = 60
    user_agent: str = "depression-kg-template/1.0"
    sqlite_path: str = str(OUTPUT_DIR / "depression_kg.db")
    seed_terms_path: str = str(DATA_DIR / "seed_terms.json")
    canonical_map_path: str = str(DATA_DIR / "canonical_map.csv")
    schema_path: str = str(PACKAGE_DIR / "schema.sql")
    sample_pubmed_path: str = str(DATA_DIR / "sample" / "pubmed_docs.json")
    sample_ctgov_path: str = str(DATA_DIR / "sample" / "ctgov_docs.json")


SETTINGS = Settings()
