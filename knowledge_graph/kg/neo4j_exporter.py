from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from depression_kg.config import SETTINGS


class Neo4jExporter:
    """
    将SQLite数据库导出为Neo4j可直接导入的CSV文件
    
    使用方法:
        exporter = Neo4jExporter()
        exporter.export_to_neo4j("output_dir")
    """
    
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path or SETTINGS.sqlite_path)
    
    def export_to_neo4j(
        self,
        output_dir: str | Path,
        include_documents: bool = True,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        generated_files = {}
        
        # 导出实体节点
        nodes_file = self._export_entity_nodes(conn, output_dir)
        generated_files["entity_nodes"] = nodes_file
        print(f"Export entity nodes: {nodes_file}")
        
        # 导出关系边
        rels_file = self._export_relations(conn, output_dir)
        generated_files["relations"] = rels_file
        print(f"Export relations: {rels_file}")
        
        if include_documents:
            doc_file = self._export_document_nodes(conn, output_dir)
            generated_files["document_nodes"] = doc_file
            print(f"Export document nodes: {doc_file}")
        
        script_file = self._generate_import_script(output_dir, generated_files)
        generated_files["import_script"] = script_file
        
        conn.close()
        return generated_files
    
    def _export_entity_nodes(self, conn: sqlite3.Connection, output_dir: Path) -> Path:
        df = pd.read_sql_query("SELECT entity_id, canonical_name, entity_type, aliases_json, description, source FROM entities", conn)
        
        df.columns = [":ID", "name", ":LABEL", "aliases", "description", "source"]
        df["aliases"] = df["aliases"].apply(lambda x: ", ".join(json.loads(x)) if x and x != "[]" else "")
        df[":LABEL"] = df[":LABEL"].apply(lambda x: f"Entity:{x.title()}" if x else "Entity")
        
        file_path = output_dir / "neo4j_entity_nodes.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def _export_document_nodes(self, conn: sqlite3.Connection, output_dir: Path) -> Path:
        df = pd.read_sql_query("SELECT doc_id, source, source_id, title, abstract, url FROM documents", conn)
        
        df.columns = [":ID", "source", "source_id", "title", "abstract", "url"]
        df[":LABEL"] = df["source"].apply(lambda x: f"Document:{x.title()}")
        df["abstract"] = df["abstract"].apply(lambda x: (x[:2000] + "...") if x and len(x) > 2000 else x)
        
        file_path = output_dir / "neo4j_document_nodes.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def _export_relations(self, conn: sqlite3.Connection, output_dir: Path) -> Path:
        df = pd.read_sql_query("SELECT relation_id, head_entity_id, relation_type, tail_entity_id, evidence_text, source, confidence FROM relations", conn)
        
        df.columns = [":ID", ":START_ID", ":TYPE", ":END_ID", "evidence_text", "source", "confidence"]
        df["evidence_text"] = df["evidence_text"].apply(lambda x: (x[:1000] + "...") if x and len(x) > 1000 else x)
        
        file_path = output_dir / "neo4j_relations.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def _generate_import_script(self, output_dir: Path, files: dict[str, Path]) -> Path:
        script = """// Neo4j Import Guide
// Run with cypher-shell -f neo4j_import_guide.cql
// (or execute block-by-block in Neo4j Browser).

// 0. Optional: clear existing data
MATCH (n) DETACH DELETE n;

// 1. Safe-to-rerun constraints and indexes
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE INDEX entity_name_idx IF NOT EXISTS
FOR (e:Entity) ON (e.name);

CREATE INDEX document_title_idx IF NOT EXISTS
FOR (d:Document) ON (d.title);

// 2. Import entity nodes
LOAD CSV WITH HEADERS FROM 'file:///neo4j_entity_nodes.csv' AS row
MERGE (e:Entity {id: row.`:ID`})
SET e.name = row.name,
    e.raw_label = row.`:LABEL`,
    e.aliases = row.aliases,
    e.description = row.description,
    e.source = row.source;

// 3. Import document nodes
LOAD CSV WITH HEADERS FROM 'file:///neo4j_document_nodes.csv' AS row
MERGE (d:Document {id: row.`:ID`})
SET d.source = row.source,
    d.source_id = row.source_id,
    d.title = row.title,
    d.abstract = row.abstract,
    d.url = row.url,
    d.raw_label = row.`:LABEL`;

// 4. Import relations
LOAD CSV WITH HEADERS FROM 'file:///neo4j_relations.csv' AS row
MATCH (h:Entity {id: row.`:START_ID`}), (t:Entity {id: row.`:END_ID`})
MERGE (h)-[r:RELATED_TO {id: row.`:ID`}]->(t)
SET r.rel_type = row.`:TYPE`,
    r.evidence_text = row.evidence_text,
    r.source = row.source,
    r.confidence = toFloat(row.confidence);

// 5. Quick checks
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC;
MATCH ()-[r]->() RETURN type(r) AS rel_label, count(*) AS count ORDER BY count DESC;
"""

        file_path = output_dir / "neo4j_import_guide.cql"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(script)
        return file_path


if __name__ == "__main__":
    exporter = Neo4jExporter()
    output_dir = SETTINGS.OUTPUT_DIR / "neo4j_export"
    files = exporter.export_to_neo4j(output_dir)
    print("\nExport completed!")
    print("Files:", files)

