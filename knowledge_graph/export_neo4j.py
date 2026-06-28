# Neo4j Export Script
# Usage: .venv\Scripts\python.exe export_neo4j.py

from pathlib import Path
from depression_kg.neo4j_exporter import Neo4jExporter
from depression_kg.config import SETTINGS

if __name__ == "__main__":
    print("=" * 60)
    print("Neo4j Export Tool")
    print("=" * 60)
    
    # 输出目录
    output_dir = Path("output/neo4j_export")
    
    print(f"\nSource DB: {SETTINGS.sqlite_path}")
    print(f"Output Dir: {output_dir}")
    print()
    
    # 执行导出
    exporter = Neo4jExporter()
    files = exporter.export_to_neo4j(output_dir)
    
    print("\n" + "=" * 60)
    print("Export Completed!")
    print("=" * 60)
    print("\nGenerated Files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    
    print("\n" + "=" * 60)
    print("Neo4j Import Instructions")
    print("=" * 60)
    print("""
1. Copy CSV files to Neo4j import directory:
   - Windows: C:\\Users\\<user>\\Documents\\Neo4j\\<name>\\import\\
   - Linux/Mac: /var/lib/neo4j/import/

2. Open Neo4j Browser (http://localhost:7474)

3. Run these commands:
   MATCH (n) DETACH DELETE n;
   
   LOAD CSV WITH HEADERS FROM 'file:///neo4j_entity_nodes.csv' AS row
   CREATE (e:Entity) SET e = row;
   
   LOAD CSV WITH HEADERS FROM 'file:///neo4j_relations.csv' AS row
   MATCH (h {id: row.`:START_ID`}), (t {id: row.`:END_ID`})
   CREATE (h)-[r]->(t) SET r = row;
   
   CREATE INDEX FOR (e:Entity) ON (e.name);
""")
