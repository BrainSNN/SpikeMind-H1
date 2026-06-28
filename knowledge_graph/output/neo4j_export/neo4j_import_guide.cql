// Neo4j Import Guide
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
