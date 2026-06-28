[CmdletBinding()]
param(
    [string]$Neo4jUri = "bolt://localhost:7687",
    [string]$Database = "neo4j",
    [string]$Neo4jUser = "neo4j",
    [string]$Neo4jPassword = $env:NEO4J_PASSWORD,
    [string]$ExportDir = "",
    [string]$ImportDir = "",
    [string]$CypherShellPath = "",
    [int]$BatchSize = 1000,
    [bool]$IncludeDocuments = $true,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptBaseDir = if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $PSScriptRoot
}
else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}

if ([string]::IsNullOrWhiteSpace($ExportDir)) {
    $ExportDir = Join-Path $scriptBaseDir "output\neo4j_export"
}

function Resolve-CypherShell {
    param([string]$PreferredPath)

    if (-not [string]::IsNullOrWhiteSpace($PreferredPath)) {
        if (Test-Path -LiteralPath $PreferredPath) {
            return (Resolve-Path -LiteralPath $PreferredPath).Path
        }
        throw "Cypher shell path not found: $PreferredPath"
    }

    $cmd = Get-Command cypher-shell -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }

    $cmdBat = Get-Command cypher-shell.bat -ErrorAction SilentlyContinue
    if ($null -ne $cmdBat) {
        return $cmdBat.Source
    }

    throw "Cannot find cypher-shell. Add it to PATH or pass -CypherShellPath."
}

function Resolve-Neo4jImportDir {
    param([string]$PreferredPath)

    if (-not [string]::IsNullOrWhiteSpace($PreferredPath)) {
        if (Test-Path -LiteralPath $PreferredPath) {
            return (Resolve-Path -LiteralPath $PreferredPath).Path
        }
        throw "Neo4j import directory not found: $PreferredPath"
    }

    $desktopDbmsRoot = Join-Path $env:USERPROFILE ".Neo4jDesktop\relate-data\dbmss"
    if (Test-Path -LiteralPath $desktopDbmsRoot) {
        $latestDbms = Get-ChildItem -LiteralPath $desktopDbmsRoot -Directory |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -ne $latestDbms) {
            $candidate = Join-Path $latestDbms.FullName "import"
            if (Test-Path -LiteralPath $candidate) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }
        }
    }

    throw "Cannot auto-detect Neo4j import directory. Pass -ImportDir explicitly."
}

function Ensure-FileExists {
    param([string]$PathToCheck)

    if (-not (Test-Path -LiteralPath $PathToCheck)) {
        throw "Required file not found: $PathToCheck"
    }
}

if ([string]::IsNullOrWhiteSpace($Neo4jPassword)) {
    throw "Missing Neo4j password. Pass -Neo4jPassword or set NEO4J_PASSWORD env var."
}

$resolvedExportDir = (Resolve-Path -LiteralPath $ExportDir).Path
$resolvedImportDir = Resolve-Neo4jImportDir -PreferredPath $ImportDir
$resolvedCypherShell = Resolve-CypherShell -PreferredPath $CypherShellPath

$filesToCopy = @(
    "neo4j_entity_nodes.csv",
    "neo4j_relations.csv"
)
if ($IncludeDocuments) {
    $filesToCopy += "neo4j_document_nodes.csv"
}

Write-Host "== Neo4j One-Click Import ==" -ForegroundColor Cyan
Write-Host "Export dir: $resolvedExportDir"
Write-Host "Import dir: $resolvedImportDir"
Write-Host "Cypher shell: $resolvedCypherShell"
Write-Host "Target: $Neo4jUri / db=$Database"

foreach ($name in $filesToCopy) {
    Ensure-FileExists -PathToCheck (Join-Path $resolvedExportDir $name)
}

foreach ($name in $filesToCopy) {
    $src = Join-Path $resolvedExportDir $name
    $dst = Join-Path $resolvedImportDir $name
    if ($DryRun) {
        Write-Host "[DryRun] Copy $src -> $dst"
    }
    else {
        Copy-Item -LiteralPath $src -Destination $dst -Force
        Write-Host "Copied: $name"
    }
}

$entityQuery = @'
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_entity_nodes.csv' AS row
CALL {
    WITH row
    MERGE (e:Entity {id: row.`:ID`})
    SET e.name = row.name,
        e.type = row.`:LABEL`,
        e.aliases = row.aliases,
        e.description = row.description,
        e.source = row.source
} IN TRANSACTIONS OF __BATCH__ ROWS;
'@

$documentQuery = @'
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_document_nodes.csv' AS row
CALL {
    WITH row
    MERGE (d:Document {id: row.`:ID`})
    SET d.source = row.source,
        d.source_id = row.source_id,
        d.title = row.title,
        d.abstract = row.abstract,
        d.url = row.url
} IN TRANSACTIONS OF __BATCH__ ROWS;
'@

$relationQuery = @'
LOAD CSV WITH HEADERS FROM 'file:///neo4j_relations.csv' AS row
CALL {
    WITH row
    MATCH (h:Entity {id: row.`:START_ID`})
    MATCH (t:Entity {id: row.`:END_ID`})
    MERGE (h)-[r:RELATION {id: row.`:ID`}]->(t)
    SET r.rel_type = row.`:TYPE`,
        r.evidence = row.evidence_text,
        r.source = row.source,
        r.confidence = toFloat(row.confidence)
} IN TRANSACTIONS OF __BATCH__ ROWS;

CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title);

MATCH (n) RETURN labels(n) AS labels, count(*) AS count ORDER BY count DESC;
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count ORDER BY count DESC;
'@

$combinedQuery = $entityQuery
if ($IncludeDocuments) {
    $combinedQuery += "`n`n$documentQuery"
}
$combinedQuery += "`n`n$relationQuery"
$combinedQuery = $combinedQuery.Replace("__BATCH__", [string]$BatchSize)

$queryFile = Join-Path $env:TEMP ("neo4j_import_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".cypher")
Set-Content -LiteralPath $queryFile -Value $combinedQuery -Encoding UTF8

if ($DryRun) {
    Write-Host "[DryRun] Query file: $queryFile"
    Write-Host "[DryRun] Command:"
    Write-Host "  $resolvedCypherShell -a $Neo4jUri -d $Database -u $Neo4jUser -p ******** -f $queryFile"
    exit 0
}

Write-Host "Running cypher-shell import..." -ForegroundColor Yellow
& $resolvedCypherShell -a $Neo4jUri -d $Database -u $Neo4jUser -p $Neo4jPassword -f $queryFile

if ($LASTEXITCODE -ne 0) {
    throw "cypher-shell failed with exit code $LASTEXITCODE"
}

Write-Host "Import finished successfully." -ForegroundColor Green
Write-Host "Query file kept at: $queryFile"
