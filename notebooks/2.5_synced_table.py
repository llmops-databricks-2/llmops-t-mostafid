# Databricks notebook source
"""Lecture 2.5 (my use case): sync funding call chunks to Lakebase and search with pgvector."""

import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.pipelines import UpdateInfoState
from databricks.sdk.service.postgres import (
    NewPipelineSpec,
    SyncedTable,
    SyncedTableState,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.vector_search import LakebasePgVectorManager

# COMMAND ----------
# Spark session and config (same pattern as 2.2)
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
    spark = SparkSession.builder.getOrCreate()
else:
    from databricks.connect import DatabricksSession

    env = get_env()
    spark = DatabricksSession.builder.serverless().getOrCreate()

cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

source_table = f"{cfg.catalog}.{cfg.schema}.funding_call_chunks"
synced_table = f"{cfg.catalog}.{cfg.schema}.funding_call_chunks_synced"
postgres_branch = f"projects/{cfg.lakebase_project_id}/branches/production"

logger.info(f"Source table: {source_table}")
logger.info(f"Synced table: {synced_table}")
logger.info(f"Lakebase project: {cfg.lakebase_project_id}")

# COMMAND ----------
# 1. Validate the Delta source: needs an 'id' key, an 'embedding' column and Change Data Feed

source_columns = {field.name for field in spark.table(source_table).schema.fields}
if "id" not in source_columns:
    raise ValueError(f"{source_table} must contain an 'id' primary-key column.")
if "embedding" not in source_columns:
    raise ValueError(f"{source_table} has no 'embedding' column. Run 2.4_embeddings.py first.")

spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
logger.info(f"Validated source table: {source_table}")

# COMMAND ----------
# 2. Enable pgvector in Lakebase (must happen before the synced table is created)

pgvector = LakebasePgVectorManager(config=cfg, workspace=w)
pgvector.ensure_extension()

# COMMAND ----------
# 3. Create the Synced Table (first run) or start a sync (later runs)

synced_table_name = f"synced_tables/{synced_table}"


def synced_table_exists() -> bool:
    """Return True if the synced table already exists."""
    try:
        w.postgres.get_synced_table(name=synced_table_name)
        return True
    except NotFound:
        return False


if not synced_table_exists():
    w.postgres.create_synced_table(
        synced_table=SyncedTable(
            spec=SyncedTableSyncedTableSpec(
                source_table_full_name=source_table,
                branch=postgres_branch,
                postgres_database="databricks_postgres",
                primary_key_columns=["id"],
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                create_database_objects_if_missing=True,
                new_pipeline_spec=NewPipelineSpec(
                    budget_policy_id=cfg.usage_policy_id,
                    storage_catalog=cfg.catalog,
                    storage_schema=cfg.schema,
                ),
                type_overrides=[
                    SyncedTableSyncedTableSpecTypeOverride(
                        column_name="embedding",
                        pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
                        size=1024,
                    )
                ],
            )
        ),
        synced_table_id=synced_table,
    ).wait()
    logger.info(f"Created Synced Table: {synced_table}")
    update_id = None
else:
    pipeline_id = w.postgres.get_synced_table(name=synced_table_name).status.pipeline_id
    update_id = w.pipelines.start_update(pipeline_id=pipeline_id).update_id
    logger.info(f"Sync started: pipeline {pipeline_id}, update {update_id}")

# COMMAND ----------
# 4. Wait until the copy is up to date

online_states = {
    SyncedTableState.SYNCED_TABLE_ONLINE,
    SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
}
failed_states = {
    SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
    SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
}

while True:
    status = w.postgres.get_synced_table(name=synced_table_name).status
    if update_id:
        update = w.pipelines.get_update(pipeline_id=status.pipeline_id, update_id=update_id).update
        if update.state == UpdateInfoState.COMPLETED:
            break
        if update.state in {UpdateInfoState.FAILED, UpdateInfoState.CANCELED}:
            raise RuntimeError(f"Sync {update_id} ended in state {update.state}")
    elif status.detailed_state in online_states:
        break
    if status.detailed_state in failed_states:
        raise RuntimeError(f"Synced Table failed: {status.detailed_state} | {status.message}")
    logger.info(f"Waiting for sync: {status.detailed_state}")
    time.sleep(30)

logger.info(f"Synced Table is up to date: {status.detailed_state}")

# COMMAND ----------
# 5. Look at the Lakebase copy (this is Postgres, so we query it through pgvector.connection())

with pgvector.connection() as conn:
    count = conn.execute(f"SELECT count(*) FROM {pgvector.table_name}").fetchone()[0]
    rows = conn.execute(
        f"""SELECT id, call_id, is_open, pg_typeof(embedding) AS embedding_type, left(text, 80)
            FROM {pgvector.table_name}
            LIMIT 3"""
    ).fetchall()

logger.info(f"Rows in Lakebase: {count}")
for row in rows:
    logger.info(row)


# COMMAND ----------
# 6. Index the copy for keyword search (GIN index, like the teacher's ensure_indexes)

pgvector.ensure_indexes()


def show(title: str, results: list[dict]) -> None:
    """Print search results in a short, readable form."""
    logger.info(title)
    for position, row in enumerate(results, 1):
        scores = f"score={float(row['score']):.3f}"
        if "vector_score" in row:
            scores += f" (vector={float(row['vector_score']):.3f}, text={float(row['text_score']):.3f})"
        logger.info(f"  {position}. {row['call_id']} | open={row['is_open']} | deadline={row['deadline']} | {scores}")
        logger.info(f"     {row['title'][:90]}")


# COMMAND ----------
# 7. Semantic search: by meaning

query = "satellite-based flood monitoring and early warning"
show(f"SEMANTIC: {query}", pgvector.search(query=query, num_results=5))

# COMMAND ----------
# 8. Filtered search: same question, only calls you can still apply to

show(f"FILTERED (is_open=True): {query}", pgvector.search(query=query, num_results=5, is_open=True))

# COMMAND ----------
# 9. Hybrid search vs. semantic on a question with an exact name

query = "Copernicus emergency management service"
show(f"SEMANTIC: {query}", pgvector.search(query=query, num_results=5))
show(f"HYBRID: {query}", pgvector.hybrid_search(query=query, num_results=5))
