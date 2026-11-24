# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.5: Synced Tables and pgvector on Lakebase
# MAGIC
# MAGIC The chunks table already holds an `embedding` column (written by
# MAGIC `process_chunks` and backfilled in Lecture 2.4). A Lakebase Synced Table
# MAGIC copies it into Postgres, and we search it natively with pgvector.
# MAGIC
# MAGIC ```text
# MAGIC arxiv_chunks_table (Unity Catalog Delta: text + embedding)
# MAGIC          │  Synced Table (managed, read-only copy)
# MAGIC          ▼
# MAGIC Lakebase Postgres: arxiv_chunks_synced
# MAGIC          ├── HNSW cosine index on the embedding
# MAGIC          ├── metadata filters
# MAGIC          └── PostgreSQL full-text ranking
# MAGIC ```
# MAGIC
# MAGIC No embeddings are generated or inserted here: Databricks owns the copy, we
# MAGIC own the indexes and the queries.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.vector_search import LakebasePgVectorManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

source_table = f"{cfg.catalog}.{cfg.schema}.arxiv_chunks_table"
synced_table = f"{cfg.catalog}.{cfg.schema}.arxiv_chunks_synced"
postgres_branch = f"projects/{cfg.lakebase_project_id}/branches/production"

logger.info(f"Environment: {env}")
logger.info(f"Source table: {source_table}")
logger.info(f"Synced table: {synced_table}")
logger.info(f"Lakebase project: {cfg.lakebase_project_id}")
logger.info(f"Embedding endpoint: {cfg.embedding_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validate the Delta source
# MAGIC
# MAGIC Synced Tables require a stable, non-null primary key. The chunk pipeline
# MAGIC creates `id` for every chunk, so it is used as the synchronization key.
# MAGIC Triggered and continuous synchronization also require Delta Change Data
# MAGIC Feed to be enabled on the source table.

# COMMAND ----------

source_columns = {field.name for field in spark.table(source_table).schema.fields}
if "id" not in source_columns:
    raise ValueError(f"{source_table} must contain an 'id' primary-key column.")
if "embedding" not in source_columns:
    raise ValueError(f"{source_table} has no 'embedding' column. Run Lecture 2.4 first.")

spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

logger.info(f"Validated source table and enabled Change Data Feed: {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create or sync the Synced Table
# MAGIC
# MAGIC The first run creates the synced table, which performs the initial
# MAGIC synchronization. Later runs sync the existing table, so new chunks (for
# MAGIC example from the daily data pipeline) reach Lakebase.
# MAGIC
# MAGIC - `type_overrides` syncs `embedding` as a native pgvector `vector(1024)`
# MAGIC   column. Without it, Delta `ARRAY` columns land in Postgres as `JSONB`.
# MAGIC - `new_pipeline_spec` sets the usage policy on the sync pipeline and the
# MAGIC   catalog/schema where the pipeline keeps its checkpoints and event logs.
# MAGIC - `TRIGGERED` is useful for a classroom because changes are synchronized
# MAGIC   when requested. Use `SNAPSHOT` for a complete replacement on each run, or
# MAGIC   `CONTINUOUS` to replicate the source continuously.
# MAGIC
# MAGIC Creating a synced table does not install Postgres extensions, so we enable
# MAGIC pgvector first: the `vector` type override needs it.

# COMMAND ----------

# The manager obtains a short-lived Lakebase database credential through the
# Databricks SDK. It does not store a password in the notebook.
pgvector = LakebasePgVectorManager(config=cfg, workspace=w)
pgvector.ensure_extension()

# COMMAND ----------

import time

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

synced_table_name = f"synced_tables/{synced_table}"


def synced_table_exists() -> bool:
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
    # A Synced Table runs on a Lakeflow pipeline; starting a pipeline update runs a
    # sync, like "Sync now" in the UI. The synced tables documentation describes only
    # the UI button and the Lakeflow Jobs task, so this uses the Pipelines API.
    pipeline_id = w.postgres.get_synced_table(name=synced_table_name).status.pipeline_id
    update_id = w.pipelines.start_update(pipeline_id=pipeline_id).update_id
    logger.info(f"Sync started: pipeline {pipeline_id}, update {update_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wait for the synchronization
# MAGIC
# MAGIC The destination is read-only: write changes to `arxiv_chunks_table`, not to
# MAGIC the Lakebase copy. A `TRIGGERED` table only syncs later source changes when a
# MAGIC sync is started, as above, with **Sync now** on the synced table's Overview
# MAGIC tab, or with a Lakeflow Jobs *Database Table Sync* task.

# COMMAND ----------

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
        # Wait for the sync we started, not just for the table to be online
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

logger.info(f"Synced Table is up to date: {status.detailed_state} | {status.message}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Query the Lakebase mirror
# MAGIC
# MAGIC This table is in Lakebase Postgres, not in the Spark SQL catalog. Query it
# MAGIC through the Lakebase connection after it reaches `ONLINE`. Thanks to the
# MAGIC type override, `embedding` is a pgvector `vector`.

# COMMAND ----------

with pgvector.connection() as conn:
    rows = conn.execute(
        f"""SELECT id, arxiv_id, title, year, pg_typeof(embedding) AS embedding_type, text
            FROM {pgvector.table_name}
            LIMIT 5"""
    ).fetchall()

for row in rows:
    logger.info(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Index the Synced Table for retrieval
# MAGIC
# MAGIC A Synced Table accepts `SELECT`, `CREATE INDEX`, and `DROP TABLE`, but no
# MAGIC writes. The table is owned by an internal `databricks_writer_<dbid>` role; if
# MAGIC `CREATE INDEX` is denied, a `databricks_superuser` must grant you index
# MAGIC permissions. We add:
# MAGIC
# MAGIC - a GIN index on `to_tsvector('english', title || ' ' || text)` for keyword
# MAGIC   search; queries use the same expression, so Postgres can use the index
# MAGIC
# MAGIC The call is idempotent: rerun it if the indexes are missing, for example
# MAGIC after the Synced Table was recreated.

# COMMAND ----------

pgvector.ensure_indexes()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Semantic search
# MAGIC
# MAGIC The query is embedded with the same endpoint as the documents. PostgreSQL
# MAGIC ranks rows by cosine distance and returns the highest-scoring chunks.

# COMMAND ----------

query = "What are the latest techniques in machine learning?"
results = pgvector.search(query=query, num_results=5)

logger.info(f"Query: {query}")
for position, row in enumerate(results, 1):
    logger.info(
        f"{position}. {row.get('title', 'N/A')} | arXiv: {row.get('arxiv_id', 'N/A')} | score={float(row['score']):.4f}"
    )
    logger.info(f"   {row.get('text', '')[:250]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Semantic search with metadata filtering

# COMMAND ----------

query = "neural networks and deep learning"
year = 2026
filtered_results = pgvector.search(query=query, num_results=3, year=year)

logger.info(f"Query: {query} | year={year}")
for position, row in enumerate(filtered_results, 1):
    logger.info(
        f"{position}. {row.get('title', 'N/A')} | year={row.get('year', 'N/A')} | score={float(row['score']):.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Hybrid search: vector similarity + full-text ranking
# MAGIC
# MAGIC Hybrid search combines semantic similarity with PostgreSQL's
# MAGIC `ts_rank_cd()` full-text score. This helps with exact terms, acronyms, and
# MAGIC model names that semantic search alone can underweight.

# COMMAND ----------

query = "transformer architecture attention mechanism"
hybrid_results = pgvector.hybrid_search(query=query, num_results=5)

logger.info(f"Hybrid query: {query}")
for position, row in enumerate(hybrid_results, 1):
    logger.info(
        f"{position}. {row.get('title', 'N/A')} | "
        f"score={float(row['score']):.4f} | "
        f"vector={float(row['vector_score']):.4f} | "
        f"text={float(row['text_score']):.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Compare with `ai_search`: Lakebase and Databricks AI Search
# MAGIC
# MAGIC The `ai_search` SQL function (Beta) queries knowledge sources by type:
# MAGIC `vector_search` for Databricks AI Search indexes (Lecture 2.4) and
# MAGIC `lakebase_search` for Lakebase synced tables. The same question is asked of
# MAGIC each source below.
# MAGIC See [ai_search](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_search).
# MAGIC
# MAGIC `lakebase_search` needs **Lakebase Search** on the project (project
# MAGIC **Settings** > **Lakebase Search** > **Enable Lakebase Search**). Enabling it
# MAGIC restarts the project's computes and cannot be turned off.
# MAGIC
# MAGIC ![Enable Lakebase Search in the project settings](images/lakebase_search_settings.png)
# MAGIC
# MAGIC Then the `lakebase_vector` extension (it builds on pgvector, so our queries
# MAGIC above keep working) and a `lakebase_ann` index on the `vector(1024)` column.

# COMMAND ----------

with pgvector.connection() as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {pgvector.synced_table_name}_embedding_ann
        ON {pgvector.table_name}
        USING lakebase_ann (embedding vector_cosine_ops)
        """
    )
logger.info("lakebase_vector extension and lakebase_ann index ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks AI Search index (Lecture 2.4)

# COMMAND ----------

ai_search_index = f"{cfg.catalog}.{cfg.schema}.arxiv_chunks_ai_search_managed"

ai_search_result = spark.sql(f"""
    SELECT ai_search(
        'transformer architecture attention mechanism',
        PARSE_JSON('[{{
            "type": "vector_search",
            "config": {{
                "index_name": "{ai_search_index}",
                "text_col": "text",
                "doc_uri_col": "id"
            }}
        }}]')
    ) AS result
""")
display(ai_search_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lakebase synced table

# COMMAND ----------

lakebase_result = spark.sql(f"""
    SELECT ai_search(
        'transformer architecture attention mechanism',
        PARSE_JSON('[{{
            "type": "lakebase_search",
            "config": {{
                "index_name": "{synced_table}",
                "text_col": "text",
                "doc_uri_col": "id",
                "embedding_column": "embedding",
                "embedding_model": "{cfg.embedding_endpoint}",
                "lakebase_project_id": "{cfg.lakebase_project_id}"
            }}
        }}]')
    ) AS result
""")
display(lakebase_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Inspect the indexes

# COMMAND ----------

with pgvector.connection() as conn:
    table_count = conn.execute(f"SELECT count(*) FROM {pgvector.table_name}").fetchone()[0]
    indexes = conn.execute(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = %s AND tablename = %s
        ORDER BY indexname
        """,
        (cfg.schema, pgvector.synced_table_name),
    ).fetchall()

logger.info(f"Synced rows: {table_count}")
for index_name, definition in indexes:
    logger.info(f"{index_name}: {definition}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC 1. Keep `arxiv_chunks_table`, with its `embedding` column, as the source of truth.
# MAGIC 2. A Synced Table replicates it into Lakebase; never write to the copy.
# MAGIC 3. Index the copy: HNSW (cosine) on `embedding`, GIN on the full-text expression.
# MAGIC 4. Combine vector, metadata, and full-text ranking in SQL.
# MAGIC
# MAGIC For a new embedding model, re-embed the Delta table, refresh the Synced Table,
# MAGIC and make sure the dimension matches the `vector(1024)` type override.
