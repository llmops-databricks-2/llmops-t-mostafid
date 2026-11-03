# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Databricks AI Search
# MAGIC
# MAGIC Databricks AI Search, formerly Databricks Vector Search, is Databricks'
# MAGIC managed vector retrieval service. It indexes a Unity Catalog Delta table
# MAGIC and keeps the index synchronized with it.
# MAGIC
# MAGIC ```text
# MAGIC arxiv_chunks_table (Unity Catalog Delta: text + embedding)
# MAGIC          │
# MAGIC          ├── Delta Sync index, managed embeddings      (AI Search embeds `text`)
# MAGIC          └── Delta Sync index, self-managed embeddings (reads `embedding`)
# MAGIC                    │
# MAGIC                    ▼
# MAGIC          similarity_search(query_text=... | query_vector=...)
# MAGIC ```
# MAGIC
# MAGIC Lecture 2.5 serves the same `embedding` column from Lakebase with pgvector.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

source_table = f"{cfg.catalog}.{cfg.schema}.arxiv_chunks_table"
ai_search_endpoint = f"{env}-arxiv-ai-search"
managed_index = f"{cfg.catalog}.{cfg.schema}.arxiv_chunks_ai_search_managed"
self_managed_index = f"{cfg.catalog}.{cfg.schema}.arxiv_chunks_ai_search_self_managed"

if not cfg.usage_policy_id:
    raise ValueError("usage_policy_id is missing in project_config.yml")

logger.info(f"Environment: {env}")
logger.info(f"Source table: {source_table}")
logger.info(f"Embedding endpoint: {cfg.embedding_endpoint}")
logger.info(f"AI Search endpoint: {ai_search_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Prepare the Delta source
# MAGIC
# MAGIC A Delta Sync index reads from a Unity Catalog table. The source needs a
# MAGIC stable primary key (`id`, created by our chunking pipeline) and Change Data
# MAGIC Feed for incremental synchronization.
# MAGIC
# MAGIC `process_chunks` writes an `embedding` column with `ai_query`. Chunks written
# MAGIC before that column existed have no embedding, so we backfill them here.

# COMMAND ----------

source_columns = {field.name for field in spark.table(source_table).schema.fields}
if "id" not in source_columns:
    raise ValueError(f"{source_table} must contain an 'id' primary-key column.")
if "embedding" not in source_columns:
    spark.sql(f"ALTER TABLE {source_table} ADD COLUMNS (embedding ARRAY<FLOAT>)")

spark.sql(f"""
    MERGE INTO {source_table} AS target
    USING (
        SELECT id, CAST(ai_query('{cfg.embedding_endpoint}', text) AS ARRAY<FLOAT>) AS embedding
        FROM {source_table}
        WHERE embedding IS NULL
    ) AS source
    ON target.id = source.id
    WHEN MATCHED THEN UPDATE SET target.embedding = source.embedding
""")

spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
logger.info(f"Validated AI Search source: {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create or reuse an AI Search endpoint
# MAGIC
# MAGIC The Python client still exposes the historical `VectorSearchClient` name;
# MAGIC the managed product is now called Databricks AI Search. The usage policy
# MAGIC attributes the endpoint's cost, as for our serverless jobs.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

ai_search = VectorSearchClient()

if not ai_search.endpoint_exists(name=ai_search_endpoint):
    ai_search.create_endpoint_and_wait(
        name=ai_search_endpoint,
        endpoint_type="STANDARD",
        usage_policy_id=cfg.usage_policy_id,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the two Delta Sync indexes
# MAGIC
# MAGIC | Option | Index reads | Who embeds documents | Who embeds queries |
# MAGIC |---|---|---|---|
# MAGIC | Managed embeddings | `text` | AI Search, with `embedding_model_endpoint_name` | AI Search (`query_text`) |
# MAGIC | Self-managed embeddings | `embedding` | Our pipeline (`process_chunks`, `ai_query`) | Us (`query_vector`) |
# MAGIC
# MAGIC Managed embeddings are the least code. Self-managed embeddings reuse the
# MAGIC vectors already in Delta, so AI Search and Lakebase (Lecture 2.5) search the
# MAGIC same vectors, and the embedding model is a pipeline decision.
# MAGIC
# MAGIC An existing index is synchronized instead of created again: `TRIGGERED`
# MAGIC indexes only pick up source changes (for example new chunks from the daily
# MAGIC data pipeline) when we call `sync()`. A new index runs its first sync as part
# MAGIC of creation.

# COMMAND ----------

index_specs = {
    # Option A: AI Search embeds the text column
    managed_index: {
        "embedding_source_column": "text",
        "embedding_model_endpoint_name": cfg.embedding_endpoint,
    },
    # Option B: AI Search indexes our precomputed embedding column
    self_managed_index: {
        "embedding_vector_column": "embedding",
        "embedding_dimension": 1024,
    },
}

for index_name, index_spec in index_specs.items():
    if not ai_search.index_exists(endpoint_name=ai_search_endpoint, index_name=index_name):
        ai_search.create_delta_sync_index(
            endpoint_name=ai_search_endpoint,
            source_table_name=source_table,
            index_name=index_name,
            pipeline_type="TRIGGERED",
            primary_key="id",
            usage_policy_id=cfg.usage_policy_id,
            **index_spec,
        )
    else:
        ai_search.get_index(index_name=index_name).sync()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Search options
# MAGIC
# MAGIC All queries below go through `similarity_search`. We wait until both
# MAGIC indexes are ready and their syncs have finished, then compare the options on
# MAGIC the managed index first:
# MAGIC
# MAGIC | Option | Parameter | What it does |
# MAGIC |---|---|---|
# MAGIC | Semantic (ANN) | `query_type="ANN"` (default) | Nearest neighbours by embedding similarity |
# MAGIC | Hybrid | `query_type="HYBRID"` | Combines vector similarity with keyword (BM25) matching |
# MAGIC | Metadata filters | `filters={...}` | Restricts candidates before ranking |
# MAGIC | Score threshold | `score_threshold=` | Drops weak matches instead of always returning `num_results` |
# MAGIC | Reranking | `reranker=DatabricksReranker(...)` | Reorders the candidates with a reranking model over chosen columns |

# COMMAND ----------

from databricks.vector_search.reranker import DatabricksReranker


def show(title: str, results: dict) -> None:
    logger.info(title)
    columns = [column["name"] for column in results["manifest"]["columns"]]
    for row in results["result"].get("data_array", []):
        record = dict(zip(columns, row))
        logger.info(f"  {record.get('score', 0):.4f} | {record.get('year')} | {record.get('title', '')[:80]}")


query = "Chunking strategies for document processing"
columns = ["id", "arxiv_id", "title", "year", "text"]

managed = ai_search.get_index(index_name=managed_index)
self_managed = ai_search.get_index(index_name=self_managed_index)
managed.wait_until_ready(wait_for_updates=True)
self_managed.wait_until_ready(wait_for_updates=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Semantic search (ANN)

# COMMAND ----------

show(
    "ANN",
    managed.similarity_search(query_text=query, columns=columns, num_results=5),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Hybrid search
# MAGIC
# MAGIC Keyword matching helps with exact terms, acronyms, and model names that
# MAGIC semantic search alone can underweight.

# COMMAND ----------

show(
    "HYBRID",
    managed.similarity_search(query_text=query, columns=columns, num_results=5, query_type="HYBRID"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Metadata filters
# MAGIC
# MAGIC Filters are a dictionary on standard endpoints. The key is a column name,
# MAGIC optionally followed by an operator:
# MAGIC
# MAGIC | Filter | Meaning |
# MAGIC |---|---|
# MAGIC | `{"year": 2026}` | equal to |
# MAGIC | `{"year": [2025, 2026]}` | in the list |
# MAGIC | `{"year >=": 2025}` | comparison: `<`, `<=`, `>`, `>=` |
# MAGIC | `{"arxiv_id NOT": "2501.00001"}` | not equal to |
# MAGIC | `{"title LIKE": "agent"}` | contains the token |
# MAGIC
# MAGIC Several keys in one dictionary are combined with `AND`.

# COMMAND ----------

show(
    "Filter: year = 2026",
    managed.similarity_search(query_text=query, columns=columns, num_results=5, filters={"year": 2026}),
)
show(
    "Filter: year >= 2025 AND title contains 'agent'",
    managed.similarity_search(
        query_text=query,
        columns=columns,
        num_results=5,
        filters={"year >=": 2025, "title LIKE": "agent"},
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Score threshold

# COMMAND ----------

show(
    "ANN with score_threshold=0.6",
    managed.similarity_search(query_text=query, columns=columns, num_results=10, score_threshold=0.6),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Reranking
# MAGIC
# MAGIC The reranker reorders the retrieved candidates using the text of the chosen
# MAGIC columns. Here we combine it with hybrid retrieval and a filter.

# COMMAND ----------

show(
    "HYBRID + filter + reranker",
    managed.similarity_search(
        query_text=query,
        columns=columns,
        filters={"year": 2026},
        num_results=5,
        query_type="HYBRID",
        reranker=DatabricksReranker(columns_to_rerank=["text", "title"]),
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.6 Self-managed index: we embed the query
# MAGIC
# MAGIC The self-managed index has no embedding model, so we embed the query with
# MAGIC the model that produced the `embedding` column and pass `query_vector`.
# MAGIC Hybrid search also needs `query_text` for the keyword part.

# COMMAND ----------

client = w.serving_endpoints.get_open_ai_client()
query_vector = client.embeddings.create(model=cfg.embedding_endpoint, input=query).data[0].embedding
show(
    "Self-managed ANN",
    self_managed.similarity_search(query_vector=query_vector, columns=columns, num_results=5),
)
show(
    "Self-managed HYBRID",
    self_managed.similarity_search(
        query_vector=query_vector,
        query_text=query,
        columns=columns,
        num_results=5,
        query_type="HYBRID",
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Search versus Lakebase pgvector
# MAGIC
# MAGIC | Concern | Databricks AI Search (2.4) | Lakebase pgvector (2.5) |
# MAGIC |---|---|---|
# MAGIC | Index owner | Databricks managed service | Our PostgreSQL indexes on a Synced Table |
# MAGIC | Source sync | Delta Sync index | Lakebase Synced Table |
# MAGIC | Embeddings | Managed by AI Search, or our `embedding` column | Our `embedding` column |
# MAGIC | Query API | `similarity_search` / AI Search APIs | PostgreSQL SQL through `psycopg` |
# MAGIC | Custom SQL and joins | Limited to index query capabilities | Native PostgreSQL |
# MAGIC | Course default | Comparison | Used by the remaining lectures |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup reminder
# MAGIC
# MAGIC AI Search endpoints and indexes are managed resources. Delete them from the
# MAGIC Databricks UI or with the AI Search client when the comparison is complete.
