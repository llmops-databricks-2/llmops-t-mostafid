# Databricks notebook source
"""part 1: add embeddings to the funding call chunks."""

import os

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from arxiv_curator.config import get_env, load_config

# COMMAND ----------
# Spark session and config
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
source_table = f"{cfg.catalog}.{cfg.schema}.funding_call_chunks"
logger.info(f"Source table: {source_table}")
logger.info(f"Embedding endpoint: {cfg.embedding_endpoint}")

# COMMAND ----------
# 1. Add an embedding to every chunk

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

# COMMAND ----------
# 2. Check the embeddings

chunks = spark.table(source_table)
chunks.select(
    F.count("*").alias("chunks"),
    F.count("embedding").alias("with_embedding"),
    F.min(F.size("embedding")).alias("min_dim"),
    F.max(F.size("embedding")).alias("max_dim"),
).show()

chunks.select("id", F.slice("embedding", 1, 5).alias("first_5_numbers")).show(3, truncate=False)
