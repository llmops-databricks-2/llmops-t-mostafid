# Databricks notebook source
"""Lecture 2.2 (my use case): turn funding call texts into chunks."""

import os
import re

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

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

cfg = load_config("../project_config.yml", env)  # read the ymal file and pick the setting for env (dev,acc,prd)
CALLS_TABLE = f"{cfg.catalog}.{cfg.schema}.funding_calls"  # is the input(the raw funding calls)
CHUNKS_TABLE = f"{cfg.catalog}.{cfg.schema}.funding_call_chunks"  # is the output,
logger.info(f"Source: {CALLS_TABLE} -> chunks: {CHUNKS_TABLE}")

# COMMAND ----------
# 1. Inspect the documents: how long are the call texts?
calls_df = spark.table(CALLS_TABLE)

logger.info("Description length in characters:")
calls_df.select(F.length("description").alias("chars")).summary("count", "min", "50%", "max").show()

logger.info("First 300 characters of 3 calls:")
calls_df.select("call_id", F.substring("description", 1, 300).alias("description_start")).show(3, truncate=False)


# COMMAND ----------
# 2. Chunk the call texts: sentence-based, chosen in 2.3
# (fixed-size cut 57-74 % of chunks mid-word; sections were too long)

MAX_SENTENCES = 5  # sentences per chunk, ~260 tokens on average


def sentence_chunking(text: str, max_sentences: int = MAX_SENTENCES) -> list[str]:
    """Create chunks based on sentence boundaries (the teacher's function from 2.3)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk = [], []
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Spark style, DataProcessor: a UDF + explode, runs in Databricks
chunk_text_udf = F.udf(sentence_chunking, ArrayType(StringType()))

chunks_df = (
    calls_df.filter(F.col("description").isNotNull())
    .withColumn("chunks", chunk_text_udf(F.col("description")))
    .select(
        "call_id",
        "title",
        "deadline",
        "status",
        "is_open",
        "url",
        F.posexplode("chunks").alias("chunk_id", "text"),  # one row per chunk + its number
    )
    .withColumn("id", F.concat_ws("_", F.col("call_id"), F.col("chunk_id").cast("string")))
)

logger.info(f"Created {chunks_df.count()} chunks from {calls_df.count()} calls")

# COMMAND ----------
# 3. Save the chunks to a Delta table with Change Data Feed (needed later for the Lakebase sync)

chunks_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CHUNKS_TABLE)
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
logger.info(f"Saved chunks to {CHUNKS_TABLE} (Change Data Feed enabled)")

# COMMAND ----------
# 4. Inspect the chunk table

saved = spark.table(CHUNKS_TABLE)
saved.printSchema()

logger.info("Chunk length in characters:")
saved.select(F.length("text").alias("chars")).summary("count", "min", "50%", "max").show()

logger.info("Chunks per call:")
saved.groupBy("call_id").count().summary("min", "50%", "max").show()

logger.info("Example chunks of one call:")
saved.filter(F.col("call_id") == saved.first().call_id).select("id", "text").orderBy("chunk_id").show(3, truncate=120)
