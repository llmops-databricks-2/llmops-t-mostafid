# Databricks notebook source
"""Lecture 2.3 (my use case): compare chunking strategies on the funding call texts."""

import os
import re
from functools import partial

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from arxiv_curator.config import get_env, load_config

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
calls_df = spark.table(f"{cfg.catalog}.{cfg.schema}.funding_calls").filter("description IS NOT NULL")
chunks_df = spark.table(f"{cfg.catalog}.{cfg.schema}.funding_call_chunks")

# COMMAND ----------
# 1. Statistics of the current chunks (from 2.2: fixed 1000 characters, overlap 200)

current = chunks_df.select(
    F.count("*").alias("total_chunks"),
    F.avg(F.length("text")).alias("avg_length"),
    F.min(F.length("text")).alias("min_length"),
    F.max(F.length("text")).alias("max_length"),
).collect()[0]

logger.info(f"Current chunks: {current['total_chunks']}")
logger.info(f"  Average length: {current['avg_length']:.0f} characters (~{current['avg_length'] / 4:.0f} tokens)")
logger.info(f"  Min / max length: {current['min_length']} / {current['max_length']} characters")

# COMMAND ----------
# 2. Chunking strategies (fixed sized, sentences)


def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Create fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def sentence_chunking(text: str, max_sentences: int = 5) -> list[str]:
    """Create chunks based on sentence boundaries."""
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


# My own strategy: every call has the same headings, so one section = one chunk
SECTION_PATTERN = r"(?=\b(?:Expected Outcome|Scope|Specific Topic Conditions):)"


def section_chunking(text: str) -> list[str]:
    """Split a call text at its section headings (Expected Outcome, Scope, ...)."""
    return [part.strip() for part in re.split(SECTION_PATTERN, text) if part.strip()]


# COMMAND ----------
# 3. Run every strategy on all calls and compare


def strategy_stats(name: str, func: callable):  # noqa: ANN201
    """Apply a chunking function to all calls and return one row of statistics."""
    chunk_udf = F.udf(func, ArrayType(StringType()))
    texts = calls_df.select(F.explode(chunk_udf("description")).alias("text"))
    length = F.length("text")
    return texts.select(
        F.lit(name).alias("strategy"),
        F.count("*").alias("chunks"),
        F.round(F.avg(length)).alias("avg_chars"),
        F.min(length).alias("min_chars"),
        F.max(length).alias("max_chars"),
        F.round(F.avg(length) / 4).alias("avg_tokens"),
        F.round(100 * F.avg(F.col("text").rlike("^[a-z]").cast("int")), 1).alias("pct_start_mid_word"),
    )


strategies = {
    "fixed_500": partial(fixed_size_chunking, chunk_size=500, overlap=50),
    "fixed_1000": partial(fixed_size_chunking, chunk_size=1000, overlap=200),
    "fixed_2000": partial(fixed_size_chunking, chunk_size=2000, overlap=200),
    "sentence_5": partial(sentence_chunking, max_sentences=5),
    "section": section_chunking,
}

comparison = None
for name, func in strategies.items():
    row = strategy_stats(name, func)
    comparison = row if comparison is None else comparison.unionByName(row)

logger.info("Comparison of chunking strategies (all calls):")
comparison.show(truncate=False)

# COMMAND ----------
# 4. Look at the section chunks of one call

example = calls_df.first()
logger.info(f"Section chunks of {example.call_id}:")
for i, section in enumerate(section_chunking(example.description)):
    logger.info(f"  [{i}] {len(section)} chars: {section[:120]}...")
