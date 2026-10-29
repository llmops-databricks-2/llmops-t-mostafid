# Databricks notebook source
"""Ingest Horizon Europe funding calls into a Delta table."""

import json
import os
from datetime import datetime

import requests
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from arxiv_curator.config import get_env, load_config

# COMMAND ----------
# Create Spark session
# In Databricks (notebook/job): normal SparkSession
# On my laptop: Databricks Connect with serverless compute
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    # In Databricks: env comes from the job's base parameters ("dev" when run by hand)
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
    spark = SparkSession.builder.getOrCreate()
else:
    # On my laptop: env from the ENV variable (default "dev")
    from databricks.connect import DatabricksSession

    env = get_env()
    spark = DatabricksSession.builder.serverless().getOrCreate()

cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "funding_calls"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------
# Fetch funding calls from the EU Funding & Tenders Portal search API (public, no key)

API_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"

# Search terms for traffic monitoring and disaster management
SEARCH_TERMS = [
    "flood",
    "wildfire",
    "disaster resilience",
    "earth observation",
    "copernicus",
    "traffic monitoring",
    "mobility",
]


def first(metadata: dict, key: str) -> str | None:
    """Return the first value of a metadata field (the API returns lists)."""
    value = metadata.get(key)
    return value[0] if value else None


def fetch_funding_calls(text: str, page_size: int = 20) -> list[dict]:
    """Fetch open and forthcoming grant calls that match a search text."""
    params = {"apiKey": "SEDIA", "text": text, "pageSize": page_size, "pageNumber": 1}
    query = {
        "bool": {
            "must": [
                {"terms": {"type": ["1", "2"]}},  # grants only
                {"terms": {"status": ["31094501", "31094502"]}},  # forthcoming, open
            ]
        }
    }
    files = {
        "query": ("blob", json.dumps(query), "application/json"),
        "languages": ("blob", json.dumps(["en"]), "application/json"),
    }
    response = requests.post(API_URL, params=params, files=files, timeout=30)
    response.raise_for_status()

    calls = []
    for result in response.json().get("results", []):
        m = result.get("metadata", {})
        calls.append(
            {
                "call_id": first(m, "identifier"),
                "call_identifier": first(m, "callIdentifier"),
                "title": first(m, "title"),
                "description": first(m, "description"),
                "deadline_raw": first(m, "deadlineDate"),
                "status_code": first(m, "status"),
                "url": result.get("url"),
                "source": "horizon_europe",
                "search_term": text,
                "ingestion_timestamp": datetime.now().isoformat(),
            }
        )
    return calls


all_calls = []
for term in SEARCH_TERMS:
    calls = fetch_funding_calls(term)
    logger.info(f"'{term}': {len(calls)} calls")
    all_calls.extend(calls)

logger.info(f"Fetched {len(all_calls)} calls in total (before removing duplicates)")
logger.info(f"Sample: {all_calls[0]['call_id']} - {all_calls[0]['title']}")

# COMMAND ----------
# Preprocess: create a Spark DataFrame and clean it

raw_schema = StructType(
    [
        StructField("call_id", StringType(), True),
        StructField("call_identifier", StringType(), True),
        StructField("title", StringType(), True),
        StructField("description", StringType(), True),
        StructField("deadline_raw", StringType(), True),
        StructField("status_code", StringType(), True),
        StructField("url", StringType(), True),
        StructField("source", StringType(), True),
        StructField("search_term", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
    ]
)

raw_df = spark.createDataFrame(all_calls, schema=raw_schema)

clean_df = (
    raw_df
    # 1. Drop rows without id or title
    .filter(F.col("call_id").isNotNull() & F.col("title").isNotNull())
    # 2. Remove duplicates (the same call is found by several search terms)
    .dropDuplicates(["call_id"])
    # 3. Remove HTML tags and extra spaces from the description
    .withColumn("description", F.regexp_replace("description", "<[^>]+>", " "))
    .withColumn("description", F.trim(F.regexp_replace("description", r"\s+", " ")))
    # 4. Turn the deadline text into a real date
    .withColumn("deadline", F.to_date(F.substring("deadline_raw", 1, 10)))
    # 5. Readable status
    .withColumn(
        "status",
        F.when(F.col("status_code") == "31094501", "forthcoming")
        .when(F.col("status_code") == "31094502", "open")
        .otherwise("other"),
    )
    # 6. Our own open flag: the API status is not always up to date
    .withColumn("is_open", F.col("deadline") >= F.current_date())
    .drop("deadline_raw", "status_code")
)

logger.info(f"Rows before cleaning: {raw_df.count()}, after cleaning: {clean_df.count()}")
logger.info(f"Open calls (deadline in the future): {clean_df.filter('is_open').count()}")

# COMMAND ----------
# Create Delta Table in Unity Catalog
# Store the funding call metadata in a Delta table for downstream processing.

# Columns for lecture 2 (same idea as the teacher's arxiv_papers table)
final_df = clean_df.withColumn("processed", F.lit(None).cast("long")).withColumn(
    "volume_path", F.lit(None).cast("string")
)

table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

final_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table_path)

logger.info(f"Created Delta table: {table_path}")
logger.info(f"Records: {final_df.count()}")

# COMMAND ----------
# Verify the Data

calls_df = spark.table(table_path)

logger.info(f"Table: {table_path}")
logger.info(f"Total calls: {calls_df.count()}")
logger.info("Schema:")
calls_df.printSchema()

logger.info("Sample records:")
calls_df.select("call_id", "title", "deadline", "status", "is_open").show(5, truncate=50)

# COMMAND ----------
# Data Statistics

logger.info("Calls by status:")
calls_df.groupBy("status", "is_open").count().show()

logger.info("Calls by search term:")
calls_df.groupBy("search_term").count().orderBy("count", ascending=False).show()

logger.info("Next deadlines:")
calls_df.filter("is_open").select("call_id", "title", "deadline").orderBy("deadline").show(5, truncate=60)
