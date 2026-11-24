# Databricks notebook source

from datetime import datetime

import arxiv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from arxiv_curator.config import get_env, load_config

# COMMAND ----------
# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "arxiv_papers"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------
# Fetch arXiv Paper Metadata from arXiv API: https://arxiv.org/help/api/index


def fetch_arxiv_papers(query: str = "cat:cs.AI OR cat:cs.LG", max_results: int = 100):
    """Fetch arXiv papers using the arXiv API.

    Args:
        query: arXiv search query (default: AI and ML papers)
        max_results: Maximum number of papers to fetch

    Returns:
        List of paper metadata dictionaries

    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        paper = {
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],  # Array to match reference code
            "summary": result.summary,
            "published": int(result.published.strftime("%Y%m%d%H%M")),  # Long to match reference code
            "updated": result.updated.isoformat() if result.updated else None,
            "categories": ", ".join(result.categories),
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "ingestion_timestamp": datetime.now().isoformat(),
            "processed": None,  # Will be set in Lecture 2.2
            "volume_path": None,  # Will be set in Lecture 2.2
        }
        papers.append(paper)

    return papers


logger.info("Fetching arXiv papers...")
papers = fetch_arxiv_papers(query="cat:cs.AI OR cat:cs.LG", max_results=10)
logger.info(f"Fetched {len(papers)} papers")
logger.info("Sample paper:")
logger.info(f"Title: {papers[0]['title']}")
logger.info(f"Authors: {papers[0]['authors']}")
logger.info(f"arXiv ID: {papers[0]['arxiv_id']}")
logger.info(f"PDF URL: {papers[0]['pdf_url']}")

# COMMAND ----------
# Create Delta Table in Unity Catalog
# Store the arXiv paper metadata in a Delta table for downstream processing.

# Define schema
schema = StructType(
    [
        StructField("arxiv_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("authors", ArrayType(StringType()), True),  # Array to match reference code
        StructField("summary", StringType(), True),
        StructField("published", LongType(), True),  # Long to match reference code
        StructField("updated", StringType(), True),
        StructField("categories", StringType(), True),
        StructField("pdf_url", StringType(), True),
        StructField("primary_category", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),  # Long to match reference code
        StructField("volume_path", StringType(), True),  # Will be set in Lecture 2.2
    ]
)

# Create DataFrame
df = spark.createDataFrame(papers, schema=schema)

# Write to Delta table
table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table_path)

logger.info(f"Created Delta table: {table_path}")
logger.info(f"Records: {df.count()}")

# COMMAND ----------
# Verify the Data

# Read back the table
papers_df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

logger.info(f"Table: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
logger.info(f"Total papers: {papers_df.count()}")
logger.info("Schema:")
papers_df.printSchema()

logger.info("Sample records:")
papers_df.select("arxiv_id", "title", "primary_category", "published").show(5, truncate=50)

# COMMAND ----------
# Data Statistics

logger.info("Papers by primary category:")
papers_df.groupBy("primary_category").count().orderBy("count", ascending=False).show()

logger.info("Most recent papers:")
papers_df.select("title", "published", "arxiv_id").orderBy("published", ascending=False).show(5, truncate=60)
