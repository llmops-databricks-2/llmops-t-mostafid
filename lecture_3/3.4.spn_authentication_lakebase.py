# Databricks notebook source
import os
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.memory import LakebaseMemory

spark = SparkSession.builder.getOrCreate()
cfg = load_config("../project_config.yml", get_env(spark))

w = WorkspaceClient()

project_id = cfg.lakebase_project_id

# Service principal credentials are Unity Catalog secrets in the project schema
# (requires READ SECRET on the secrets, schema or catalog).
os.environ["LAKEBASE_SP_CLIENT_ID"] = dbutils.secrets.get(catalog=cfg.catalog, schema=cfg.schema, key="client_id")
os.environ["LAKEBASE_SP_CLIENT_SECRET"] = dbutils.secrets.get(
    catalog=cfg.catalog, schema=cfg.schema, key="client_secret"
)
os.environ["LAKEBASE_SP_HOST"] = w.config.host

# COMMAND ----------
memory = LakebaseMemory(
    project_id=project_id,
)

# COMMAND ----------

# Create a test session
session_id = f"test-session-{uuid4()}"

# Save some messages
test_messages = [
    {"role": "user", "content": "What are recent papers on transformers?"},
    {"role": "assistant", "content": "Here are some recent papers on transformer architectures..."},
    {"role": "user", "content": "Tell me more about the first one"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"✓ Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

# Load messages back
loaded_messages = memory.load_messages(session_id)
logger.info(f"Loaded {len(loaded_messages)} messages:")
for msg in loaded_messages:
    logger.info(f"  {msg['role']}: {msg['content'][:50]}...")
