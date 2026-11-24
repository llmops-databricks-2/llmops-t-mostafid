# Databricks notebook source

# MAGIC %md
# MAGIC # Lecture 3.3: Session Memory with Lakebase
# MAGIC
# MAGIC This notebook shows how an agent can persist conversation history in
# MAGIC Lakebase PostgreSQL. The vector-search table stores research content, while
# MAGIC this session table stores messages and session state.
# MAGIC
# MAGIC ```text
# MAGIC Agent conversation → session_messages (JSONB) → load history on the next turn
# MAGIC ```
# MAGIC
# MAGIC The notebook demonstrates both the SQL/`psycopg` approach and the reusable
# MAGIC `LakebaseMemory` class used by the agent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Connect to Lakebase
# MAGIC
# MAGIC We reuse the configured Lakebase project from `project_config.yml`. The
# MAGIC Databricks SDK discovers the project endpoint and generates a short-lived
# MAGIC database credential, so the notebook does not store a database password.

# Project
#  └── Branches (main, development, staging, etc.)
#        ├── Computes (R/W compute)
#        ├── Roles (Postgres roles)
#        └── Databases (Postgres databases)

# COMMAND ----------

import json
import urllib.parse
from uuid import uuid4

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import PostgresAPI
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
cfg = load_config("../project_config.yml", get_env(spark))

w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project_id = cfg.lakebase_project_id

project = pg_api.get_project(name=f"projects/{project_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Create the session table
# MAGIC
# MAGIC Each row represents one message in a conversation. `session_id` groups
# MAGIC messages into a conversation, `message_data` stores the role and content as
# MAGIC JSONB, and the index makes loading one session efficient.

# COMMAND ----------
# Get endpoint, host, and generate credential
default_branch = next(iter(pg_api.list_branches(parent=project.name)))
endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
host = endpoint.status.hosts.host

# Get username (works with user credentials in notebook)
user = w.current_user.me()
pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)
username = urllib.parse.quote_plus(user.user_name)
conn_string = f"postgresql://{username}:{pg_credential.token}@{host}:5432/databricks_postgres?sslmode=require"


# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Save messages directly with SQL
# MAGIC
# MAGIC This small example makes the storage pattern explicit before we use the
# MAGIC application helper. A real agent would save messages after each turn or at
# MAGIC a controlled checkpoint.

# COMMAND ----------
with psycopg.connect(conn_string) as conn:
    # Create session_messages table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
        ON session_messages(session_id)
    """)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Load a conversation
# MAGIC
# MAGIC The agent can retrieve prior messages by `session_id`, ordered by creation
# MAGIC time, and add them to the next model call as conversation context.

# COMMAND ----------

test_session_id = f"test-session-{uuid4()}"
test_messages = [
    {"role": "user", "content": "Hello, what can you help me with?"},
    {"role": "assistant", "content": "I can help you find research papers."},
    {"role": "user", "content": "Find papers about LLM reasoning"},
]

with psycopg.connect(conn_string) as conn:
    for msg in test_messages:
        conn.execute(
            "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
            (test_session_id, json.dumps(msg)),
        )

# COMMAND ----------

# Load messages back
with psycopg.connect(conn_string) as conn:
    result = conn.execute(
        """
        SELECT message_data, created_at FROM session_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        """,
        (test_session_id,),
    ).fetchall()

    logger.info(f"Loaded {len(result)} messages:")
    for row in result:
        logger.info(f"  [{row[1]}] {row[0]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Use the reusable memory class
# MAGIC
# MAGIC `LakebaseMemory` wraps the connection and SQL details so the agent does not
# MAGIC need to manage database credentials or table statements directly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test with LakebaseMemory Class

# COMMAND ----------

from arxiv_curator.memory import LakebaseMemory

memory = LakebaseMemory(project_id=project_id, workspace=w)
memory.ensure_schema()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Save and load through the helper
# MAGIC
# MAGIC These calls demonstrate the interface used by the agent: save a list of
# MAGIC messages for a session, then load them later to restore conversational
# MAGIC context.

# COMMAND ----------

# Test save
session_id = f"memory-test-{uuid4()}"
messages = [
    {"role": "user", "content": "What papers discuss transformer architectures?"},
    {"role": "assistant", "content": "Here are some relevant papers..."},
]

memory.save_messages(session_id, messages)
logger.info(f"Saved messages to session: {session_id}")

# COMMAND ----------

# Test load
loaded = memory.load_messages(session_id)
logger.info(f"Loaded {len(loaded)} messages:")
for msg in loaded:
    logger.info(f"  {msg['role']}: {msg['content'][:50]}...")
