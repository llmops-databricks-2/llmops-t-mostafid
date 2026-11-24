# Databricks notebook source
"""Lecture 3.3 (my use case): conversation memory in Lakebase, tested on its own."""

import os
import uuid

from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import get_env, load_config
from arxiv_curator.memory import LakebaseMemory

# COMMAND ----------
# Config
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
else:
    env = get_env()

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
# 1. Create the memory helper and the session table (safe to rerun)

memory = LakebaseMemory(project_id=cfg.lakebase_project_id, workspace=WorkspaceClient())
memory.ensure_schema()
logger.info("Table session_messages ready")

# COMMAND ----------
# 2. Save one conversation turn under a new session id

session_id = str(uuid.uuid4())  # a random, unique id for this conversation
memory.save_messages(
    session_id,
    [
        {"role": "user", "content": "Which calls fit road traffic monitoring?"},
        {"role": "assistant", "content": "HORIZON-CL5-2027-06-D6-04 (CCAM), deadline 2027-10-07."},
    ],
)
logger.info(f"Saved 2 messages for session {session_id}")

# COMMAND ----------
# 3. Load it back, like the agent will at the start of the next turn

for message in memory.load_messages(session_id):
    logger.info(f"{message['role']}: {message['content']}")

# A different session id must return nothing (histories never mix)
logger.info(f"Other session: {memory.load_messages(str(uuid.uuid4()))}")
