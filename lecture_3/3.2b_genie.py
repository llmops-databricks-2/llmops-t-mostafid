# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2b: Genie Space Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Deploying a Genie space with Declarative Automation Bundles
# MAGIC - Starting conversations with Genie
# MAGIC - Using Genie for natural language queries
# MAGIC
# MAGIC **What is Genie?**
# MAGIC - Databricks Genie is an AI-powered data analyst
# MAGIC - Converts natural language questions to SQL queries
# MAGIC - Executes queries and returns results
# MAGIC - Can be integrated with agents via MCP
# MAGIC
# MAGIC **Where does the Genie space come from?**
# MAGIC The space is a bundle resource defined in `resources/genie_space.yml`. It uses the
# MAGIC `warehouse_id` from `project_config.yml` (mirrored as a bundle variable in `databricks.yml`).
# MAGIC The lecture3-genie-mcp job passes the deployed space ID as the `genie_space_id`
# MAGIC parameter. When running interactively without the widget, the deployed dev space is used.

# COMMAND ----------
import json

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_widget

spark = SparkSession.builder.getOrCreate()
# The job passes the ID of the space deployed by the bundle. The default is the dev space
# (`databricks bundle summary` shows it) for interactive runs without the widget.
genie_space_id = get_widget(spark, "genie_space_id", default="01f1d2d392811a448b1126db9a70b2ea")

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the Deployed Genie Space

# COMMAND ----------

if not genie_space_id:
    raise ValueError(
        "genie_space_id is not set. Run this notebook through the lecture3-genie-mcp job, "
        "or set the genie_space_id widget to the ID of the deployed arxiv-curator-space."
    )

space = w.genie.get_space(space_id=genie_space_id, include_serialized_space=True)
logger.info(f"Genie Space ID: {space.space_id}")
logger.info(f"Warehouse ID: {space.warehouse_id}")
logger.info(f"Space config: {json.loads(space.serialized_space)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Start a Conversation
# MAGIC
# MAGIC Ask Genie a natural language question about the data.

# COMMAND ----------

conversation = w.genie.start_conversation_and_wait(space_id=space.space_id, content="Find the last 10 papers published")

conversation.as_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Continue the Conversation
# MAGIC
# MAGIC Ask follow-up questions in the same conversation.

# COMMAND ----------

message = w.genie.create_message_and_wait(
    space_id=space.space_id,
    conversation_id=conversation.conversation_id,
    content="Return the list of authors of the last 10 papers published",
)

message.as_dict()

# COMMAND ----------
