# Databricks notebook source
# /// script
# dependencies = [
#   "..",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Lecture 1.4: External Models Through the AI Gateway
# MAGIC
# MAGIC This notebook calls an external model (OpenAI) through the Databricks AI Gateway. The gateway
# MAGIC objects are deployed centrally for the course, as two governed Unity Catalog objects:
# MAGIC
# MAGIC - a **model provider service**, `course_ops.gateway.openai`, which holds the OpenAI key *by
# MAGIC   reference* to the Unity Catalog secret `course_ops.secrets.openai_api_key` and lists the
# MAGIC   models it may route to (`gpt-4o-mini`);
# MAGIC - a **model service**, `course_ops.gateway.openai_demo`, the name you call, which routes to
# MAGIC   that provider service, with a rate limit.
# MAGIC
# MAGIC You never see the key. You authenticate to Databricks, and the gateway calls OpenAI with the
# MAGIC key of the provider service. Running this notebook creates no gateway object and no token.
# MAGIC
# MAGIC **Checked: October 13, 2026 (AWS).** See
# MAGIC [model provider services](https://docs.databricks.com/aws/en/ai-gateway/model-provider-services)
# MAGIC and [query model services](https://docs.databricks.com/aws/en/ai-gateway/query-model-services).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the Model Service
# MAGIC
# MAGIC The OpenAI client points at the workspace's AI Gateway, `/ai-gateway/mlflow/v1`, and `model`
# MAGIC is the model service's name. The workspace client authenticates you to Databricks: its token is
# MAGIC sent to the gateway, never to OpenAI, and no new token is created.
# MAGIC
# MAGIC To call the model service you need `EXECUTE` on it, plus `USE CATALOG` on `course_ops` and
# MAGIC `USE SCHEMA` on `course_ops.gateway`. The demonstration service grants `EXECUTE` to the
# MAGIC teachers only.

# COMMAND ----------

workspace = WorkspaceClient()
client = OpenAI(
    base_url=f"{workspace.config.host}/ai-gateway/mlflow/v1",
    api_key=workspace.config.authenticate()["Authorization"].removeprefix("Bearer "),
)

response = client.chat.completions.create(
    model="course_ops.gateway.openai_demo",
    messages=[{"role": "user", "content": "In one sentence, what is an AI gateway?"}],
    max_tokens=256,
)
logger.info(f"Answer from {response.model}: {response.choices[0].message.content}")
