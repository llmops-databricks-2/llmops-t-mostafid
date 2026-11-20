# Databricks notebook source
"""Lecture 3.1 funding call tools for the agent."""

import json
import os

from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import get_env, load_config
from arxiv_curator.mcp import ToolInfo
from arxiv_curator.vector_search import LakebasePgVectorManager

# COMMAND ----------
# Config and Lakebase search helper (no Spark needed: tools query Lakebase directly)
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
else:
    env = get_env()

cfg = load_config("../project_config.yml", env)
pgvector = LakebasePgVectorManager(config=cfg, workspace=WorkspaceClient())

# COMMAND ----------
# 1. search_calls: find open calls that match a research topic


def search_calls(query: str, num_results: int = 5) -> str:
    """Find OPEN funding calls that match a research topic.

    Code-controlled rules (from the lecture 2 findings):
    - only open calls (is_open = true), the model cannot turn this off
    - one result per call: fetch extra chunks, keep the best chunk per call
    """
    rows = pgvector.search(query=query, num_results=num_results * 4, is_open=True)

    best_per_call = {}
    for row in rows:  # rows are sorted best first, so the first chunk per call is its best
        if row["call_id"] not in best_per_call:
            best_per_call[row["call_id"]] = {
                "call_id": row["call_id"],
                "title": row["title"],
                "deadline": str(row["deadline"]),
                "status": row["status"],
                "url": row["url"],
                "excerpt": row["text"][:300] + "...",
                "score": round(float(row["score"]), 3),
            }
        if len(best_per_call) == num_results:
            break

    return json.dumps(list(best_per_call.values()), indent=2)


search_calls_spec = {
    "type": "function",
    "function": {
        "name": "search_calls",
        "description": (
            "Search OPEN EU funding calls (Horizon Europe) by research topic, using semantic "
            "search over the call texts. Returns one entry per call with title, deadline, "
            "status, link, a text excerpt and a relevance score. Use it when the user "
            "describes their research and wants matching funding opportunities."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic, e.g. 'satellite-based flood monitoring'",
                },
                "num_results": {
                    "type": "integer",
                    "description": "How many calls to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

search_calls_tool = ToolInfo(name="search_calls", spec=search_calls_spec, exec_fn=search_calls)

# COMMAND ----------
# Test the tool on its own (before giving it to a model)

TEST_QUERIES = [
    "road traffic monitoring with cameras and sensors",
    "traffic flow estimation from aerial or drone imagery",
    "intelligent transport systems for road safety",
]
for test_query in TEST_QUERIES:
    logger.info(f"QUERY: {test_query}")
    for call in json.loads(search_calls(test_query, num_results=3)):
        logger.info(f"  {call['score']} | {call['deadline']} | {call['call_id']} | {call['title'][:80]}")
