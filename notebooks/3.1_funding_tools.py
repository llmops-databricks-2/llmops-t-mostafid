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


# COMMAND ----------
# 2. get_call_details: all facts about ONE call (the model already knows the call_id)


def get_call_details(call_id: str) -> str:
    """Return title, deadline, status, link and the start of the call text for one call."""
    with pgvector.connection() as conn:
        rows = conn.execute(
            f"""SELECT call_id, title, deadline, status, is_open, url, text
                FROM {pgvector.table_name}
                WHERE call_id = %s
                ORDER BY chunk_id""",
            (call_id,),
        ).fetchall()

    if not rows:
        return json.dumps({"error": f"No call found with id {call_id}"})

    first = rows[0]
    full_text = " ".join(row[6] for row in rows)
    details = {
        "call_id": first[0],
        "title": first[1],
        "deadline": str(first[2]),
        "status": first[3],
        "is_open": first[4],
        "url": first[5],
        "text_start": full_text[:1500] + "...",
    }
    return json.dumps(details, indent=2)


get_call_details_spec = {
    "type": "function",
    "function": {
        "name": "get_call_details",
        "description": (
            "Get the details of ONE funding call by its call_id: title, deadline, status, "
            "whether it is open, link, and the start of the call text (expected outcome, scope). "
            "Use it after search_calls, when the user asks about a specific call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "call_id": {
                    "type": "string",
                    "description": "The call id, e.g. 'HORIZON-CL5-2027-06-D6-04'",
                },
            },
            "required": ["call_id"],
        },
    },
}

get_call_details_tool = ToolInfo(name="get_call_details", spec=get_call_details_spec, exec_fn=get_call_details)

# COMMAND ----------
# 3. Tool registry (same as the teacher's section 6): all tools in one place


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_all_specs(self) -> list[dict]:
        """Get all tool specifications (what the model reads)."""
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict) -> str:
        """Run a tool by name, like the agent will when the model asks for it."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name].exec_fn(**args)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools)


registry = ToolRegistry()
registry.register(search_calls_tool)
registry.register(get_call_details_tool)

# COMMAND ----------
# 4. Execute tools by name (this is exactly what the agent will do)

logger.info(registry.execute("get_call_details", {"call_id": "HORIZON-CL5-2027-06-D6-04"}))
logger.info(registry.execute("get_call_details", {"call_id": "DOES-NOT-EXIST"}))
