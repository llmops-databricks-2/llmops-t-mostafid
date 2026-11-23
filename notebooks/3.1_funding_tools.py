# Databricks notebook source
"""Lecture 3.1 (my use case): funding call tools and a simple agent."""

import json
import os

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

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
    - only good matches (score >= min_score)
    - one result per call: fetch extra chunks, keep the best chunk per call
    """
    min_score = 0.6  # weaker matches are not returned at all
    rows = pgvector.search(query=query, num_results=num_results * 4, is_open=True)
    rows = [row for row in rows if float(row["score"]) >= min_score]

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
# 4. Test the tools on their own (before giving them to a model)

TEST_QUERIES = [
    "road traffic monitoring with cameras and sensors",
    "traffic flow estimation from aerial or drone imagery",
    "intelligent transport systems for road safety",
]
for test_query in TEST_QUERIES:
    logger.info(f"QUERY: {test_query}")
    calls = json.loads(registry.execute("search_calls", {"query": test_query, "num_results": 3}))
    if not calls:
        logger.info("  (no match with score >= 0.6)")
    for call in calls:
        logger.info(f"  {call['score']} | {call['deadline']} | {call['call_id']} | {call['title'][:80]}")

logger.info(registry.execute("get_call_details", {"call_id": "DOES-NOT-EXIST"}))

# COMMAND ----------
# 5. The agent: model + tools + loop (the teacher's SimpleAgent, with my tools)

SYSTEM_PROMPT = """You are Funding Scout, an assistant for researchers in road traffic monitoring
and disaster management. You help them find EU funding calls that fit their research.

Rules:
- Use search_calls to find calls; use get_call_details for facts about one specific call.
- Only recommend calls returned by the tools. Never invent calls, deadlines or links.
- For each recommended call give: title, call_id, deadline, link, and one sentence on why it fits.
- All calls returned by search_calls have a deadline in the future, so the user can apply to them.
  Status "open" = submissions already accepted; "forthcoming" = opens soon, preparation can start now.
- If search_calls returns an empty list, say there is no strong match in the current data."""


class SimpleAgent:
    """A simple agent that can call tools in a loop."""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        workspace = WorkspaceClient()
        self._client = OpenAI(  # my existing login, no new token created
            api_key=workspace.config.authenticate()["Authorization"].removeprefix("Bearer "),
            base_url=f"{workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
        )

    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM."""
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Chat with the agent, allowing tool calls."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _iteration in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )
            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Calling tool: {tool_name}({tool_args})")
                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:  # noqa: BLE001  the model gets the error instead of a crash
                        result = f"Error: {e}"
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
            else:
                return assistant_message.content

        return "Max iterations reached."


agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_calls_tool, get_call_details_tool],
)

# COMMAND ----------
# 6. Test the agent

questions = [
    "I work on road traffic monitoring with cameras and sensors. Which open calls fit me?",
    "What is the deadline and the link of HORIZON-CL5-2027-06-D6-04?",
]
for question in questions:
    logger.info("=" * 80)
    logger.info(f"USER: {question}")
    logger.info(f"AGENT: {agent.chat(question)}")
