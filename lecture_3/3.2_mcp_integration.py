# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2: MCP and Lakebase pgvector Tools
# MAGIC
# MAGIC This notebook keeps the MCP integration pattern for managed Databricks
# MAGIC services such as Genie, while exposing the migrated Lakebase pgvector
# MAGIC search through the same `ToolInfo` interface used by the agent.
# MAGIC
# MAGIC ```text
# MAGIC LLM agent
# MAGIC   ├── search_papers ToolInfo → Lakebase pgvector
# MAGIC   └── Genie MCP tools → Databricks managed MCP server
# MAGIC ```

# COMMAND ----------

import asyncio
import json

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, get_widget, load_config
from arxiv_curator.mcp import ToolInfo, create_mcp_tools
from arxiv_curator.vector_search import LakebasePgVectorManager

nest_asyncio.apply()

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
# The job passes the ID of the space deployed by the bundle. The default is the dev space
# (`databricks bundle summary` shows it) for interactive runs without the widget.
genie_space_id = get_widget(spark, "genie_space_id", default="01f1d2d392811a448b1126db9a70b2ea")
w = WorkspaceClient()

pgvector = LakebasePgVectorManager(config=cfg, workspace=w)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. MCP versus custom tools
# MAGIC
# MAGIC MCP remains useful for managed services such as Genie. The old Databricks
# MAGIC Vector Search MCP endpoint is intentionally not used here because the
# MAGIC project now stores embeddings in Lakebase pgvector.

# COMMAND ----------


def search_papers(query: str, num_results: int = 5, year_filter: str | None = None) -> str:
    """Search Lakebase pgvector using hybrid vector and keyword ranking."""
    year = None
    year_from = year_to = None
    if year_filter:
        if "-" in year_filter:
            year_from, year_to = [int(value.strip()) for value in year_filter.split("-", 1)]
            if year_from > year_to:
                year_from, year_to = year_to, year_from
        else:
            year = int(year_filter.strip())
    results = pgvector.hybrid_search(
        query=query,
        num_results=num_results,
        year=year,
        year_from=year_from,
        year_to=year_to,
    )
    return json.dumps(
        [
            {
                "title": row.get("title", "N/A"),
                "arxiv_id": row.get("arxiv_id", "N/A"),
                "authors": row.get("authors", "N/A"),
                "year": row.get("year", "N/A"),
                "excerpt": row.get("text", "")[:500],
                "score": round(float(row.get("score", 0.0)), 4),
            }
            for row in results
        ],
        indent=2,
    )


search_papers_tool = ToolInfo(
    name="search_papers",
    spec={
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search arXiv papers using Lakebase pgvector hybrid search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 5},
                    "year_filter": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    exec_fn=search_papers,
)

logger.info(search_papers_tool.exec_fn("machine learning and neural networks", 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Genie Space MCP
# MAGIC
# MAGIC Genie remains a managed MCP server. It is independent of the pgvector
# MAGIC search tool and can be included in the same agent tool list.
# MAGIC
# MAGIC The space ID comes from the `genie_space_id` job parameter, which the bundle
# MAGIC fills from `resources/genie_space.yml`. When running interactively without the
# MAGIC widget, the deployed dev space is used.

# COMMAND ----------

genie_tools: list[ToolInfo] = []
if genie_space_id:
    host = w.config.host
    genie_url = f"{host}/api/2.0/mcp/genie/{genie_space_id}"
    genie_client = DatabricksMCPClient(server_url=genie_url, workspace_client=w)

    logger.info(f"Genie MCP URL: {genie_url}")
    for tool in genie_client.list_tools():
        logger.info(f"Genie tool: {tool.name}")

    genie_tools = asyncio.run(create_mcp_tools(w, [genie_url]))
else:
    logger.warning("No genie_space_id parameter set, skipping Genie MCP tools")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Combined tool registry

# COMMAND ----------

tools = [search_papers_tool, *genie_tools]
tools_dict = {tool.name: tool for tool in tools}

logger.info(f"Loaded {len(tools)} tools: {list(tools_dict)}")
for tool in tools:
    logger.info(json.dumps(tool.spec, indent=2))

# COMMAND ----------

# Execute the pgvector tool directly.
logger.info(tools_dict["search_papers"].exec_fn(query="deep learning architectures"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Using the combined tools with an agent

# COMMAND ----------

from openai import OpenAI


class SimpleAgent:
    """Small OpenAI tool-calling loop for pgvector and MCP tools."""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
        )

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=[tool.spec for tool in self.tools.values()],
            )
            assistant_message = response.choices[0].message
            if not assistant_message.tool_calls:
                return assistant_message.content

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in assistant_message.tool_calls
                    ],
                }
            )
            for call in assistant_message.tool_calls:
                args = json.loads(call.function.arguments)
                result = self.tools[call.function.name].exec_fn(**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(result),
                    }
                )
        return "Max iterations reached."


# COMMAND ----------

agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="Use search_papers for arXiv research questions and cite the returned papers.",
    tools=tools,
)

logger.info(agent.chat("Find recent papers about transformer attention mechanisms."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Unity Catalog Functions through MCP
# MAGIC
# MAGIC A Unity Catalog Functions MCP server exposes governed UC functions as discoverable tools. The agent can call an approved function through MCP without importing or reimplementing its Python logic.

# COMMAND ----------

# Call the add_numbers function created in Notebook 3.1c through managed MCP.
# The agent discovers the governed UC function as a tool instead of importing
# or reimplementing the function locally.
uc_function_url = f"{w.config.host.rstrip('/')}/api/2.0/mcp/functions/{cfg.catalog}/{cfg.schema}/add_numbers"
uc_function_tools = asyncio.run(create_mcp_tools(w, [uc_function_url]))

logger.info(f"Discovered UC MCP tools: {[tool.name for tool in uc_function_tools]}")
uc_add_numbers = next(tool for tool in uc_function_tools if tool.name.endswith("add_numbers"))
uc_result = uc_add_numbers.exec_fn(a=10, b=3)
logger.info(f"MCP call result: {uc_result}")
