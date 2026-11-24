# Databricks notebook source
"""Lecture 3 (my use case): funding call tools, Genie via MCP, and an agent with memory."""

import json
import os
import uuid

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from loguru import logger
from openai import OpenAI

from arxiv_curator.config import get_env, load_config
from arxiv_curator.mcp import ToolInfo, create_mcp_tools
from arxiv_curator.memory import LakebaseMemory
from arxiv_curator.vector_search import LakebasePgVectorManager

# Which tests to run (switch on/off to keep runs short)
RUN_TOOL_TESTS = False  # sections 1-4: tools on their own
RUN_AGENT_TESTS = False  # section 6: agent with 2 tools
RUN_GENIE_TEST = False  # section 7: agent with Genie (slow, ~40 s)
RUN_MEMORY_TEST = True  # section 8: follow-up question with memory

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

MIN_SCORE = 0.6  # code-controlled: weaker matches are not returned at all


def search_calls(query: str, num_results: int = 5) -> str:
    """Find OPEN funding calls that match a research topic.

    Code-controlled rules (from the lecture 2 findings):
    - only open calls (is_open = true), the model cannot turn this off
    - only matches with a score >= MIN_SCORE
    - one result per call: fetch extra chunks, keep the best chunk per call
    """
    rows = pgvector.search(query=query, num_results=num_results * 4, is_open=True)
    rows = [row for row in rows if float(row["score"]) >= MIN_SCORE]

    best_per_call = {}
    for row in rows:  # rows are sorted best first, so the first chunk per call is its best
        if row["call_id"] not in best_per_call:
            best_per_call[row["call_id"]] = {
                "call_id": row["call_id"],
                "title": row["title"],
                "deadline": str(row["deadline"]),
                # Code-controlled wording: no raw EU status, so the model can't misread "forthcoming"
                "submission": "accepting submissions now"
                if row["status"] == "open"
                else "opens soon, preparation can start now",
                "can_apply": True,
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
            "Search AVAILABLE EU funding calls (Horizon Europe: status open or forthcoming) by research topic, using semantic "
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

if RUN_TOOL_TESTS:
    TEST_QUERIES = [
        "road traffic monitoring with cameras and sensors",
        "traffic flow estimation from aerial or drone imagery",
        "intelligent transport systems for road safety",
    ]
    for test_query in TEST_QUERIES:
        logger.info(f"QUERY: {test_query}")
        calls = json.loads(registry.execute("search_calls", {"query": test_query, "num_results": 3}))
        if not calls:
            logger.info(f"  (no match with score >= {MIN_SCORE})")
        for call in calls:
            logger.info(f"  {call['score']} | {call['deadline']} | {call['call_id']} | {call['title'][:80]}")

    logger.info(registry.execute("get_call_details", {"call_id": "HORIZON-CL5-2027-06-D6-04"}))
    logger.info(registry.execute("get_call_details", {"call_id": "DOES-NOT-EXIST"}))

# COMMAND ----------
# 5. The agent: model + tools + loop (the teacher's SimpleAgent, with my tools and memory)

SYSTEM_PROMPT = """You are Funding Scout, an assistant for researchers in road traffic monitoring
and disaster management. You help them find EU funding calls that fit their research.

Rules:
- Use search_calls to find calls by research topic; use get_call_details for facts about one call.
- Use the Genie tool for table questions: counting calls, listing or sorting them by deadline.
- Only recommend calls returned by the tools. Never invent calls, deadlines or links.
- For each recommended call give: title, call_id, deadline, link, and one sentence on why it fits.
- All calls returned by search_calls have a deadline in the future, so the user can apply to them.
  Status "open" = submissions already accepted; "forthcoming" = opens soon, preparation can start now.
- If search_calls returns an empty list, say there is no strong match in the current data."""


class SimpleAgent:
    """A simple agent that can call tools in a loop, with optional Lakebase memory."""

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo],
        memory: LakebaseMemory | None = None,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self.memory = memory
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

    def chat(self, user_message: str, session_id: str | None = None, max_iterations: int = 5) -> str:
        """Chat with the agent; with a session_id, earlier turns are loaded and this turn is saved."""
        # Code-controlled: always load the history of this session first
        history = self.memory.load_messages(session_id) if self.memory and session_id else []
        logger.info(f"Loaded {len(history)} earlier messages")

        user_entry = {"role": "user", "content": user_message}
        messages = [{"role": "system", "content": self.system_prompt}, *history, user_entry]

        for iteration in range(max_iterations):
            # Code-controlled: in the last round no tools are offered, so the model must answer
            last_round = iteration == max_iterations - 1
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=None if last_round or not self._tools_dict else self.get_tool_specs(),
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
                answer = assistant_message.content
                # Code-controlled: always save the question and the final answer (not the tool steps)
                if self.memory and session_id:
                    self.memory.save_messages(session_id, [user_entry, {"role": "assistant", "content": answer}])
                return answer

        return "Max iterations reached."


# COMMAND ----------
# 6. Test the agent with my 2 tools (no memory)

if RUN_AGENT_TESTS:
    agent = SimpleAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_calls_tool, get_call_details_tool],
    )
    for question in [
        "I work on road traffic monitoring with cameras and sensors. Which open calls fit me?",
        "What is the deadline and the link of HORIZON-CL5-2027-06-D6-04?",
    ]:
        logger.info("=" * 80)
        logger.info(f"USER: {question}")
        logger.info(f"AGENT: {agent.chat(question)}")

# COMMAND ----------
# 7. Genie as a third tool, through MCP (teacher's 3.2 sections 2 + 3)

if RUN_GENIE_TEST:
    w = WorkspaceClient()
    genie_url = f"{w.config.host.rstrip('/')}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    genie_client = DatabricksMCPClient(server_url=genie_url, workspace_client=w)

    logger.info(f"Genie MCP URL: {genie_url}")
    for tool in genie_client.list_tools():
        logger.info(f"Genie tool: {tool.name}")

    genie_tools = create_mcp_tools(w, [genie_url])

    full_agent = SimpleAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_calls_tool, get_call_details_tool, *genie_tools],
    )
    question = "How many open calls close before June 2027, and which ones close first?"
    logger.info("=" * 80)
    logger.info(f"USER: {question}")
    logger.info(f"AGENT: {full_agent.chat(question)}")

# COMMAND ----------
# 8. Memory: a follow-up question that only works if turn 1 is remembered

if RUN_MEMORY_TEST:
    memory = LakebaseMemory(project_id=cfg.lakebase_project_id, workspace=WorkspaceClient())
    memory.ensure_schema()

    memory_agent = SimpleAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_calls_tool, get_call_details_tool],
        memory=memory,
    )

    session_id = str(uuid.uuid4())
    for question in [
        "Which open calls fit research on intelligent transport systems for road safety?",
        "What is the deadline and the link of the first call you recommended?",
    ]:
        logger.info("=" * 80)
        logger.info(f"USER: {question}")
        logger.info(f"AGENT: {memory_agent.chat(question, session_id=session_id)}")
