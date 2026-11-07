# Databricks notebook source
# /// script
# dependencies = [
#   "..",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# ///
"""Experiment with the course chat model through the AI Gateway for funding questions."""

import os
import time

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

# COMMAND ----------
# Connect to the AI Gateway (authentication through WorkspaceClient, no token in code)

workspace = WorkspaceClient()
client = OpenAI(
    base_url=f"{workspace.config.host}/ai-gateway/mlflow/v1",
    api_key=workspace.config.authenticate()["Authorization"].removeprefix("Bearer "),
)
MODEL = "course_ops.gateway.chat"

# COMMAND ----------
# Load 5 open calls from my Delta table to use as context in one prompt

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    spark = SparkSession.builder.getOrCreate()
else:
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.serverless().getOrCreate()

calls = (
    spark.table("llmops_dev.t_mostafid.funding_calls")
    .filter("is_open")
    .select("call_id", "title", "deadline")
    .limit(5)
    .collect()
)
calls_text = "\n".join(f"- {c.call_id}: {c.title} (deadline {c.deadline})" for c in calls)

# COMMAND ----------
# Prompts for my use case

PROMPTS = {
    "general_knowledge": ("Which EU funding programmes support research on satellite-based flood monitoring?"),
    "specific_open_calls": (
        "List 3 Horizon Europe calls that are open right now for drone-based traffic monitoring, "
        "with their exact deadlines."
    ),
    "grounded_ranking": (
        "I am a researcher working on disaster management with satellite imagery. "
        "From the following open calls, pick the 2 that fit me best and explain why "
        f"in one sentence each:\n{calls_text}"
    ),
    "summarise": (
        "In 3 bullet points, explain what a researcher should check before applying to a Horizon Europe call."
    ),
}


def ask(prompt: str) -> dict:
    """Send one prompt and return the answer, token usage and latency."""
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low",
        max_tokens=1024,
    )
    latency = time.time() - start
    return {
        "answer": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_s": round(latency, 2),
        "finish_reason": response.choices[0].finish_reason,
    }


# COMMAND ----------
# Run the experiments and record the results

results = {}
for name, prompt in PROMPTS.items():
    result = ask(prompt)
    results[name] = result
    logger.info(f"=== {name} ===")
    logger.info(f"Answer:\n{result['answer']}")
    logger.info(
        f"Tokens: {result['total_tokens']} "
        f"(prompt {result['prompt_tokens']}, completion {result['completion_tokens']}) | "
        f"latency {result['latency_s']} s | finish_reason {result['finish_reason']}"
    )

# COMMAND ----------
# Summary

total_tokens = sum(r["total_tokens"] for r in results.values())
avg_latency = sum(r["latency_s"] for r in results.values()) / len(results)
logger.info(f"Prompts: {len(results)} | total tokens: {total_tokens} | avg latency: {avg_latency:.2f} s")
