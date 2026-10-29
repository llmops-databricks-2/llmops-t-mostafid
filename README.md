# Usecase: Funding Scout for Traffic Monitoring & Disaster Management

An LLMOps project on Databricks: an AI agent that helps researchers in
**traffic monitoring and disaster management** find and evaluate relevant
funding calls from **Horizon Europe** and **ESA**.

## Problem and users

Researchers working with satellite, aerial and drone imagery (PhD students,
postdocs, research groups, small EO startups) need funding, but calls are
spread across several portals, are long, and have different deadlines,
budgets and eligibility rules. Finding the few calls that fit a specific
topic, such as flood mapping or traffic flow estimation, takes hours.

## What it does

- **Collect:** a Databricks job loads funding calls into a Delta table.
- **Search:** the agent finds the calls that match the user's research.
- **Explain:** for each call, it gives the deadline, budget and why it fits.
- **Evaluate and monitor:** the agent is traced and evaluated with MLflow.

## Example questions

- Which open calls fit drone-based traffic monitoring?
- Are there Horizon Europe calls on flood or wildfire early warning with satellite data?
- Which calls close in the next 3 months?
- What is the budget and eligibility for call X?

## Data sources

| Source | What it provides |
|---|---|
| EU Funding & Tenders Portal | Horizon Europe topics on disaster resilience, mobility and space |
| ESA | Earth Observation funding opportunities |

Start small: about 30–80 calls.

## Data contract

| Column | Description |
|---|---|
| call_id | Unique id of the call |
| source | `horizon_europe` or `esa` |
| title | Title of the call |
| programme | Programme or cluster |
| description | Call text |
| deadline | Submission deadline |
| budget | Budget, when available |
| url | Link to the original call |
| ingested_at | When the call was collected |

## Architecture

```
Funding portals (Horizon Europe, ESA)
        │
        ▼
Ingestion job ──► Delta table in llmops_dev.t_mostafid
        │
        ▼
Embeddings (course_ops.gateway.embed) ──► Lakebase pgvector
        │
        ▼
Agent (course_ops.gateway.chat) + tools: search calls, call details, deadline check
        │
        ▼
MLflow tracing, evaluation and monitoring
