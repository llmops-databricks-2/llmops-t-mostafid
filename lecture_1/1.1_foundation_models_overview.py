# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.1: Foundation Models Overview
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Foundation models on Databricks
# MAGIC - Unity Catalog model services and the Unified Responses API
# MAGIC - Databricks hosted models
# MAGIC - External models
# MAGIC - Pricing comparison
# MAGIC
# MAGIC **Documentation and pricing checked: October 12, 2026 (AWS).**
# MAGIC Model availability depends on your workspace, region, permissions, and model lifecycle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Foundation Models on Databricks
# MAGIC
# MAGIC A foundation model is pretrained on broad data and can be adapted to many tasks.
# MAGIC Chat, reasoning, vision, and embedding models have different input/output contracts.
# MAGIC In this course, we query **Unity Gateway model services**, governed objects in Unity Catalog.
# MAGIC A service name such as `course_ops.gateway.chat` identifies the catalog, schema, and service.
# MAGIC Its routing configuration selects the underlying model; the service name is not a model version.
# MAGIC See [model services](https://docs.databricks.com/aws/en/ai-gateway/model-services).
# MAGIC
# MAGIC The underlying models and serving options include:
# MAGIC
# MAGIC 1. **Foundation Model APIs (Pay-per-token)**
# MAGIC    - Serverless, fully managed
# MAGIC    - Pay only for what you use
# MAGIC    - No infrastructure management
# MAGIC    - Open-weight examples: Meta Llama, OpenAI GPT OSS, Google Gemma, Qwen, and DeepSeek
# MAGIC    - Selected proprietary OpenAI GPT, Anthropic Claude, and Google Gemini models are also hosted by Databricks
# MAGIC    - Priority pay-per-token is an opt-in, higher-priced tier for supported models and latency-sensitive requests
# MAGIC
# MAGIC 2. **Provisioned Throughput**
# MAGIC    - Dedicated capacity
# MAGIC    - A Foundation Model APIs capacity option, not a separate model family
# MAGIC    - On-demand capacity or fixed-term reserved capacity, depending on the model
# MAGIC    - Eligibility is narrower than the hosted model catalog; the two capacity options have different model lists
# MAGIC    - Supports serving already fine-tuned models for supported architectures; serving does not train them
# MAGIC
# MAGIC 3. **External Models**
# MAGIC    - Models hosted outside Databricks, accessed through a Unity Catalog model provider service
# MAGIC    - Providers include OpenAI, Azure OpenAI, Anthropic, Amazon Bedrock, and Google Gemini Enterprise
# MAGIC    - Provider credentials are managed centrally; callers authenticate to Databricks
# MAGIC    - Unified access and centralized credential management/governance
# MAGIC    - A provider name alone does not tell you whether a model is Databricks-hosted or external
# MAGIC
# MAGIC 4. **AI Functions for batch inference**
# MAGIC    - Use functions such as `ai_query` to apply models to data in SQL and pipelines
# MAGIC
# MAGIC DBRX is no longer a current serving example: its pay-per-token offering retired on April 30, 2025,
# MAGIC and provisioned throughput retired on December 19, 2025.
# MAGIC
# MAGIC Sources: [Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/),
# MAGIC [hosted models](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview),
# MAGIC [model providers](https://docs.databricks.com/aws/en/ai-gateway/model-provider-services),
# MAGIC [model lifecycle](https://docs.databricks.com/aws/en/machine-learning/retired-models-policy).

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Models Behind Unity Gateway Services
# MAGIC
# MAGIC Examples in the current [Databricks-hosted model catalog](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview):
# MAGIC
# MAGIC | Provider | Example models |
# MAGIC |----------|----------------|
# MAGIC | OpenAI | GPT-6 Astra, GPT OSS 120B, GPT OSS 20B |
# MAGIC | Anthropic | Claude Sonnet 5.5, Claude Opus 5 |
# MAGIC | Google | Gemini 3.8 Flash, Gemini 3.1 Pro Preview, Gemma 3 12B |
# MAGIC | Alibaba Cloud | Qwen3.5 122B A10B |
# MAGIC | Moonshot AI | Kimi K3 |
# MAGIC
# MAGIC This is a selection, not a guarantee that every model is enabled in our workspace.
# MAGIC Check [model specifications](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models)
# MAGIC and [regional availability](https://docs.databricks.com/aws/en/unity-gateway/model-region-availability).
# MAGIC
# MAGIC ### Discover the course service
# MAGIC
# MAGIC In **Catalog Explorer**, open `course_ops` → `gateway` → `chat`, or find it in **AI Gateway**.
# MAGIC Inspect its routing destinations to see the actual model/provider and any fallbacks.
# MAGIC System-provided services live in `system.ai`; this course uses our custom service instead.
# MAGIC To query it, your identity needs `USE CATALOG` on `course_ops`, `USE SCHEMA` on
# MAGIC `course_ops.gateway`, and `EXECUTE` on `course_ops.gateway.chat`.
# MAGIC See [discovery and permissions](https://docs.databricks.com/aws/en/ai-gateway/govern-model-services).

# COMMAND ----------

workspace_client = WorkspaceClient()
model_service_name = "course_ops.gateway.chat"

logger.info(f"Workspace: {workspace_client.config.host}")
logger.info(f"Unity Catalog model service: {model_service_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Chat Through the Unified Responses API
# MAGIC
# MAGIC `WorkspaceClient` uses configured Databricks authentication to create a token valid for 20 minutes.
# MAGIC The authenticated identity needs permission to create tokens.
# MAGIC The token authenticates to Databricks, not directly to the model provider. This example uses the
# MAGIC existing service and makes a billable inference request; it creates a temporary token, but no service.
# MAGIC
# MAGIC Use `/ai-gateway/mlflow/v1` with the fully qualified service name in `model`.
# MAGIC Databricks recommends the **Unified Responses API** for provider-independent requests.
# MAGIC Here, `input` contains conversation history, `max_output_tokens` limits output, and
# MAGIC `response.output_text` collects the generated text. Usage fields are `input_tokens` and `output_tokens`.
# MAGIC See [query model services](https://docs.databricks.com/aws/en/ai-gateway/query-model-services).

# COMMAND ----------

databricks_token = workspace_client.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=databricks_token,
    base_url=f"{workspace_client.config.host}/ai-gateway/mlflow/v1",
)

response = client.responses.create(
    model=model_service_name,
    max_output_tokens=256,
    input=[
        {"role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello! How can I assist you today?"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "What is Databricks?"}],
        },
    ],
)

logger.info("Response:")
logger.info(response.output_text)
if response.usage is not None:
    logger.info(f"Tokens used: {response.usage.total_tokens}")
    logger.info(f"Input tokens: {response.usage.input_tokens}")
    logger.info(f"Output tokens: {response.usage.output_tokens}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pricing Comparison
# MAGIC
# MAGIC `course_ops.gateway.chat` is a routing service, not a pricing SKU. Check its destination model,
# MAGIC capacity option, and applicable gateway charges before estimating the cost of our chat requests.
# MAGIC The examples below are separate pricing exercises, not the measured cost of that service.
# MAGIC
# MAGIC ### Foundation Model APIs (Pay-per-token)
# MAGIC
# MAGIC Selected standard-tier rates, checked October 12, 2026; not a complete model catalog.
# MAGIC Units are **DBUs per million tokens**, not dollars:
# MAGIC
# MAGIC | Model | Input (DBU/1M tokens) | Output (DBU/1M tokens) |
# MAGIC |-------|----------------------|------------------------|
# MAGIC | Qwen3.5 122B A10B | 3.143 | 31.429 |
# MAGIC | GPT OSS 120B | 2.143 | 8.571 |
# MAGIC | Gemma 3 12B | 2.143 | 7.143 |
# MAGIC | GPT OSS 20B | 1.000 | 4.286 |
# MAGIC
# MAGIC ### Provisioned Throughput
# MAGIC
# MAGIC **Check eligibility before comparing costs.** A model being queryable through Unity Gateway
# MAGIC does not imply that it supports provisioned throughput in this workspace.
# MAGIC
# MAGIC - **On-demand:** no term commitment; eligible architectures vary by region.
# MAGIC   GPT OSS 120B/20B and Gemma 3 12B remain documented examples.
# MAGIC   See [on-demand deployment](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis)
# MAGIC   and the [regional support matrix](https://docs.databricks.com/aws/en/unity-gateway/model-region-availability).
# MAGIC - **Reserved:** a separate, curated list. The current AWS guide lists GLM 5.3, GLM 5.3 Flash,
# MAGIC   GLM 5.2 (legacy), Kimi K3, DeepSeek V4.1 Flash, and Qwen3.5 122B A10B
# MAGIC   (Public Preview; account-team enablement). GPT OSS 120B is not on that reserved list.
# MAGIC   The creation flow determines what you can reserve.
# MAGIC   See [reserved throughput eligibility](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/reserved-provisioned-throughput#supported-models).
# MAGIC
# MAGIC Reserved capacity starts at 50 model units, in increments of 50, for a prepaid 1- or 3-month term.
# MAGIC You pay for the full term even when idle and cannot cancel midway. Overflow is rate-limited
# MAGIC unless you configure a fallback; pay-per-token overflow is charged separately.
# MAGIC
# MAGIC #### On-demand pricing examples
# MAGIC
# MAGIC On-demand rates below are **DBU/hour per 50 model units**:
# MAGIC
# MAGIC | Model | DBU/hour per 50 model units |
# MAGIC |-------|----------------------------|
# MAGIC | Qwen3.5 122B A10B (Public Preview) | 85.714 |
# MAGIC | GPT OSS 120B | 71.429 |
# MAGIC | Gemma 3 12B | 71.429 |
# MAGIC | GPT OSS 20B | 53.571 |
# MAGIC
# MAGIC Source: [Foundation Model Serving pricing](https://www.databricks.com/product/pricing/foundation-model-serving).
# MAGIC Currency cost = DBUs consumed × your applicable currency/DBU SKU price.
# MAGIC Check cloud, region, contract, serving tier, and reservation terms. Proprietary models have a
# MAGIC [separate pricing page](https://www.databricks.com/product/pricing/proprietary-foundation-model-serving).
# MAGIC
# MAGIC Model units are capacity units, not tokens/second. Benchmark the required allocation for your
# MAGIC model and request shape. On-demand allocation limits are model-specific; reserved capacity
# MAGIC uses the 50-unit minimum described above. A pricing entry alone does not establish eligibility.
# MAGIC
# MAGIC ### Cost Calculation Example: Pay-per-token vs On-demand
# MAGIC
# MAGIC Compare the **same model (GPT OSS 120B)** with a 60% input / 40% output token mix.
# MAGIC This is an **on-demand** exercise, conditional on regional availability and valid allocation limits.
# MAGIC It is not a reserved-throughput quote or a claim that our chat service supports dedicated capacity.
# MAGIC The 100-model-unit allocation below is an illustrative fixed allocation, not a sizing recommendation.
# MAGIC This estimates serving DBUs only, excluding discounts and additional services.
# MAGIC For a reserved estimate, choose an eligible model, its reservation rate, and the full term's
# MAGIC billable hours; a 24-hour calculation understates the commitment. Include fallback usage separately.

# COMMAND ----------


def calculate_api_cost(
    input_tokens: int,
    output_tokens: int,
    input_dbu_per_1m: float,
    output_dbu_per_1m: float,
) -> float:
    """Calculate DBU cost for pay-per-token API."""
    input_cost = (input_tokens / 1_000_000) * input_dbu_per_1m
    output_cost = (output_tokens / 1_000_000) * output_dbu_per_1m
    return input_cost + output_cost


def calculate_provisioned_cost(hours: float, model_units: int, dbu_per_hour_per_50_units: float) -> float:
    """Estimate on-demand DBUs for a constant allocation over the given hours."""
    return hours * (model_units / 50) * dbu_per_hour_per_50_units


pricing_model_name = "GPT OSS 120B"
input_dbu_per_1m = 2.143
output_dbu_per_1m = 8.571
dbu_per_hour_per_50_units = 71.429
hours = 24
model_units = 100
input_tokens = 6_000_000
output_tokens = 4_000_000

api_cost = calculate_api_cost(input_tokens, output_tokens, input_dbu_per_1m, output_dbu_per_1m)
provisioned_cost = calculate_provisioned_cost(hours, model_units, dbu_per_hour_per_50_units)
input_fraction = input_tokens / (input_tokens + output_tokens)
blended_dbu_per_1m = input_fraction * input_dbu_per_1m + (1 - input_fraction) * output_dbu_per_1m
break_even_tokens = provisioned_cost / blended_dbu_per_1m * 1_000_000

logger.info(f"{pricing_model_name}: {input_tokens + output_tokens:,} tokens over {hours}h")
logger.info(f"Pay-per-token: {api_cost:.2f} DBUs")
logger.info(f"On-demand provisioned ({model_units} model units): {provisioned_cost:.2f} DBUs")
logger.info(f"Theoretical break-even at this token mix: {break_even_tokens:,.0f} tokens")
logger.info("Benchmark whether the allocated capacity can handle that volume and latency target.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Key Differences Summary
# MAGIC
# MAGIC These are backend choices behind a model service; callers use its Unity Catalog name.
# MAGIC
# MAGIC | Aspect | Pay-per-token Foundation APIs | Provisioned Throughput | External Models |
# MAGIC |--------|----------------|------------------------|-----------------|
# MAGIC | **Pricing** | Input/output tokens; tier-dependent | Allocated model units × time; term-dependent | Provider billing plus applicable Databricks charges |
# MAGIC | **Performance** | Shared capacity | Dedicated capacity | Varies by provider |
# MAGIC | **Latency** | Standard or priority tier | Sized to workload; benchmark | Provider-dependent |
# MAGIC | **Custom weights** | Preconfigured models | Supported architectures/variants | Provider-dependent |
# MAGIC | **Best For** | Variable workloads | High-volume, predictable | Specific model requirements |
# MAGIC | **Setup** | Model availability and query permissions | Requires provisioning | Requires provider configuration and credentials |
# MAGIC | **Eligibility** | Available models in workspace | Separate on-demand/reserved lists; region and enablement apply | Provider/model support |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Choosing the Right Option
# MAGIC
# MAGIC **Use pay-per-token when:**
# MAGIC - You have variable or unpredictable workloads
# MAGIC - You're prototyping or experimenting
# MAGIC - You want zero infrastructure management
# MAGIC - Consider priority mode for supported models when standard-tier latency is insufficient
# MAGIC
# MAGIC **Use Provisioned Throughput when:**
# MAGIC - Your exact model is eligible for the chosen on-demand or reserved option in your workspace
# MAGIC - You have high, predictable volume
# MAGIC - You need dedicated capacity sized for your throughput and latency targets
# MAGIC - You need to serve fine-tuned weights from a supported architecture
# MAGIC - Benchmarks and utilization justify paying for allocated capacity
# MAGIC
# MAGIC **Use External Models when:**
# MAGIC - You need specific models not available on Databricks
# MAGIC - You have existing contracts with providers
# MAGIC - You need specific model capabilities
# MAGIC
# MAGIC **Use AI Functions when:**
# MAGIC - You need batch enrichment or inference directly over tables
# MAGIC - For Unity Gateway integration, current `ai_query` support is limited to system-provided
# MAGIC   `system.ai` model services, not custom services such as `course_ops.gateway.chat`.
# MAGIC   See [batch inference limitations](https://docs.databricks.com/aws/en/ai-gateway/query-model-services#limitations).
# MAGIC
# MAGIC Recheck the linked model catalog, lifecycle policy, and pricing before each course run.
