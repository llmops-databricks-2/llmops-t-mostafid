# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.2: Provisioned Throughput Deployment
# MAGIC
# MAGIC Deploy an **on-demand** endpoint using the Databricks SDK, create a Unity Catalog model
# MAGIC service that routes to it, query through Unity Gateway, and clean up after the lab.
# MAGIC **Checked: October 12, 2026 (AWS).**
# MAGIC
# MAGIC Running all cells with the default flags only calculates a cost estimate.
# MAGIC Enable deployment, inference, and cleanup individually as you work through the notebook.
# MAGIC No reservation is purchased by this lab.

# COMMAND ----------

from datetime import timedelta
from urllib.parse import quote

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import PtEndpointCoreConfig, PtServedModel
from loguru import logger
from openai import OpenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. On-demand and Reserved Are Different
# MAGIC
# MAGIC On-demand capacity has no fixed term. This lab uses GPT OSS 120B, which uses model units.
# MAGIC Legacy Llama 3.x deployments use throughput bands instead; do not reuse their configuration.
# MAGIC Eligibility and allocation increments are checked through the model optimization API.
# MAGIC See [on-demand deployment](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis)
# MAGIC and [model units](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/model-units).
# MAGIC
# MAGIC Reserved capacity has a separate, narrower model list: GLM 5.3/5.3 Flash, legacy GLM 5.2,
# MAGIC Kimi K3, DeepSeek V4.1 Flash, and preview Qwen3.5 122B A10B. Availability varies by workspace.
# MAGIC Its prepaid 1- or 3-month commitments cannot be cancelled midway. Active reservations prevent
# MAGIC endpoint deletion. The reserved creation guide documents a UI workflow; the SDK call below
# MAGIC does not select a reservation term. See [reserved capacity](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/reserved-provisioned-throughput).
# MAGIC
# MAGIC A model being queryable through Unity Gateway does not establish on-demand eligibility.
# MAGIC Check the [regional matrix](https://docs.databricks.com/aws/en/unity-gateway/model-region-availability).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure the Lab
# MAGIC
# MAGIC Use distinct lab resource names. This notebook leaves `course_ops.gateway.chat` unchanged.
# MAGIC Confirm the registered model name/version in your workspace before enabling deployment.
# MAGIC The model below is the example in Databricks' on-demand guide.
# MAGIC
# MAGIC Authenticate the SDK for the specified workspace (for example, `DATABRICKS_TOKEN`).
# MAGIC You need access to the source model and permission to create serving endpoints. Creating the
# MAGIC gateway service requires access to its parent catalog/schema, `CREATE SERVICE`, and access
# MAGIC to its destination. Callers need `EXECUTE` on the service plus `USE CATALOG`/`USE SCHEMA`.
# MAGIC See [model service setup and permissions](https://docs.databricks.com/aws/en/ai-gateway/model-services).

# COMMAND ----------

WORKSPACE_URL = "https://dbc-88df4c2c-4033.cloud.databricks.com"
MODEL_NAME = "system.ai.gpt-oss-120b"
MODEL_VERSION = "1"
ENDPOINT_NAME = "course-gpt-oss-120b-on-demand"
SERVICE_SCHEMA = "course_ops.gateway"
SERVICE_ID = "provisioned_chat"
MODEL_SERVICE_NAME = f"{SERVICE_SCHEMA}.{SERVICE_ID}"
CAPACITY_CHUNKS = 2
RUN_DEPLOYMENT = False
RUN_INFERENCE = False

workspace_client = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Estimate On-demand Cost
# MAGIC
# MAGIC GPT OSS 120B: **71.429 DBU/hour per 50 model units**.
# MAGIC Source: [Foundation Model Serving pricing](https://www.databricks.com/product/pricing/foundation-model-serving).
# MAGIC DBUs are not dollars: multiply by your applicable currency/DBU SKU price.
# MAGIC Add gateway charges separately. This fixed-allocation estimate assumes burst scaling is disabled.
# MAGIC
# MAGIC The 50-unit example is arithmetic, not a sizing recommendation. Deployment derives its allocation
# MAGIC from the model's supported chunk size. There is no universal tokens/second-per-unit conversion.
# MAGIC Billing continues for allocated capacity while idle; finishing inference does not stop it.

# COMMAND ----------


def estimate_on_demand_cost(model_units: int, allocated_hours: float, dbu_per_hour_per_50_units: float) -> float:
    """Estimate DBUs for a constant on-demand allocation, including idle hours."""
    if model_units <= 0 or allocated_hours < 0 or dbu_per_hour_per_50_units <= 0:
        raise ValueError("Capacity and rate must be positive; allocated hours cannot be negative.")
    return (model_units / 50) * allocated_hours * dbu_per_hour_per_50_units


DBU_PER_HOUR_PER_50_UNITS = 71.429
example_dbu = estimate_on_demand_cost(50, 2, DBU_PER_HOUR_PER_50_UNITS)
logger.info(f"Illustrative 50 units for 2 hours: {example_dbu:.3f} DBUs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check Eligibility and Create the Endpoint
# MAGIC
# MAGIC Set `RUN_DEPLOYMENT=True`, then run configuration and deployment cells.
# MAGIC The SDK exposes `create_provisioned_throughput_endpoint`. Its typed request uses
# MAGIC `provisioned_model_units`, not `workload_size` or legacy throughput-band fields.
# MAGIC See the [SDK reference](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/serving/serving_endpoints.html).
# MAGIC
# MAGIC The optimization lookup uses the SDK's authenticated REST client. Unsupported models or
# MAGIC missing model-unit information stop deployment. Existing lab names also stop creation:
# MAGIC inspect/reuse them deliberately rather than overwriting another endpoint or service.
# MAGIC Deployment may still fail because of permissions or regional capacity.

# COMMAND ----------

if RUN_DEPLOYMENT:
    workspace_client = WorkspaceClient(host=WORKSPACE_URL)
    if not isinstance(CAPACITY_CHUNKS, int) or isinstance(CAPACITY_CHUNKS, bool) or CAPACITY_CHUNKS < 1:
        raise ValueError("CAPACITY_CHUNKS must be a positive integer.")

    optimization = workspace_client.api_client.do(
        "GET",
        f"/api/2.0/serving-endpoints/get-model-optimization-info/"
        f"{quote(MODEL_NAME, safe='')}/{quote(MODEL_VERSION, safe='')}",
    )
    if not optimization.get("optimizable"):
        raise ValueError(f"{MODEL_NAME} version {MODEL_VERSION} is not eligible.")
    chunk_size = optimization.get("model_unit_chunk_size")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("No valid model-unit chunk size returned; do not use throughput-band values.")

    desired_model_units = CAPACITY_CHUNKS * chunk_size
    logger.info(f"Requested on-demand allocation: {desired_model_units} model units")
    if MODEL_NAME == "system.ai.gpt-oss-120b":
        hourly_dbu = estimate_on_demand_cost(desired_model_units, 1, DBU_PER_HOUR_PER_50_UNITS)
        logger.info(f"Estimated fixed-capacity rate: {hourly_dbu:.3f} DBUs/hour")

    try:
        workspace_client.serving_endpoints.get(ENDPOINT_NAME)
    except NotFound:
        pass
    else:
        raise ValueError(f"Endpoint {ENDPOINT_NAME} already exists; inspect it or choose a new lab name.")

    try:
        workspace_client.api_client.do(
            "GET", f"/api/2.1/unity-catalog/model-services/{quote(MODEL_SERVICE_NAME, safe='')}"
        )
    except NotFound:
        pass
    else:
        raise ValueError(f"Service {MODEL_SERVICE_NAME} already exists; choose a new lab name.")

    deployment = workspace_client.serving_endpoints.create_provisioned_throughput_endpoint(
        name=ENDPOINT_NAME,
        config=PtEndpointCoreConfig(
            served_entities=[
                PtServedModel(
                    entity_name=MODEL_NAME,
                    entity_version=MODEL_VERSION,
                    provisioned_model_units=desired_model_units,
                    burst_scaling_enabled=False,
                )
            ]
        ),
    )
    deployed_endpoint = deployment.result(timeout=timedelta(minutes=30))
    logger.info(f"Endpoint deployment complete: {deployed_endpoint.name}")
else:
    logger.info("Deployment skipped. Set RUN_DEPLOYMENT=True when ready to allocate capacity.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create the Unity Catalog Model Service
# MAGIC
# MAGIC The pinned SDK (0.118.0) lacks typed gateway service helpers, so use its authenticated
# MAGIC `api_client.do` with the [model-service REST API](https://docs.databricks.com/api/ai-gateway/v1/create-model-service).
# MAGIC The route targets `serving-endpoints/{name}`, with all traffic going to the lab endpoint.
# MAGIC This creates the gateway service programmatically; no UI step is required.
# MAGIC If service creation fails after endpoint creation, the endpoint remains allocated; inspect it
# MAGIC and use the cleanup cell when abandoning the lab.

# COMMAND ----------

if RUN_DEPLOYMENT:
    model_service = workspace_client.api_client.do(
        "POST",
        "/api/2.1/unity-catalog/model-services",
        query={"parent": f"schemas/{SERVICE_SCHEMA}", "model_service_id": SERVICE_ID},
        body={
            "comment": "Lecture 1.2: on-demand provisioned throughput lab",
            "config": {
                "routing": {
                    "destinations": [
                        {
                            "name": "primary",
                            "destination_type": "DESTINATION_TYPE_PROVISIONED_THROUGHPUT_FOUNDATION_MODEL",
                            "provisioned_throughput_config": {
                                "model_serving_endpoint": f"serving-endpoints/{ENDPOINT_NAME}"
                            },
                            "traffic_percentage": 100,
                        }
                    ]
                }
            },
        },
    )
    logger.info(f"Created Unity Catalog service: {MODEL_SERVICE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Query Through Unity Gateway
# MAGIC
# MAGIC Set `RUN_INFERENCE=True` after deployment and service creation succeed.
# MAGIC With deployment disabled, you can also query an already-created lab service.
# MAGIC The authenticated Databricks SDK creates a 20-minute token for the OpenAI client.
# MAGIC Your workspace must permit personal access token creation. No OpenAI API key is needed.
# MAGIC See [Unified Responses API](https://docs.databricks.com/aws/en/ai-gateway/query-model-services).

# COMMAND ----------

if RUN_INFERENCE:
    if workspace_client is None:
        workspace_client = WorkspaceClient(host=WORKSPACE_URL)
    databricks_token = workspace_client.tokens.create(lifetime_seconds=1200).token_value
    if not databricks_token:
        raise ValueError("Databricks did not return a token. Check token creation permissions.")

    client = OpenAI(
        api_key=databricks_token,
        base_url=f"{WORKSPACE_URL}/ai-gateway/mlflow/v1",
    )
    response = client.responses.create(
        model=MODEL_SERVICE_NAME,
        max_output_tokens=256,
        input=[{"role": "user", "content": "Explain the benefits of provisioned throughput for LLMs."}],
    )
    logger.info(response.output_text)
    if response.usage is not None:
        logger.info(f"Input tokens: {response.usage.input_tokens}")
        logger.info(f"Output tokens: {response.usage.output_tokens}")
        logger.info(f"Total tokens: {response.usage.total_tokens}")
else:
    logger.info("Inference skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Monitor and Clean Up
# MAGIC
# MAGIC Use endpoint metrics for measured request rate, errors, latency, and time-to-first-token.
# MAGIC SDK endpoint configuration is not a throughput benchmark.
# MAGIC
# MAGIC After the lab, verify `MODEL_SERVICE_NAME` and `ENDPOINT_NAME`, then set `RUN_CLEANUP=True`
# MAGIC **in the next cell**. Cleanup deletes those named resources, even after configuration is rerun
# MAGIC or the kernel is restarted. Run imports and configuration first after a restart.
# MAGIC The service is deleted before the endpoint. Missing resources are reported and skipped;
# MAGIC permission or other API errors are raised, not treated as successful cleanup.
# MAGIC On-demand charges continue until the allocated endpoint is removed.
# MAGIC Reserved endpoints have different lifecycle restrictions; do not apply this cleanup to them.

# COMMAND ----------

RUN_CLEANUP = False


def cleanup_lab_resources(client: WorkspaceClient, service_name: str, endpoint_name: str) -> None:
    """Delete the explicitly selected lab service and endpoint; tolerate absent resources."""
    logger.info(f"Cleaning up service {service_name} and endpoint {endpoint_name}")
    try:
        client.api_client.do("DELETE", f"/api/2.1/unity-catalog/model-services/{quote(service_name, safe='')}")
    except NotFound:
        logger.info(f"Lab service already absent: {service_name}")
    else:
        logger.info(f"Deleted lab service: {service_name}")

    try:
        client.serving_endpoints.delete(endpoint_name)
    except NotFound:
        logger.info(f"Lab endpoint already absent: {endpoint_name}")
    else:
        logger.info(f"Deleted lab endpoint: {endpoint_name}")


if RUN_CLEANUP:
    cleanup_client = WorkspaceClient(host=WORKSPACE_URL)
    cleanup_lab_resources(cleanup_client, MODEL_SERVICE_NAME, ENDPOINT_NAME)
else:
    logger.info("Cleanup skipped: set RUN_CLEANUP=True in this cell and rerun it to delete the lab resources.")
