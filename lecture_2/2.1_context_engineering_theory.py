# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.1: Context Engineering Theory
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is context engineering
# MAGIC - RAG (Retrieval Augmented Generation) fundamentals
# MAGIC - Context window limitations
# MAGIC - Strategies for effective context engineering
# MAGIC - Trade-offs and best practices

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Context Engineering?
# MAGIC
# MAGIC **Context Engineering** is the practice of providing relevant information to LLMs to improve their responses.
# MAGIC
# MAGIC ### Why Context Matters
# MAGIC
# MAGIC - LLMs have knowledge cutoff dates
# MAGIC - Need access to private/proprietary data
# MAGIC - Reduce hallucinations
# MAGIC - Improve accuracy and relevance
# MAGIC - Enable domain-specific applications

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RAG (Retrieval Augmented Generation)
# MAGIC
# MAGIC ### RAG Pipeline Overview
# MAGIC
# MAGIC ```
# MAGIC User Query
# MAGIC     ↓
# MAGIC Query Embedding
# MAGIC     ↓
# MAGIC Vector Search (Retrieve relevant documents)
# MAGIC     ↓
# MAGIC Context Assembly
# MAGIC     ↓
# MAGIC LLM Prompt (Query + Context)
# MAGIC     ↓
# MAGIC Generated Response
# MAGIC ```
# MAGIC
# MAGIC ### Benefits of RAG
# MAGIC
# MAGIC 1. **Up-to-date Information**: Access to latest data
# MAGIC 2. **Domain Knowledge**: Incorporate specialized information
# MAGIC 3. **Reduced Hallucinations**: Grounded in retrieved facts
# MAGIC 4. **Cost-Effective**: No need to fine-tune for every use case
# MAGIC 5. **Transparency**: Can cite sources

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Context Window Limitations
# MAGIC
# MAGIC ### Token Limits by Model
# MAGIC
# MAGIC Models available on Databricks (October 2026). Context windows change often: check the
# MAGIC [supported models](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models)
# MAGIC page and the providers' documentation before relying on a number.
# MAGIC
# MAGIC | Model | Provider | Context Window | Notes |
# MAGIC |-------|----------|---------------|-------|
# MAGIC | Claude Opus 5.5 / Sonnet 5.5 / Haiku 5.5 | Anthropic | 1M tokens | ~750K words |
# MAGIC | Kimi K3 | Moonshot AI | 1M tokens | ~750K words |
# MAGIC | GLM 5.3 | Zhipu AI | 1,048,576 tokens | ~786K words |
# MAGIC | Grok 4.6 | xAI | 500K tokens | ~375K words |
# MAGIC | GPT OSS 120B / 20B | OpenAI (open weights) | 128K tokens | ~96K words |
# MAGIC
# MAGIC Large windows do not remove the need for retrieval: cost and latency grow with every token,
# MAGIC and embedding models have far smaller windows than chat models, which bounds chunk size.

# COMMAND ----------

from loguru import logger

# COMMAND ----------


# Example: Token estimation
def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 characters per token."""
    return len(text) // 4


# Example texts
short_text = "Hello, world!"
long_text = "This is a much longer piece of text that would be used in a real application. " * 100

logger.info(f"Short text: {estimate_tokens(short_text)} tokens")
logger.info(f"Long text: {estimate_tokens(long_text)} tokens")

# Context window example
context_window = 128_000  # GPT OSS 120B
system_prompt_tokens = 200
user_query_tokens = 100
max_output_tokens = 2000

available_for_context = context_window - system_prompt_tokens - user_query_tokens - max_output_tokens
logger.info(f"\nAvailable tokens for context: {available_for_context:,}")
logger.info(f"Approximate words: {available_for_context * 0.75:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Retrieval Strategies Overview
# MAGIC
# MAGIC ### Three Main Approaches
# MAGIC
# MAGIC **1. Semantic Search (Vector Search)**
# MAGIC - Convert documents to embeddings (numerical vectors)
# MAGIC - Find documents with similar vector representations
# MAGIC - **Pros**: Captures meaning, handles synonyms, cross-lingual
# MAGIC - **Cons**: May miss exact keyword matches
# MAGIC
# MAGIC **2. Hybrid Search**
# MAGIC - Combines semantic search (embeddings) + keyword search (BM25)
# MAGIC - Merges results using fusion algorithms
# MAGIC - **Pros**: Best of both worlds - meaning + exact matches
# MAGIC - **Cons**: More complex, slightly slower
# MAGIC
# MAGIC **3. Reranking**
# MAGIC - Retrieve more candidates (e.g., top 20-50)
# MAGIC - Use a cross-encoder model to rerank
# MAGIC - Return top-k after reranking
# MAGIC - **Pros**: Higher precision
# MAGIC - **Cons**: Additional computation cost
# MAGIC
# MAGIC **Note**: We'll implement these with actual code in **Notebook 2.4**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Enhancement Technique: Query Rewriting
# MAGIC
# MAGIC Before retrieval, enhance the query by generating variations:
# MAGIC - Use different terminology
# MAGIC - Make it more specific or general
# MAGIC - Focus on different aspects
# MAGIC
# MAGIC This improves recall by searching with multiple phrasings.

# COMMAND ----------

# Example: Query rewriting for better retrieval
from databricks.sdk import WorkspaceClient
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

w = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()
cfg = load_config("../project_config.yml", get_env(spark))

# The course chat model service in Unity AI Gateway (course_ops.gateway.chat),
# called with a Databricks token valid for 20 minutes.
databricks_token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=databricks_token,
    base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
)

MODEL_NAME = cfg.llm_endpoint
logger.info(f"Using model: {MODEL_NAME}")


def rewrite_query(original_query: str) -> list[str]:
    """Generate query variations for better retrieval."""
    prompt = f"""Given this search query, generate 3 alternative phrasings that would help retrieve relevant information:

Original query: {original_query}

Generate 3 variations that:
1. Use different terminology
2. Are more specific or more general
3. Focus on different aspects

Return only the 3 variations, one per line."""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        max_output_tokens=200,
    )

    variations = response.output_text.strip().split("\n")
    return [v.strip() for v in variations if v.strip()]


# Example
original = "How do I deploy a model in Databricks?"
variations = rewrite_query(original)

logger.info(f"Original: {original}\n")
logger.info("Variations:")
for i, var in enumerate(variations, 1):
    logger.info(f"{i}. {var}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Context Quality vs Quantity
# MAGIC
# MAGIC ### The Trade-off
# MAGIC
# MAGIC - **More context** = More information, but:
# MAGIC   - Higher cost (more tokens)
# MAGIC   - Slower inference
# MAGIC   - Risk of "lost in the middle" problem
# MAGIC   - More noise
# MAGIC
# MAGIC - **Less context** = Faster and cheaper, but:
# MAGIC   - May miss important information
# MAGIC   - Less comprehensive answers

# COMMAND ----------


# Example: Context ordering strategies
def order_context_by_relevance(chunks: list[dict]) -> list[dict]:
    """Order chunks to avoid 'lost in the middle' problem.

    Strategy: Most relevant at start, second-most at end, rest in middle.
    """
    if len(chunks) <= 2:
        return chunks

    # Assume chunks are already sorted by relevance score
    ordered = []

    # Most relevant at start
    ordered.append(chunks[0])

    # Least relevant in middle
    if len(chunks) > 2:
        ordered.extend(chunks[2:-1])

    # Second most relevant at end
    if len(chunks) > 1:
        ordered.append(chunks[1])

    return ordered


# Example chunks (with mock relevance scores)
chunks = [
    {"text": "Most relevant chunk", "score": 0.95},
    {"text": "Second most relevant", "score": 0.88},
    {"text": "Third relevant", "score": 0.75},
    {"text": "Fourth relevant", "score": 0.70},
    {"text": "Fifth relevant", "score": 0.65},
]

ordered = order_context_by_relevance(chunks)
logger.info("Ordered chunks:")
for i, chunk in enumerate(ordered, 1):
    logger.info(f"{i}. {chunk['text']} (score: {chunk['score']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Context Compression Techniques
# MAGIC
# MAGIC When context is too large for the window, compress it:
# MAGIC
# MAGIC ### Technique 1: Extractive Summarization
# MAGIC - Select most important sentences/chunks
# MAGIC - Fast, preserves original wording
# MAGIC - Simple scoring: TF-IDF, position, length
# MAGIC
# MAGIC ### Technique 2: Abstractive Summarization
# MAGIC - Use LLM to generate summaries
# MAGIC - More concise and coherent
# MAGIC - Slower, costs tokens

# COMMAND ----------


def summarize_chunk(text: str, max_length: int = 100) -> str:
    """Summarize a text chunk using LLM."""
    prompt = f"""Summarize the following text in {max_length} words or less, preserving key information:

{text}

Summary:"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        max_output_tokens=max_length * 2,  # Rough token estimate
    )

    return response.output_text.strip()


# Example
long_text = """
Databricks is a unified analytics platform that combines data engineering,
data science, and machine learning. It provides a collaborative environment
for data teams to work together on big data and AI projects. The platform
is built on top of Apache Spark and offers features like Delta Lake for
reliable data lakes, MLflow for machine learning lifecycle management,
and Unity Catalog for unified governance. Databricks supports multiple
programming languages including Python, SQL, R, and Scala, making it
accessible to various types of data professionals.
"""

summary = summarize_chunk(long_text, max_length=50)
logger.info(f"Original length: {len(long_text)} chars")
logger.info(f"Summary length: {len(summary)} chars")
logger.info(f"\nSummary:\n{summary}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Technique 3: Chunk Selection & Filtering
# MAGIC - Retrieve more chunks than needed
# MAGIC - Filter by relevance threshold
# MAGIC - Remove duplicates or near-duplicates
# MAGIC - **Pros**: Reduces noise
# MAGIC - **Cons**: May filter out useful context
# MAGIC
# MAGIC **Note**: For reranking with cross-encoder models, see **Notebook 2.4**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Metadata Filtering
# MAGIC
# MAGIC Enhance retrieval with metadata filters:
# MAGIC
# MAGIC - **Date filters**: Recent documents only
# MAGIC - **Source filters**: Specific departments/teams
# MAGIC - **Type filters**: Documentation vs code vs discussions
# MAGIC - **Access control**: User permissions
# MAGIC - **Language**: Specific languages

# COMMAND ----------
import json

# Example: Metadata structure
example_document = {
    "id": "doc_123",
    "text": "Content of the document...",
    "embedding": [0.1, 0.2, 0.3],  # Vector embedding
    "metadata": {
        "source": "documentation",
        "date": "2024-01-15",
        "author": "data-team",
        "department": "engineering",
        "language": "en",
        "tags": ["databricks", "mlops", "deployment"],
        "access_level": "internal",
    },
}

logger.info("Example document with metadata:")

logger.info(json.dumps(example_document, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Prompt Engineering for RAG
# MAGIC
# MAGIC ### Effective RAG Prompts
# MAGIC
# MAGIC ```
# MAGIC System: You are a helpful assistant. Use the provided context
# MAGIC to answer questions. If the answer is not in the context,
# MAGIC say "I don't have enough information to answer that."
# MAGIC
# MAGIC Context:
# MAGIC [Retrieved documents here]
# MAGIC
# MAGIC Question: [User question]
# MAGIC
# MAGIC Answer:
# MAGIC ```

# COMMAND ----------


def create_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """Create a RAG prompt with context."""
    context = "\n\n".join([f"[Document {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    return prompt


# Example
query = "What is Databricks?"
context_chunks = [
    "Databricks is a unified analytics platform for data engineering and data science.",
    "The platform is built on Apache Spark and provides collaborative notebooks.",
]

prompt = create_rag_prompt(query, context_chunks)
logger.info(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluation Metrics for Context Engineering
# MAGIC
# MAGIC ### Retrieval Metrics
# MAGIC - **Precision@K**: Proportion of retrieved docs that are relevant
# MAGIC - **Recall@K**: Proportion of relevant docs that are retrieved
# MAGIC - **MRR (Mean Reciprocal Rank)**: Position of first relevant result
# MAGIC - **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality
# MAGIC
# MAGIC ### End-to-End Metrics
# MAGIC - **Answer Relevance**: Is the answer relevant to the question?
# MAGIC - **Faithfulness**: Is the answer grounded in the context?
# MAGIC - **Context Relevance**: Is the retrieved context relevant?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Best Practices Summary
# MAGIC
# MAGIC ### Do:
# MAGIC 1. Chunk documents appropriately (more in next notebook)
# MAGIC 2. Use high-quality embeddings
# MAGIC 3. Implement metadata filtering
# MAGIC 4. Order context strategically
# MAGIC 5. Monitor retrieval quality
# MAGIC 6. Provide clear instructions in prompts
# MAGIC 7. Handle cases where context doesn't contain the answer
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Exceed context window limits
# MAGIC 2. Include irrelevant information
# MAGIC 3. Ignore the "lost in the middle" problem
# MAGIC 4. Forget to cite sources
# MAGIC 5. Assume all retrieved docs are relevant
