# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Simple RAG with Lakebase pgvector
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is RAG (Retrieval-Augmented Generation)?
# MAGIC - Using Lakebase pgvector for document retrieval
# MAGIC - Enriching prompts with retrieved context
# MAGIC - Building a simple Q&A system
# MAGIC
# MAGIC **RAG Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Lakebase pgvector (retrieve relevant documents)
# MAGIC     ↓
# MAGIC Build Prompt (question + context)
# MAGIC     ↓
# MAGIC LLM (generate answer)
# MAGIC     ↓
# MAGIC Response
# MAGIC ```

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.vector_search import LakebasePgVectorManager

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# Create OpenAI client for Databricks
client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value,
    base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
)

pgvector = LakebasePgVectorManager(config=cfg, workspace=w)

logger.info(f"✓ Connected to workspace: {w.config.host}")
logger.info(f"✓ Using LLM endpoint: {cfg.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Lakebase pgvector Retrieval
# MAGIC
# MAGIC First, let's create a function to retrieve relevant documents from our vector search index.

# COMMAND ----------


def retrieve_documents(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve relevant documents from Lakebase pgvector.

    Args:
        query: The search query
        num_results: Number of documents to retrieve

    Returns:
        List of document dictionaries with title, text, and metadata

    """
    results = pgvector.hybrid_search(query=query, num_results=num_results)
    documents = []
    for row in results:
        documents.append(
            {
                "text": row.get("text", ""),
                "title": row.get("title", "N/A"),
                "arxiv_id": row.get("arxiv_id", "N/A"),
                "authors": row.get("authors", "N/A"),
                "year": row.get("year", "N/A"),
            }
        )

    return documents


# COMMAND ----------

# Test retrieval
query = "transformer attention mechanisms"
docs = retrieve_documents(query, num_results=3)

logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
for i, doc in enumerate(docs, 1):
    logger.info(f"\n{i}. {doc['title']}")
    logger.info(f"   ArXiv ID: {doc['arxiv_id']}")
    logger.info(f"   Text preview: {doc['text'][:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Building the RAG Prompt
# MAGIC
# MAGIC Now let's create a function that builds a prompt with the retrieved context.

# COMMAND ----------


def build_rag_prompt(question: str, documents: list[dict]) -> str:
    """Build a prompt with retrieved context.

    Args:
        question: The user's question
        documents: List of retrieved documents

    Returns:
        Formatted prompt string

    """
    # Format context from documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"""
Document {i}: {doc["title"]}
ArXiv ID: {doc["arxiv_id"]}
Content: {doc["text"]}
""")

    context = "\n---\n".join(context_parts)

    prompt = f"""You are a helpful research assistant. Answer the question based on the provided context from research papers.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based on the provided context
- If the context doesn't contain enough information, say so
- Cite the relevant paper titles when making claims
- Be concise but thorough

ANSWER:"""

    return prompt


# COMMAND ----------

# Test prompt building
test_prompt = build_rag_prompt("What is attention in transformers?", docs)
logger.info("Built RAG prompt:")
logger.info(f"Prompt length: {len(test_prompt)} characters")
logger.info(f"Preview:\n{test_prompt[:500]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. RAG Query Function
# MAGIC
# MAGIC Combine retrieval and generation into a single function.

# COMMAND ----------


def rag_query(question: str, num_docs: int = 5) -> dict:
    """Answer a question using RAG.

    Args:
        question: The user's question
        num_docs: Number of documents to retrieve

    Returns:
        Dictionary with answer and sources

    """
    # Step 1: Retrieve relevant documents
    logger.info(f"Retrieving documents for: '{question}'")
    documents = retrieve_documents(question, num_results=num_docs)
    logger.info(f"Retrieved {len(documents)} documents")

    # Step 2: Build prompt with context
    prompt = build_rag_prompt(question, documents)

    # Step 3: Generate answer with LLM
    logger.info("Generating answer...")
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    answer = response.choices[0].message.content

    # Return answer with sources
    return {
        "question": question,
        "answer": answer,
        "sources": [{"title": doc["title"], "arxiv_id": doc["arxiv_id"]} for doc in documents],
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test RAG System

# COMMAND ----------

# Test with a research question
result = rag_query("What are the key innovations in transformer architectures?")

logger.info("=" * 80)
logger.info(f"Question: {result['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result['answer']}")
logger.info("\nSources:")
for src in result["sources"]:
    logger.info(f"  - {src['title']} ({src['arxiv_id']})")

# COMMAND ----------

# Test with another question
result2 = rag_query("How do large language models handle reasoning tasks?")

logger.info("=" * 80)
logger.info(f"Question: {result2['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result2['answer']}")
logger.info("\nSources:")
for src in result2["sources"]:
    logger.info(f"  - {src['title']} ({src['arxiv_id']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. RAG with Conversation History
# MAGIC
# MAGIC Extend RAG to support multi-turn conversations.

# COMMAND ----------


class SimpleRAG:
    """Simple RAG system with conversation history."""

    def __init__(self, llm_endpoint: str, vector_store: LakebasePgVectorManager):
        self.llm_endpoint = llm_endpoint
        self.vector_store = vector_store
        self.conversation_history = []

        # Initialize clients
        self.w = WorkspaceClient()
        self.client = OpenAI(
            api_key=self.w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{self.w.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
        )

    def retrieve(self, query: str, num_results: int = 5) -> list[dict]:
        """Retrieve relevant documents."""
        results = self.vector_store.hybrid_search(query=query, num_results=num_results)
        documents = []
        for row in results:
            documents.append(
                {
                    "text": row.get("text", ""),
                    "title": row.get("title", "N/A"),
                    "arxiv_id": row.get("arxiv_id", "N/A"),
                }
            )
        return documents

    def chat(self, question: str, num_docs: int = 3) -> str:
        """Chat with RAG, maintaining conversation history."""
        # Retrieve documents
        documents = self.retrieve(question, num_results=num_docs)

        # Build context
        context = "\n\n".join([f"[{doc['title']}]: {doc['text']}" for doc in documents])

        # Build system message with context
        system_message = f"""You are a helpful research assistant. Use the following context from research papers to answer questions.

CONTEXT:
{context}

If the context doesn't contain relevant information, say so. Always cite paper titles when making claims."""

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": question})

        # Build messages for LLM
        messages = [{"role": "system", "content": system_message}] + self.conversation_history

        # Generate response
        response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# COMMAND ----------

# Create RAG instance
rag = SimpleRAG(llm_endpoint=cfg.llm_endpoint, vector_store=pgvector)

logger.info("✓ SimpleRAG initialized")

# COMMAND ----------

# Multi-turn conversation
logger.info("Starting multi-turn RAG conversation...")
logger.info("=" * 80)

# First question
q1 = "What is attention in neural networks?"
a1 = rag.chat(q1)
logger.info(f"Q: {q1}")
logger.info(f"A: {a1}\n")

# COMMAND ----------

# Follow-up question (uses conversation history)
q2 = "How does self-attention differ from cross-attention?"
a2 = rag.chat(q2)
logger.info(f"Q: {q2}")
logger.info(f"A: {a2}\n")

# COMMAND ----------

# Another follow-up
q3 = "What are the computational costs?"
a3 = rag.chat(q3)
logger.info(f"Q: {q3}")
logger.info(f"A: {a3}")
