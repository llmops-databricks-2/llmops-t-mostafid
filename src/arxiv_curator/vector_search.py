"""Lakebase pgvector search over the funding call chunks Synced Table.

Adapted from the course's vector_search.py: same connection, extension, index and
scoring logic; only the table name, the columns and the filter (is_open instead of
year) are changed for the Funding Scout use case.
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import ProjectConfig

# Columns returned by every search (my funding call chunk table)
RESULT_COLUMNS = "id, call_id, title, deadline, status, is_open, url, text"


class LakebasePgVectorManager:
    """Search funding call chunk embeddings in a Lakebase Synced Table with pgvector.

    The Synced Table is a read-only copy of funding_call_chunks, created with a
    type override so that `embedding` is a pgvector vector column. Full-text
    search uses an expression index, built on the same expression the queries use.
    """

    synced_table_name = "funding_call_chunks_synced"

    def __init__(
        self,
        config: ProjectConfig | None = None,
        table_name: str | None = None,
        *,
        catalog: str | None = None,
        schema: str | None = None,
        lakebase_project_id: str | None = None,
        embedding_model: str | None = None,
        workspace: WorkspaceClient | None = None,
    ) -> None:
        self.config = config
        self.catalog = config.catalog if config else catalog
        self.schema = config.schema if config else schema
        self.lakebase_project_id = config.lakebase_project_id if config else lakebase_project_id
        self.embedding_model = embedding_model or (config.embedding_endpoint if config else None)
        if not self.catalog or not self.schema or not self.lakebase_project_id:
            raise ValueError("catalog, schema, and lakebase_project_id are required")
        if not self.embedding_model:
            raise ValueError("embedding_model is required")
        # Synced Tables land in a Postgres schema named after the Unity Catalog schema
        self.table_name = table_name or f'"{self.schema}"."{self.synced_table_name}"'
        self._connection_string: str | None = None
        self.workspace = workspace

        # Shared by the GIN index and the queries, so Postgres can use the index
        self.text_expr = "to_tsvector('english', coalesce(title, '') || ' ' || text)"

    def _get_connection_string(self) -> str:
        """Create a short-lived Lakebase connection string (no stored password)."""
        if self._connection_string:
            return self._connection_string

        workspace = self.workspace or WorkspaceClient()
        postgres_api = workspace.postgres
        endpoint_name = os.environ.get("LAKEBASE_ENDPOINT")
        if endpoint_name:
            host = os.environ.get("PGHOST")
            database = os.environ.get("PGDATABASE", "databricks_postgres")
        else:
            project_name = f"projects/{self.lakebase_project_id}"
            branch = next(iter(postgres_api.list_branches(parent=project_name)))
            endpoint = next(iter(postgres_api.list_endpoints(parent=branch.name)))
            endpoint_name = endpoint.name
            host = endpoint.status.hosts.host
            database = "databricks_postgres"
        credential = postgres_api.generate_database_credential(endpoint=endpoint_name)
        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        username = urllib.parse.quote_plus(client_id or workspace.current_user.me().user_name)

        self._connection_string = f"postgresql://{username}:{credential.token}@{host}:5432/{database}?sslmode=require"
        return self._connection_string

    def connection(self) -> psycopg.Connection:
        """Open a Lakebase connection using a fresh workspace credential."""
        self._connection_string = None
        return psycopg.connect(self._get_connection_string())

    def ensure_extension(self) -> None:
        """Enable lakebase_vector (installs pgvector with CASCADE).

        Must run before the Synced Table with a vector column is created: creating a
        synced table does not install extensions.
        """
        with self.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")
        logger.info("lakebase_vector extension ready")

    def ensure_indexes(self) -> None:
        """Create the full-text (GIN) index on the Synced Table. Safe to rerun."""
        with self.connection() as conn:
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.synced_table_name}_search_gin
                ON {self.table_name} USING gin (({self.text_expr}))
                """
            )
        logger.info(f"Full-text index ready on {self.table_name}")

    @staticmethod
    def _vector_literal(vector: list[float] | tuple[float, ...]) -> str:
        """Format a Python vector for PostgreSQL's vector input syntax."""
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def embed_query(self, query: str) -> list[float]:
        """Turn the user's question into a vector, with the same model as the chunks."""
        workspace = self.workspace or WorkspaceClient()
        client = workspace.serving_endpoints.get_open_ai_client()
        response = client.embeddings.create(model=self.embedding_model, input=query)
        if not response.data or not response.data[0].embedding:
            raise ValueError("The embedding endpoint returned no vector")
        return list(response.data[0].embedding)

    def _fetch(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def search(
        self,
        query: str,
        num_results: int = 5,
        is_open: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search (cosine similarity), optionally only open calls."""
        query_vector = self._vector_literal(self.embed_query(query))
        where = "WHERE embedding IS NOT NULL"
        params: list[Any] = [query_vector]
        if is_open is not None:
            where += " AND is_open = %s"
            params.append(is_open)
        params.extend([query_vector, num_results])
        sql = f"""
            SELECT {RESULT_COLUMNS},
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.table_name}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        return self._fetch(sql, params)

    def hybrid_search(
        self,
        query: str,
        num_results: int = 5,
        is_open: bool | None = None,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Combine cosine similarity (meaning) and PostgreSQL full-text ranking (keywords)."""
        query_vector = self._vector_literal(self.embed_query(query))
        text_weight = 1.0 - vector_weight
        where = "WHERE embedding IS NOT NULL"
        params: list[Any] = [query_vector, query]
        if is_open is not None:
            where += " AND is_open = %s"
            params.append(is_open)
        params.extend([vector_weight, text_weight, num_results])
        sql = f"""
            WITH ranked AS (
                SELECT {RESULT_COLUMNS},
                       1 - (embedding <=> %s::vector) AS vector_score,
                       ts_rank_cd({self.text_expr}, plainto_tsquery('english', %s)) AS text_score
                FROM {self.table_name}
                {where}
            )
            SELECT *, (%s * vector_score + %s * text_score) AS score
            FROM ranked
            ORDER BY score DESC
            LIMIT %s
        """
        return self._fetch(sql, params)
