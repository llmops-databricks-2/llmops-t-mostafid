import json
import os
import urllib.parse
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import PostgresAPI
from loguru import logger
from psycopg_pool import ConnectionPool


class LakebaseMemory:
    """Handles session message persistence using Lakebase (PostgreSQL)."""

    def __init__(
        self,
        project_id: str,
        workspace: WorkspaceClient | None = None,
    ) -> None:
        self.project_id = project_id
        self.workspace = workspace
        self._pool: ConnectionPool | None = None

    def _get_connection_string(self) -> str:
        """Build connection string for Lakebase.

        Supports two authentication modes:
        - SPN (production): Needs LAKEBASE_SP_CLIENT_ID, LAKEBASE_SP_CLIENT_SECRET, LAKEBASE_SP_HOST
        - User (local testing): Uses default WorkspaceClient auth
        """
        # Use dedicated Lakebase SPN env vars to avoid overriding
        # the default WorkspaceClient auth used by MCP tools
        client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID") or os.environ.get("DATABRICKS_CLIENT_ID")
        client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET") or os.environ.get("DATABRICKS_CLIENT_SECRET")
        host = os.environ.get("LAKEBASE_SP_HOST") or os.environ.get("DATABRICKS_HOST")

        if client_id and client_secret and host:
            w = WorkspaceClient(
                host=host,
                client_id=client_id,
                client_secret=client_secret,
            )
        else:
            w = self.workspace or WorkspaceClient()

        pg_api = PostgresAPI(w.api_client)

        # Determine username based on auth type
        if client_id:
            username = client_id
        else:
            user = w.current_user.me()
            username = urllib.parse.quote_plus(user.user_name)

        # Apps expose the selected Lakebase endpoint and host as resources.
        endpoint_name = os.environ.get("LAKEBASE_ENDPOINT")
        if endpoint_name:
            host = os.environ.get("PGHOST")
            database = os.environ.get("PGDATABASE", "databricks_postgres")
        else:
            project_parent = f"projects/{self.project_id}"
            default_branch = next(iter(pg_api.list_branches(parent=project_parent)))
            endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
            endpoint_name = endpoint.name
            host = endpoint.status.hosts.host
            database = "databricks_postgres"
        pg_credential = pg_api.generate_database_credential(endpoint=endpoint_name)

        return f"postgresql://{username}:{pg_credential.token}@{host}:5432/{database}?sslmode=require"

    def ensure_schema(self) -> None:
        """Create the session table and index if they do not exist."""
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
                ON session_messages(session_id)
                """
            )

    def _get_pool(self) -> ConnectionPool:
        """Get or create connection pool."""
        if self._pool is None:
            conn_string = self._get_connection_string()
            self._pool = ConnectionPool(conninfo=conn_string, min_size=1, max_size=5)
        return self._pool

    def _reset_pool(self) -> None:
        """Reset pool to force new credentials on next use."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load previous messages for a session."""
        try:
            with self._get_pool().connection() as conn:
                result = conn.execute(
                    """
                    SELECT message_data FROM session_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
                return [row[0] for row in result]
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to load session messages: {e}")
            return []

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append messages to a session."""
        try:
            with self._get_pool().connection() as conn:
                for msg in messages:
                    conn.execute(
                        "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
                        (session_id, json.dumps(msg)),
                    )
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to save session messages: {e}")
