"""Configuration management for Arxiv Curator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class ProjectConfig(BaseModel):
    """Project configuration model."""

    usage_policy_id: str | None = Field(None, description="Usage policy id")
    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="LLM endpoint name")
    embedding_endpoint: str = Field(..., description="Embedding endpoint name")
    warehouse_id: str = Field(..., description="Warehouse ID")
    experiment_path: str = Field(..., description="MLflow experiment path")
    lakebase_project_id: str = Field(..., description="Lakebase project ID")
    agent_app_name: str | None = Field(None, description="Databricks App name")
    system_prompt: str = Field(
        "You are a helpful assistant that answers questions about arXiv papers.",
        description="Agent system prompt",
    )

    @property
    def experiment_name(self) -> str:
        """Expose the configured MLflow experiment using the agent's name."""
        return self.experiment_path

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> ProjectConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance

        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        env_config = config_data[env]
        if "usage_policy_id" in config_data:
            env_config["usage_policy_id"] = config_data["usage_policy_id"]

        return cls(**env_config)

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"


class ModelConfig(BaseModel):
    """Model configuration."""

    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: int = Field(2000, description="Maximum tokens")
    top_p: float = Field(0.95, description="Top-p sampling parameter")


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""

    embedding_dimension: int = Field(1024, description="Embedding dimension")
    similarity_metric: str = Field("cosine", description="Similarity metric")
    num_results: int = Field(5, description="Number of results to return")


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    chunk_size: int = Field(512, description="Chunk size in tokens")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    separator: str = Field("\n\n", description="Separator for chunking")


def load_config(config_path: str = "project_config.yml", env: str = "dev") -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        ProjectConfig instance

    """
    return ProjectConfig.from_yaml(config_path, env)


def get_env(spark: SparkSession) -> str:
    """Get current environment from dbutils widget, falling back to ENV variable or 'dev'.

    Returns:
        Environment name (dev, acc, or prd)

    """
    try:
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"


def get_widget(spark: SparkSession, name: str, default: str | None = None) -> str | None:
    """Get a notebook widget or job parameter value, falling back to a default.

    Returns:
        The widget value, or the default if the widget is missing or empty

    """
    try:
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(spark)
        return dbutils.widgets.get(name) or default
    except Exception:
        return default
