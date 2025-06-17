"""Configuration Settings."""

from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Project Information
    PROJECT_NAME: str = "StackAI Takehome assignment"
    PROJECT_DESCRIPTION: str = "StackAI Takehome assignment - coding a vector search engine from scratch using FastAPI and Pydantic."
    VERSION: str = "0.1.0"
    API_ROUTER: str = "/api"
    API_VERSION: str = "v1"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # Security
    SECRET_KEY: str | None = "your-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    # Trusted Hosts
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1", "testserver"]

    # Database Configuration
    DATABASE_URL: str | None = None
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_USER: str = "postgres"
    DATABASE_PASSWORD: str = "postgres"
    DATABASE_NAME: str = "stackai"

    # Vector Search Configuration
    VECTOR_DIMENSION: int = 1536
    VECTOR_INDEX_TYPE: str = "lsh"  # "bruteforce" or "lsh"
    VECTOR_DISTANCE_METRIC: str = "cosine"  # "cosine" or "euclidean"

    # LSH specific settings (if VECTOR_INDEX_TYPE is 'lsh')
    LSH_NUM_TABLES: int = 10
    LSH_NUM_HYPERPLANES: int = 10

    # Cohere Configuration
    COHERE_API_KEY: str | None = "nGccwptQNrRyeoLc6Hs0NnLtcdU2zQ7drJvjRdza"
    COHERE_MODEL: str = "embed-v4.0"
    COHERE_INPUT_TYPE: str = "search_query"

    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @model_validator(mode="after")
    def assemble_db_connection(self) -> Settings:
        """Assemble database URL from individual components if not provided."""
        if self.DATABASE_URL is None:
            self.DATABASE_URL = (
                f"postgresql://{self.DATABASE_USER}:"
                f"{self.DATABASE_PASSWORD}@"
                f"{self.DATABASE_HOST}:"
                f"{self.DATABASE_PORT}/"
                f"{self.DATABASE_NAME}"
            )
        return self

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production", "testing"]
        if v.lower() not in allowed_environments:
            msg = f"Environment must be one of: {allowed_environments}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("VECTOR_INDEX_TYPE")
    @classmethod
    def validate_vector_index_type(cls, v: str) -> str:
        """Validate vector index type setting."""
        allowed_types = ["bruteforce", "lsh"]
        if v.lower() not in allowed_types:
            msg = f"VECTOR_INDEX_TYPE must be one of: {allowed_types}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("COHERE_API_KEY")
    @classmethod
    def validate_cohere_api_key(cls, v: str | None) -> str:
        """Validate that Cohere API key is provided."""
        if v is None or v.strip() == "":
            msg = (
                "COHERE_API_KEY must be provided in environment variables or .env file"
            )
            raise ValueError(msg)
        return v.strip()

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str | None) -> str:
        """Validate that Secret key is provided."""
        if v is None or v.strip() == "":
            msg = "SECRET_KEY must be provided in environment variables or .env file"
            raise ValueError(msg)
        return v.strip()

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == "testing"


settings = Settings()
