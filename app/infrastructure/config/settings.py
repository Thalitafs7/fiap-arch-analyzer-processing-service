from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic (opcional — mantido por compatibilidade)
    anthropic_api_key: str = ""

    # OpenAI
    openai_api_key: str

    # PostgreSQL
    postgres_user: str = "hackathon"
    postgres_password: str = "hackathon123"
    postgres_db: str = "hackathon_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_connection_string: str = (
        "postgresql+psycopg://hackathon:hackathon123@localhost:5432/hackathon_db"
    )

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "hackathon-diagrams"

    # SQS
    sqs_queue_url: str = ""

    # SOAT — sistema externo que origina as análises
    soat_base_url: str = ""

    # Report Agent — backend de geração de relatório
    # Valores: "langchain" | "finetuned_api" | "finetuned_local"
    report_model_backend: str = "langchain"

    # LLM — modelo e endpoint configuráveis (suporta OpenAI, Groq, etc.)
    llm_model: str = "gpt-4o"
    llm_base_url: str = ""          # vazio = OpenAI padrão; preenchido = Groq/outro
    llm_vision_model: str = ""      # modelo para Vision; se vazio usa llm_model

    # Fine-tuning — HuggingFace Inference API
    huggingface_api_token: str = ""
    huggingface_endpoint_url: str = ""

    # Fine-tuning — modelo local (requer GPU)
    local_model_path: str = ""
    base_model_id: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # RabbitMQ
    rabbitmq_url: str = "amqp://hackathon:hackathon123@rabbitmq:5672/"
    rabbitmq_exchange: str = "reports.events"
    outbox_poll_interval_seconds: int = 2

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def sqlalchemy_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
