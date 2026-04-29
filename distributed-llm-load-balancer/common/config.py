# common/config.py
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:1b")

    # System
    num_workers: int = Field(default=4)
    num_users: int = Field(default=100)
    lb_strategy: str = Field(default="round_robin")

    # Fault tolerance
    worker_failure_simulation: bool = Field(default=True)
    failure_after_n_requests: int = Field(default=50)
    heartbeat_interval: float = Field(default=2.0)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global singleton
config = Settings()
