from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class MLPipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # Reproducibility
    RANDOM_SEED: int = 42

    # Database & Tracking
    DATABASE_URL: str = (
        "postgresql+psycopg2://promosense_user:promosense_pass@localhost:5432/promosense"
    )
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "promosense_causal_demand"

    # Methodological Defaults
    DEFAULT_ASSUMED_MARGIN_PCT: float = 0.30
    DEFAULT_CAMPAIGN_COST_PER_HH: float = 0.50


ml_settings = MLPipelineSettings()
