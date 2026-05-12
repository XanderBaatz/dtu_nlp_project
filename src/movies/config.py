from pathlib import Path

from tools.config.settings import Settings as BaseSettings


class Settings(BaseSettings):
    """Environment variables settings.

    Examples:
        >>> from tools.config import Settings
        >>> from tools.logger import Logger, LogType
        >>>
        >>> settings = Settings()
        >>> logger = Logger(
        >>>     name=__name__,
        >>>     log_type=LogType.LOCAL if settings.IS_LOCAL else LogType.GOOGLE_CLOUD
        >>> )

    """

    LOCAL_EMBEDDING: bool = True
    LOCAL_EMBED_MODEL: str = "all-MiniLM-L6-v2"
    RRF_K: int = 60
    SPACY_MODEL: str = "en_core_web_sm"
    MINIMUM_VOTES: int = 1000

    # CampusAI / OpenAI-compatible LLM & embedding API
    CAMPUSAI_API_KEY: str = ""
    """CampusAI API key. Must be set in .env.local."""

    CAMPUSAI_API_URL: str = "https://api.campusai.compute.dtu.dk/v1/"
    CAMPUSAI_MODEL: str = "gpt-4o-mini"
    CAMPUSAI_EMBED_MODEL: str = "nomic-embed-text"
    BATCH_SIZE: int = 128

    # OpenAI API key
    OPENAI_API_KEY: str = ""

    # Google
    GOOGLE_API_KEY: str = ""

    # Kaggle
    KAGGLE_API_TOKEN: str = ""

    # Token limit
    MAX_WRITERS: int = 2
    MAX_DIRECTORS: int = 2
    MAX_STARS: int = 10
    MAX_DESC_LENGTH: int = 1000

    # Paths
    DATA_ROOT: Path | None = None
    DATA_NAME: str = "imdb_movies"

    @property
    def base_dir(self) -> Path:
        """Project root."""
        return Path(__file__).resolve().parents[2]

    @property
    def data_root(self) -> Path:
        """Root data directory."""
        return self.DATA_ROOT or self.base_dir / "data"

    @property
    def raw_dir(self) -> Path:
        """Raw data directory."""
        return self.data_root / self.DATA_NAME / "raw"

    @property
    def processed_dir(self) -> Path:
        """Processed data directory."""
        return self.data_root / self.DATA_NAME / "processed"

    @property
    def processed_path(self) -> Path:
        """Processed data file."""
        return self.processed_dir / "movies.parquet"

    @property
    def index_dir(self) -> Path:
        """Index directory for embeddings etc."""
        return self.data_root / self.DATA_NAME / "index"

    @property
    def index_ids_path(self) -> Path:
        """Index IDs."""
        return self.index_dir / "index_ids.npy"

    @property
    def user_added(self) -> Path:
        """User added movies (movies that were not already part of the dataset)."""
        return self.data_root / "user" / "added_movies.parquet"

    @property
    def cpi_path(self) -> Path:
        """Seasonally adjusted CPI (CPIAUCSL) CSV from FRED."""
        return self.data_root / "cpi" / "CPIAUCSL.csv"
