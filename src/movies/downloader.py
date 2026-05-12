# Downloader to download and clean dataset
from pathlib import Path  # noqa: TC003

import kagglehub
import polars as pl

from movies.config import Settings
from movies.preprocess import (
    add_adjusted_gross,
    add_weighted_rating,
    clean_numeric_columns,
    convert_duration_to_minutes,
    convert_list_like_columns,
)
from tools.logger import Logger, LogType

# Initialize setting and logger
settings = Settings()
logger = Logger(
    name=__name__,
    log_type=LogType.LOCAL,
)

DATA_ROOT = settings.data_root
RAW_DIR = settings.raw_dir
PROCESSED_DIR = settings.processed_dir
PROCESSED_PATH = settings.processed_path


def _load_kaggle_dataset() -> pl.DataFrame:
    """Download IMDb dataset with top 500-600 movies from each year."""
    path = kagglehub.dataset_download(
        handle="raedaddala/top-500-600-movies-of-each-year-from-1960-to-2024",
        path="final_dataset.parquet",
        output_dir=RAW_DIR,
    )

    return pl.read_parquet(path)


def process_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Process dataset (dataframe)."""
    df = df.clone()

    # Remove unused columns
    cols_to_drop = [c for c in ["movie_link", "awards_content"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    # Remove rows for which have NaN for specific columns
    df = df.drop_nulls(subset=["title", "genres"])

    # Convert list-like columns
    df = convert_list_like_columns(
        df=df,
        columns=[
            "genres",
            "languages",
            "countries_origin",
            "production_companies",
            "stars",
            "directors",
            "writers",
            "filming_locations",
        ],
    )

    # Change duration from str to int
    df = convert_duration_to_minutes(df, column="duration")

    # Clean numeric columns
    df = clean_numeric_columns(
        df=df,
        columns=[
            "votes",
            "budget",
            "opening_weekend_gross",
            "gross_worldwide",
            "gross_us_canada",
        ],
    )

    df = add_weighted_rating(df=df)
    df = add_adjusted_gross(df=df, cpi_path=settings.cpi_path)

    return df


class MovieDownloader:
    """Download, process and save movie dataset.

    Usage
    -----
    >>> downloader = MovieDownloader()
    >>> df = downloader.run()          # download + merge + save
    >>> df = downloader.load()         # load existing processed CSV
    """

    def __init__(
        self,
        processed_path: Path = PROCESSED_PATH,
    ) -> None:
        """Initialize processed path."""
        self._path = processed_path

    def __call__(self) -> pl.DataFrame:
        """If called without method, run normally."""
        return self.run()

    def run(
        self,
        force: bool = False,  # noqa: FBT001, FBT002
    ) -> pl.DataFrame:
        """Download and process source, then return a dataframe."""
        if self._path.exists() and not force:
            logger.info(
                "Processed dataset already exists at %s - skipping download.",
                self._path,
            )

            return self.load()

        # Download dataset from Kaggle
        df = _load_kaggle_dataset()

        # Process dataset
        df = process_dataframe(df=df)

        # Save processed dataset (save as parquet to save disk space)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self._path)
        logger.info("Saved processed dataset -> %s (%d movies)", self._path, len(df))

        return df

    def load(self) -> pl.DataFrame:
        """Load the processed dataset from disk."""
        if not self._path.exists():
            msg = (
                f"Processed dataset not found at {self._path}. "
                "Run MovieDownloader().run() first."
            )
            raise FileNotFoundError(msg)

        return pl.read_parquet(self._path)


if __name__ == "__main__":
    print(f"Data root: {DATA_ROOT}")  # noqa: T201
    downloader = MovieDownloader()
    df = downloader.run()
    print(df.head())  # noqa: T201
