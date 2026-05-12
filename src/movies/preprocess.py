import ast
import re
from pathlib import Path  # noqa: TC003

import polars as pl


def convert_list_like_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    def safe_parse_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (
                ValueError,
                SyntaxError,
            ):
                return []
        return []

    for col in columns:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).map_elements(
                    safe_parse_list, return_dtype=pl.List(pl.String)
                )
            )

    return df


def convert_duration_to_minutes(
    df: pl.DataFrame, column: str = "duration"
) -> pl.DataFrame:
    def parse_duration(x):
        h = re.search(r"(\d+)\s*h", str(x))
        m = re.search(r"(\d+)\s*m", str(x))

        hours = int(h.group(1)) if h else 0
        minutes = int(m.group(1)) if m else 0

        return hours * 60 + minutes

    return df.with_columns(
        pl.col(column).map_elements(parse_duration, return_dtype=pl.Int64)
    )


def clean_numeric_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    def clean(x):
        if isinstance(x, (int, float)):
            return float(x)

        x = str(x).replace(",", "").replace("$", "").strip()

        # Handle K/M shorthand
        if x.endswith("K"):
            return float(x[:-1]) * 1_000
        if x.endswith("M"):
            return float(x[:-1]) * 1_000_000

        try:
            return float(x)
        except ValueError:
            return None

    for col in columns:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).map_elements(clean, return_dtype=pl.Float64)
            )

    return df


def add_adjusted_gross(
    df: pl.DataFrame,
    cpi_path: Path,
    gross_col: str = "gross_us_canada",
) -> pl.DataFrame:
    """Add CPI-adjusted gross revenue column.

    Computes: adjusted_gross = gross * (CPI_latest / CPI_release_year)

    Movies without gross data get null (not 0), so they are not penalised
    in the scoring blend.

    Parameters
    ----------
    df:
        Movie dataframe with ``release_date`` (Date) and a gross column.
    cpi_path:
        Path to CPIAUCSL monthly CSV from FRED (columns: observation_date, CPIAUCSL).
    gross_col:
        Which gross column to adjust (default: gross_worldwide).

    """
    if not cpi_path.exists():
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("adjusted_gross"))

    cpi = pl.read_csv(cpi_path, try_parse_dates=True)

    # Build a year → mean annual CPI lookup
    year_cpi = (
        cpi.with_columns(pl.col("observation_date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("CPIAUCSL").mean().alias("cpi"))
        .sort("year")
    )

    # Latest available CPI value
    cpi_latest: float = cpi["CPIAUCSL"].tail(1).item()

    # Join on release year
    df_with_year = df.with_columns(
        pl.col("release_date").dt.year().alias("__release_year")
    )
    df_with_year = df_with_year.join(
        year_cpi.rename({"year": "__release_year", "cpi": "__cpi_release"}),
        on="__release_year",
        how="left",
    )

    df_with_year = df_with_year.with_columns(
        (pl.col(gross_col) * (cpi_latest / pl.col("__cpi_release"))).alias(
            "adjusted_gross"
        )
    ).drop(["__release_year", "__cpi_release"])

    return df_with_year


def add_weighted_rating(
    df: pl.DataFrame,
    min_votes_quantile: float = 0.90,
) -> pl.DataFrame:
    """Add IMDb-style weighted rating column.

    WR = (v / (v + m)) * R + (m / (v + m)) * C
    """
    # Global mean rating
    C = df.select(pl.col("rating").mean()).item()

    # Minimum votes required
    m = df.select(pl.col("votes").quantile(min_votes_quantile)).item()

    return df.with_columns(
        (
            (pl.col("votes") / (pl.col("votes") + m)) * pl.col("rating")
            + (m / (pl.col("votes") + m)) * C
        ).alias("weighted_rating")
    )
