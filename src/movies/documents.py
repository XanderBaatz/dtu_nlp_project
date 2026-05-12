import polars as pl

from movies.config import Settings

settings = Settings()


def build_documents(df: pl.DataFrame) -> pl.DataFrame:
    """Build a flat text document for each movie, used by BM25 and embedding indexes.

    The document format is:
        Title: <title> | Desc: <description> | Genres: <genres> | Cast: <stars>
        | Directors: <directors> | Writers: <writers>
    """
    df = df.with_columns(
        pl.col("writers")
        .fill_null([])
        .list.slice(0, settings.MAX_WRITERS)
        .list.join(", ")
        .alias("writers_str"),
        pl.col("directors")
        .fill_null([])
        .list.slice(0, settings.MAX_DIRECTORS)
        .list.join(", ")
        .alias("directors_str"),
        pl.col("stars")
        .fill_null([])
        .list.slice(0, settings.MAX_STARS)
        .list.join(", ")
        .alias("stars_str"),
        pl.col("genres").fill_null([]).list.join(", ").alias("genres_str"),
    )

    return df.with_columns(
        pl.concat_str(
            [
                pl.lit("Title: "),
                pl.col("title").fill_null(""),
                pl.lit(" | Desc: "),
                pl.col("description")
                .fill_null("")
                .str.slice(0, settings.MAX_DESC_LENGTH),
                pl.lit(" | Genres: "),
                pl.col("genres_str"),
                pl.lit(" | Cast: "),
                pl.col("stars_str"),
                pl.lit(" | Directors: "),
                pl.col("directors_str"),
                pl.lit(" | Writers: "),
                pl.col("writers_str"),
            ],
            separator="",
        ).alias("document")
    )
