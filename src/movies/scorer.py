"""Pure scoring and ranking functions used by MovieDatabase.search()."""

import numpy as np
import polars as pl

from movies.config import Settings

_RRF_K = Settings().RRF_K


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise an array to zero mean and unit variance."""
    return (x - x.mean()) / (x.std() + 1e-8)


def rrf(rankings: list[list[int]], n_docs: int, k: int = _RRF_K) -> np.ndarray:
    """Reciprocal Rank Fusion over multiple ranked lists of document indices.

    Parameters
    ----------
    rankings:
        Each element is a list of doc indices sorted by descending relevance.
    n_docs:
        Total number of documents in the corpus.
    k:
        RRF smoothing constant (default 60).

    Returns
    -------
    np.ndarray
        Shape ``(n_docs,)`` RRF scores — higher is better.

    """
    scores = np.zeros(n_docs, dtype=np.float64)
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return scores


def person_boost(people: list[str], df: pl.DataFrame) -> np.ndarray:
    """Return a binary boost array: 1.0 where a queried person is in cast/crew.

    Parameters
    ----------
    people:
        Lowercase full names to match against (directors, stars, writers).
    df:
        Candidate result dataframe with list-typed ``directors``, ``stars``,
        ``writers`` columns.

    """
    boost = np.zeros(len(df), dtype=np.float64)
    if not people:
        return boost

    people_set = set(people)
    for i, row in enumerate(df.iter_rows(named=True)):
        combined = " ".join(
            (row.get("directors") or [])
            + (row.get("stars") or [])
            + (row.get("writers") or [])
        ).lower()
        if any(p in combined for p in people_set):
            boost[i] = 1.0
    return boost


def genre_boost(genres: list[str], df: pl.DataFrame) -> np.ndarray:
    """Return a binary boost array: 1.0 where movie genres overlap with queried genres.

    Parameters
    ----------
    genres:
        Lowercase genre strings extracted from the query.
    df:
        Candidate result dataframe with a list-typed ``genres`` column.

    """
    boost = np.zeros(len(df), dtype=np.float64)
    if not genres:
        return boost

    query_genres = {g.lower() for g in genres}
    for i, row in enumerate(df.iter_rows(named=True)):
        movie_genres = {g.lower() for g in (row.get("genres") or [])}
        if query_genres & movie_genres:
            boost[i] = 1.0
    return boost
