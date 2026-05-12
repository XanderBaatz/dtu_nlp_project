"""MovieDatabase: build, persist and search the hybrid BM25 + embedding index."""

import pickle
import re
from pathlib import Path

import numpy as np
import polars as pl

from movies.config import Settings
from movies.documents import build_documents
from movies.embedder import Embedder, get_embedder
from movies.nlp import QueryExpander, QueryProcessor
from movies.schema import SearchResult
from movies.scorer import genre_boost, person_boost, rrf, zscore
from tools.logger import Logger, LogType

settings = Settings()
logger = Logger(name=__name__, log_type=LogType.LOCAL)

PROCESSED_PATH = settings.processed_path
INDEX_DIR = settings.index_dir
BM25_PATH = INDEX_DIR / "bm25.pkl"
EMBED_PATH = INDEX_DIR / "embeddings.npy"
INDEX_IDS_PATH = settings.index_ids_path
USER_ADDED = settings.user_added


class MovieDatabase:
    """Local movie database with BM25 (sparse) and embedding (dense) search.

    Typical usage
    -------------
    >>> db = MovieDatabase()
    >>> db.load_or_build()
    >>> results = db.search("melancholic space movie")
    """

    def __init__(
        self,
        processed_path: Path = PROCESSED_PATH,
        index_dir: Path = INDEX_DIR,
    ) -> None:
        self._processed_path: Path = processed_path
        self._index_dir: Path = index_dir
        self._nlp = QueryProcessor()
        self._expander = QueryExpander()
        self._embedder: Embedder = get_embedder(settings)

        self._df: pl.DataFrame | None = None
        self._embeddings: np.ndarray | None = None
        self._bm25 = None

        # Person-name reverse index (built after loading/building)
        self._person_row_map: dict[str, list[int]] = {}
        self._person_names: list[str] = []
        self._person_name_re: re.Pattern | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the BM25 + embedding index from the processed parquet and save it."""
        if not self._processed_path.exists():
            msg = (
                f"Processed parquet not found at {self._processed_path}. "
                "Run MovieDownloader().run() first."
            )
            raise FileNotFoundError(msg)

        self._df = self._load_base_dataframe()
        logger.info("Building index for %d movies...", len(self._df))

        corpus = build_documents(self._df).get_column("document").to_list()
        self._build_bm25(corpus)
        self._embeddings = self._embedder.embed_documents(corpus)
        self._build_person_index()
        self._save()

    def load(self) -> None:
        """Load a previously built index from disk."""
        missing = [p for p in (BM25_PATH, EMBED_PATH, INDEX_IDS_PATH) if not p.exists()]
        if missing:
            msg = f"Index files missing: {missing}"
            raise FileNotFoundError(msg)

        self._embeddings = np.load(EMBED_PATH)
        index_ids = np.load(INDEX_IDS_PATH, allow_pickle=True).tolist()

        with Path.open(BM25_PATH, "rb") as fh:
            self._bm25 = pickle.load(fh)  # noqa: S301

        df = self._load_base_dataframe()
        order_df = pl.DataFrame({"id": index_ids, "__order": np.arange(len(index_ids))})
        self._df = (
            df.join(order_df, on="id", how="inner").sort("__order").drop("__order")
        )
        self._build_person_index()
        logger.info("Index loaded (%d movies)", len(self._df))

    def load_or_build(self) -> None:
        """Load existing index or build it if not present."""
        try:
            self.load()
        except FileNotFoundError:
            self.build()

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        weighted_rating_alpha: float = 0.10,
        popularity_alpha: float = 0.05,
        expand: bool = True,
    ) -> list[SearchResult]:
        """Hybrid BM25 + semantic search with optional LLM query expansion.

        Parameters
        ----------
        query:
            Free-text user query.
        top_k:
            Number of results to return.
        alpha:
            Dense/sparse balance — 0 = pure BM25, 1 = pure semantic.
        weighted_rating_alpha:
            Weight given to IMDb weighted rating in the final score (0-1).
        popularity_alpha:
            Weight given to CPI-adjusted gross revenue (0-1).
        expand:
            Whether to use LLM query expansion (requires API key in settings).

        """
        self._assert_loaded()

        parsed = self._nlp.process(query)
        known_persons = self._extract_known_persons(query)
        all_people = list({p.lower() for p in parsed.people} | set(known_persons))

        expansion_tokens = self._expander.expand(query) if expand else []
        bm25_tokens = list(dict.fromkeys(parsed.tokens + expansion_tokens))

        n_docs = len(self._df)
        bm25_ranking = self._bm25_search(bm25_tokens)
        dense_ranking = self._dense_search(query)

        bm25_copies = max(1, round((1 - alpha) * 10))
        dense_copies = max(1, round(alpha * 10))
        rrf_scores = rrf(
            rankings=bm25_ranking * bm25_copies + dense_ranking * dense_copies,
            n_docs=n_docs,
        )

        # Build candidate pool: top RRF + person matches + expansion BM25 hits
        candidate_k = max(top_k * 5, 50)
        candidate_arr = np.argsort(rrf_scores)[::-1][:candidate_k]

        person_indices = self._person_candidate_indices(all_people)
        if person_indices:
            candidate_arr = np.union1d(candidate_arr, person_indices)

        if expansion_tokens:
            expansion_top = np.array(
                self._bm25_search(expansion_tokens)[0][:candidate_k]
            )
            candidate_arr = np.union1d(candidate_arr, expansion_top)

        candidates = candidate_arr.tolist()

        result_df = (
            self._df.with_row_index("idx")
            .filter(pl.col("idx").is_in(candidates))
            .with_columns(
                pl.col("idx")
                .replace(
                    pl.Series(candidates, dtype=pl.UInt32),
                    pl.Series(rrf_scores[candidate_arr].tolist()),
                )
                .alias("retrieval_score")
            )
        )

        # Final scoring: retrieval + rating + popularity (adj. gross) + hard boosts
        retrieval = zscore(result_df["retrieval_score"].to_numpy())
        rating = zscore(result_df["weighted_rating"].fill_null(0).to_numpy())
        p_boost = person_boost(all_people, result_df)
        g_boost = genre_boost(parsed.genres, result_df)

        # Popularity: CPI-adjusted gross — null → 0 so missing films aren't penalised
        # by the z-score (they land at the mean after fill_null with the median).
        gross_series = (
            result_df["adjusted_gross"]
            if "adjusted_gross" in result_df.columns
            else None
        )
        if gross_series is not None and gross_series.drop_nulls().len() > 0:
            gross_median = gross_series.median()
            popularity = zscore(gross_series.fill_null(gross_median).to_numpy())
        else:
            popularity = np.zeros(len(result_df))

        retrieval_w = 1.0 - weighted_rating_alpha - popularity_alpha
        final_scores = (
            retrieval_w * retrieval
            + weighted_rating_alpha * rating
            + popularity_alpha * popularity
            + 2.0 * p_boost
            + 1.0 * g_boost
        )

        result_df = (
            result_df.with_columns(pl.Series("final_score", final_scores))
            .sort("final_score", descending=True)
            .head(top_k)
        )

        return [
            SearchResult(
                title=row["title"],
                release_date=row["release_date"].year,
                duration=row["duration"],
                rating=row["rating"],
                genre=", ".join(row["genres"] or []),
                description=row["description"],
                directors=", ".join(row["directors"] or []),
                stars=", ".join(row["stars"] or []),
                imdb_id=row["id"],
                retrieval_score=round(row["final_score"], 4),
            )
            for row in result_df.iter_rows(named=True)
        ]

    def by_id(self, imdb_id: str) -> SearchResult | None:
        """Return a movie by its IMDb ID, or None if not found."""
        self._assert_loaded()
        row_df = self._df.filter(pl.col("id") == imdb_id)
        if len(row_df) == 0:
            return None
        row = row_df.row(0, named=True)
        return SearchResult(
            title=row["title"],
            release_date=row["release_date"].year,
            duration=row["duration"],
            rating=row["rating"],
            genre=", ".join(row["genres"] or []),
            description=row["description"],
            directors=", ".join(row["directors"] or []),
            stars=", ".join(row["stars"] or []),
            imdb_id=row["id"],
            retrieval_score=0.0,
        )

    def lookup(self, title: str, year: int | None = None) -> SearchResult | None:
        """Find a movie by title (and optionally year), returning the closest match."""
        self._assert_loaded()
        title_lower = title.lower()
        matches = self._df.filter(pl.col("title").str.to_lowercase() == title_lower)
        if year is not None:
            year_matches = matches.filter(pl.col("release_date").dt.year() == year)
            if len(year_matches) > 0:
                matches = year_matches
        if len(matches) == 0:
            return None
        row = matches.row(0, named=True)
        return SearchResult(
            title=row["title"],
            release_date=row["release_date"].year,
            duration=row["duration"],
            rating=row["rating"],
            genre=", ".join(row["genres"] or []),
            description=row["description"],
            directors=", ".join(row["directors"] or []),
            stars=", ".join(row["stars"] or []),
            imdb_id=row["id"],
            retrieval_score=0.0,
        )

    def similar(
        self,
        imdb_id: str,
        top_k: int = 10,
        weighted_rating_alpha: float = 0.10,
        popularity_alpha: float = 0.05,
    ) -> list[SearchResult]:
        """Return top-k movies similar to the given movie, re-ranked with rating
        and CPI-adjusted popularity signals (same blend as search).
        """
        self._assert_loaded()
        if self._embeddings is None:
            return []

        idx_df = self._df.with_row_index("idx").filter(pl.col("id") == imdb_id)
        if len(idx_df) == 0:
            return []
        idx: int = idx_df["idx"][0]

        q_emb = self._embeddings[idx]
        similarities = self._embeddings @ q_emb
        similarities[idx] = -1.0  # exclude the query movie itself

        # Fetch a larger candidate pool, then re-rank
        candidate_k = max(top_k * 5, 50)
        top_indices = np.argsort(similarities)[::-1][:candidate_k].tolist()

        result_df = self._df.with_row_index("idx").filter(
            pl.col("idx").is_in(top_indices)
        )

        sim_scores = np.array([similarities[i] for i in result_df["idx"].to_list()])
        retrieval = zscore(sim_scores)
        rating = zscore(result_df["weighted_rating"].fill_null(0).to_numpy())

        gross_series = (
            result_df["adjusted_gross"]
            if "adjusted_gross" in result_df.columns
            else None
        )
        if gross_series is not None and gross_series.drop_nulls().len() > 0:
            gross_median = gross_series.median()
            popularity = zscore(gross_series.fill_null(gross_median).to_numpy())
        else:
            popularity = np.zeros(len(result_df))

        retrieval_w = 1.0 - weighted_rating_alpha - popularity_alpha
        final_scores = (
            retrieval_w * retrieval
            + weighted_rating_alpha * rating
            + popularity_alpha * popularity
        )

        result_df = (
            result_df.with_columns(pl.Series("final_score", final_scores))
            .sort("final_score", descending=True)
            .head(top_k)
        )

        return [
            SearchResult(
                title=row["title"],
                release_date=row["release_date"].year,
                duration=row["duration"],
                rating=row["rating"],
                genre=", ".join(row["genres"] or []),
                description=row["description"],
                directors=", ".join(row["directors"] or []),
                stars=", ".join(row["stars"] or []),
                imdb_id=row["id"],
                retrieval_score=round(row["final_score"], 4),
            )
            for row in result_df.iter_rows(named=True)
        ]

    @property
    def size(self) -> int:
        """Number of movies currently indexed."""
        return len(self._df) if self._df is not None else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_base_dataframe(self) -> pl.DataFrame:
        base_df = pl.read_parquet(self._processed_path)
        if USER_ADDED.exists():
            user_df = pl.read_parquet(USER_ADDED)
            logger.info("Loaded %d user-added movies", len(user_df))
            base_df = pl.concat([base_df, user_df], how="vertical_relaxed")
        return base_df.unique(subset=["id"], keep="last")

    def _build_bm25(self, doc_texts: list[str]) -> None:
        import bm25s

        tokenized = self._nlp.tokenize_batch(doc_texts)
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokenized)

    def _save(self) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        with Path.open(BM25_PATH, "wb") as fh:
            pickle.dump(self._bm25, fh)
        np.save(str(EMBED_PATH), self._embeddings)
        np.save(INDEX_IDS_PATH, self._df.get_column("id").to_numpy())
        logger.info("Index saved to %s", self._index_dir)

    def _bm25_search(self, tokens: list[str]) -> list[list[int]]:
        scores = self._bm25.get_scores(tokens)
        return [np.argsort(scores)[::-1].tolist()]

    def _dense_search(self, query: str) -> list[list[int]]:
        if self._embeddings is None or not np.any(self._embeddings):
            return []
        q_emb = self._embedder.embed_query(query)
        similarities = self._embeddings @ q_emb
        return [np.argsort(similarities)[::-1].tolist()]

    def _build_person_index(self) -> None:
        """Build a word-boundary regex and reverse map: name -> row indices."""
        people_cols = ["directors", "stars", "writers"]
        name_to_rows: dict[str, list[int]] = {}

        for col in people_cols:
            if col not in self._df.columns:
                continue
            for row_idx, names in enumerate(self._df[col].to_list()):
                for name in names or []:
                    key = name.lower().strip()
                    if key:
                        name_to_rows.setdefault(key, []).append(row_idx)

        # Only keep multi-word names to avoid single-token noise
        self._person_row_map = {
            k: v for k, v in name_to_rows.items() if len(k.split()) >= 2
        }
        self._person_names = sorted(self._person_row_map, key=len, reverse=True)
        self._person_name_re = (
            re.compile(
                "|".join(r"\b" + re.escape(n) + r"\b" for n in self._person_names)
            )
            if self._person_names
            else None
        )

    def _extract_known_persons(self, query: str) -> list[str]:
        """Find database-known person names that appear verbatim in the query."""
        if not self._person_name_re:
            return []
        return list({m.group() for m in self._person_name_re.finditer(query.lower())})

    def _person_candidate_indices(self, people: list[str]) -> list[int]:
        """Return all row indices of movies featuring any of the given people."""
        indices: set[int] = set()
        for person in people:
            indices.update(self._person_row_map.get(person, []))
        return list(indices)

    def _assert_loaded(self) -> None:
        if self._df is None or self._bm25 is None:
            msg = (
                "Index not loaded. "
                "Call MovieDatabase().load() or .load_or_build() first."
            )
            raise RuntimeError(msg)
