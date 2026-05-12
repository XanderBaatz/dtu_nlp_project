import re
from dataclasses import dataclass, field
from functools import lru_cache

import polars as pl
import spacy

from movies.config import Settings
from tools.logger import Logger, LogType

_EXPAND_SYSTEM = (
    "You are a movie search assistant. "
    "The user will give you a vague or descriptive movie query. "
    "Your task: output 5-10 keywords that would help a keyword search engine "
    "find the right movie(s) - include likely movie titles, character names, "
    "franchise names, director names, and genre terms. "
    "Output ONLY the keywords separated by spaces, no punctuation, no explanation."
)

# Initialize setting and logger
settings = Settings()
logger = Logger(
    name=__name__,
    log_type=LogType.LOCAL,
)

SPACY_MODEL = settings.SPACY_MODEL

_df = pl.read_parquet(settings.processed_path)
_KNOWN_GENRES = set(_df.get_column("genres").explode().drop_nulls().str.to_lowercase())

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@lru_cache(maxsize=1)
def _get_nlp() -> spacy.Language | None:
    """Load the spaCy model (cached after the first call)."""
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            SPACY_MODEL,
            SPACY_MODEL,
        )
        raise


@dataclass
class ParsedQuery:
    """Structured representation of a user query after NLP processing."""

    raw: str
    """Original query text."""

    tokens: list[str] = field(default_factory=list)
    """Lemmatised, lowercased, stop-word-free tokens for BM25."""

    people: list[str] = field(default_factory=list)
    """Person names extracted by NER (for cast/crew matching)."""

    genres: list[str] = field(default_factory=list)
    """Genre keywords found in the query."""

    years: list[int] = field(default_factory=list)
    """Four-digit years mentioned in the query."""

    @property
    def bm25_query(self) -> str:
        """Space-joined token string ready for BM25 retrieval."""
        return " ".join(self.tokens)


class QueryProcessor:
    """Preprocess natural-language movie queries with spaCy."""

    def __init__(self) -> None:
        pass

    def process(self, query: str) -> ParsedQuery:
        """Run the full NLP pipeline on query."""
        nlp = _get_nlp()
        doc = nlp(query)

        people = [
            ent.text.strip() for ent in doc.ents if ent.label_ in {"PERSON", "ORG"}
        ]

        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        genres = list(_KNOWN_GENRES & query_tokens)

        years = [int(m.group()) for m in _YEAR_RE.finditer(query)]

        tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]

        tokens = [t.lower() for t in tokens]

        return ParsedQuery(
            raw=query,
            tokens=tokens,
            people=people,
            genres=genres,
            years=years,
        )

    def tokenize(self, text: str) -> list[str]:
        nlp = _get_nlp()
        doc = nlp(text.lower())

        return [
            t.lemma_
            for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space and len(t) > 1
        ]

    def tokenize_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
    ) -> list[list[str]]:
        nlp = _get_nlp()

        results: list[list[str]] = []

        for doc in nlp.pipe(
            texts,
            batch_size=batch_size,
            disable=["ner", "parser"],
        ):
            tokens = [
                t.lemma_.lower()
                for t in doc
                if not t.is_stop and not t.is_punct and not t.is_space and len(t) > 1
            ]
            results.append(tokens)

        return results


class QueryExpander:
    """Expand a vague query into concrete movie keywords using an LLM.

    Falls back to the original query tokens when no API key is configured.
    """

    def __init__(self) -> None:
        self._settings = Settings()

    def expand(self, query: str) -> list[str]:
        """Return a list of expansion keywords for *query*.

        On any error (missing key, network failure) the original query is
        returned unchanged so the rest of the pipeline is unaffected.
        """
        api_key = self._settings.CAMPUSAI_API_KEY or self._settings.OPENAI_API_KEY
        if not api_key:
            logger.debug("QueryExpander: no API key configured, skipping expansion")
            return []

        try:
            from openai import OpenAI

            base_url = (
                self._settings.CAMPUSAI_API_URL
                if self._settings.CAMPUSAI_API_KEY
                else None
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
            model = (
                self._settings.CAMPUSAI_MODEL
                if self._settings.CAMPUSAI_API_KEY
                else "gpt-4o-mini"
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _EXPAND_SYSTEM},
                    {"role": "user", "content": query},
                ],
                max_tokens=80,
                temperature=0.2,
            )
            raw = response.choices[0].message.content or ""
            keywords = raw.lower().split()
            logger.debug("QueryExpander: %r → %r", query, keywords)

        except (
            OSError,
            ValueError,
            RuntimeError,
        ):
            logger.debug("QueryExpander: expansion failed, continuing without it")
            return []

        else:
            return keywords

        return []
