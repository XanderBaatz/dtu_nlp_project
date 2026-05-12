"""LLM-powered RAG modules using DSPy (CampusAI / OpenAI-compatible backend)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from movies.config import Settings
from tools.logger import Logger, LogType

if TYPE_CHECKING:
    from movies.schema import SearchResult

settings = Settings()
logger = Logger(name=__name__, log_type=LogType.LOCAL)


# ---------------------------------------------------------------------------
# DSPy LM configuration
# ---------------------------------------------------------------------------


def _get_lm() -> dspy.LM:
    """Return a configured DSPy LM (CampusAI or OpenAI fallback)."""
    if settings.CAMPUSAI_API_KEY:
        return dspy.LM(
            model=f"openai/{settings.CAMPUSAI_MODEL}",
            api_key=settings.CAMPUSAI_API_KEY,
            api_base=settings.CAMPUSAI_API_URL,
        )
    if settings.OPENAI_API_KEY:
        return dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
        )
    msg = "No LLM API key configured. Set CAMPUSAI_API_KEY or OPENAI_API_KEY."
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------


class ExplainRecommendations(dspy.Signature):
    """Given a user query and a list of movie recommendations, write a concise
    explanation (2-4 sentences) of why these movies match the query.
    """

    query: str = dspy.InputField(desc="The user's search query")
    movies: str = dspy.InputField(desc="Comma-separated list of recommended titles")
    explanation: str = dspy.OutputField(desc="Why these movies match the query")


class GuessMostLikelyMovie(dspy.Signature):
    """Given a plot/description and a shortlist of candidate movies, identify
    the single most likely match and explain your reasoning.
    """

    description: str = dspy.InputField(desc="User's plot description or hints")
    candidates: str = dspy.InputField(
        desc="Numbered list of candidate movies with descriptions"
    )
    title: str = dspy.OutputField(desc="Title of the most likely matching movie")
    imdb_id: str = dspy.OutputField(desc="IMDb ID of the most likely match")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why this matches")


# ---------------------------------------------------------------------------
# DSPy modules
# ---------------------------------------------------------------------------


class Recommender(dspy.Module):
    """Generate a natural-language explanation for a set of recommendations."""

    def __init__(self) -> None:
        self.explain = dspy.ChainOfThought(ExplainRecommendations)

    def forward(self, query: str, movies: str) -> dspy.Prediction:
        return self.explain(query=query, movies=movies)


class MovieGuesser(dspy.Module):
    """Guess the most likely movie from a description and a candidate list."""

    def __init__(self) -> None:
        self.guess = dspy.ChainOfThought(GuessMostLikelyMovie)

    def forward(self, description: str, candidates: str) -> dspy.Prediction:
        return self.guess(description=description, candidates=candidates)


# ---------------------------------------------------------------------------
# Public helper functions (called from main.py)
# ---------------------------------------------------------------------------


def explain_recommendations(query: str, results: list[SearchResult]) -> str:
    """Return an LLM-generated explanation for why the results match the query.

    Falls back to an empty string if no LLM key is configured.
    """
    try:
        lm = _get_lm()
    except RuntimeError:
        logger.warning("No LLM key configured; skipping recommendation explanation.")
        return ""

    with dspy.context(lm=lm):
        movie_titles = ", ".join(r.title for r in results)
        module = Recommender()
        pred = module(query=query, movies=movie_titles)
        return pred.explanation


def guess_movie_from_description(
    description: str,
    candidates: list[SearchResult],
) -> dict[str, Any]:
    """Use LLM reasoning to pick the single best matching movie from candidates.

    Returns a dict with keys: title, imdb_id, reasoning, candidates.
    Falls back to the top search result if no LLM key is configured.
    """
    fallback: dict[str, Any] = {
        "title": candidates[0].title if candidates else "",
        "imdb_id": candidates[0].imdb_id if candidates else "",
        "reasoning": "LLM not available; returning top search result.",
        "candidates": [c.to_dict() for c in candidates],
    }

    try:
        lm = _get_lm()
    except RuntimeError:
        logger.warning("No LLM key configured; returning top result as guess.")
        return fallback

    candidate_str = "\n".join(
        f"{i + 1}. {r.title} ({r.release_date}) [{r.imdb_id}] "
        f"– {r.description[:120]}..."
        for i, r in enumerate(candidates)
    )

    with dspy.context(lm=lm):
        module = MovieGuesser()
        try:
            pred = module(description=description, candidates=candidate_str)
            return {
                "title": pred.title,
                "imdb_id": pred.imdb_id,
                "reasoning": pred.reasoning,
                "candidates": [c.to_dict() for c in candidates],
            }
        except Exception:
            logger.exception("DSPy guess failed; returning top result.")
            return fallback
