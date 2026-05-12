"""FastAPI application – movie discovery API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from movies.config import Settings
from tools.logger import Logger, LogType

settings = Settings()
logger = Logger(name=__name__, log_type=LogType.LOCAL)

_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load (or build) the movie index on startup."""
    from movies.database import MovieDatabase

    logger.info("Initializing movie database...")
    db = MovieDatabase()
    db.load_or_build()
    _state["db"] = db
    logger.info("Startup complete – %d movies in index", db.size)
    yield
    logger.info("Shutting down.")
    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan, **settings.fastapi_kwargs)

app.add_middleware(
    middleware_class=TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts,
)
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

PREFIX = settings.api_prefix_v1


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5


class GuessRequest(BaseModel):
    description: str


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _db() -> Any:
    db = _state.get("db")
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")
    return db


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(f"{PREFIX}/health", summary="Health check")
async def health() -> dict[str, Any]:
    """Return API health status and current index size."""
    return {
        "status": "ok",
        "index_size": _db().size,
        "version": settings.version,
    }


@app.get(f"{PREFIX}/search", summary="Semantic + BM25 hybrid search")
async def search(
    query: str = Query(..., description="Free-text search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    alpha: float = Query(
        0.5, ge=0.0, le=1.0, description="Dense/sparse balance (0=BM25, 1=semantic)"
    ),
    weighted_rating_alpha: float = Query(
        0.10, ge=0.0, le=1.0, description="Weight for IMDb rating"
    ),
    popularity_alpha: float = Query(
        0.05, ge=0.0, le=1.0, description="Weight for CPI-adjusted gross revenue"
    ),
    expand: bool = Query(True, description="Use LLM query expansion"),
) -> dict[str, Any]:
    """Hybrid BM25 + semantic search with optional LLM query expansion."""
    db = _db()
    results = db.search(
        query=query,
        top_k=top_k,
        alpha=alpha,
        weighted_rating_alpha=weighted_rating_alpha,
        popularity_alpha=popularity_alpha,
        expand=expand,
    )
    return {
        "query": query,
        "total": len(results),
        "results": [r.to_dict() for r in results],
    }


@app.get(f"{PREFIX}/lookup", summary="Resolve title to IMDb ID")
async def lookup(
    title: str = Query(..., description="Movie title"),
    year: int | None = Query(None, description="Release year (optional)"),
) -> dict[str, Any]:
    """Find a movie by title and optional release year."""
    db = _db()
    result = db.lookup(title=title, year=year)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")
    return result.to_dict()


@app.get(f"{PREFIX}/movies/{{imdb_id}}", summary="Get movie by IMDb ID")
async def get_movie(imdb_id: str) -> dict[str, Any]:
    """Return a single movie by its IMDb ID."""
    db = _db()
    result = db.by_id(imdb_id=imdb_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Movie '{imdb_id}' not found.")
    return result.to_dict()


@app.get(f"{PREFIX}/movies/{{imdb_id}}/similar", summary="Find similar movies")
async def similar_movies(
    imdb_id: str,
    top_k: int = Query(10, ge=1, le=100),
    weighted_rating_alpha: float = Query(0.10, ge=0.0, le=1.0),
    popularity_alpha: float = Query(0.05, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Return movies most similar to the given IMDb ID, re-ranked with rating
    and CPI-adjusted popularity signals.
    """
    db = _db()
    if db.by_id(imdb_id) is None:
        raise HTTPException(status_code=404, detail=f"Movie '{imdb_id}' not found.")
    results = db.similar(
        imdb_id=imdb_id,
        top_k=top_k,
        weighted_rating_alpha=weighted_rating_alpha,
        popularity_alpha=popularity_alpha,
    )
    return {
        "imdb_id": imdb_id,
        "total": len(results),
        "results": [r.to_dict() for r in results],
    }


@app.post(f"{PREFIX}/recommend", summary="LLM-powered recommendation with explanation")
async def recommend(body: RecommendRequest) -> dict[str, Any]:
    """Search for movies and generate an LLM explanation for why they match."""
    from movies.rag import explain_recommendations

    db = _db()
    results = db.search(query=body.query, top_k=body.top_k, expand=True)
    explanation = explain_recommendations(query=body.query, results=results)
    return {
        "query": body.query,
        "explanation": explanation,
        "results": [r.to_dict() for r in results],
    }


@app.post(f"{PREFIX}/movies/guess", summary="Guess movie from plot description")
async def guess_movie(body: GuessRequest) -> dict[str, Any]:
    """Use LLM reasoning to guess which movie matches a plot description."""
    from movies.rag import guess_movie_from_description

    db = _db()
    candidates = db.search(query=body.description, top_k=5, expand=True)
    result = guess_movie_from_description(
        description=body.description, candidates=candidates
    )
    return result


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
