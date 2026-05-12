from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Single movie search result."""

    title: str
    release_date: int  # year
    duration: int
    rating: float
    genre: str
    description: str
    directors: str
    stars: str
    imdb_id: str
    retrieval_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "release_date": self.release_date,
            "duration": self.duration,
            "rating": self.rating,
            "genre": self.genre,
            "description": self.description,
            "directors": self.directors,
            "stars": self.stars,
            "imdb_id": self.imdb_id,
            "retrieval_score": round(self.retrieval_score, 4),
        }
