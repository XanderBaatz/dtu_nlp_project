# DTU NLP Project - Movie Discovery API

A hybrid movie search and recommendation system built with BM25, sentence embeddings, LLM query expansion, and CPI-adjusted popularity ranking. ~62,000 IMDb movies from 1960–2024.

**Dataset:** [Kaggle - Top 500–600 Movies per Year (1960–2024)](https://www.kaggle.com/datasets/raedaddala/top-500-600-movies-of-each-year-from-1960-to-2024/data)


---

## Quick Start

### 1. Install dependencies
```sh
uv sync
```

### 2. Configure environment
Copy `.env` and fill in your credentials in `.env.local` (never committed):
```sh
cp .env .env.local
```

Minimum required in `.env.local`:
```ini
CAMPUSAI_API_KEY=your-key-here        # for LLM query expansion + DSPy
KAGGLE_API_TOKEN=your-token-here      # only needed to re-download the dataset
```

### 3. Download and process the dataset
Only needed once (or when you want to refresh the data):
```sh
uv run python -m movies.downloader
```
This downloads the Kaggle dataset, cleans it, adds weighted ratings and CPI-adjusted gross, and saves it to `data/imdb_movies/processed/movies.parquet`.

### 4. Build the search index
Also only needed once (or after re-downloading):
```sh
uv run python -c "from movies.database import MovieDatabase; db = MovieDatabase(); db.build()"
```
This builds and saves the BM25 index (`bm25.pkl`), embedding matrix (`embeddings.npy`), and index ID map to `data/imdb_movies/index/`.

### 5. Start the API
```sh
uv run uvicorn src.movies.main:app --reload --port 8000
```
Interactive Swagger UI: **http://127.0.0.1:8000/docs**


---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IS_LOCAL` | `true` | Switches logger to coloured console output |
| `CAMPUSAI_API_KEY` | _(empty)_ | CampusAI API key for LLM expansion + DSPy |
| `CAMPUSAI_API_URL` | `https://api.campusai.compute.dtu.dk/v1/` | API base URL |
| `CAMPUSAI_MODEL` | `google/gemma-4-26b-a4b` | LLM model name |
| `CAMPUSAI_EMBED_MODEL` | `nomic/nomic-embed-text` | Embedding model (used when `LOCAL_EMBEDDING=false`) |
| `LOCAL_EMBEDDING` | `true` | Use local `all-MiniLM-L6-v2` model instead of API |
| `KAGGLE_API_TOKEN` | _(empty)_ | Kaggle token for dataset download |
| `OPENAI_API_KEY` | _(empty)_ | Optional fallback if no CampusAI key |


---

## Docker

Build:
```sh
docker build -f dockerfiles/api.dockerfile -t movies-api .
```

Run:
```sh
docker run -p 8002:8002 -v $(pwd)/data:/app/data \
  --env-file .env \
  -e CAMPUSAI_API_KEY=$CAMPUSAI_API_KEY \
  movies-api
```

---

## API Endpoints

All endpoints are prefixed with `/api/v1`. Full docs at `/docs`.

### `GET /api/v1/health`
Health check - returns index size and version.
```sh
curl http://127.0.0.1:8000/api/v1/health
```

### `GET /api/v1/search`
Hybrid BM25 + semantic search with LLM query expansion and smart re-ranking.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | required | Free-text search query |
| `top_k` | `10` | Number of results (1–100) |
| `alpha` | `0.5` | Dense/sparse balance: `0`=pure BM25, `1`=pure semantic |
| `weighted_rating_alpha` | `0.10` | Weight for IMDb rating signal |
| `popularity_alpha` | `0.05` | Weight for CPI-adjusted box office signal |
| `expand` | `true` | Enable LLM query expansion |

```sh
curl "http://127.0.0.1:8000/api/v1/search?query=wizard+with+round+glasses&top_k=5"
curl "http://127.0.0.1:8000/api/v1/search?query=Meryl+Streep+drama&expand=false"
curl "http://127.0.0.1:8000/api/v1/search?query=space+thriller&alpha=0.8&popularity_alpha=0.1"
```

### `GET /api/v1/movies/{imdb_id}`
Fetch a single movie by its IMDb ID.
```sh
curl http://127.0.0.1:8000/api/v1/movies/tt0120338    # Titanic
```

### `GET /api/v1/movies/{imdb_id}/similar`
Find movies similar to a given film, re-ranked with the same rating + popularity blend as search.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | `10` | Number of results |
| `weighted_rating_alpha` | `0.10` | Weight for IMDb rating |
| `popularity_alpha` | `0.05` | Weight for CPI-adjusted gross |

```sh
curl "http://127.0.0.1:8000/api/v1/movies/tt0120338/similar?top_k=5"
```

### `GET /api/v1/lookup`
Resolve a movie title (and optional year) to its IMDb ID and metadata.
```sh
curl "http://127.0.0.1:8000/api/v1/lookup?title=Titanic&year=1997"
```

### `POST /api/v1/recommend`
Search for movies and generate an LLM explanation (via DSPy `ChainOfThought`) for why they match the query.
```sh
curl -X POST http://127.0.0.1:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "melancholic sci-fi about loneliness in space", "top_k": 5}'
```

### `POST /api/v1/movies/guess`
Given a plot description, search for candidates and use the LLM to reason about which movie it is.
```sh
curl -X POST http://127.0.0.1:8000/api/v1/movies/guess \
  -H "Content-Type: application/json" \
  -d '{"description": "A boy learns he is a wizard and goes to a school for magic"}'
```


---

## Architecture

### Source layout
```
src/movies/
  config.py       - Pydantic Settings (paths, API keys, tuning params)
  downloader.py   - Kaggle download + dataset processing
  preprocess.py   - Cleaning, normalization, weighted rating, CPI-adjusted gross
  documents.py    - Builds flat text documents from movie rows (used by both indexes)
  embedder.py     - Embedder protocol: LocalEmbedder (sentence-transformers) or OpenAIEmbedder
  nlp.py          - QueryProcessor (spaCy), QueryExpander (LLM), ParsedQuery
  scorer.py       - Pure scoring functions: zscore, rrf, person_boost, genre_boost
  database.py     - MovieDatabase: build/load/search/similar/lookup/by_id
  rag.py          - DSPy modules: Recommender (ChainOfThought), MovieGuesser
  schema.py       - SearchResult dataclass
  main.py         - FastAPI app + all endpoints
```

### Dataset processing ([downloader.py](src/movies/downloader.py), [preprocess.py](src/movies/preprocess.py))

- Downloads ~62,000 movies from Kaggle (top 500–600 per year, 1960–2024)
- Normalizes list columns (genres, cast, directors, writers), duration strings → minutes, budget/gross strings → floats
- Adds IMDb-style **weighted rating** (Bayesian average against the dataset mean):

$$\mathrm{WR} = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$$

where $R$ = movie rating, $v$ = vote count, $m$ = 90th-percentile vote threshold, $C$ = dataset mean rating.

- Adds **CPI-adjusted gross** using seasonally adjusted CPIAUCSL data (FRED, `data/cpi/CPIAUCSL.csv`):

$$\mathrm{AdjustedGross} = \mathrm{gross\_us\_canada} \times \frac{\mathrm{CPI}_\text{today}}{\mathrm{CPI}_\text{release year}}$$

This makes box-office revenue comparable across decades - Star Wars (1977) at $775M nominal becomes ~$4.2B in today's dollars. Unfortunately only about 1/3 of the dataset are non-null in `gross_us_canada`, so the null rows are given the median/mean gross.

### Indexing ([database.py](src/movies/database.py))

Two indexes are built from a single flat document per movie:
```
Title: Titanic | Desc: A love story aboard... | Genres: Drama, Romance | Cast: Leonardo DiCaprio, Kate Winslet | Directors: James Cameron
```

1. **Sparse - BM25** (`bm25s`): lemmatized, lowercased tokens via spaCy
2. **Dense - sentence embeddings** (`all-MiniLM-L6-v2` locally, or `nomic-embed-text` via CampusAI): L2-normalized cosine similarity

A **reverse person index** is also built at load time: `lowercase_name → [row_indices]` for all directors, stars, and writers. Multi-word names are matched with word-boundary regex against the raw query, bypassing spaCy NER for reliable cast lookup.

### Query Processing ([nlp.py](src/movies/nlp.py))

Three stages before search:
1. **spaCy** (`en_core_web_sm`): lemmatization, stop-word removal, named entity recognition
2. **Genre detection**: matched against known genre vocabulary extracted from the dataset
3. **LLM query expansion** (`QueryExpander`): an OpenAI-compatible call turns a vague description like `"wizard with round glasses"` into concrete keywords - `harry potter hogwarts gryffindor fantasy magic` - which are fed into an additional BM25 pass

### Hybrid Search + Re-ranking ([database.py](src/movies/database.py))

```
Query
  │
  ├─ BM25 ranking (sparse)         ─┐
  └─ Embedding cosine sim (dense)   ┼─→ Reciprocal Rank Fusion (RRF) → candidate pool
                                   ─┘
          │
          ├─ Inject: person-matched rows (reverse person index)
          └─ Inject: top BM25 hits on LLM expansion tokens
                                   │
                           Re-score candidates:
                             z(retrieval) × (1 − α_rating − α_pop)
                           + z(weighted_rating) × α_rating
                           + z(adjusted_gross)  × α_pop
                           + person_boost  (+2.0 if cast/crew match)
                           + genre_boost   (+1.0 if genre match)
                                   │
                                Top-k results
```

`alpha` controls BM25/dense balance (`0`=pure BM25, `1`=pure semantic). `weighted_rating_alpha` (default 0.10) and `popularity_alpha` (default 0.05) blend in the quality and box-office signals; the remainder goes to retrieval relevance.

### LLM Reasoning ([rag.py](src/movies/rag.py))

Uses DSPy `ChainOfThought` with two signatures:
- **`Recommender`** - generates a natural-language explanation for why a set of retrieved movies matches a query (`/recommend`)
- **`MovieGuesser`** - given a plot description and a candidate shortlist, reasons about which specific film is being described (`/movies/guess`)

Falls back gracefully (top search result / empty explanation) if no API key is configured.


