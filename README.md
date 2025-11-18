# Dota 2 Draft Analysis System

## Description

The system analyzes draft screenshots and predicts winners based on:

- team statistics
- hero win rates for each team
- match history and form trends
- draft analysis

## Installation

### 1. Create environment

```zsh
conda create -n dd python=3.11 -y
conda activate dd
```

### 2. Install project

```zsh
pip install -e . --config-settings editable_mode=strict
```

### 3. Setup API keys

Create files in `configs/`:

**`configs/.google_api_key.env`**:

```env
GOOGLE_API_KEY=ur_google_api_key
```

**`configs/.langchain_api_key.env`** (optional, for LangSmith):

```env
LANGCHAIN_API_KEY=ur_langchain_api_key
```

## Architecture

```zsh
┌─────────────────┐
│  OpenDota API   │  ← Dota 2 data source
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  data loading   │
│  and caching    │  
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  chroma db      │  ← Vector database
│                 │     - Text embeddings
│                 │     - CLIP image embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Analysing pipeline                      │
│                                         │
│  1. extract teams/                      │
│     players from screenshot             │
│                                         │
│  2. check player                        │
│     names using player data             │
│                                         │
│  3. get heroes                          │
│                                         │
│  4. get team statistics,                │
│     match history, player data          │
│                                         │
│  5. final analysis &                    │
│     winner prediction                   │
└─────────────────────────────────────────┘
```

## Data Sources

### OpenDota API

The system uses [OpenDota API](https://docs.opendota.com/) to fetch:

heroes, teams, pro players, pro matches

## Chroma

using huggingface embeddings(`sentence-transformers/all-MiniLM-L6-v2`) for text documents and clip for hero images embaddings

## Get data

### fetch data from opendota

```zsh
dota-drafts fetch_all
```

### fetch pro matches

```zsh
dota-drafts pro_matches
```

### build rag

```zsh
dota-drafts build_rag
```

## Tests

[![hero-classification-comparison.png](https://i.postimg.cc/6pJTSBbd/hero-classification-comparison.png)](https://postimg.cc/56gxQWV0)

### FISSURE PLAYGROUND 2 Playoffs

| Round | Winner | Loser | Predict | Score | Succes Predict |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **UPPER BRACKET** | | | | | |
| Quarterfinals | **Team Yandex** | Tundra Esports | 0 - 3 | 2 - 1 | 1/3 |
| | **MOUZ** | Team Falcons | 0 - 2 | 2 - 0 | 0/2 |
| | **Team Spirit** | HEROIC | 1 - 1 | 2 - 0 | 1/2 |
| | **BetBoom Team** | Team Liquid | 1 - 2 | 2 - 1 | 2/3 |
| Semifinals | **Team Yandex** | MOUZ | 1 - 2 | 2 - 1 | 1/3 |
| | **BetBoom Team** | Team Spirit | 1 - 2 | 2 - 1 | 1/3 |
| Final | **BetBoom Team** | Team Yandex | 3 - 0 | 2 - 1 | 2/3 |
| | *BetBoom Team to Grand Final* | *Team Yandex to Lower Bracket Final* | | | |
| **LOWER BRACKET** | | | | | |
| Round 1 | **Team Falcons** | Tundra Esports | 3 - 0 | 2 - 1 | 2/3 |
| | **Team Liquid** | HEROIC | 1 - 2 | 2 - 1 | 1/3 |
| Quarterfinals | **Team Falcons** | Team Spirit | 3 - 0 | 2 - 1 | 2/3 |
| | **Team Liquid** | MOUZ | 3 - 0 | 2 - 1 | 2/3 |
| Semifinal | **Team Falcons** | Team Liquid | 3 - 0 | 2 - 1 | 2/3 |
| Final | **Team Falcons** | Team Yandex | 2 - 0 | 2 - 0 | 2/2 |
| | *Team Falcons to Grand Final* | *Team Yandex eliminated* | | | |
| **GRAND FINAL** | | | | | |
| | **Team Falcons** | BetBoom Team | 4 - 0 | 3 - 1 | 3/4 |

[![output.png](https://i.postimg.cc/BZctLrmz/output.png)](<https://postimg.cc/z3vq4cZw>)

[![Screen-Recording2025-11-18at7-37-10PMonline-video-cutter-comonline-video-cutter-com-ezgif-com-optimi.gif](https://i.postimg.cc/HnqNXHXZ/Screen-Recording2025-11-18at7-37-10PMonline-video-cutter-comonline-video-cutter-com-ezgif-com-optimi.gif)](https://postimg.cc/ctcTGpDw)

## Links

- [OpenDota API Documentation](https://docs.opendota.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangChain Documentation](https://python.langchain.com/)
