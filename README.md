# Dream Journal Analyzer 🌙

An NLP-powered dream journal that analyzes your dreams over time to find patterns, recurring symbols, emotional trends, and hidden connections.

## Features

- **Dream Logging** — Add dreams with dates and optional tags
- **Symbol Extraction** — Automatically identifies key symbols, people, places, and objects
- **Emotion Analysis** — Classifies each dream's emotional tone
- **Pattern Detection** — Tracks recurring symbols, co-occurrences, and temporal trends
- **Visualization** — Interactive dashboard with word clouds, emotion timelines, and symbol networks
- **Dream Similarity Search** — Find dreams similar to any entry

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run app.py
```

## Project Structure

```
dream_journal/
├── app.py                 # Streamlit dashboard
├── journal.py             # Dream entry storage & retrieval
├── nlp_pipeline.py        # NLP processing (entities, emotions, embeddings)
├── analyzer.py            # Pattern analysis & insights
├── visualizations.py      # Charts and graphs
├── dream_symbols.py       # Common dream symbol dictionary
├── requirements.txt
└── data/
    └── dreams.json        # Local dream storage
```

## Tech Stack

- Python 3.9+
- spaCy (NLP & entity extraction)
- transformers / zero-shot classification (emotion analysis)
- sentence-transformers (dream embeddings & similarity)
- scikit-learn (clustering)
- Streamlit (web dashboard)
- Plotly (interactive visualizations)
