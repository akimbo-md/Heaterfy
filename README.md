# 🔥 Heaterfy — Intelligent Playlist Recommendation Engine

Heaterfy is a machine learning-powered music recommendation system designed to enhance playlist curation by analyzing a playlist's "vibe" and generating 20 personalized track suggestions that match the style, mood, and genre diversity of the original list.

by Aidan Mark (Akimbo), DJ 'Oscar De Leon' NUMP, Trevor ‘Track Whisperer’ Farquhar

---

## Key Features

- **Multi-Model Recommendation Pipeline**  
  Uses a combination of models (XGBoost, Ridge Regression, GLM, K-Means, Neural Net) to evaluate musical fit from different angles.

- **Genre Graph Embeddings (16D)**  
  Leverages EveryNoise’s 2D genre map to build a 16-dimensional genre graph using Node2Vec, allowing nuanced understanding of genre similarity.

- **Audio Feature Analysis**  
  Tracks are scored based on audio characteristics like danceability, energy, tempo, and valence using both cosine similarity and regression.

- **Metadata Matching (GLM)**  
  Tracks are boosted if they share artists, albums, or record labels with the original playlist.

- **Temporal Alignment**  
  Applies dynamic year-based penalties or bonuses based on playlist era to ensure time-consistent recommendations.

- **Cluster-Based Diversity**  
  Uses K-Means to preserve genre and style diversity, ensuring recommendations represent all "flavors" of the original playlist.

- **Streamlit Interface (optional)**  
  Lightweight front-end for generating and reviewing recommendations in a user-friendly format.

---

## How It Works

1. Load playlist and catalog
2. Clean data and fill missing genre info
3. Embed genres using pretrained 16D vectors
4. Score tracks using multiple models
5. Adjust scores with year penalties and bonuses
6. Blend scores into a final ranking
7. Recommend 100 tracks based on cluster proportions

---

## Launching

```bash
streamlit run web_app.py
