# Spotify Hybrid Recommender System

A production-ready hybrid recommendation engine that combines **content-based filtering** (using audio features) and **collaborative filtering** (based on user listening history). The system allows users to control the blending factor (α ∈ [0,1]) via a **diversity slider (1–10)** in the Streamlit app, where `α = 1 - (diversity / 10)`.

---

## Overview

This project implements a full end-to-end recommender system for Spotify-like music discovery. It leverages:
- **Content-based filtering**: Recommendations based on track metadata and audio features (e.g., danceability, energy, tempo).
- **Collaborative filtering**: User-item interactions modeled via similarity or matrix factorization.
- **Hybrid blending**: Weighted combination of both scores using a user-defined α parameter.

The system is designed with **MLOps best practices**, including:
- Data stored in **AWS S3**
- Versioned with **DVC**
- CI/CD via **GitHub Actions**
- Containerized with **Docker**
- Interactive **Streamlit UI** for real-time recommendations

---

## Key Features

- Interactive **Streamlit web app** with:
  - Song + artist input
  - Filtering mode selection (Content / Collaborative / Hybrid)
  - Diversity slider (1–10) → controls α
  - Adjustable number of recommendations (k)
- Data preprocessing via `data_cleaning.py` and `transform_filtered_data.py`
- Modular recommendation logic:
  - `content_based_filtering.py`
  - `collaborative_filtering.py`
  - `hybrid_recommendations.py`
- DVC pipeline (`dvc.yaml`) for reproducible data and model stages
- CI/CD pipeline (`.github/workflows/ci.yaml`) with automated testing and Docker build
- Docker image pushed to **Docker Hub** on merge to `main`
- Evaluation-ready notebooks for **Precision@K, Recall@K, NDCG**

---

## Tech Stack

| Component              | Technology                              |
|------------------------|-----------------------------------------|
| Web Interface          | Streamlit                               |
| ML Framework           | Scikit-learn, Pandas, NumPy, SciPy      |
| Data Versioning        | DVC                                     |
| Cloud Storage          | AWS S3                                  |
| CI/CD                  | GitHub Actions                          |
| Containerization       | Docker                                  |
| API / App              | Streamlit (with embedded logic)         |
| Datasets               | `Music Info.csv`, `User Listening History.csv` |

---

## Initialization

```bash
# Clone the repository
git clone https://github.com/ShubhamWaghmare11/Spotify-Hybrid-Recommender-System.git
cd Spotify-Hybrid-Recommender-System

# Install dependencies
pip install -r requirements.txt

# Pull data using DVC (requires AWS credentials configured)
dvc pull

# Run the DVC pipeline to process data
dvc repro

# Start the Streamlit app
streamlit run app.py
```

