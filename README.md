# Spotify Breakout Predictor

A machine learning project that predicts breakout potential for songs on Spotify using multi-platform engagement metrics, achieving 97% accuracy with Random Forest classification.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Het415/spotify-breakout-predictor/blob/main/spotifytrend.ipynb)

## Overview

This project analyzes the most streamed songs on Spotify in 2024 to identify emerging tracks with breakout potential. By combining streaming data with cross-platform engagement metrics from TikTok, YouTube, Shazam, and other platforms, the model predicts which songs are poised to become viral hits despite not yet having mainstream popularity.

## Problem Statement

The music industry needs tools to identify breakout artists and tracks early, before they reach mainstream success. This project addresses that need by:
- Predicting which songs have viral potential based on social media traction
- Identifying tracks with high TikTok engagement but low mainstream popularity
- Providing data-driven insights for A&R professionals, playlist curators, and music marketers

## Key Features

- **Multi-Platform Analysis**: Integrates data from Spotify, TikTok, YouTube, Shazam, Apple Music, and more
- **Feature Engineering**: Creates velocity metrics (streams per day) and virality ratios
- **Binary Classification**: Identifies "breakout candidates" using Random Forest
- **High Accuracy**: Achieves 97% accuracy with 0.99 ROC-AUC score
- **Comprehensive EDA**: Correlation analysis, temporal patterns, and platform relationships
- **Data Export**: Generates clean CSV with breakout probabilities for dashboard deployment

## Dataset

The dataset contains 4,595 tracks with 28+ features spanning multiple streaming platforms:

### Core Metrics
- **Spotify**: Streams, Playlist Count, Playlist Reach, Popularity Score
- **TikTok**: Posts, Likes, Views
- **YouTube**: Views, Likes, Playlist Reach
- **Shazam**: Recognition counts
- **Apple Music**: Playlist Count
- **Other Platforms**: Deezer, Amazon, Pandora, AirPlay

### Engineered Features
- `Days_Since_Release`: Days between release date and current date
- `Stream_Velocity`: Daily streaming rate (Streams / Days Since Release)
- `Virality_Ratio`: TikTok Views / Spotify Streams
- `Release_Month`: Month of release for seasonal analysis

### Target Variable
- `is_breakout`: Binary label indicating breakout potential
  - **Criteria**: Spotify Popularity < 75 AND TikTok Views >= 75th percentile
  - **Logic**: High social media engagement but not yet mainstream success

## Tech Stack

**Languages & Libraries:**
- Python 3.x
- pandas - Data manipulation and cleaning
- NumPy - Numerical operations
- Matplotlib & Seaborn - Visualizations
- scikit-learn - Machine learning (Random Forest, metrics)
- Streamlit - Dashboard deployment (optional)
- Plotly - Interactive visualizations

**Environment:**
- Jupyter Notebook / Google Colab

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Het415/spotify-breakout-predictor.git
cd spotify-breakout-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
```

3. Open the notebook:
```bash
jupyter notebook spotifytrend.ipynb
```

## Project Structure

```
spotify-breakout-predictor/
│
├── spotifytrend.ipynb               # Main analysis notebook
├── df_cleaned.csv                   # Processed dataset with predictions
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── data/
    └── most_streamed_spotify_songs_2024.csv  # Raw dataset
```

## Methodology

### 1. Data Acquisition & Preprocessing
- Loaded 2024 Spotify streaming data with 4,600 tracks
- Cleaned numeric columns (removed commas, converted to float)
- Parsed release dates and calculated days since release
- Created stream velocity metric (streams per day)

### 2. Missing Value Handling
- **Dropped**: TIDAL Popularity, Soundcloud Streams, SiriusXM Spins (>70% missing)
- **Zero-filled**: TikTok metrics, Pandora streams (assuming no activity)
- **Median imputation**: Spotify Popularity (804 missing values)
- **Removed**: 5 rows with missing artist information
- Final dataset: 4,595 tracks with complete features

### 3. Target Variable Creation
- Identified "breakout candidates" using dual criteria:
  - Low mainstream popularity: Spotify Popularity < 75
  - High social engagement: TikTok Views >= 75th percentile
- Result: 1,149 breakout candidates (25% of dataset)

### 4. Exploratory Data Analysis
- **Platform Correlations**: Analyzed relationships between streaming platforms
- **Temporal Patterns**: Examined popularity by release month
- **Feature Distributions**: Identified key patterns in engagement metrics
- **Virality Analysis**: Explored TikTok's influence on streaming success

### 5. Feature Engineering & Selection
Selected 12 predictive features:
- Track Score, Spotify Playlist Count, Spotify Playlist Reach
- YouTube Views, TikTok Views, TikTok Posts, TikTok Likes
- Apple Music Playlist Count, AirPlay Spins, Shazam Counts
- Stream Velocity, Days Since Release

### 6. Model Training
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: 100 estimators, max_depth=10
- **Train-Test Split**: 80/20 with stratified sampling
- **Random State**: 42 (for reproducibility)

### 7. Model Evaluation
- **Accuracy**: 97%
- **ROC-AUC Score**: 0.99
- **Precision** (Breakout class): 90%
- **Recall** (Breakout class): 94%
- **F1-Score** (Breakout class): 92%

## Results

The Random Forest model demonstrates exceptional performance in identifying breakout tracks:

**Performance Metrics:**
```
              precision    recall  f1-score   support
           0       0.99      0.98      0.98       740
           1       0.90      0.94      0.92       179
    accuracy                           0.97       919
   macro avg       0.95      0.96      0.95       919
weighted avg       0.97      0.97      0.97       919

ROC-AUC Score: 0.99
```

**Key Insights:**
1. **TikTok Dominance**: TikTok engagement is the strongest predictor of breakout potential
2. **Velocity Matters**: Stream velocity (daily streaming rate) is more predictive than total streams
3. **Cross-Platform Signals**: Shazam counts and YouTube views provide complementary signals
4. **Timing**: Tracks released in certain months show higher breakout rates
5. **Playlist Power**: Spotify playlist reach correlates with sustained growth

**Business Applications:**
- **Early Detection**: Identify viral tracks 2-4 weeks before mainstream success
- **Playlist Curation**: Data-driven selections for "viral hits" and "rising stars" playlists
- **Marketing Budget**: Allocate resources to high-potential tracks
- **A&R Scouting**: Discover emerging artists before competitors

## Usage

### Making Predictions

```python
import pandas as pd
import pickle

# Load the trained model
# with open('breakout_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# Prepare new track data
new_track = {
    'Track Score': [85.5],
    'Spotify Playlist Count': [1200],
    'Spotify Playlist Reach': [2500000],
    'YouTube Views': [5000000],
    'TikTok Views': [15000000],
    'TikTok Posts': [8500],
    'TikTok Likes': [350000],
    'Apple Music Playlist Count': [85],
    'AirPlay Spins': [1200],
    'Shazam Counts': [45000],
    'Stream_Velocity': [250000],
    'Days_Since_Release': [45]
}

df_new = pd.DataFrame(new_track)

# Predict breakout probability
prediction = rf_model.predict(df_new)
probability = rf_model.predict_proba(df_new)[:, 1]

print(f"Breakout Prediction: {'YES' if prediction[0] == 1 else 'NO'}")
print(f"Breakout Probability: {probability[0]:.1%}")
```

### Identifying Top Breakout Candidates

```python
# Load processed data
df = pd.read_csv('df_cleaned.csv')

# Get top 10 breakout candidates
top_breakouts = df[df['is_breakout'] == 1].nlargest(10, 'Breakout_Probability')

print("Top 10 Breakout Tracks:")
print(top_breakouts[['Track', 'Artist', 'Breakout_Probability', 'TikTok Views', 'Spotify Popularity']])
```

## Model Interpretation

**Feature Importance (Top 5):**
1. TikTok Views - Strongest predictor of viral potential
2. Stream Velocity - Indicates momentum and growth rate
3. Shazam Counts - Shows genuine listener interest
4. Spotify Playlist Reach - Reflects algorithmic promotion
5. YouTube Views - Cross-platform validation signal

## Future Enhancements

### Model Improvements
- **Ensemble Methods**: Combine Random Forest with Gradient Boosting and Neural Networks
- **Time Series Analysis**: Predict growth trajectories over time
- **Artist Features**: Incorporate artist follower counts and historical performance
- **Sentiment Analysis**: Analyze lyrics and social media comments
- **Audio Features**: Add Spotify API audio features (danceability, energy, valence)

### Feature Engineering
- **Engagement Ratios**: Posts-to-views, likes-to-views for engagement quality
- **Growth Metrics**: Week-over-week streaming growth rates
- **Genre Classification**: Category-specific breakout patterns
- **Geographic Data**: Region-specific viral trends

### Deployment
- **Real-Time API**: Flask/FastAPI endpoint for live predictions
- **Streamlit Dashboard**: Interactive visualization of breakout candidates
- **Scheduled Updates**: Automated daily data pipeline
- **Alert System**: Notifications for new high-probability breakouts

## Limitations & Considerations

- **Data Recency**: Model trained on 2024 data; trends evolve rapidly
- **Platform Bias**: Heavily weighted toward TikTok influence
- **Definition**: "Breakout" criteria is subjective and adjustable
- **Survivorship Bias**: Dataset contains only already-successful tracks
- **External Factors**: Cannot account for marketing budgets, radio play, or cultural events

## Contributing

Contributions are welcome! Areas for contribution:
- Additional data sources (radio airplay, concert data, social sentiment)
- Alternative model architectures
- Dashboard improvements
- Feature engineering ideas

Please open an issue first to discuss proposed changes.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset source: Spotify Most Streamed Songs 2024
- Inspiration from music analytics and viral trend prediction research
- scikit-learn and pandas communities

## Contact

Het - [GitHub Profile](https://github.com/Het415)

Project Link: [https://github.com/Het415/spotify-breakout-predictor](https://github.com/Het415/spotify-breakout-predictor)

---

**Note**: This project is for educational and research purposes. Predictions should be combined with domain expertise and market knowledge for business decisions.
