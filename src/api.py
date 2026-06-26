# src/api.py
# FastAPI service — serves anomaly scores and sentiment results as REST API

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sys
import os

sys.path.append("src")
from database import (
    load_data_to_db,
    get_top_contributors_by_commits,
    get_pr_merge_time_analysis,
    get_code_churn_analysis,
    get_commit_activity_by_day,
    get_hourly_productivity_score
)

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────

app = FastAPI(
    title="Developer Productivity Intelligence API",
    description="Behavioral analytics and anomaly detection for developer teams",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────

def load_results():
    """Load all processed results"""
    results = {}

    # Anomaly results
    anomaly_path = "data/processed/anomaly_results.csv"
    if os.path.exists(anomaly_path):
        results["anomaly"] = pd.read_csv(anomaly_path)

    # Sentiment results
    sentiment_path = "data/processed/sentiment_summary.csv"
    if os.path.exists(sentiment_path):
        results["sentiment"] = pd.read_csv(sentiment_path)

    # Features
    features_path = "data/processed/features.csv"
    if os.path.exists(features_path):
        results["features"] = pd.read_csv(features_path)

    # Raw data
    results["commits"] = pd.read_csv("data/raw/commits.csv")
    results["prs"] = pd.read_csv("data/raw/pull_requests.csv")
    results["contributors"] = pd.read_csv("data/raw/contributors.csv")

    return results


# Load on startup
DATA = load_results()
load_data_to_db()

# ─────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    total_developers: int
    total_commits: int
    total_prs: int


class DeveloperRisk(BaseModel):
    author_login: str
    anomaly_score: float
    reconstruction_error: float
    final_risk: str


class SentimentResult(BaseModel):
    author: str
    total_prs: int
    dominant_sentiment: str
    avg_confidence: float
    frustrated_count: int
    blocked_count: int
    confident_count: int
    sentiment_risk: bool


class TeamSummary(BaseModel):
    total_developers: int
    high_risk_count: int
    medium_risk_count: int
    normal_count: int
    avg_merge_hours: float
    total_commits: int
    anomaly_rate: float


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_developers=len(DATA.get("contributors", [])),
        total_commits=len(DATA.get("commits", [])),
        total_prs=len(DATA.get("prs", []))
    )


@app.get("/team/summary", response_model=TeamSummary)
def get_team_summary():
    """Get overall team health summary"""
    anomaly_df = DATA.get("anomaly")
    commits_df = DATA.get("commits")
    prs_df = DATA.get("prs")

    if anomaly_df is None:
        raise HTTPException(
            status_code=404,
            detail="Anomaly results not found. Run ml_models.py first."
        )

    merged_prs = prs_df[prs_df["merged"] == True]
    avg_merge = merged_prs["time_to_merge_hours"].mean()

    risk_counts = anomaly_df["final_risk"].value_counts()

    return TeamSummary(
        total_developers=len(anomaly_df),
        high_risk_count=int(risk_counts.get("HIGH RISK", 0)),
        medium_risk_count=int(risk_counts.get("MEDIUM RISK", 0)),
        normal_count=int(risk_counts.get("NORMAL", 0)),
        avg_merge_hours=round(float(avg_merge), 2),
        total_commits=len(commits_df),
        anomaly_rate=round(
            float((anomaly_df["final_risk"] != "NORMAL").mean()), 3
        )
    )


@app.get("/developers/anomalies")
def get_all_anomalies():
    """Get anomaly scores for all developers"""
    anomaly_df = DATA.get("anomaly")

    if anomaly_df is None:
        raise HTTPException(
            status_code=404,
            detail="Anomaly results not found. Run ml_models.py first."
        )

    return {
        "total": len(anomaly_df),
        "developers": anomaly_df[[
            "author_login",
            "anomaly_score",
            "reconstruction_error",
            "final_risk"
        ]].to_dict(orient="records")
    }


@app.get("/developers/{username}/risk")
def get_developer_risk(username: str):
    """Get risk assessment for a specific developer"""
    anomaly_df = DATA.get("anomaly")
    features_df = DATA.get("features")

    if anomaly_df is None:
        raise HTTPException(status_code=404, detail="Anomaly data not found")

    dev_anomaly = anomaly_df[
        anomaly_df["author_login"] == username
    ]

    if dev_anomaly.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Developer '{username}' not found"
        )

    dev_features = features_df[
        features_df["author_login"] == username
    ] if features_df is not None else None

    result = dev_anomaly.iloc[0].to_dict()

    if dev_features is not None and not dev_features.empty:
        result["features"] = dev_features.iloc[0].to_dict()

    # Clean NaN values
    for key, value in result.items():
        if isinstance(value, float) and np.isnan(value):
            result[key] = None

    return result


@app.get("/developers/sentiment/all")
def get_all_sentiment():
    """Get sentiment analysis for all developers"""
    sentiment_df = DATA.get("sentiment")

    if sentiment_df is None:
        raise HTTPException(
            status_code=404,
            detail="Sentiment results not found. Run nlp_sentiment.py first."
        )

    return {
        "total": len(sentiment_df),
        "at_risk": int(sentiment_df["sentiment_risk"].sum()),
        "developers": sentiment_df.to_dict(orient="records")
    }


@app.get("/developers/{username}/sentiment")
def get_developer_sentiment(username: str):
    """Get sentiment for a specific developer"""
    sentiment_df = DATA.get("sentiment")

    if sentiment_df is None:
        raise HTTPException(status_code=404, detail="Sentiment data not found")

    dev_sentiment = sentiment_df[sentiment_df["author"] == username]

    if dev_sentiment.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Developer '{username}' not found in sentiment data"
        )

    return dev_sentiment.iloc[0].to_dict()


@app.get("/analytics/commits/by-day")
def commits_by_day():
    """Commit activity by day of week"""
    return get_commit_activity_by_day().to_dict(orient="records")


@app.get("/analytics/commits/by-hour")
def commits_by_hour():
    """Hourly productivity with rolling average"""
    return get_hourly_productivity_score().to_dict(orient="records")


@app.get("/analytics/contributors/top")
def top_contributors():
    """Top contributors by commit count"""
    return get_top_contributors_by_commits().to_dict(orient="records")


@app.get("/analytics/prs/merge-times")
def pr_merge_times():
    """PR merge time analysis per developer"""
    return get_pr_merge_time_analysis().to_dict(orient="records")


@app.get("/analytics/code/churn")
def code_churn():
    """Code churn analysis per developer"""
    return get_code_churn_analysis().to_dict(orient="records")


@app.get("/team/high-risk")
def get_high_risk_developers():
    """Get all developers flagged as high or medium risk"""
    anomaly_df = DATA.get("anomaly")
    sentiment_df = DATA.get("sentiment")

    if anomaly_df is None:
        raise HTTPException(status_code=404, detail="Anomaly data not found")

    at_risk = anomaly_df[
        anomaly_df["final_risk"] != "NORMAL"
    ].copy()

    results = []
    for _, row in at_risk.iterrows():
        entry = {
            "author_login": row["author_login"],
            "anomaly_risk": row["final_risk"],
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "reconstruction_error": round(
                float(row["reconstruction_error"]), 4
            )
        }

        # Add sentiment if available
        if sentiment_df is not None:
            dev_sentiment = sentiment_df[
                sentiment_df["author"] == row["author_login"]
            ]
            if not dev_sentiment.empty:
                entry["sentiment_risk"] = bool(
                    dev_sentiment.iloc[0]["sentiment_risk"]
                )
                entry["dominant_sentiment"] = dev_sentiment.iloc[0][
                    "dominant_sentiment"
                ]

        results.append(entry)

    return {
        "total_at_risk": len(results),
        "developers": results
    }