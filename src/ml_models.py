# src/ml_models.py
# ML Layer - Isolation Forest + LSTM Autoencoder for anomaly detection

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import warnings
import os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────

def engineer_features():
    """Create 20+ behavioral features from raw data"""
    print("Engineering features...")

    commits_df = pd.read_csv("data/raw/commits.csv")
    pr_df = pd.read_csv("data/raw/pull_requests.csv")

    # ── Commit features per developer ──
    commit_features = commits_df.groupby("author_login").agg(
        total_commits=("sha", "count"),
        avg_additions=("additions", "mean"),
        avg_deletions=("deletions", "mean"),
        avg_total_changes=("total_changes", "mean"),
        max_changes=("total_changes", "max"),
        std_changes=("total_changes", "std"),
        avg_hour=("hour_of_day", "mean"),
        std_hour=("hour_of_day", "std"),
        late_night_commits=("hour_of_day", lambda x: (x >= 22).sum()),
        early_morning_commits=("hour_of_day", lambda x: (x <= 6).sum()),
        weekend_commits=("day_of_week", lambda x: x.isin(["Saturday", "Sunday"]).sum()),
        unique_days=("day_of_week", "nunique")
    ).reset_index()

    # ── PR features per developer ──
    pr_features = pr_df.groupby("author").agg(
        total_prs=("pr_number", "count"),
        merged_prs=("merged", "sum"),
        avg_merge_hours=("time_to_merge_hours", "mean"),
        std_merge_hours=("time_to_merge_hours", "std"),
        avg_first_review_hours=("first_review_hours", "mean"),
        avg_pr_additions=("additions", "mean"),
        avg_pr_deletions=("deletions", "mean"),
        avg_changed_files=("changed_files", "mean"),
        avg_review_comments=("review_comments", "mean"),
        total_review_comments=("review_comments", "sum")
    ).reset_index().rename(columns={"author": "author_login"})

    # ── Merge all features ──
    features_df = commit_features.merge(
        pr_features, on="author_login", how="left"
    )

    # ── Fill missing values ──
    features_df = features_df.fillna(0)

    # ── Derived features ──
    features_df["merge_rate"] = (
        features_df["merged_prs"] /
        features_df["total_prs"].replace(0, 1)
    )
    features_df["churn_ratio"] = (
        features_df["avg_deletions"] /
        features_df["avg_additions"].replace(0, 1)
    )
    features_df["off_hours_ratio"] = (
        (features_df["late_night_commits"] + features_df["early_morning_commits"]) /
        features_df["total_commits"].replace(0, 1)
    )
    features_df["pr_size_score"] = (
        features_df["avg_pr_additions"] + features_df["avg_pr_deletions"]
    )

    print(f"Features engineered for {len(features_df)} developers")
    print(f"Total features: {len(features_df.columns) - 1}")

    os.makedirs("data/processed", exist_ok=True)
    features_df.to_csv("data/processed/features.csv", index=False)

    return features_df


# ─────────────────────────────────────────
# MODEL 1 — ISOLATION FOREST
# ─────────────────────────────────────────

def train_isolation_forest(features_df):
    """Train Isolation Forest for anomaly detection"""
    print("\n" + "="*50)
    print("Training Isolation Forest...")
    print("="*50)

    # Select numeric feature columns
    feature_cols = [
        "total_commits", "avg_additions", "avg_deletions",
        "avg_total_changes", "max_changes", "std_changes",
        "avg_hour", "std_hour", "late_night_commits",
        "early_morning_commits", "weekend_commits", "unique_days",
        "total_prs", "merged_prs", "avg_merge_hours",
        "avg_changed_files", "avg_review_comments",
        "merge_rate", "churn_ratio", "off_hours_ratio", "pr_size_score"
    ]

    X = features_df[feature_cols].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Start MLflow experiment
    mlflow.set_experiment("developer_anomaly_detection")

    with mlflow.start_run(run_name="isolation_forest"):
        # Train model
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            max_features=0.8
        )
        model.fit(X_scaled)

        # Get anomaly scores
        anomaly_scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)

        # -1 = anomaly, 1 = normal → convert to 0/1
        is_anomaly = (predictions == -1).astype(int)

        # Add results to dataframe
        results_df = features_df[["author_login"]].copy()
        results_df["anomaly_score"] = anomaly_scores
        results_df["is_anomaly"] = is_anomaly
        results_df["risk_level"] = pd.cut(
            anomaly_scores,
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=["High Risk", "Medium Risk", "Normal"]
        )

        # Log to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", 0.1)
        mlflow.log_param("max_features", 0.8)
        mlflow.log_metric("anomalies_detected", int(is_anomaly.sum()))
        mlflow.log_metric("anomaly_rate", float(is_anomaly.mean()))
        mlflow.sklearn.log_model(model, "isolation_forest_model")

        print(f"\nAnomalies detected: {is_anomaly.sum()} / {len(is_anomaly)}")
        print(f"Anomaly rate: {is_anomaly.mean():.1%}")
        print("\nAnomaly Results:")
        print(results_df[["author_login", "anomaly_score",
                          "is_anomaly", "risk_level"]].to_string())

    return results_df, scaler, feature_cols


# ─────────────────────────────────────────
# MODEL 2 — LSTM AUTOENCODER
# ─────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for temporal anomaly detection"""

    def __init__(self, input_size, hidden_size=32, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Decoder
        self.decoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Repeat hidden state for decoder input
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)

        # Decode
        decoder_output, _ = self.decoder(decoder_input)

        # Reconstruct
        reconstruction = self.output_layer(decoder_output)
        return reconstruction


def prepare_sequences(X_scaled, seq_length=3):
    """Prepare sequences for LSTM"""
    sequences = []
    for i in range(len(X_scaled) - seq_length + 1):
        sequences.append(X_scaled[i:i + seq_length])
    return np.array(sequences)


def train_lstm_autoencoder(features_df, feature_cols):
    """Train LSTM Autoencoder"""
    print("\n" + "="*50)
    print("Training LSTM Autoencoder...")
    print("="*50)

    X = features_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare sequences
    seq_length = 3
    sequences = prepare_sequences(X_scaled, seq_length)
    X_tensor = torch.FloatTensor(sequences)

    input_size = X_scaled.shape[1]

    with mlflow.start_run(run_name="lstm_autoencoder"):
        # Initialize model
        model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=32,
            num_layers=2
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        epochs = 50
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

        # Get reconstruction errors
        model.eval()
        with torch.no_grad():
            reconstructions = model(X_tensor)
            reconstruction_errors = torch.mean(
                (X_tensor - reconstructions) ** 2, dim=[1, 2]
            ).numpy()

        # Threshold = mean + 1.5 std
        threshold = reconstruction_errors.mean() + 1.5 * reconstruction_errors.std()
        is_anomaly_lstm = (reconstruction_errors > threshold).astype(int)

        # Map back to developers
        n_devs = len(features_df)
        lstm_results = features_df[["author_login"]].copy()

        # Pad results to match original developer count
        errors_padded = np.zeros(n_devs)
        anomaly_padded = np.zeros(n_devs)

        for i in range(len(reconstruction_errors)):
            if i < n_devs:
                errors_padded[i] = reconstruction_errors[i]
                anomaly_padded[i] = is_anomaly_lstm[i]

        lstm_results["reconstruction_error"] = errors_padded
        lstm_results["lstm_anomaly"] = anomaly_padded.astype(int)
        lstm_results["lstm_risk"] = lstm_results["reconstruction_error"].apply(
            lambda x: "High Risk" if x > threshold else "Normal"
        )

        # Log to MLflow
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden_size", 32)
        mlflow.log_param("seq_length", seq_length)
        mlflow.log_param("threshold", float(threshold))
        mlflow.log_metric("final_loss", float(losses[-1]))
        mlflow.log_metric("lstm_anomalies", int(anomaly_padded.sum()))

        print(f"\nLSTM Anomaly threshold: {threshold:.6f}")
        print(f"LSTM Anomalies detected: {int(anomaly_padded.sum())}")
        print("\nLSTM Results:")
        print(lstm_results.to_string())

    return lstm_results, model


# ─────────────────────────────────────────
# COMBINE RESULTS
# ─────────────────────────────────────────

def combine_anomaly_results(if_results, lstm_results):
    """Combine both models into final anomaly report"""
    print("\n" + "="*50)
    print("Combining Model Results...")
    print("="*50)

    combined = if_results.merge(lstm_results, on="author_login")

    # Final risk score — flagged by EITHER model
    combined["final_anomaly"] = (
        (combined["is_anomaly"] == 1) |
        (combined["lstm_anomaly"] == 1)
    ).astype(int)

    combined["final_risk"] = combined.apply(
        lambda row: "HIGH RISK" if (
            row["is_anomaly"] == 1 and row["lstm_anomaly"] == 1
        ) else "MEDIUM RISK" if (
            row["is_anomaly"] == 1 or row["lstm_anomaly"] == 1
        ) else "NORMAL",
        axis=1
    )

    os.makedirs("data/processed", exist_ok=True)
    combined.to_csv("data/processed/anomaly_results.csv", index=False)

    print("\nFinal Combined Anomaly Report:")
    print(combined[[
        "author_login", "anomaly_score",
        "reconstruction_error", "final_risk"
    ]].to_string())

    print(f"\nSummary:")
    print(combined["final_risk"].value_counts().to_string())

    return combined


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("Phase 4 — ML Anomaly Detection Pipeline")
    print("="*50)

    # Step 1 - Feature engineering
    features_df = engineer_features()

    # Step 2 - Isolation Forest
    if_results, scaler, feature_cols = train_isolation_forest(features_df)

    # Step 3 - LSTM Autoencoder
    lstm_results, lstm_model = train_lstm_autoencoder(features_df, feature_cols)

    # Step 4 - Combine results
    final_results = combine_anomaly_results(if_results, lstm_results)

    print("\n✅ Phase 4 Complete!")
    print("Results saved to data/processed/anomaly_results.csv")
    print("MLflow experiments logged — run 'mlflow ui' to view")