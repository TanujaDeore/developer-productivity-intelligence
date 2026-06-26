# src/nlp_sentiment.py
# NLP Layer - Developer Sentiment Analysis using DistilBERT

import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import mlflow
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# SENTIMENT LABELS
# ─────────────────────────────────────────

LABELS = ["confident", "collaborative", "frustrated", "blocked", "neutral"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# ─────────────────────────────────────────
# CREATE TRAINING DATA
# ─────────────────────────────────────────

def create_training_data():
    """Create labeled training dataset for developer sentiment"""
    print("Creating training dataset...")

    training_examples = [
        # CONFIDENT
        ("LGTM, merging this now", "confident"),
        ("This looks great, approved", "confident"),
        ("Perfect implementation, exactly what we needed", "confident"),
        ("Clean solution, no issues found", "confident"),
        ("Approved, well done", "confident"),
        ("This is the right approach, merging", "confident"),
        ("Excellent work, tests pass", "confident"),
        ("Good to go, ship it", "confident"),
        ("Solid implementation, approved", "confident"),
        ("No concerns, looks good to merge", "confident"),
        ("Tests are passing, code is clean", "confident"),
        ("Nice fix, this resolves the issue", "confident"),
        ("Well structured, merging now", "confident"),
        ("This is correct, approved", "confident"),
        ("Good catch, fix looks right", "confident"),

        # COLLABORATIVE
        ("What do you think about this approach?", "collaborative"),
        ("Should we consider an alternative here?", "collaborative"),
        ("Open to suggestions on this implementation", "collaborative"),
        ("Would love your feedback on this", "collaborative"),
        ("Any thoughts on how to improve this?", "collaborative"),
        ("Let me know if this direction makes sense", "collaborative"),
        ("Happy to discuss the tradeoffs here", "collaborative"),
        ("Could you review this when you get a chance?", "collaborative"),
        ("Is there a better way to handle this case?", "collaborative"),
        ("Wanted to get your input before merging", "collaborative"),
        ("Feel free to suggest changes", "collaborative"),
        ("Let me know if you see any issues", "collaborative"),
        ("What are your thoughts on this design?", "collaborative"),
        ("Looking for feedback on the architecture here", "collaborative"),
        ("Should we handle the edge case differently?", "collaborative"),

        # FRUSTRATED
        ("I already fixed this in the previous PR", "frustrated"),
        ("This was discussed and decided before", "frustrated"),
        ("We keep going back and forth on this", "frustrated"),
        ("I thought we agreed on this approach", "frustrated"),
        ("This is the third time I'm making this change", "frustrated"),
        ("Why is this being reverted again?", "frustrated"),
        ("This keeps breaking every time we touch it", "frustrated"),
        ("I've addressed this comment multiple times", "frustrated"),
        ("We've had this discussion before", "frustrated"),
        ("This shouldn't need another round of review", "frustrated"),
        ("The same issue keeps coming up", "frustrated"),
        ("I don't understand why this keeps changing", "frustrated"),
        ("This has been blocking for too long", "frustrated"),
        ("We're going in circles here", "frustrated"),
        ("This is getting unnecessarily complicated", "frustrated"),

        # BLOCKED
        ("Not sure how to proceed here", "blocked"),
        ("Waiting for input before I can continue", "blocked"),
        ("Stuck on this, need help", "blocked"),
        ("Can't move forward without clarification", "blocked"),
        ("Unsure about the right approach here", "blocked"),
        ("Need guidance on how to handle this", "blocked"),
        ("Blocked until we resolve the dependency", "blocked"),
        ("Can someone help me understand this?", "blocked"),
        ("I'm not sure this is the right direction", "blocked"),
        ("Waiting on approval to proceed", "blocked"),
        ("Need someone to unblock this", "blocked"),
        ("Not confident about this implementation", "blocked"),
        ("Stuck on the edge case here", "blocked"),
        ("Can't reproduce the issue locally", "blocked"),
        ("Need more context to complete this", "blocked"),

        # NEUTRAL
        ("Updated the tests", "neutral"),
        ("Fixed the typo in the comment", "neutral"),
        ("Refactored as requested", "neutral"),
        ("Added the missing null check", "neutral"),
        ("Updated dependencies", "neutral"),
        ("Removed unused imports", "neutral"),
        ("Renamed variable for clarity", "neutral"),
        ("Updated documentation", "neutral"),
        ("Applied the suggested changes", "neutral"),
        ("Addressed review comments", "neutral"),
        ("Changed the method signature", "neutral"),
        ("Added logging statement", "neutral"),
        ("Updated config file", "neutral"),
        ("Moved file to correct location", "neutral"),
        ("Fixed merge conflict", "neutral"),
    ]

    df = pd.DataFrame(training_examples, columns=["text", "label"])
    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"Training examples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


# ─────────────────────────────────────────
# TOKENIZE DATA
# ─────────────────────────────────────────

def tokenize_data(df, tokenizer):
    """Tokenize training data"""
    dataset = Dataset.from_pandas(df[["text", "label_id"]])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    tokenized.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized


# ─────────────────────────────────────────
# TRAIN BERT MODEL
# ─────────────────────────────────────────

def train_sentiment_model(df):
    """Fine-tune DistilBERT for developer sentiment"""
    print("\n" + "="*50)
    print("Fine-tuning DistilBERT...")
    print("="*50)

    # Load tokenizer and model
    print("Loading DistilBERT model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Tokenize
    tokenized_dataset = tokenize_data(df, tokenizer)

    # Split train/eval
    split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")

    # Training arguments
    os.makedirs("models/sentiment", exist_ok=True)
    training_args = TrainingArguments(
        output_dir="models/sentiment",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="logs/sentiment",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Train
    print("\nTraining started...")
    trainer.train()
    print("Training complete!")

    return model, tokenizer


# ─────────────────────────────────────────
# RUN INFERENCE ON REAL PR COMMENTS
# ─────────────────────────────────────────

def analyze_pr_sentiment(model, tokenizer):
    """Run sentiment analysis on real PR comments"""
    print("\n" + "="*50)
    print("Analyzing real PR comments...")
    print("="*50)

    pr_df = pd.read_csv("data/raw/pull_requests.csv")

    # Use PR titles as proxy for comments (real comments need extra API call)
    texts = pr_df["title"].fillna("").tolist()
    authors = pr_df["author"].tolist()

    model.eval()
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, (text, author) in enumerate(zip(texts, authors)):
        if not text.strip():
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        results.append({
            "author": author,
            "text": text[:100],
            "predicted_sentiment": ID2LABEL[predicted_class],
            "confidence": round(confidence, 3),
            "confident_score": round(probs[0][0].item(), 3),
            "collaborative_score": round(probs[0][1].item(), 3),
            "frustrated_score": round(probs[0][2].item(), 3),
            "blocked_score": round(probs[0][3].item(), 3),
            "neutral_score": round(probs[0][4].item(), 3),
        })

    results_df = pd.DataFrame(results)

    # Summary by author
    sentiment_summary = results_df.groupby("author").agg(
        total_prs=("text", "count"),
        dominant_sentiment=("predicted_sentiment", lambda x: x.mode()[0]),
        avg_confidence=("confidence", "mean"),
        frustrated_count=("predicted_sentiment",
                         lambda x: (x == "frustrated").sum()),
        blocked_count=("predicted_sentiment",
                      lambda x: (x == "blocked").sum()),
        confident_count=("predicted_sentiment",
                        lambda x: (x == "confident").sum()),
    ).reset_index()

    # Risk flag — frustrated or blocked
    sentiment_summary["sentiment_risk"] = (
        sentiment_summary["frustrated_count"] +
        sentiment_summary["blocked_count"]
    ) > 0

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv("data/processed/pr_sentiment.csv", index=False)
    sentiment_summary.to_csv(
        "data/processed/sentiment_summary.csv", index=False
    )

    print(f"\nAnalyzed {len(results_df)} PR titles")
    print("\nSentiment Distribution:")
    print(results_df["predicted_sentiment"].value_counts().to_string())
    print("\nPer-Developer Sentiment Summary:")
    print(sentiment_summary.to_string())

    return results_df, sentiment_summary


# ─────────────────────────────────────────
# LOG TO MLFLOW
# ─────────────────────────────────────────

def log_nlp_results(results_df, sentiment_summary):
    """Log NLP results to MLflow"""
    mlflow.set_experiment("developer_sentiment_analysis")

    with mlflow.start_run(run_name="distilbert_sentiment"):
        sentiment_counts = results_df[
            "predicted_sentiment"
        ].value_counts().to_dict()

        for sentiment, count in sentiment_counts.items():
            mlflow.log_metric(f"count_{sentiment}", count)

        mlflow.log_metric(
            "pct_at_risk",
            sentiment_summary["sentiment_risk"].mean()
        )
        mlflow.log_metric("total_analyzed", len(results_df))
        mlflow.log_param("model", "distilbert-base-uncased")
        mlflow.log_param("num_labels", len(LABELS))
        mlflow.log_param("labels", str(LABELS))

        print("\nResults logged to MLflow")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("Phase 5 — NLP Sentiment Analysis Pipeline")
    print("="*50)

    # Step 1 - Create training data
    training_df = create_training_data()

    # Step 2 - Train model
    model, tokenizer = train_sentiment_model(training_df)

    # Step 3 - Analyze real PR comments
    results_df, sentiment_summary = analyze_pr_sentiment(model, tokenizer)

    # Step 4 - Log to MLflow
    log_nlp_results(results_df, sentiment_summary)

    print("\n✅ Phase 5 Complete!")
    print("Sentiment results saved to data/processed/")
    print("MLflow results logged")