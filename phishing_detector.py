"""
phishing_detector.py
----------------------------------------------------------------------
🛡️  PHISHING EMAIL DETECTION MODEL  🛡️
----------------------------------------------------------------------
A complete Scikit-learn pipeline that:
  1. Loads an email dataset (text + label)
  2. Extracts hand-crafted features (URLs, suspicious keywords, etc.)
     AND TF-IDF text features
  3. Trains a RandomForest classifier
  4. Reports Accuracy, Precision, Recall, F1 and a styled Confusion Matrix
  5. Lets you test the model live on your own email text

Run:
    python generate_dataset.py     # creates emails_dataset.csv (first time only)
    python phishing_detector.py    # trains + evaluates + interactive demo
----------------------------------------------------------------------
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from feature_engineering import build_feature_dataframe

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

DATASET_PATH = "emails_dataset.csv"
MODEL_PATH = "phishing_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
SCALER_PATH = "feature_scaler.joblib"

PHISH_COLOR = "bold red"
SAFE_COLOR = "bold green"
TITLE_COLOR = "bold cyan"


# ----------------------------------------------------------------------
# Pretty printing helpers (fall back to plain print if 'rich' missing)
# ----------------------------------------------------------------------
def banner():
    text = r"""
██████╗ ██╗  ██╗██╗███████╗██╗  ██╗██╗███╗   ██╗ ██████╗
██╔══██╗██║  ██║██║██╔════╝██║  ██║██║████╗  ██║██╔════╝
██████╔╝███████║██║███████╗███████║██║██╔██╗ ██║██║  ███╗
██╔═══╝ ██╔══██║██║╚════██║██╔══██║██║██║╚██╗██║██║   ██║
██║     ██║  ██║██║███████║██║  ██║██║██║ ╚████║╚██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
        🛡️   E M A I L   D E T E C T I O N   M O D E L   🛡️
"""
    if RICH:
        console.print(Panel.fit(text, style=TITLE_COLOR, border_style="bright_magenta"))
    else:
        print(text)


def step(msg):
    if RICH:
        console.print(f"\n[bold yellow]▶ {msg}[/bold yellow]")
    else:
        print(f"\n>> {msg}")


def success(msg):
    if RICH:
        console.print(f"[bold green]✅ {msg}[/bold green]")
    else:
        print(f"[OK] {msg}")


def info(msg):
    if RICH:
        console.print(f"[bold blue]ℹ {msg}[/bold blue]")
    else:
        print(f"[i] {msg}")


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_data():
    if not os.path.exists(DATASET_PATH):
        info("Dataset not found — generating a synthetic dataset for you...")
        from generate_dataset import build_dataset
        df = build_dataset(n_per_class=300)
        df.to_csv(DATASET_PATH, index=False)
    else:
        df = pd.read_csv(DATASET_PATH)
    return df


# ----------------------------------------------------------------------
# 2. Feature extraction (TF-IDF + handcrafted, combined)
# ----------------------------------------------------------------------
def build_features(df, vectorizer=None, scaler=None, fit=True):
    texts = df["text"].astype(str)

    # --- TF-IDF text features ---
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=2000, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)

    # --- Hand-crafted features ---
    hand_feats = build_feature_dataframe(texts)
    if fit:
        scaler = StandardScaler()
        hand_scaled = scaler.fit_transform(hand_feats)
    else:
        hand_scaled = scaler.transform(hand_feats)

    combined = hstack([tfidf_matrix, csr_matrix(hand_scaled)])
    return combined, vectorizer, scaler, hand_feats


# ----------------------------------------------------------------------
# 3. Train
# ----------------------------------------------------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------
# 4. Evaluate + styled confusion matrix
# ----------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if RICH:
        table = Table(title="📊 Model Performance", box=box.ROUNDED, border_style="bright_magenta")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Score", style="bold white")
        table.add_row("Accuracy", f"{acc*100:.2f}%")
        table.add_row("Precision", f"{prec*100:.2f}%")
        table.add_row("Recall", f"{rec*100:.2f}%")
        table.add_row("F1-Score", f"{f1*100:.2f}%")
        console.print(table)
    else:
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall:    {rec*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")

    print("\nFull classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    plot_confusion_matrix(cm)
    plot_roc_curve(y_test, y_proba)

    return acc, cm


def plot_confusion_matrix(cm):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,
        linewidths=2,
        linecolor="black",
        xticklabels=["Safe", "Phishing"],
        yticklabels=["Safe", "Phishing"],
        annot_kws={"size": 18, "weight": "bold", "color": "white"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, weight="bold", labelpad=12)
    ax.set_ylabel("True Label", fontsize=12, weight="bold", labelpad=12)
    ax.set_title("Confusion Matrix — Phishing Detection", fontsize=15, weight="bold", pad=18, color="#00FFAA")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=180, facecolor=fig.get_facecolor())
    success("Confusion matrix saved to confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#00FFAA", lw=3, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#00FFAA")
    ax.set_xlabel("False Positive Rate", fontsize=12, weight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, weight="bold")
    ax.set_title("ROC Curve — Phishing Detector", fontsize=15, weight="bold", color="#FF6E6E")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=180, facecolor=fig.get_facecolor())
    success("ROC curve saved to roc_curve.png")
    plt.close()


# ----------------------------------------------------------------------
# 5. Predict on new/raw email text
# ----------------------------------------------------------------------
def predict_email(text, model, vectorizer, scaler):
    df = pd.DataFrame({"text": [text]})
    X, _, _, hand_feats = build_features(df, vectorizer, scaler, fit=False)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = "Phishing" if pred == 1 else "Safe"
    confidence = proba[pred] * 100
    return label, confidence, hand_feats.iloc[0].to_dict()


def print_prediction(text, label, confidence, feats):
    color = PHISH_COLOR if label == "Phishing" else SAFE_COLOR
    icon = "🚨" if label == "Phishing" else "✅"

    if RICH:
        panel_text = (
            f"[{color}]{icon}  RESULT: {label.upper()}[/{color}]\n"
            f"Confidence: [bold]{confidence:.2f}%[/bold]\n\n"
            f"[dim]URLs found: {feats['num_urls']} | "
            f"Suspicious keywords: {feats['num_suspicious_words']} | "
            f"Has IP-based URL: {bool(feats['has_ip_url'])}[/dim]"
        )
        console.print(Panel(panel_text, title="Email Analysis", border_style=color, expand=False))
    else:
        print(f"{icon} RESULT: {label} ({confidence:.2f}% confidence)")
        print(f"   URLs: {feats['num_urls']} | Suspicious words: {feats['num_suspicious_words']}")


# ----------------------------------------------------------------------
# Interactive demo loop
# ----------------------------------------------------------------------
def interactive_demo(model, vectorizer, scaler):
    step("Interactive Demo — paste an email below to classify it (type 'exit' to quit)")
    sample = (
        "Subject: Urgent: Verify Your Account Now\n"
        "Dear customer, we noticed suspicious activity. Click here "
        "http://secure-paypa1-verify.com/login to verify your identity "
        "within 24 hours or your account will be suspended."
    )
    info("Example email being tested automatically first:")
    print(sample)
    label, conf, feats = predict_email(sample, model, vectorizer, scaler)
    print_prediction(sample, label, conf, feats)

    while True:
        try:
            user_text = input("\n📧 Paste email text (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.lower() in ("exit", "quit", ""):
            break
        label, conf, feats = predict_email(user_text, model, vectorizer, scaler)
        print_prediction(user_text, label, conf, feats)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    banner()

    step("Loading dataset...")
    df = load_data()
    success(f"Loaded {len(df)} emails  |  Phishing: {(df['label']==1).sum()}  Safe: {(df['label']==0).sum()}")

    step("Extracting features (TF-IDF + URL/keyword analysis)...")
    X, vectorizer, scaler, _ = build_features(df, fit=True)
    y = df["label"].values
    success(f"Feature matrix shape: {X.shape}")

    step("Splitting train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    success(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    step("Training RandomForest classifier...")
    t0 = time.time()
    model = train_model(X_train, y_train)
    success(f"Model trained in {time.time()-t0:.2f}s")

    step("Evaluating model performance...")
    evaluate_model(model, X_test, y_test)

    step("Saving model artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    success(f"Saved: {MODEL_PATH}, {VECTORIZER_PATH}, {SCALER_PATH}")

    interactive_demo(model, vectorizer, scaler)

    if RICH:
        console.print(Panel.fit("🎉 Done! Thanks for using the Phishing Detector.", style="bold magenta"))
    else:
        print("\nDone!")


if __name__ == "__main__":
    main()
