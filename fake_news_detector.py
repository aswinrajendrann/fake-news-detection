"""
Fake News Detection System
----------------------------
A binary text classifier that distinguishes real news from fake news
using TF-IDF vectorization and classic ML models.

Dataset: Kaggle "Fake and Real News Dataset"
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Setup:
1. Download Fake.csv and True.csv from the Kaggle link above
2. Place both files in the same folder as this script
3. Run: python fake_news_detector.py
"""

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_data(fake_path="Fake.csv", true_path="True.csv"):
    """Load and label the two CSV files into one dataframe."""
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0  # 0 = fake
    true_df["label"] = 1  # 1 = real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Combine title + text for a richer feature set
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["content", "label"]]


def build_pipeline(use_feature_selection=True, k_best=5000):
    """
    Build the preprocessing + model pipeline.

    TF-IDF handles tokenization + vectorization in one step.
    SelectKBest (chi2) trims the feature space to the most informative
    terms only — this is the preprocessing optimization that reduces
    inference time.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,      # ignore overly common terms
        min_df=5,        # ignore overly rare terms
        ngram_range=(1, 1)
    )

    selector = SelectKBest(chi2, k=k_best) if use_feature_selection else None

    return vectorizer, selector


def main():
    print("Loading data...")
    df = load_data()
    print(f"Total samples: {len(df)}  |  Fake: {(df.label==0).sum()}  Real: {(df.label==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer, selector = build_pipeline(use_feature_selection=True, k_best=5000)

    # ---- Baseline: full TF-IDF, no feature selection ----
    print("\n--- Baseline (no feature selection) ---")
    vec_baseline = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=5)
    Xtr_base = vec_baseline.fit_transform(X_train)
    Xte_base = vec_baseline.transform(X_test)

    model_base = LogisticRegression(max_iter=1000)
    model_base.fit(Xtr_base, y_train)

    start = time.time()
    preds_base = model_base.predict(Xte_base)
    baseline_inference_time = time.time() - start

    acc_base = accuracy_score(y_test, preds_base)
    f1_base = f1_score(y_test, preds_base)
    print(f"Baseline  -> Accuracy: {acc_base:.4f}  F1: {f1_base:.4f}  Inference time: {baseline_inference_time:.4f}s")
    print(f"Baseline feature count: {Xtr_base.shape[1]}")

    # ---- Optimized: TF-IDF + chi2 feature selection ----
    print("\n--- Optimized (with feature selection) ---")
    Xtr_full = vectorizer.fit_transform(X_train)
    Xte_full = vectorizer.transform(X_test)

    Xtr_sel = selector.fit_transform(Xtr_full, y_train)
    Xte_sel = selector.transform(Xte_full)

    model_opt = LogisticRegression(max_iter=1000)
    model_opt.fit(Xtr_sel, y_train)

    start = time.time()
    preds_opt = model_opt.predict(Xte_sel)
    optimized_inference_time = time.time() - start

    acc_opt = accuracy_score(y_test, preds_opt)
    f1_opt = f1_score(y_test, preds_opt)
    print(f"Optimized -> Accuracy: {acc_opt:.4f}  F1: {f1_opt:.4f}  Inference time: {optimized_inference_time:.4f}s")
    print(f"Optimized feature count: {Xtr_sel.shape[1]}")

    speedup = (1 - optimized_inference_time / baseline_inference_time) * 100
    print(f"\nInference time change vs baseline: {speedup:.1f}%")

    print("\n--- Classification Report (Optimized model) ---")
    print(classification_report(y_test, preds_opt, target_names=["Fake", "Real"]))

    # ---- Try Naive Bayes too for comparison ----
    print("\n--- Naive Bayes (optimized features) ---")
    nb = MultinomialNB()
    nb.fit(Xtr_sel, y_train)
    nb_preds = nb.predict(Xte_sel)
    print(f"Naive Bayes -> Accuracy: {accuracy_score(y_test, nb_preds):.4f}  F1: {f1_score(y_test, nb_preds):.4f}")


if __name__ == "__main__":
    main()
