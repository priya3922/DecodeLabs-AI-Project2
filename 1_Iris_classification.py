"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit - Batch 2026

Goal: Build a classification model on the Iris dataset using KNN.
Follows the IPO framework from the slides:
  INPUT   -> Iris dataset, feature scaling
  PROCESS -> Train-test split, KNN algorithm
  OUTPUT  -> Confusion matrix, F1 score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# STEP 1: INPUT — Load and understand the dataset
# ---------------------------------------------------------------------------
iris = load_iris()
X = iris.data                     # 150 samples x 4 features
y = iris.target                   # 150 labels (0, 1, 2)
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)
print(f"Samples: {df.shape[0]}, Features: {X.shape[1]}, Classes: {len(target_names)}")
print("\nClass distribution:")
print(df["species"].value_counts())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# ---------------------------------------------------------------------------
# STEP 2: PROCESS (part A) — Train-test split
#   Shuffle first (train_test_split does this by default) to remove order bias.
#   stratify=y keeps the 3 classes balanced in both sets (matches "Balanced" note).
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 80/20 split, as shown in "THE FULL ARCHITECTURE" slide
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n" + "=" * 60)
print("STEP 2: TRAIN-TEST SPLIT")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# STEP 2 (part B) — Feature scaling ("The Gatekeeper Rule")
#   Fit scaler ONLY on training data, then apply to both sets.
#   This avoids leaking test-set information into training.
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaling applied (StandardScaler: mean=0, variance=1)")
print(f"Train mean (approx): {X_train_scaled.mean(axis=0).round(2)}")
print(f"Train std  (approx): {X_train_scaled.std(axis=0).round(2)}")

# ---------------------------------------------------------------------------
# STEP 3: Tuning the engine — choose the best K by testing a range
#   and picking the "elbow" (lowest error rate).
# ---------------------------------------------------------------------------
k_range = range(1, 21)
error_rates = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(np.mean(pred_k != y_test))

# Per the slides, K=1 is prone to overfitting/noise, so we pick the smallest
# error rate among K >= 3 (a more stable choice than the raw minimum).
candidates = [(k, err) for k, err in zip(k_range, error_rates) if k >= 3]
best_k = min(candidates, key=lambda pair: pair[1])[0]

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), error_rates, marker="o", linestyle="--", color="steelblue")
plt.scatter([best_k], [error_rates[best_k - 1]], color="orangered", s=120, zorder=5, label=f"Best K = {best_k}")
plt.title("Choosing K: Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.xticks(list(k_range))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/k_tuning.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("STEP 3: CHOOSING K")
print("=" * 60)
print(f"Best K found: {best_k} (lowest error rate = {error_rates[best_k - 1]:.4f})")

# ---------------------------------------------------------------------------
# STEP 4: PROCESS — Train final KNN model with best K
#   (INSTANTIATE -> FIT -> PREDICT, as in "THE WORKFLOW: SCIKIT-LEARN" slide)
# ---------------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# ---------------------------------------------------------------------------
# STEP 5: OUTPUT — Validation (Confusion Matrix, F1 Score, Accuracy)
#   Accuracy alone can be a "mirage" on imbalanced data, so we report
#   precision, recall, and F1 too (Iris is balanced, but this is best practice).
# ---------------------------------------------------------------------------
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, target_names=target_names)

print("\n" + "=" * 60)
print("STEP 5: OUTPUT VALIDATION")
print("=" * 60)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names
)
plt.title(f"Confusion Matrix (K={best_k}, Accuracy={acc:.2%})")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.savefig("/home/claude/confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved plots: k_tuning.png, confusion_matrix.png")
print("\nDone. Model ready.")
