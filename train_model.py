"""
train_model.py
--------------
Run this in the terminal to see the full training result without the website:
    python train_model.py
It prints a comparison table and saves charts into the 'outputs' folder
(useful as figures for the project report).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ml_pipeline as ml

os.makedirs("outputs", exist_ok=True)
df = ml.load_data()
print("Dataset shape:", df.shape)
print(df[ml.TARGET].value_counts(), "\n")

fitted, results, best, _ = ml.train_and_evaluate(df)

print(f"{'Model':22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'CV Acc':>9}")
for n, r in results.items():
    print(f"{n:22}{r['accuracy']:10.3f}{r['precision']:11.3f}{r['recall']:9.3f}{r['f1']:8.3f}{r['cv_accuracy']:9.3f}")
print("\nBest model:", best)

# Chart 1: confusion matrix of the best model
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(results[best]["confusion"], cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(ml.LABELS)
ax.set_yticks(range(3)); ax.set_yticklabels(ml.LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix - {best}")
for i in range(3):
    for j in range(3):
        ax.text(j, i, results[best]["confusion"][i, j], ha="center", va="center")
fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png", dpi=150)

# Chart 2: model comparison
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(results.keys(), [r["accuracy"] for r in results.values()], color="#4C78A8")
ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Model Comparison")
fig.tight_layout(); fig.savefig("outputs/model_comparison.png", dpi=150)

# Chart 3: feature importance (Random Forest)
imp = ml.feature_importance(fitted["Random Forest"])
fig, ax = plt.subplots(figsize=(6, 4))
imp.sort_values().plot.barh(ax=ax, color="#F58518")
ax.set_title("Feature Importance (Random Forest)")
fig.tight_layout(); fig.savefig("outputs/feature_importance.png", dpi=150)
print("Charts saved in outputs/")
