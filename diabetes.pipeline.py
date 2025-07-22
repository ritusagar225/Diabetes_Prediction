import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import sklearn

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv("C:\\Users\\saura\\Downloads\\diabetes.csv")

# -----------------------------------------------------------
# 2. IMPUTE MISSING VALUES
# -----------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_nan] = df[zero_as_nan].replace(0, np.nan)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

# -----------------------------------------------------------
# 3. VISUAL EXPLORATION
# -----------------------------------------------------------
def visual_exploration(df: pd.DataFrame) -> None:
    sns.countplot(x="Outcome", data=df, hue="Outcome", palette="Set2", legend=False)
    plt.title("Outcome Count (0 = Non‑Diabetic | 1 = Diabetic)")
    plt.show()

    df_outcome1 = df[df["Outcome"] == 1]
    for col in df.columns[:-1]:
        plt.figure()
        plt.hist(df_outcome1[col], bins=20, edgecolor="black", color="green")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True, ls=":")
        plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="viridis")
    plt.title("Correlation Heat‑map")
    plt.tight_layout()
    plt.show()

    scatter_pairs = [("Glucose", "BMI"), ("Age", "BloodPressure"), ("Insulin", "BMI")]
    for x_col, y_col in scatter_pairs:
        plt.figure()
        plt.scatter(df[x_col], df[y_col], c=df["Outcome"], cmap="coolwarm", alpha=0.7)
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, ls=":")
        plt.show()

    sns.pairplot(df, hue="Outcome", palette="husl", diag_kind="kde")
    plt.suptitle("Pair Plot of Diabetes Dataset (Colored by Outcome)", y=1.02)
    plt.show()

# -----------------------------------------------------------
# 4. PREPROCESS DATA WITH SMOTE
# -----------------------------------------------------------
def preprocess(df: pd.DataFrame):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return (X_train_resampled, X_test, y_train_resampled, y_test), scaler

# -----------------------------------------------------------
# 5. DEFINE MODELS
# -----------------------------------------------------------
def define_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "K‑Nearest Neighbors": GridSearchCV(
            KNeighborsClassifier(),
            param_grid={"n_neighbors": list(range(1, 31))},
            cv=5,
            n_jobs=-1,
            scoring="f1",
        ),
        "Support Vector Machine": GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid={"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]},
            cv=5,
            n_jobs=-1,
            scoring="f1",
        ),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid={"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10]},
            cv=5,
            n_jobs=-1,
            scoring="f1",
        ),
    }

# -----------------------------------------------------------
# 6. TRAIN & EVALUATE
# -----------------------------------------------------------
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
            }
        )
    return pd.DataFrame(results).round(4)

# -----------------------------------------------------------
# 7. COMPARE MODELS
# -----------------------------------------------------------
def plot_comparison(results_df: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(results_df))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for idx, m in enumerate(metrics):
        plt.bar(x + idx * width, results_df[m], width, label=m)

    plt.xticks(x + width * 1.5, results_df["Model"], rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# 8. BEST MODEL DETAILS
# -----------------------------------------------------------
def best_model_report(models, results_df, X_test, y_test):
    best_name = results_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]
    y_best = best_model.predict(X_test)

    print(f"\n>>> Best Model: {best_name}\n")
    print(classification_report(y_test, y_best))

    cm = confusion_matrix(y_test, y_best)
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(f"Confusion Matrix – {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Diabetes Classification Pipeline")
    parser.add_argument("--data", type=Path, default=Path("diabetes.csv"), help="Path to the diabetes CSV file")
    parser.add_argument("--skip-plots", action="store_true", help="Skip exploratory plots to save time")
    args = parser.parse_args()

    df = load_data(args.data)
    df = impute_missing(df)

    if not args.skip_plots:
        visual_exploration(df)

    (X_train, X_test, y_train, y_test), scaler = preprocess(df)

    models = define_models()
    results_df = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\n=== Evaluation Summary ===")
    print(results_df)

    plot_comparison(results_df)
    best_model_report(models, results_df, X_test, y_test)

    best_name = results_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]

    if hasattr(best_model, "best_estimator_"):
        best_model_to_save = best_model.best_estimator_
    else:
        best_model_to_save = best_model

    joblib.dump({"scaler": scaler, "model": best_model_to_save, "sklearn_version": sklearn.__version__}, "best_model.joblib")
    print("\nArtifacts saved to best_model.joblib")

if __name__ == "__main__":
    main()
