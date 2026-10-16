"""
Train, tune and compare three models, track every run in MLflow, and save the winner.

  1. Split the clean data: 80% training, 20% test. The test set stays locked until step 5.
  2. Tune each model with 5-fold cross-validation (CV) on the training set only.
  3. Pick the winner by CV F2 score (recall counts twice as much as precision).
  4. Tune the winner's decision threshold for F2, again on training data only.
  5. Score the final model once on the test set, apply the quality gate, then save it.

Run:       python -m src.train
See runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.features import FEATURES, TARGET, build_preprocessor

plt.switch_backend("Agg")  # draw charts without a screen (needed later in CI)

DATA_PATH = Path("data/processed/heart_disease_clean.csv")
MODEL_DIR = Path("models")
FIG_DIR = Path("report/figures")
EXPERIMENT = "heart-disease-classifier"
RANDOM_STATE = 42
MIN_TEST_ROC_AUC = 0.80  # quality gate: a model below this is never saved

F2 = make_scorer(fbeta_score, beta=2)
SCORING = {"f2": F2, "recall": "recall", "precision": "precision",
           "accuracy": "accuracy", "roc_auc": "roc_auc"}

# Each model, plus the settings to try. GridSearchCV tries every combination (8 per model).
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1, 10],
         "model__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [200, 400],
         "model__max_depth": [None, 5],
         "model__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def compute_metrics(y_true, proba, threshold=0.5) -> dict:
    """Score disease probabilities: 'disease' when probability >= threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f2": fbeta_score(y_true, pred, beta=2),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def evaluation_figure(y_true, proba, threshold, data_label):
    """Confusion matrix + ROC curve, side by side."""
    pred = (proba >= threshold).astype(int)
    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["No disease", "Disease"],
        cmap="Blues", colorbar=False, ax=ax_cm)
    ax_cm.set_title(f"Confusion matrix ({data_label}, threshold {threshold:.2f})")
    RocCurveDisplay.from_predictions(y_true, proba, plot_chance_level=True, ax=ax_roc)
    ax_roc.set_title(f"ROC curve ({data_label})")
    fig.tight_layout()
    return fig


def log_search(name, search, X_train, y_train, cv, data_version) -> dict:
    """Log one model's grid search to the active MLflow run and return its CV scores."""
    res, best = search.cv_results_, search.best_index_

    # One child run per settings combination, so the whole tuning process is visible.
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}-{i:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({f"cv_{m}": res[f"mean_test_{m}"][i] for m in SCORING})

    cv_scores = {f"cv_{m}": float(res[f"mean_test_{m}"][best]) for m in SCORING}
    cv_scores["cv_f2_std"] = float(res["std_test_f2"][best])

    mlflow.set_tag("data_version", data_version)
    mlflow.log_params({"model_type": name, "cv_folds": cv.get_n_splits(),
                       **search.best_params_})
    mlflow.log_metrics(cv_scores)

    # Plots use each fold's held-out predictions, so the test set stays untouched.
    proba = cross_val_predict(search.best_estimator_, X_train, y_train,
                              cv=cv, method="predict_proba")[:, 1]
    fig = evaluation_figure(y_train, proba, 0.5, "training CV")
    mlflow.log_figure(fig, "cv_evaluation.png")
    plt.close(fig)

    mlflow.log_text(pd.DataFrame(res).to_csv(index=False), "cv_results.csv")
    mlflow.sklearn.log_model(search.best_estimator_, name="model", serialization_format="pickle")
    return cv_scores


def save_model(model, metadata) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {MODEL_DIR / 'model.joblib'} (version {metadata['model_version']})")


def main() -> None:
    # Step 1: split once; stratify keeps the same disease ratio in both parts.
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2,
        stratify=df[TARGET], random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    data_version = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()[:10]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)

    # Steps 2-3: tune every model, then pick the best CV F2.
    results = {}
    for name, (estimator, grid) in CANDIDATES.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring=SCORING, refit="f2", cv=cv, n_jobs=-1)
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            cv_scores = log_search(name, search, X_train, y_train, cv, data_version)
        results[name] = (search, cv_scores)
        print(f"{name:20s} CV F2 {cv_scores['cv_f2']:.3f} ±{cv_scores['cv_f2_std']:.3f} | "
              f"recall {cv_scores['cv_recall']:.3f} | ROC-AUC {cv_scores['cv_roc_auc']:.3f}")

    winner = max(results, key=lambda n: results[n][1]["cv_f2"])
    search, cv_scores = results[winner]

    # Step 4: a model normally says "disease" at >= 50% probability.
    # Find the cut-off with the best F2, using CV on the training data only.
    final_model = TunedThresholdClassifierCV(search.best_estimator_, scoring=F2, cv=cv)

    with mlflow.start_run(run_name=f"final-{winner}") as run:
        final_model.fit(X_train, y_train)
        threshold = float(final_model.best_threshold_)

        # Step 5: the test set is used exactly once, here.
        proba = final_model.predict_proba(X_test)[:, 1]
        test_scores = {f"test_{k}": float(v)
                       for k, v in compute_metrics(y_test, proba, threshold).items()}

        mlflow.set_tags({"stage": "final", "data_version": data_version})
        mlflow.log_params({"model_type": winner, "decision_threshold": round(threshold, 3),
                           **search.best_params_})
        mlflow.log_metrics({**cv_scores, **test_scores})
        fig = evaluation_figure(y_test, proba, threshold, "test set")
        mlflow.log_figure(fig, "test_evaluation.png")
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "05_final_model_test.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.sklearn.log_model(final_model, name="model", serialization_format="pickle")

    print(f"\nWinner: {winner} (highest CV F2)")
    print(f"Decision threshold: {threshold:.2f} (default is 0.50)")
    print("Test set: " + " | ".join(
        f"{k.removeprefix('test_')} {v:.3f}" for k, v in test_scores.items()))

    if test_scores["test_roc_auc"] < MIN_TEST_ROC_AUC:
        raise SystemExit(f"Quality gate FAILED: test ROC-AUC {test_scores['test_roc_auc']:.3f} "
                         f"< {MIN_TEST_ROC_AUC}. Model NOT saved.")

    save_model(final_model, {
        "model_version": datetime.now(timezone.utc).strftime("%Y.%m.%d-%H%M"),
        "model_type": winner,
        "decision_threshold": round(threshold, 4),
        "features": FEATURES,
        "best_params": search.best_params_,
        "cv_metrics": {k: round(v, 4) for k, v in cv_scores.items()},
        "test_metrics": {k: round(v, 4) for k, v in test_scores.items()},
        "data_version": data_version,
        "mlflow_run_id": run.info.run_id,
        "sklearn_version": sklearn.__version__,
    })


if __name__ == "__main__":
    main()