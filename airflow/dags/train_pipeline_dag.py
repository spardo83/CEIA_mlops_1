"""DAG de entrenamiento/registro en MLflow alineado con la notebook tp.ipynb.

Usa los CSV procesados por data_treatment_dag (target: occupancy_level) y entrena
modelos de clasificación, logueando métricas y artefactos en MLflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime


DATA_DIR = Path("/opt/airflow/data/processed")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TARGET = "occupancy_level"
ORDERED_LABELS = ["zero", "low", "mid", "high"]
MODEL_NAME = os.getenv("MODEL_NAME", "airbnb-occupancy-classifier")

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@dag(schedule=None, start_date=datetime(2025, 1, 1), catchup=False, tags=["training", "mlflow", "mlops"])
def train_pipeline_dag():
    """Run preprocessing, train multiple models, and promote the selected one."""
    @task()
    def ensure_data() -> Dict[str, str]:
        """Ensure processed train/test CSV files exist before training."""
        if not TRAIN_PATH.exists() or not TEST_PATH.exists():
            raise FileNotFoundError("Processed CSV files not found. Run data_treatment_dag first.")
        return {"train_path": str(TRAIN_PATH), "test_path": str(TEST_PATH)}

    @task()
    def train_and_log(model_name: str, paths: Dict[str, str]) -> Dict[str, float]:
        """Train a model, log metrics/artifacts to MLflow, and return summary metrics."""
        import pandas as pd
        import mlflow
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.decomposition import PCA
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        experiment_name = "airbnb-occupancy-classifier"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        train_df = pd.read_csv(paths["train_path"])
        test_df = pd.read_csv(paths["test_path"])

        if TARGET not in train_df or TARGET not in test_df:
            raise ValueError(f"Target column {TARGET} not found in processed CSVs")

        feature_cols = [c for c in train_df.columns if c != TARGET]
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET]

        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ]
        )

        models = {
            "logreg": LogisticRegression(max_iter=1000, multi_class="auto"),
            "gboost": GradientBoostingClassifier(random_state=42),
            "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42),
            "pca_logreg": Pipeline(steps=[
                ("pca", PCA(n_components=0.95, random_state=42, svd_solver="full")),
                ("clf", LogisticRegression(max_iter=1000, multi_class="auto"))
            ]),
        }

        # Ensemble needs instantiated estimators
        if model_name == "ensemble":
            clf1 = LogisticRegression(max_iter=1000, multi_class="auto")
            clf2 = GradientBoostingClassifier(random_state=42)
            clf3 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
            estimator = VotingClassifier(
                estimators=[('lr', clf1), ('gb', clf2), ('mlp', clf3)],
                voting='soft'
            )
        elif model_name == "simplenn":
            # Special case for PyTorch
            estimator = None
        else:
            estimator = models[model_name]

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            mlflow.log_params({"model_name": model_name})
            proba = None
            
            if model_name == "simplenn":
                # PyTorch
                import torch
                from torch import nn
                from torch.utils.data import DataLoader, TensorDataset
                from sklearn.preprocessing import LabelEncoder

                class SimpleNN(nn.Module):
                    def __init__(self, in_features, out_features):
                        """Simple feed-forward network for multi-class classification."""
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(in_features, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, out_features)
                        )

                    def forward(self, x):
                        return self.net(x)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                X_train_t = torch.tensor(preprocess.fit_transform(X_train).astype('float32'))
                X_test_t = torch.tensor(preprocess.transform(X_test).astype('float32'))
                
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                y_test_enc = le.transform(y_test)
                
                y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
                
                train_ds = TensorDataset(X_train_t, y_train_t)
                train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
                
                model = SimpleNN(X_train_t.shape[1], len(le.classes_)).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                model.train()
                for epoch in range(20):
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    logits_test = model(X_test_t.to(device))
                    proba = torch.softmax(logits_test, dim=1).cpu().numpy()
                    preds_enc = proba.argmax(axis=1)
                    preds = le.inverse_transform(preds_enc)

                mlflow.pytorch.log_model(model, "model")
                
            else:
                # Sklearn
                pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
                
                signature = infer_signature(X_train.head(), pipeline.predict(X_train.head()))
                mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature)

            acc = accuracy_score(y_test, preds)
            f1_macro = f1_score(y_test, preds, average="macro")
            labels_order: List[str] = [l for l in ORDERED_LABELS if l in set(y_train)] + [
                l for l in sorted(pd.unique(y_train)) if l not in ORDERED_LABELS
            ]
            label_to_ord = {label: idx for idx, label in enumerate(labels_order)}
            y_test_ord = pd.Series(y_test).map(label_to_ord).astype(float)
            preds_ord = pd.Series(preds).map(label_to_ord).astype(float)
            mae = mean_absolute_error(y_test_ord, preds_ord)
            mlflow.log_metrics({"accuracy": acc, "f1_macro": f1_macro, "mae": mae})

            # Confusion matrix 
            cm = confusion_matrix(y_test, preds, labels=labels_order)
            mlflow.log_dict(
                {"labels": labels_order, "matrix": cm.tolist()},
                f"confusion_matrix_{model_name}.json",
            )

            # ROC curves data (one-vs-rest) when probabilities are available
            if proba is not None and len(labels_order) > 1 and proba.shape[1] == len(labels_order):
                y_test_bin = label_binarize(y_test, classes=labels_order)
                auc_ovr = roc_auc_score(y_test_bin, proba, multi_class="ovr")
                mlflow.log_metric("roc_auc_ovr", auc_ovr)

                roc_data = {}
                for idx, label in enumerate(labels_order):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, idx], proba[:, idx])
                    roc_data[str(label)] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                mlflow.log_dict(roc_data, f"roc_curve_{model_name}.json")

        return {
            "model_name": model_name,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "mae": mae,
            "run_id": run_id,
        }

    @task()
    def select_best(results: List[Dict[str, float]]) -> Dict[str, str]:
        """Pick the model with the highest MAE and return its identifiers."""
        if not results:
            raise ValueError("No training results were provided")
        best = max(results, key=lambda r: r.get("mae", float("-inf")))
        return {
            "best_model": best["model_name"],
            "mae": best["mae"],
            "run_id": best["run_id"],
        }

    @task()
    def promote_best(best: Dict[str, str]) -> Dict[str, str]:
        """Register the selected model and transition it to Production."""
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        client = MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)
        model_uri = f"runs:/{best['run_id']}/model"
        registered = client.register_model(model_uri=model_uri, name=MODEL_NAME)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        return {
            "promoted_model": MODEL_NAME,
            "version": registered.version,
            "mae": best["mae"],
        }

    run_preprocess = TriggerDagRunOperator(
        task_id="run_data_treatment",
        trigger_dag_id="data_treatment_dag",
        wait_for_completion=True,
    )

    data_paths = ensure_data()
    train_results = []
    for name in ["logreg", "gboost", "mlp", "pca_logreg", "ensemble", "simplenn"]:
        train_task = train_and_log.override(task_id=f"train_{name}")(name, data_paths)
        run_preprocess >> data_paths >> train_task
        train_results.append(train_task)

    best = select_best(train_results)
    promote_best(best)


dag = train_pipeline_dag()
