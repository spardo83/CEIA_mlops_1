"""DAG de entrenamiento/registro en MLflow alineado con la notebook tp.ipynb.

Usa los CSV procesados por data_treatment_dag (target: occupancy_level) y entrena
modelos de clasificación, logueando métricas y artefactos en MLflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from airflow.decorators import dag, task
from pendulum import datetime


DATA_DIR = Path("/opt/airflow/data/processed")
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
TARGET = "occupancy_level"
ORDERED_LABELS = ["zero", "low", "mid", "high"]
MODEL_NAME = os.getenv("MODEL_NAME", "airbnb-occupancy-classifier")

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["training", "mlflow", "mlops"],
)
def train_pipeline_dag():
    """Run preprocessing, train multiple models, and promote the selected one."""

    @task()
    def ensure_data() -> Dict[str, str]:
        """Ensure processed train/test Parquet files exist before training."""
        if not TRAIN_PATH.exists() or not TEST_PATH.exists():
            raise FileNotFoundError(
                "Processed Parquet files not found. Run data_treatment_dag first."
            )
        return {"train_path": str(TRAIN_PATH), "test_path": str(TEST_PATH)}

    @task()
    def prepare_features(paths: Dict[str, str]) -> Dict[str, str]:
        """Centralized feature engineering: load data, preprocess, and save matrices."""
        import pandas as pd
        import numpy as np
        from scipy import sparse
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        import joblib

        train_df = pd.read_parquet(paths["train_path"])
        test_df = pd.read_parquet(paths["test_path"])

        if TARGET not in train_df or TARGET not in test_df:
            raise ValueError(f"Target column {TARGET} not found in processed files")

        feature_cols = [c for c in train_df.columns if c != TARGET]
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET]

        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X_train.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

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

        X_train_processed = preprocess.fit_transform(X_train)
        X_test_processed = preprocess.transform(X_test)

        # Ensure sparse format
        if not sparse.issparse(X_train_processed):
            X_train_processed = sparse.csr_matrix(X_train_processed)
        if not sparse.issparse(X_test_processed):
            X_test_processed = sparse.csr_matrix(X_test_processed)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # Save artifacts
        out_dir = Path(paths["train_path"]).parent
        sparse.save_npz(out_dir / "X_train.npz", X_train_processed)
        sparse.save_npz(out_dir / "X_test.npz", X_test_processed)
        np.save(out_dir / "y_train.npy", y_train_enc)
        np.save(out_dir / "y_test.npy", y_test_enc)
        np.save(out_dir / "classes.npy", le.classes_)
        # Save preprocessor
        joblib.dump(preprocess, out_dir / "preprocessor.joblib")

        return {
            "X_train": str(out_dir / "X_train.npz"),
            "X_test": str(out_dir / "X_test.npz"),
            "y_train": str(out_dir / "y_train.npy"),
            "y_test": str(out_dir / "y_test.npy"),
            "classes": str(out_dir / "classes.npy"),
            "preprocessor": str(out_dir / "preprocessor.joblib"),
        }

    @task()
    def train_and_log(model_name: str, paths: Dict[str, str]) -> Dict[str, float]:
        """Train a model using preprocessed data, log metrics/artifacts to MLflow."""
        import numpy as np
        import pandas as pd
        import mlflow
        from scipy import sparse
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
        from sklearn.preprocessing import label_binarize
        from sklearn.neural_network import MLPClassifier
        from sklearn.decomposition import TruncatedSVD
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        experiment_name = "airbnb-occupancy-classifier"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        # Load preprocessed data
        X_train = sparse.load_npz(paths["X_train"])
        X_test = sparse.load_npz(paths["X_test"])
        y_train = np.load(paths["y_train"])
        y_test = np.load(paths["y_test"])
        classes = np.load(paths["classes"], allow_pickle=True)

        models = {
            "logreg": LogisticRegression(max_iter=1000, multi_class="auto"),
            "gboost": GradientBoostingClassifier(
                random_state=42, n_iter_no_change=10, validation_fraction=0.1
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=400,
                random_state=42,
                early_stopping=True,
            ),
            "pca_logreg": Pipeline(
                steps=[
                    ("pca", TruncatedSVD(n_components=50, random_state=42)),
                    ("clf", LogisticRegression(max_iter=1000, multi_class="auto")),
                ]
            ),
        }

        # Ensemble needs instantiated estimators
        if model_name == "ensemble":
            clf1 = LogisticRegression(max_iter=1000, multi_class="auto")
            clf2 = GradientBoostingClassifier(
                random_state=42, n_iter_no_change=10, validation_fraction=0.1
            )
            clf3 = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=400,
                random_state=42,
                early_stopping=True,
            )
            estimator = VotingClassifier(
                estimators=[("lr", clf1), ("gb", clf2), ("mlp", clf3)], voting="soft"
            )
        elif model_name == "simplenn":
            estimator = None
        else:
            estimator = models[model_name]

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            mlflow.log_params({"model_name": model_name})
            proba = None

            if model_name == "simplenn":
                # PyTorch with Early Stopping
                import torch
                from torch import nn
                from torch.utils.data import DataLoader, Dataset, random_split

                class SimpleNN(nn.Module):
                    def __init__(self, in_features, out_features):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(in_features, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, out_features),
                        )

                    def forward(self, x):
                        return self.net(x)

                class SparseDataset(Dataset):
                    def __init__(self, X, y=None):
                        self.X = X
                        self.y = y

                    def __len__(self):
                        return self.X.shape[0]

                    def __getitem__(self, idx):
                        x_data = self.X[idx].toarray().squeeze()
                        x_tensor = torch.tensor(x_data, dtype=torch.float32)
                        if self.y is not None:
                            return x_tensor, torch.tensor(self.y[idx], dtype=torch.long)
                        return x_tensor

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                full_ds = SparseDataset(X_train, y_train)
                # Split for validation (early stopping)
                train_size = int(0.9 * len(full_ds))
                val_size = len(full_ds) - train_size
                train_ds, val_ds = random_split(full_ds, [train_size, val_size])

                train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

                model = SimpleNN(X_train.shape[1], len(classes)).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                best_val_loss = float("inf")
                patience = 5
                trigger_times = 0

                for epoch in range(50):  # Max epochs
                    model.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()

                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            logits = model(xb)
                            loss = criterion(logits, yb)
                            val_loss += loss.item() * xb.size(0)
                    val_loss /= len(val_ds)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        trigger_times = 0
                    else:
                        trigger_times += 1
                        if trigger_times >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

                # Inference on test set
                test_ds = SparseDataset(X_test)
                test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

                all_logits = []
                with torch.no_grad():
                    for xb in test_loader:
                        xb = xb.to(device)
                        logits = model(xb)
                        all_logits.append(logits)

                logits_test = torch.cat(all_logits, dim=0)
                proba = torch.softmax(logits_test, dim=1).cpu().numpy()
                preds_enc = proba.argmax(axis=1)
                preds = preds_enc  # Already encoded

                mlflow.pytorch.log_model(model, "model")

            else:
                # Sklearn
                estimator.fit(X_train, y_train)
                preds = estimator.predict(X_test)
                proba = (
                    estimator.predict_proba(X_test)
                    if hasattr(estimator, "predict_proba")
                    else None
                )

                signature = infer_signature(
                    X_train[:5].toarray(), estimator.predict(X_train[:5])
                )
                mlflow.sklearn.log_model(
                    estimator, "model", signature=signature
                )
            
            # Log preprocessor
            if "preprocessor" in paths and os.path.exists(paths["preprocessor"]):
                mlflow.log_artifact(paths["preprocessor"], artifact_path="preprocessor")

            # Metrics
            acc = accuracy_score(y_test, preds)
            f1_macro = f1_score(y_test, preds, average="macro")

            # Map integers to ordered float values for MAE calculation
            ordered_rank = {label: i for i, label in enumerate(ORDERED_LABELS)}

            # y_test and preds are indices into `classes`.
            # We want to convert them to ranks in ORDERED_LABELS.

            def get_rank(class_idx):
                label = classes[class_idx]
                return ordered_rank.get(label, -1)

            v_get_rank = np.vectorize(get_rank)
            y_test_rank = v_get_rank(y_test)
            preds_rank = v_get_rank(preds)

            mae = mean_absolute_error(y_test_rank, preds_rank)
            mlflow.log_metrics({"accuracy": acc, "f1_macro": f1_macro, "mae": mae})

            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            # We can log the class names
            mlflow.log_dict(
                {"labels": classes.tolist(), "matrix": cm.tolist()},
                f"confusion_matrix_{model_name}.json",
            )

            # ROC curves (one-vs-rest)
            if (
                proba is not None
                and len(classes) > 1
                and proba.shape[1] == len(classes)
            ):
                y_test_bin = label_binarize(y_test, classes=range(len(classes)))
                auc_ovr = roc_auc_score(y_test_bin, proba, multi_class="ovr")
                mlflow.log_metric("roc_auc_ovr", auc_ovr)

        return {
            "model_name": model_name,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "mae": mae,
            "run_id": run_id,
        }

    @task()
    def select_best(results: List[Dict[str, float]]) -> Dict[str, str]:
        """Pick the model with the highest MAE (wait, lowest MAE is better? Previous code picked max MAE?)
        Checking previous code: `best = max(results, key=lambda r: r.get("mae", float("-inf")))`
        MAE should be minimized. The previous code might have been buggy or I misread it.
        Let's assume we want to MINIMIZE MAE.
        Wait, if the user didn't complain about logic, I should stick to it or fix it if obvious.
        Max MAE is definitely wrong. I will fix it to MIN MAE.
        """
        if not results:
            raise ValueError("No training results were provided")
        # Fix: Select MINIMUM MAE
        best = min(results, key=lambda r: r.get("mae", float("inf")))
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

        # Register model
        registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
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

    data_paths = ensure_data()
    features_paths = prepare_features(data_paths)

    train_results = []
    previous_task = features_paths
    for name in ["logreg", "gboost", "mlp", "pca_logreg", "ensemble", "simplenn"]:
        train_task = train_and_log.override(task_id=f"train_{name}")(
            name, features_paths
        )
        previous_task >> train_task
        previous_task = train_task
        train_results.append(train_task)

    best = select_best(train_results)
    promotion = promote_best(best)

    @task()
    def trigger_api_reload(promotion_result: Dict[str, str]):
        """Call the prediction API to reload the model."""
        import requests
        try:
            # We assume internal docker network name 'prediction-api' or 'prediction_api'
            # Service name in docker-compose is prediction-api
            resp = requests.post("http://prediction-api:8000/reload-model", timeout=10)
            resp.raise_for_status()
            print(f"API Reload triggered: {resp.json()}")
        except Exception as e:
            print(f"Warning: Failed to trigger API reload: {e}")
            # Don't fail the DAG, just warn
    
    trigger_api_reload(promotion)


dag = train_pipeline_dag()
