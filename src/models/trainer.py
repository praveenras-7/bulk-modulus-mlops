"""
Model trainer for bulk modulus prediction.
Extracted from NOTEBOOK-3.ipynb
"""

import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class BulkModulusTrainer:
    """
    Trains ML models for bulk modulus prediction.

    USAGE:
        trainer = BulkModulusTrainer()
        model, metrics = trainer.train(X_train, y_train, X_val, y_val)
        trainer.save_model(model, "random_forest_v1")
    """

    # All models from NOTEBOOK-3
    MODELS = {
        "dummy"            : DummyRegressor,
        "ridge"            : Ridge,
        "random_forest"    : RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "adaboost"         : AdaBoostRegressor,
        "extra_trees"      : ExtraTreesRegressor,
        "knn"              : KNeighborsRegressor,
        "svr"              : SVR,
        "xgboost"          : XGBRegressor,
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.model_config = self.config["model"]
        self.paths_config = self.config["paths"]
        self.algorithm = self.model_config["algorithm"]
        self.random_state = self.model_config["random_state"]
        logger.info(f"BulkModulusTrainer initialized")
        logger.info(f"Algorithm   : {self.algorithm}")
        logger.info(f"Random state: {self.random_state}")

    def get_model(self, algorithm: Optional[str] = None) -> Any:
        """
        Get model instance by name.

        WHY:
        - Create any model just by name from config
        - Hyperparameters loaded from config.yaml
        - Easy to switch algorithms

        Args:
            algorithm: Model name. Uses config default if None.

        Returns:
            Untrained model instance
        """
        algorithm = algorithm or self.algorithm

        if algorithm not in self.MODELS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}\n"
                f"Available: {list(self.MODELS.keys())}"
            )

        # Get hyperparameters from config
        hyperparams = (
            self.model_config
            .get("hyperparameters", {})
            .get(algorithm, {})
        )

        # Add random_state if model supports it
        if "random_state" in self.MODELS[algorithm].__init__.__code__.co_varnames:
            hyperparams["random_state"] = self.random_state

        logger.info(f"Creating model: {algorithm}")
        logger.info(f"Hyperparameters: {hyperparams}")

        return self.MODELS[algorithm](**hyperparams)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        algorithm: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """
        Train a model and evaluate on validation set.

        WHY:
        - Single function for complete training cycle
        - Measures training time automatically
        - Evaluates immediately after training
        - Logs all results clearly

        Args:
            X_train  : Training features
            y_train  : Training targets
            X_val    : Validation features
            y_val    : Validation targets
            algorithm: Model name. Uses config default if None.

        Returns:
            model  : Trained model
            metrics: r2, mae, rmse, training time
        """
        algorithm = algorithm or self.algorithm

        logger.info(f"Starting training : {algorithm}")
        logger.info(f"Training samples  : {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Number of features: {X_train.shape[1]}")

        model = self.get_model(algorithm)

        # Train and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        logger.info(f"Training complete in {elapsed:.2f} seconds")

        # Evaluate on validation set
        metrics = self.evaluate(model, X_val, y_val)
        metrics["train_time"] = round(elapsed, 2)
        metrics["algorithm"] = algorithm
        metrics["n_train_samples"] = len(X_train)
        metrics["n_features"] = X_train.shape[1]

        logger.info("Validation Results:")
        logger.info(f"  R2   : {metrics['r2']:.4f}")
        logger.info(f"  MAE  : {metrics['mae']:.4f} GPa")
        logger.info(f"  RMSE : {metrics['rmse']:.4f} GPa")
        logger.info(f"  Time : {elapsed:.2f}s")

        return model, metrics

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Evaluate a trained model.

        Metrics:
        - R2  : 1.0 = perfect, 0.0 = predicting mean
        - MAE : Average error in GPa
        - RMSE: Penalizes large errors more

        Args:
            model: Trained model
            X    : Features
            y    : True targets

        Returns:
            Dictionary with r2, mae, rmse
        """
        y_pred = model.predict(X)

        return {
            "r2"  : round(float(r2_score(y, y_pred)), 4),
            "mae" : round(float(mean_absolute_error(y, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)
        }

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> pd.DataFrame:
        """
        Train and compare ALL models.

        WHY:
        - Replicates your NOTEBOOK-3 model comparison
        - Find best model automatically
        - Results sorted by R2 score

        Args:
            X_train: Training features
            y_train: Training targets
            X_val  : Validation features
            y_val  : Validation targets

        Returns:
            DataFrame with all results sorted by R2
        """
        logger.info(f"Training all {len(self.MODELS)} models...")

        results = []
        for name in self.MODELS:
            try:
                _, metrics = self.train(
                    X_train, y_train,
                    X_val, y_val,
                    algorithm=name
                )
                results.append(metrics)
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                results.append({
                    "algorithm" : name,
                    "r2"        : 0.0,
                    "mae"       : 999.0,
                    "rmse"      : 999.0,
                    "train_time": 0.0
                })

        df = pd.DataFrame(results)
        df = df.sort_values("r2", ascending=False).reset_index(drop=True)

        logger.info(f"Best model: {df.iloc[0]['algorithm']}")
        logger.info(f"Best R2   : {df.iloc[0]['r2']:.4f}")

        return df

    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save trained model to disk.

        WHY:
        - Persist model after training
        - Load later for predictions without retraining
        - Deploy in API

        Args:
            model     : Trained model
            model_name: Name without extension

        Returns:
            Path where model was saved
        """
        path = Path(self.paths_config["models"]) / f"{model_name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved: {path}")
        return str(path)

    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model from disk.

        Args:
            model_path: Path to .pkl file

        Returns:
            Loaded model ready for predictions
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded: {model_path}")
        return model


if __name__ == "__main__":
    trainer = BulkModulusTrainer()
    print("BulkModulusTrainer ready!")
    print()
    print("Available models:")
    for name in trainer.MODELS:
        print(f"  - {name}")
    print()
    print("NOTE: To test training, we need encoded features.")