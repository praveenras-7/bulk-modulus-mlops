"""
Model evaluator for bulk modulus prediction.
Extracted from NOTEBOOK-3.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates trained models for bulk modulus prediction.

    USAGE:
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X_test, y_test)
        evaluator.plot_predictions(y_test, y_pred, "RandomForest")
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.paths_config = self.config["paths"]
        logger.info("ModelEvaluator initialized")

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Evaluate model and return metrics.

        WHY:
        - Consistent evaluation across all models
        - Works on train, val, or test set
        - Logs results clearly

        Metrics explained:
        - R2  : 1.0 = perfect, 0.0 = just predicting mean
        - MAE : Average prediction error in GPa
        - RMSE: Penalizes large errors more than MAE

        Args:
            model       : Trained model
            X           : Feature matrix
            y           : True target values
            dataset_name: Name for logging (train/val/test)

        Returns:
            Dictionary with all metrics
        """

        logger.info(f"Evaluating on {dataset_name} set ({len(y)} samples)")

        y_pred = model.predict(X)

        r2   = r2_score(y, y_pred)
        mae  = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        max_error  = float(np.max(np.abs(y - y_pred)))
        mean_error = float(np.mean(y_pred - y))

        metrics = {
            "dataset"   : dataset_name,
            "r2"        : round(float(r2),   4),
            "mae"       : round(float(mae),  4),
            "rmse"      : round(float(rmse), 4),
            "max_error" : round(max_error,   4),
            "mean_error": round(mean_error,  4),
            "n_samples" : len(y)
        }

        logger.info(f"Results on {dataset_name}:")
        logger.info(f"  R2        : {r2:.4f}")
        logger.info(f"  MAE       : {mae:.4f} GPa")
        logger.info(f"  RMSE      : {rmse:.4f} GPa")
        logger.info(f"  Max Error : {max_error:.4f} GPa")

        return metrics

    def evaluate_all_sets(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate model on all three splits.

        WHY:
        - Compare train vs val vs test performance
        - Detect overfitting:
          train R2 >> test R2 = overfitting!
          train R2 == test R2 = good generalization!

        Returns:
            DataFrame with metrics for all splits
        """

        logger.info("Evaluating on all splits...")

        train_metrics = self.evaluate(model, X_train, y_train, "train")
        val_metrics   = self.evaluate(model, X_val,   y_val,   "val")
        test_metrics  = self.evaluate(model, X_test,  y_test,  "test")

        df = pd.DataFrame([train_metrics, val_metrics, test_metrics])

        # Check for overfitting
        gap = train_metrics["r2"] - test_metrics["r2"]
        if gap > 0.1:
            logger.warning(
                f"Possible overfitting! "
                f"Train R2={train_metrics['r2']:.4f}, "
                f"Test R2={test_metrics['r2']:.4f}, "
                f"Gap={gap:.4f}"
            )
        else:
            logger.info(
                f"Good generalization! "
                f"Train R2={train_metrics['r2']:.4f}, "
                f"Test R2={test_metrics['r2']:.4f}, "
                f"Gap={gap:.4f}"
            )

        return df

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ) -> None:
        """
        Plot predicted vs actual bulk modulus values.

        WHY:
        - Visual check of model performance
        - Perfect model = all points on diagonal line
        - Spread around line = prediction error

        Args:
            y_true    : True bulk modulus values
            y_pred    : Predicted values
            model_name: Name for plot title
            save_path : Where to save the plot
        """

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Predicted vs Actual
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--", linewidth=2,
            label="Perfect prediction"
        )

        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        ax1.set_xlabel("Actual Bulk Modulus (GPa)")
        ax1.set_ylabel("Predicted Bulk Modulus (GPa)")
        ax1.set_title(f"{model_name}\nR2={r2:.4f} | MAE={mae:.4f} GPa")
        ax1.legend()

        # Plot 2: Residuals
        ax2 = axes[1]
        residuals = y_pred - y_true
        ax2.scatter(y_true, residuals, alpha=0.3, s=10, color="coral")
        ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel("Actual Bulk Modulus (GPa)")
        ax2.set_ylabel("Residual (Predicted - Actual)")
        ax2.set_title(f"{model_name} - Residuals")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()
        plt.close()

    def compare_models(self, results_df: pd.DataFrame) -> None:
        """
        Plot comparison of all trained models.

        WHY:
        - Visual comparison of all models
        - Same as comparison plots in NOTEBOOK-3
        - Easy to see which model is best

        Args:
            results_df: DataFrame from trainer.train_all_models()
        """

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ["r2",   "mae",  "rmse"]
        titles  = [
            "R2 Score (higher is better)",
            "MAE in GPa (lower is better)",
            "RMSE in GPa (lower is better)"
        ]
        colors = ["steelblue", "coral", "mediumseagreen"]

        for ax, metric, title, color in zip(axes, metrics, titles, colors):
            ax.barh(
                results_df["algorithm"],
                results_df[metric],
                color=color,
                alpha=0.8
            )
            ax.set_title(title)
            ax.set_xlabel(metric.upper())

        plt.suptitle(
            "Model Comparison - Bulk Modulus Prediction",
            fontsize=14
        )
        plt.tight_layout()

        save_dir = Path(self.paths_config.get("reports", "docs/reports"))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "model_comparison.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison plot saved: {save_path}")

        plt.show()
        plt.close()

    def generate_report(
        self,
        model_name: str,
        metrics_df: pd.DataFrame,
        output_path: str = None
    ) -> str:
        """
        Generate a text report of model performance.

        WHY:
        - Document results for portfolio
        - Track improvements over time
        - Share with recruiters

        Args:
            model_name : Name of the model
            metrics_df : DataFrame from evaluate_all_sets()
            output_path: Where to save the report

        Returns:
            Report as string
        """

        lines = [
            "=" * 50,
            "Model Evaluation Report",
            f"Model: {model_name}",
            "=" * 50,
            "",
        ]

        for _, row in metrics_df.iterrows():
            lines.extend([
                f"Dataset  : {row['dataset'].upper()}",
                f"  R2     : {row['r2']:.4f}",
                f"  MAE    : {row['mae']:.4f} GPa",
                f"  RMSE   : {row['rmse']:.4f} GPa",
                f"  Samples: {row['n_samples']}",
                "",
            ])

        report = "\n".join(lines)
        logger.info("\n" + report)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved: {output_path}")

        return report


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print("ModelEvaluator ready!")
    print()
    print("Methods available:")
    print("  - evaluate()          : Evaluate on one dataset")
    print("  - evaluate_all_sets() : Evaluate on train/val/test")
    print("  - plot_predictions()  : Plot predicted vs actual")
    print("  - compare_models()    : Compare all models visually")
    print("  - generate_report()   : Generate text report")
    print()
    print("NOTE: To test, we need a trained model and data.")