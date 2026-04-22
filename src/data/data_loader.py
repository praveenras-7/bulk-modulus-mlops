"""
Data loader for the bulk modulus MLOps project.

WHY THIS FILE EXISTS:
- Loads raw data from CSV files
- Handles errors gracefully
- Logs every step for debugging
- Validates data after loading
- Splits data into train/val/test sets

EXTRACTED FROM: NOTEBOOK-2.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class DataLoader:
    """
    Handles all data loading operations.

    WHY A CLASS:
    - Groups all related data functions together
    - Shares config and logger across all methods
    - Easy to test and reuse

    USAGE:
        loader = DataLoader()
        df = loader.load_raw_data()
        df_train, df_val, df_test = loader.split_data(df)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize DataLoader with configuration.

        Args:
            config_path: Path to config.yaml
        """
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        logger.info("DataLoader initialized")

    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw bulk modulus data from CSV file.

        WHY:
        - Single function to load data
        - Uses config path not hardcoded path
        - Validates data after loading
        - Logs what was loaded

        Args:
            file_path: Optional path override.
                      If None, uses path from config.yaml

        Returns:
            DataFrame with raw data
        """

        if file_path is None:
            file_path = self.data_config['raw_path']

        data_file = Path(file_path)
        if not data_file.exists():
            logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please add your bulk-modulus.csv to data/raw/"
            )

        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)

        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        df = self.validate_data(df)

        return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the loaded data.

        WHY:
        - Catches data problems early
        - Better to fail here than during training
        - Logs warnings for issues found

        Checks:
        - Required columns exist
        - No missing values in key columns
        - Target column has positive values

        Args:
            df: DataFrame to validate

        Returns:
            Cleaned DataFrame
        """

        logger.info("Validating data...")

        target_col = self.data_config['target_column']
        composition_col = self.data_config['composition_column']

        # Check required columns exist
        required_cols = [target_col, composition_col]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for missing values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found missing values: {null_counts[null_counts > 0].to_dict()}")
            logger.info("Dropping rows with missing values...")
            df = df.dropna(subset=required_cols)

        # Check target values are positive
        negative_count = (df[target_col] <= 0).sum()
        if negative_count > 0:
            logger.warning(f"Found {negative_count} non-positive bulk modulus values - removing them")
            df = df[df[target_col] > 0]

        logger.info(f"Validation complete. {len(df)} valid rows remaining")
        return df

    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation and test sets.

        WHY 3 SPLITS:
        - train   : Model learns from this (70%)
        - val     : Tune hyperparameters   (10%)
        - test    : Final evaluation only  (20%)

        Using test set during training = CHEATING!
        Test set must stay unseen until final evaluation.

        Args:
            df: Full dataset

        Returns:
            Tuple of (df_train, df_val, df_test)
        """

        test_size = self.data_config['test_size']
        val_size = self.data_config['val_size']
        random_state = self.data_config['random_state']

        logger.info(f"Splitting data: test={test_size}, val={val_size}")
        logger.info(f"Total rows to split: {len(df)}")

        # First: separate test set
        df_temp, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        # Second: separate val from train
        val_size_adjusted = val_size / (1 - test_size)
        df_train, df_val = train_test_split(
            df_temp,
            test_size=val_size_adjusted,
            random_state=random_state
        )

        logger.info(f"Train : {len(df_train)} rows ({len(df_train)/len(df)*100:.1f}%)")
        logger.info(f"Val   : {len(df_val)} rows ({len(df_val)/len(df)*100:.1f}%)")
        logger.info(f"Test  : {len(df_test)} rows ({len(df_test)/len(df)*100:.1f}%)")

        return df_train, df_val, df_test

    def save_splits(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> None:
        """
        Save train/val/test splits to CSV files.

        WHY:
        - Saves splits so we do not re-split every run
        - Ensures same splits used every time
        - Other modules load these directly

        Args:
            df_train: Training data
            df_val  : Validation data
            df_test : Test data
        """

        train_path = self.data_config['train_path']
        val_path = self.data_config['val_path']
        test_path = self.data_config['test_path']

        Path(train_path).parent.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)

        logger.info(f"Saved train : {train_path}")
        logger.info(f"Saved val   : {val_path}")
        logger.info(f"Saved test  : {test_path}")

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load pre-saved train/val/test splits.

        WHY:
        - Faster than re-splitting every time
        - Ensures same splits used consistently

        Returns:
            Tuple of (df_train, df_val, df_test)
        """

        train_path = self.data_config['train_path']
        val_path = self.data_config['val_path']
        test_path = self.data_config['test_path']

        logger.info("Loading pre-saved data splits...")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        logger.info(f"Train : {len(df_train)} rows")
        logger.info(f"Val   : {len(df_val)} rows")
        logger.info(f"Test  : {len(df_test)} rows")

        return df_train, df_val, df_test


if __name__ == "__main__":
    print("Testing DataLoader...")
    print("-" * 40)

    loader = DataLoader()

    print("DataLoader created successfully!")
    print()
    print("Config loaded:")
    print(f"  Raw path  : {loader.data_config['raw_path']}")
    print(f"  Test size : {loader.data_config['test_size']}")
    print(f"  Val size  : {loader.data_config['val_size']}")
    print(f"  Target    : {loader.data_config['target_column']}")
    print(f"  Feature   : {loader.data_config['composition_column']}")
    print()
    print("NOTE: To fully test, add bulk-modulus.csv to data/raw/")
    print("DataLoader is ready!")
