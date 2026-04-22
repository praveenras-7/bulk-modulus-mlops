"""
CBFV Feature Encoder for bulk modulus prediction.

WHY THIS FILE EXISTS:
- Converts chemical compositions to numerical features
- Machine learning models need numbers, not text
- CBFV = Composition Based Feature Vector
- Extracted from NOTEBOOK-3.ipynb

HOW IT WORKS:
    Input : Fe2O3 (chemical formula string)
    Output: [15.2, 31.9, 2.47, ...] (feature vector)

STEPS:
    1. Parse composition: Fe2O3 -> {Fe:2, O:3}
    2. Get element properties: Fe -> [26, 55.8, 1.83, ...]
    3. Weighted average by composition fractions
    4. Return feature vector
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from CBFV import composition as cbfv_composition

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class CBFVEncoder:
    """
    Encodes chemical compositions as feature vectors.

    WHY A CLASS:
    - Keeps all CBFV logic in one place
    - Easy to reuse across training and inference
    - Can be tested independently

    USAGE:
        encoder = CBFVEncoder()
        X_train, y_train, formulae, skipped = encoder.encode(df_train)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize CBFVEncoder.

        Args:
            config_path: Path to config.yaml
        """
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        self.feature_config = self.config['features']
        self.target_col = self.data_config['target_column']
        self.composition_col = self.data_config['composition_column']

        logger.info("CBFVEncoder initialized")
        logger.info(f"Target column     : {self.target_col}")
        logger.info(f"Composition column: {self.composition_col}")

    def encode(
        self,
        df: pd.DataFrame,
        elem_prop: str = "oliynyk"
    ) -> Tuple[pd.DataFrame, np.ndarray, list, list]:
        """
        Encode compositions in a DataFrame to feature vectors.

        WHY THIS FUNCTION:
        - Single function to convert any DataFrame
        - Handles errors for invalid compositions
        - Logs progress
        - Returns both features AND targets

        Args:
            df       : DataFrame with composition and target columns
            elem_prop: Element property set to use
                       oliynyk is best for bulk modulus prediction

        Returns:
            X       : Feature matrix
            y       : Target array
            formulae: List of composition strings
            skipped : List of compositions that failed
        """

        logger.info(f"Encoding {len(df)} compositions...")
        logger.info(f"Using element properties: {elem_prop}")

        # CBFV needs columns named formula and target
        df_cbfv = df.rename(columns={
            self.composition_col: "formula",
            self.target_col: "target"
        })

        df_cbfv = df_cbfv[["formula", "target"]].copy()

        # Core CBFV encoding - this is from your NOTEBOOK-3
        X, y, formulae, skipped = cbfv_composition.generate_features(
            df_cbfv,
            elem_prop=elem_prop,
            drop_duplicates=False,
            extend_features=True,
            sum_feat=True
        )

        logger.info(f"Encoding complete")
        logger.info(f"Feature matrix shape : {X.shape}")
        logger.info(f"Target array shape   : {y.shape}")
        logger.info(f"Skipped compositions : {len(skipped)}")

        if len(skipped) > 0:
            logger.warning(f"Could not encode {len(skipped)} compositions")

        return X, y, formulae, skipped

    def encode_single(
        self,
        composition: str,
        elem_prop: str = "oliynyk"
    ) -> Optional[np.ndarray]:
        """
        Encode a single composition string.

        WHY:
        - Used in the API for single predictions
        - Someone sends Fe2O3 and we return a prediction

        Args:
            composition: Chemical formula string
            elem_prop  : Element property set

        Returns:
            Feature vector or None if encoding fails
        """

        logger.info(f"Encoding single composition: {composition}")

        df_single = pd.DataFrame({
            "formula": [composition],
            "target": [0.0]
        })

        try:
            X, _, _, skipped = cbfv_composition.generate_features(
                df_single,
                elem_prop=elem_prop,
                drop_duplicates=False,
                extend_features=True,
                sum_feat=True
            )

            if len(skipped) > 0:
                logger.error(f"Could not encode: {composition}")
                return None

            logger.info(f"Encoded to {X.shape[1]} features")
            return X.values[0]

        except Exception as e:
            logger.error(f"Encoding failed for {composition}: {str(e)}")
            return None

    def encode_all_splits(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        elem_prop: str = "oliynyk"
    ) -> dict:
        """
        Encode all three data splits at once.

        WHY:
        - Convenient wrapper to encode train/val/test together
        - Used in the main training pipeline
        - Returns everything needed for training

        Args:
            df_train : Training DataFrame
            df_val   : Validation DataFrame
            df_test  : Test DataFrame
            elem_prop: Element property set

        Returns:
            Dictionary with all encoded splits
        """

        logger.info("Encoding all data splits...")

        X_train, y_train, train_formulae, _ = self.encode(df_train, elem_prop)
        X_val,   y_val,   val_formulae,   _ = self.encode(df_val,   elem_prop)
        X_test,  y_test,  test_formulae,  _ = self.encode(df_test,  elem_prop)

        logger.info("All splits encoded successfully")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape  : {X_val.shape}")
        logger.info(f"X_test shape : {X_test.shape}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val"  : X_val,   "y_val"  : y_val,
            "X_test" : X_test,  "y_test" : y_test,
            "train_formulae": train_formulae,
            "val_formulae"  : val_formulae,
            "test_formulae" : test_formulae
        }


if __name__ == "__main__":
    print("Testing CBFVEncoder...")
    print("-" * 40)

    encoder = CBFVEncoder()

    print("CBFVEncoder created successfully!")
    print()
    print("Config loaded:")
    print(f"  Target col     : {encoder.target_col}")
    print(f"  Composition col: {encoder.composition_col}")
    print()
    print("NOTE: To fully test encoding, we need data files.")
    print("CBFVEncoder is ready!")
