"""
Configuration loader for the bulk modulus MLOps project.

WHY THIS FILE EXISTS:
- config.yaml has all our settings
- This file reads config.yaml and makes settings
  available as Python objects
- Every other module imports this to get settings

HOW IT WORKS:
    config = load_config()
    config['data']['raw_path']   -> data/raw/bulk-modulus.csv
    config['data']['test_size']  -> 0.2
    config['model']['algorithm'] -> random_forest
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Dictionary with all configuration settings
    """

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_data_config(config_path: str = "configs/config.yaml") -> Dict:
    """Get just the data configuration."""
    config = load_config(config_path)
    return config.get('data', {})


def get_model_config(config_path: str = "configs/config.yaml") -> Dict:
    """Get just the model configuration."""
    config = load_config(config_path)
    return config.get('model', {})


def get_feature_config(config_path: str = "configs/config.yaml") -> Dict:
    """Get just the feature configuration."""
    config = load_config(config_path)
    return config.get('features', {})


if __name__ == "__main__":
    print("Testing config loader...")
    print("-" * 40)

    config = load_config()

    print("Project :", config['project']['name'])
    print("Version :", config['project']['version'])
    print()
    print("Data settings:")
    print("  Raw path   :", config['data']['raw_path'])
    print("  Test size  :", config['data']['test_size'])
    print("  Target col :", config['data']['target_column'])
    print()
    print("Model settings:")
    print("  Algorithm  :", config['model']['algorithm'])
    print("  n_estimators:",
          config['model']['hyperparameters']['random_forest']['n_estimators'])
    print()
    print("Config loaded successfully!")
