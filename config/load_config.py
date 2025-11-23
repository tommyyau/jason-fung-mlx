#!/usr/bin/env python3
"""
Configuration Loader Utility
────────────────────────────
Provides a unified way to load configuration from training_config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_project_root() -> Path:
    """Get the project root directory (parent of config directory)."""
    # This file is in config/, so parent is project root
    config_dir = Path(__file__).parent
    return config_dir.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        Dictionary containing all configuration values.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "training_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config/training_config.yaml"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot-notation path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "training.learning_rate")
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> config = load_config()
        >>> lr = get_config_value(config, "training.learning_rate")
        >>> lr
        5e-6
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


# Convenience functions for common config access
def get_data_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get data processing configuration."""
    if config is None:
        config = load_config()
    return config.get("data", {})


def get_question_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get question generation configuration."""
    if config is None:
        config = load_config()
    return config.get("question_generation", {})


def get_answer_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get answer generation configuration."""
    if config is None:
        config = load_config()
    return config.get("answer_generation", {})


def get_training_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get training configuration."""
    if config is None:
        config = load_config()
    return config.get("training", {})


if __name__ == "__main__":
    # Test loading config
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
        print(f"\nTraining learning rate: {get_config_value(config, 'training.learning_rate')}")
        print(f"Question generation max_concurrent: {get_config_value(config, 'question_generation.max_concurrent')}")
        print(f"Data train_split: {get_config_value(config, 'data.train_split')}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")










































