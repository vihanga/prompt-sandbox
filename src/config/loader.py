"""
Configuration loader with YAML support
"""

import yaml
from pathlib import Path
from typing import Union
from .schema import PromptConfig, ExperimentConfig


class ConfigLoader:
    """Utility for loading configuration files"""

    @staticmethod
    def load_prompt_config(path: Union[str, Path]) -> PromptConfig:
        """
        Load and validate prompt configuration from YAML file

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated PromptConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config doesn't match schema
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return PromptConfig(**data)

    @staticmethod
    def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
        """
        Load and validate experiment configuration from YAML file

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated ExperimentConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config doesn't match schema
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert string paths to Path objects
        if "prompt_configs" in data:
            data["prompt_configs"] = [Path(p) for p in data["prompt_configs"]]

        if "dataset_config" in data and "path" in data["dataset_config"]:
            data["dataset_config"]["path"] = Path(data["dataset_config"]["path"])

        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return ExperimentConfig(**data)

    @staticmethod
    def save_prompt_config(config: PromptConfig, path: Union[str, Path]) -> None:
        """
        Save prompt configuration to YAML file

        Args:
            config: PromptConfig object to save
            path: Destination path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(config.dict(), f, default_flow_style=False, sort_keys=False)
