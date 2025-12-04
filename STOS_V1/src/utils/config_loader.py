import yaml
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.config_schema import AppConfig


class ConfigLoader:
    """
    Refactored ConfigLoader that returns a validated AppConfig object.
    """

    def __init__(self, config_path: str, data_config_path: Optional[str] = None):
        self.config_path = Path(config_path)
        self.data_config_path = Path(data_config_path) if data_config_path else None
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self) -> AppConfig:
        """
        Load, merge, and validate configuration.
        Returns:
            AppConfig: The validated configuration object.
        """
        # 1. Load raw dicts
        main_conf = self._load_yaml(self.config_path)
        data_conf = self._load_yaml(self.data_config_path) if self.data_config_path else {}

        # 2. Merge (data_config overrides main_conf for overlapping keys)
        # Note: In your original logic, you simple merged.
        # Here we perform a shallow merge of the root keys.
        merged_conf = {**main_conf, **data_conf}

        # 3. Environment Variable Substitution
        merged_conf = self._replace_env_variables(merged_conf)

        # 4. Validate and Parse into Pydantic Model
        try:
            config_object = AppConfig(**merged_conf)
            self.logger.info(f"Configuration loaded and validated from {self.config_path}")
            return config_object
        except Exception as e:
            self.logger.critical(f"Configuration validation failed: {e}")
            raise

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data or {}

    def _replace_env_variables(self, config: Any) -> Any:
        """Recursively replace ${VAR:default} patterns."""
        if isinstance(config, dict):
            return {k: self._replace_env_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_variables(item) for item in config]
        elif isinstance(config, str):
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

            def replacer(match):
                var_name = match.group(1)
                default_val = match.group(2) if match.group(2) is not None else ''
                return os.environ.get(var_name, default_val)

            return re.sub(pattern, replacer, config)
        else:
            return config


# Helper for quick access
def load_app_config(config_path: str = "config/config.yaml") -> AppConfig:
    # Auto-detect data_config.yaml in the same dir
    path = Path(config_path)
    data_path = path.parent / "data_config.yaml"

    loader = ConfigLoader(
        config_path=str(path),
        data_config_path=str(data_path) if data_path.exists() else None
    )
    return loader.load_config()