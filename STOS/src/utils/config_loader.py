import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigLoader:
    def __init__(self, config_path: str, data_config_path: Optional[str] = None):
        """
        Args:
            config_path: 主配置文件路径
            data_config_path: 可选的数据配置文件路径
        """
        self.config_path = Path(config_path)
        self.data_config_path = Path(data_config_path) if data_config_path else None
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        if self.data_config_path and not self.data_config_path.exists():
            raise FileNotFoundError(f"数据配置文件不存在: {self.data_config_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        返回合并后的配置字典
        """
        # 加载主配置
        config = self._load_yaml(self.config_path)
        config = self._replace_env_variables(config)
        
        # 如果有数据配置，加载并合并
        if self.data_config_path:
            data_config = self._load_yaml(self.data_config_path)
            data_config = self._replace_env_variables(data_config)
            
            # 合并：data_config 覆盖 config
            config = {**config, **data_config}
            self.logger.info(f"已加载配置: {self.config_path.name} + {self.data_config_path.name}")
        else:
            self.logger.info(f"已加载配置: {self.config_path.name}")
        
        # 应用默认值
        config = self._apply_defaults(config)
        
        return config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """加载 YAML 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise ValueError(f"配置文件为空: {path}")
        
        return data
    
    def _replace_env_variables(self, config: Any) -> Any:
        """
        递归替换环境变量
        
        格式: ${VAR_NAME} 或 ${VAR_NAME:default_value}
        """
        if isinstance(config, dict):
            return {k: self._replace_env_variables(v) for k, v in config.items()}
        
        elif isinstance(config, list):
            return [self._replace_env_variables(item) for item in config]
        
        elif isinstance(config, str):
            # 替换 ${VAR_NAME} 或 ${VAR_NAME:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.environ.get(var_name, default_value)
            
            return re.sub(pattern, replacer, config)
        
        else:
            return config
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用默认值
        """
        # 数据库默认值
        if 'database' in config:
            config['database'].setdefault('timeout', 30)
        
        # 系统默认值
        if 'system' not in config:
            config['system'] = {}
        config['system'].setdefault('mode', 'production')
        config['system'].setdefault('cycle_interval', 30)
        
        # 心跳默认值
        if 'heartbeat' not in config:
            config['heartbeat'] = {'enable_heartbeat': False}
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持点号路径）
        """
        # 这个方法需要先调用 load_config()
        if not hasattr(self, '_loaded_config'):
            raise RuntimeError("请先调用 load_config()")
        
        keys = key.split('.')
        value = self._loaded_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


def load_config(config_path: str, data_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    快速加载配置
    
    Args:
        config_path: 主配置文件路径
        data_config_path: 可选的数据配置文件路径
    
    Returns:
        合并后的配置字典
    """
    loader = ConfigLoader(config_path, data_config_path)
    return loader.load_config()
