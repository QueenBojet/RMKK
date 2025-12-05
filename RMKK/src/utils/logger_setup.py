import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def get_app_root() -> Path:
    """
    获取应用程序根目录
    
    支持：
    - 开发模式：返回项目根目录
    - 打包后：返回可执行文件所在目录
    """
    if getattr(sys, 'frozen', False):
        # 打包后的可执行文件
        return Path(sys.executable).parent
    else:
        # 开发模式：返回项目根目录（src的上一级）
        return Path(__file__).parent.parent.parent


class LoggerSetup:
    """
    日志配置管理器
    
    支持打包后的日志路径：
    - 相对路径：相对于应用程序根目录（打包后=可执行文件目录，开发=项目根目录）
    - 绝对路径：直接使用
    """

    def __init__(self, logging_config: Dict[str, Any]):
        self.config = logging_config
        self.enable_logging = logging_config.get('enable_logging', True)
        
        # 解析日志路径：支持打包后的路径
        log_path_str = logging_config.get('log_path', './logs')
        log_path = Path(log_path_str)
        
        if not log_path.is_absolute():
            # 相对路径：相对于应用程序根目录
            app_root = get_app_root()
            self.log_path = app_root / log_path
        else:
            # 绝对路径：直接使用
            self.log_path = log_path
        
        self.log_level = logging_config.get('log_level', 'INFO')
        self.log_format = logging_config.get(
            'log_format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.max_log_size_mb = logging_config.get('max_log_size_mb', 100)
        self.backup_count = logging_config.get('backup_count', 5)
        self.loggers = {}

        self._setup_root_logger()

    def _setup_root_logger(self):
        """配置根日志记录器"""
        try:
            self.log_path.mkdir(parents=True, exist_ok=True)

            root_logger = logging.getLogger()
            root_logger.setLevel(self._get_log_level(self.log_level))
            root_logger.handlers.clear()

            formatter = logging.Formatter(
                self.log_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            if self.enable_logging:
                file_handler = self._create_file_handler(formatter)
                root_logger.addHandler(file_handler)

            console_handler = self._create_console_handler(formatter)
            root_logger.addHandler(console_handler)

        except Exception as e:
            print(f"初始化日志系统失败: {e}")
            raise

    def _create_file_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """创建文件处理器"""
        log_file = self.log_path / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        max_bytes = self.max_log_size_mb * 1024 * 1024

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8',
            mode='a'
        )

        file_handler.setLevel(self._get_log_level(self.log_level))
        file_handler.setFormatter(formatter)

        return file_handler

    def _create_console_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """创建控制台处理器"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._get_log_level(self.log_level))
        console_handler.setFormatter(formatter)
        return console_handler

    def _get_log_level(self, level_name: str) -> int:
        """获取日志级别"""
        level_mapping = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_mapping.get(level_name.upper(), logging.INFO)

    def get_logger(self, name: str) -> logging.Logger:
        """获取或创建指定名称的日志记录器"""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(self._get_log_level(self.log_level))
        self.loggers[name] = logger
        return logger

    def cleanup_old_logs(self, days: int = 30):
        """清理旧日志文件"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days)

            for log_file in self.log_path.glob('*.log*'):
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        log_file.unlink()
                except Exception:
                    pass

        except Exception as e:
            logging.error(f"清理日志文件失败: {e}")


def setup_logging(config: Dict[str, Any]) -> LoggerSetup:
    """快速配置日志系统"""
    return LoggerSetup(config)


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """获取日志记录器"""
    if config:
        logger_setup = LoggerSetup(config)
        return logger_setup.get_logger(name)
    else:
        return logging.getLogger(name)