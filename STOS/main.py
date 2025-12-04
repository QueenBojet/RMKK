import sys
import signal
import argparse
from enum import Enum
from pathlib import Path
from typing import Optional
import logging
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import LoggerSetup, get_app_root
from src.utils.system_utils import SystemManager
from src.utils.windows_power import WindowsPowerManager

class MonitorState(Enum):
    """监控状态"""
    INIT = 0
    RUNNING = 1
    STOPPING = 2
    STOPPED = 3


class STOSMonitor:
    """智能拖曳作业系统"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.state = MonitorState.INIT
        self.logger = None
        self.system_manager = None
        self.power_manager = None
        
        # 加载配置
        if config_path is None:
            app_root = get_app_root()
            config_path = app_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = get_app_root() / config_path
        
        # 自动查找 data_config.yaml
        data_config_path = config_path.parent / "data_config.yaml"
        
        # 配置加载
        if data_config_path.exists():
            self.config_loader = ConfigLoader(str(config_path), str(data_config_path))
        else:
            self.config_loader = ConfigLoader(str(config_path))
        
        self.config = self.config_loader.load_config()
        
        # 设置日志
        log_config = self.config.get('output', {}).get('logging', {})
        self.logger_setup = LoggerSetup(log_config)
        self.logger = self.logger_setup.get_logger(self.__class__.__name__)
        
        # 初始化系统管理器
        project_root = Path(__file__).parent.parent
        self.system_manager = SystemManager(
            config=self.config,
            logger=self.logger,
            project_root=project_root
        )
        
        # 初始化电源管理器
        try:
            self.power_manager = WindowsPowerManager(self.logger)
        except Exception as e:
            self.logger.warning(f"电源管理器不可用: {e}")
            self.power_manager = None
        
        self.logger.info("FinStabMonitor 初始化完成")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def shutdown_handler(signum, frame):
            self.logger.info(f"收到信号 {signum}，开始关闭")
            self.shutdown()
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        # Windows 特殊处理
        if self.power_manager:
            self.power_manager.setup_console_handlers(self.shutdown)
    
    def start(self):
        """启动监控系统"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("启动齿轮箱电机监控系统")
            self.logger.info("=" * 50)
            
            # 1. 设置信号处理
            self._setup_signal_handlers()
            
            # 2. 防止系统休眠
            if self.power_manager:
                self.power_manager.prevent_system_sleep()
            
            # 3. 启动系统管理器
            self.system_manager.start()
            
            # 4. 进入主循环
            self.state = MonitorState.RUNNING
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("用户中断")
        finally:
            self.shutdown()
    
    def _main_loop(self):
        cycle_interval = self.config.get('system', {}).get('cycle_interval', 0.1)
        
        while self.state == MonitorState.RUNNING:
            self.system_manager.run_diagnosis_cycle()

            if self.state == MonitorState.RUNNING:
                import time
                time.sleep(cycle_interval)
        
        self.logger.info("主循环结束")
    
    def shutdown(self):
        if self.state == MonitorState.STOPPING or self.state == MonitorState.STOPPED:
            return
        
        self.state = MonitorState.STOPPING
        self.logger.info("开始关闭系统")
        
        # 关闭系统管理器
        if self.system_manager:
            self.system_manager.shutdown()
        
        # 恢复电源设置
        if self.power_manager:
            self.power_manager.restore_power_settings()
        
        self.state = MonitorState.STOPPED
        self.logger.info("系统已关闭")
    
    def run_once(self) -> bool:
        """运行单次诊断"""
        try:
            if self.power_manager:
                self.power_manager.prevent_system_sleep()
            
            self.system_manager.initialize_components()
            result = self.system_manager.run_diagnosis_cycle()
            
            return result
        finally:
            self.shutdown()


def parse_arguments():
    parser = argparse.ArgumentParser(description='齿轮箱电机监控系统')
    
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--mode', '-m', choices=['continuous', 'once'],
                        default='continuous', help='运行模式')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细日志')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # 创建监控器
    monitor = STOSMonitor(config_path=args.config)
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行
    try:
        if args.mode == 'once':
            success = monitor.run_once()
            return 0 if success else 1
        else:
            monitor.start()
            return 0
    except KeyboardInterrupt:
        print("\n用户中断")
        return 0
    except Exception as e:
        print(f"程序失败: {e}")
        if monitor.logger:
            monitor.logger.critical(f"程序失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())