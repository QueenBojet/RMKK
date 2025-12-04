import os
import ctypes
import threading
import time
from typing import Callable

# Windows电源管理常量
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040

# Windows控制台事件常量
CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6


class WindowsPowerManager:
    """Windows电源管理器"""

    def __init__(self, logger):
        self.logger = logger
        self.is_windows = os.name == 'nt'
        self.power_request_active = False
        self.original_execution_state = None

        # 加载Windows API
        self.kernel32 = None
        self.win32api = None

        if self.is_windows:
            try:
                self.kernel32 = ctypes.windll.kernel32
                # 尝试导入win32api
                import win32api
                self.win32api = win32api
                self.logger.info("Windows电源管理模块已加载")
            except ImportError:
                self.logger.warning("win32api模块未安装，部分功能可能受限")
            except Exception as e:
                self.logger.error(f"加载Windows API失败: {e}")

    def prevent_system_sleep(self):
        """阻止系统进入睡眠状态"""
        if not self.is_windows or not self.kernel32:
            self.logger.debug("非Windows系统或API不可用，跳过电源管理")
            return

        try:
            # 设置执行状态，阻止系统睡眠
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: 保持系统运行
            # ES_AWAYMODE_REQUIRED: 允许显示器关闭但系统保持运行
            execution_state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED

            result = self.kernel32.SetThreadExecutionState(execution_state)

            if result:
                self.power_request_active = True
                self.logger.info("已启用电源管理保护 - 系统将保持运行状态")
                self.logger.info("注意: 显示器可能会关闭，但程序将继续运行")
            else:
                self.logger.warning("设置电源管理状态失败")

        except Exception as e:
            self.logger.error(f"启用电源管理保护失败: {e}")

    def restore_power_settings(self):
        """恢复系统电源设置"""
        if not self.is_windows or not self.kernel32 or not self.power_request_active:
            return

        try:
            # 恢复默认执行状态
            result = self.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

            if result:
                self.power_request_active = False
                self.logger.info("已恢复系统电源管理设置")
            else:
                self.logger.warning("恢复电源管理设置失败")

        except Exception as e:
            self.logger.error(f"恢复电源设置失败: {e}")

    def setup_console_handlers(self, shutdown_callback: Callable):
        """设置Windows控制台事件处理器"""
        if not self.is_windows or not self.win32api:
            self.logger.debug("win32api不可用，跳过控制台事件处理器设置")
            return

        def console_ctrl_handler(ctrl_type):
            """控制台事件处理器"""
            try:
                event_name = self._get_ctrl_event_name(ctrl_type)
                self.logger.warning(f"收到Windows控制台事件: {event_name}")

                # 在新线程中执行关闭回调，避免阻塞
                shutdown_thread = threading.Thread(
                    target=shutdown_callback,
                    name=f"WindowsShutdown-{event_name}",
                    daemon=False
                )
                shutdown_thread.start()

                # 给关闭过程一些时间
                time.sleep(2)
                return True  # 表示已处理事件

            except Exception as e:
                self.logger.error(f"处理控制台事件失败: {e}")
                return False

        try:
            # 注册控制台事件处理器
            self.win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)
            self.logger.info("Windows控制台事件处理器已注册")

        except Exception as e:
            self.logger.error(f"注册控制台事件处理器失败: {e}")

    def _get_ctrl_event_name(self, ctrl_type: int) -> str:
        """获取控制台事件名称"""
        event_names = {
            CTRL_C_EVENT: "CTRL_C (Ctrl+C)",
            CTRL_BREAK_EVENT: "CTRL_BREAK (Ctrl+Break)",
            CTRL_CLOSE_EVENT: "CTRL_CLOSE (关闭控制台)",
            CTRL_LOGOFF_EVENT: "CTRL_LOGOFF (用户注销)",
            CTRL_SHUTDOWN_EVENT: "CTRL_SHUTDOWN (系统关闭)"
        }
        return event_names.get(ctrl_type, f"UNKNOWN_EVENT_{ctrl_type}")

    def keep_alive_worker(self, interval: int = 300):
        """保活工作线程 - 定期刷新电源状态"""
        if not self.is_windows or not self.kernel32:
            return

        self.logger.info(f"启动电源保活线程，刷新间隔: {interval}秒")

        while self.power_request_active:
            try:
                # 定期刷新执行状态
                execution_state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
                result = self.kernel32.SetThreadExecutionState(execution_state)

                if result:
                    self.logger.debug("电源状态已刷新")
                else:
                    self.logger.warning("刷新电源状态失败")

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"电源保活线程异常: {e}")
                time.sleep(60)  # 异常后等待1分钟再重试

    def start_keep_alive(self, interval: int = 300):
        """启动保活线程"""
        if not self.power_request_active:
            self.logger.warning("电源管理未激活，无法启动保活线程")
            return

        keep_alive_thread = threading.Thread(
            target=self.keep_alive_worker,
            args=(interval,),
            name="PowerKeepAlive",
            daemon=True
        )
        keep_alive_thread.start()
        self.logger.info("电源保活线程已启动")

    def check_power_capabilities(self) -> dict:
        """检查系统电源能力"""
        if not self.is_windows or not self.kernel32:
            return {"error": "Windows API不可用"}

        try:
            # 获取系统电源信息
            from ctypes.wintypes import DWORD, BOOLEAN

            class SYSTEM_POWER_CAPABILITIES(ctypes.Structure):
                _fields_ = [
                    ("PowerButtonPresent", BOOLEAN),
                    ("SleepButtonPresent", BOOLEAN),
                    ("LidPresent", BOOLEAN),
                    ("SystemS1", BOOLEAN),
                    ("SystemS2", BOOLEAN),
                    ("SystemS3", BOOLEAN),
                    ("SystemS4", BOOLEAN),
                    ("SystemS5", BOOLEAN),
                    ("HiberFilePresent", BOOLEAN),
                    ("FullWake", BOOLEAN),
                    ("VideoDimPresent", BOOLEAN),
                    ("ApmPresent", BOOLEAN),
                    ("UpsPresent", BOOLEAN),
                    ("ThermalControl", BOOLEAN),
                    ("ProcessorThrottle", BOOLEAN),
                    ("ProcessorMinThrottle", ctypes.c_ubyte),
                    ("ProcessorMaxThrottle", ctypes.c_ubyte),
                    ("FastSystemS4", BOOLEAN),
                    ("Hiberboot", BOOLEAN),
                    ("WakeAlarmPresent", BOOLEAN),
                    ("AoAc", BOOLEAN),
                    ("DiskSpinDown", BOOLEAN),
                    ("HiberFileType", ctypes.c_ubyte),
                    ("AoAcConnectivitySupported", BOOLEAN),
                    ("spare3", ctypes.c_ubyte * 6),
                    ("SystemBatteriesPresent", BOOLEAN),
                    ("BatteriesAreShortTerm", BOOLEAN),
                ]

            capabilities = SYSTEM_POWER_CAPABILITIES()
            success = self.kernel32.GetPwrCapabilities(ctypes.byref(capabilities))

            if success:
                return {
                    "sleep_states": {
                        "S1": bool(capabilities.SystemS1),
                        "S2": bool(capabilities.SystemS2),
                        "S3": bool(capabilities.SystemS3),
                        "S4": bool(capabilities.SystemS4),
                        "S5": bool(capabilities.SystemS5)
                    },
                    "features": {
                        "hibernate": bool(capabilities.HiberFilePresent),
                        "battery_present": bool(capabilities.SystemBatteriesPresent),
                        "thermal_control": bool(capabilities.ThermalControl),
                        "processor_throttle": bool(capabilities.ProcessorThrottle)
                    }
                }
            else:
                return {"error": "无法获取电源能力信息"}

        except Exception as e:
            self.logger.error(f"检查电源能力失败: {e}")
            return {"error": str(e)}

    def get_status(self) -> dict:
        """获取电源管理状态"""
        return {
            "is_windows": self.is_windows,
            "api_available": self.kernel32 is not None,
            "win32api_available": self.win32api is not None,
            "power_protection_active": self.power_request_active
        }

    def __del__(self):
        """析构函数 - 确保恢复电源设置"""
        try:
            self.restore_power_settings()
        except Exception:
            pass  # 忽略析构时的错误