import os
import ctypes
import threading
import time
import logging

# Re-using the exact logic from the old codebase, adapted slightly for the new logger
# Windows API Constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


class WindowsPowerManager:
    """Windows电源管理器 - Ported to V1"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_windows = os.name == 'nt'
        self.power_request_active = False
        self.kernel32 = None

        if self.is_windows:
            try:
                self.kernel32 = ctypes.windll.kernel32
            except Exception as e:
                self.logger.error(f"Failed to load kernel32: {e}")

    def prevent_system_sleep(self):
        if not self.is_windows or not self.kernel32: return

        try:
            # Prevent Sleep, Keep System Running
            execution_state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            result = self.kernel32.SetThreadExecutionState(execution_state)

            if result:
                self.power_request_active = True
                self.logger.info("Power management active: System sleep prevented.")
            else:
                self.logger.warning("Failed to set power execution state.")
        except Exception as e:
            self.logger.error(f"Error setting power state: {e}")

    def restore_power_settings(self):
        if not self.is_windows or not self.kernel32: return
        try:
            self.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            self.power_request_active = False
            self.logger.info("Power settings restored.")
        except Exception:
            pass