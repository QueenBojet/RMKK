import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
from pathlib import Path


class SpeedMonitor:
    """
    转速监测器
    
    持续监测转速信号，当转速超过阈值时触发振动数据采集
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 data_handler,
                 trigger_callback: Optional[Callable] = None):
        """
        初始化转速监测器
        
        Args:
            config: 完整配置字典
            data_handler: 数据库处理器
            trigger_callback: 触发时的回调函数
        """
        self.config = config
        self.data_handler = data_handler
        self.trigger_callback = trigger_callback
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 读取转速触发配置
        speed_config = config.get('speed_trigger', {})
        self.enabled = speed_config.get('enable', True)
        self.threshold_rpm = speed_config.get('threshold_rpm', 5.0)
        self.cooldown_seconds = speed_config.get('cooldown_seconds', 10)
        self.check_interval = speed_config.get('check_interval_seconds', 1)
        
        self.device_name = speed_config.get('device_name', 'gearbox_test')
        self.point_name = speed_config.get('point_name', 'speed_test')
        
        # 状态变量
        self._running = False
        self._monitor_thread = None
        self._last_trigger_time = None
        self._current_speed = 0.0
        
        # 采样期间转速监测
        self._is_sampling = False
        self._sampling_start_time = None
        self._sampling_duration = 0
        self._sampling_speed_data = []  # 存储采样期间的转速数据
        
        self.logger.info(f"转速监测器初始化: 阈值={self.threshold_rpm} RPM, "
                        f"冷却={self.cooldown_seconds}秒")
    
    def start(self):
        """启动转速监测"""
        if not self.enabled:
            self.logger.info("转速监测未启用")
            return
        
        if self._running:
            self.logger.warning("转速监测已在运行")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("转速监测已启动")
    
    def stop(self):
        """停止转速监测"""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("转速监测已停止")

    def start_sampling_monitoring(self, duration: float):
        """记录采样开始时间"""
        self._is_sampling = True
        self._sampling_start_time = datetime.now()
        self._sampling_duration = duration
        self._sampling_speed_data = []
        self.logger.debug(f"开始采样监测（时长: {duration}秒）")

    def get_sampling_speed_data(self) -> list:
        """获取采样期间的转速数据"""
        return self._sampling_speed_data

    def validate_sampling_speeds(self, min_valid_ratio: float) -> bool:
        """验证采样期间的转速是否有效"""
        # 计算采样结束时间
        end_time = datetime.now()
        start_time = self._sampling_start_time
        
        # 从数据库查询该时间段内的所有转速数据
        try:
            df = self.data_handler.read_signal_data(
                device_name=self.device_name,
                point_name=self.point_name,
                start_time=start_time,
                end_time=end_time,
                order='ASC'
            )
            
            if df.empty:
                self.logger.warning("采样期间未查询到转速数据")
                return False
                
            # 转换为字典列表
            self._sampling_speed_data = [
                {'speed': row['value'], 'timestamp': row['time']} 
                for _, row in df.iterrows()
            ]
            
            # 打印详细转速数据以便调试（合并为一条日志）
            speed_values = [round(d['speed'], 2) for d in self._sampling_speed_data]
            self.logger.info(f"采样期间转速数据（共 {len(self._sampling_speed_data)} 条）: {speed_values}")
            
            valid_count = sum(1 for d in self._sampling_speed_data 
                             if d['speed'] > self.threshold_rpm)
            valid_ratio = valid_count / len(self._sampling_speed_data)
            
            self.logger.info(f"转速验证: 有效比例 {valid_ratio:.2%} (阈值: {min_valid_ratio:.2%})")
            return valid_ratio >= min_valid_ratio
            
        except Exception as e:
            self.logger.error(f"验证采样转速失败: {e}")
            return False
        finally:
            # 无论验证成功与否，都重置采样状态
            self._is_sampling = False
            self.logger.info("采样监测状态已重置")
    
    def _monitor_loop(self):
        """监测循环（在独立线程中运行）"""
        self.logger.info("转速监测线程已启动")
        
        while self._running:
            try:
                # 读取当前转速
                speed = self._read_current_speed()
                
                if speed is not None:
                    self._current_speed = speed
                    
                    # 检查是否满足触发条件（只在非采样期间触发）
                    if not self._is_sampling and self._should_trigger(speed):
                        self._on_trigger()
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"转速监测循环出错: {e}", exc_info=True)
                time.sleep(self.check_interval)
    
    def _read_current_speed(self) -> Optional[float]:
        """
        从数据库读取当前转速
        
        Returns:
            当前转速值，如果读取失败返回None
        """
        try:
            # 读取最近10秒内的转速数据，使用降序排列获取最新数据
            df = self.data_handler.read_signal_data(
                device_name=self.device_name,
                point_name=self.point_name,
                time_range='10s',
                limit=10,
                order='DESC'  # 使用降序获取最新数据
            )
            
            if df.empty:
                self.logger.warning("未读取到转速数据 - 请检查Modbus采集器是否正常运行")
                return None
            
            # 降序排列，第一条就是最新数据
            latest_speed = float(df.iloc[0]['value'])
            
            # 仅使用DEBUG日志，避免刷屏
            self.logger.debug(f"当前转速: {latest_speed:.2f} RPM")
            
            return latest_speed
            
        except Exception as e:
            self.logger.error(f"读取转速数据失败: {e}")
            return None
    
    def _should_trigger(self, speed: float) -> bool:
        """
        判断是否应该触发采集
        
        Args:
            speed: 当前转速
            
        Returns:
            是否应该触发
        """
        # 检查转速是否超过阈值
        if speed < self.threshold_rpm:
            return False
        
        # 检查冷却时间
        if self._last_trigger_time is not None:
            time_since_last = datetime.now() - self._last_trigger_time
            if time_since_last.total_seconds() < self.cooldown_seconds:
                self.logger.debug(
                    f"冷却中: 距上次触发 {time_since_last.total_seconds():.1f}秒"
                )
                return False
        
        return True
    
    def _on_trigger(self):
        """触发时的处理"""
        self._last_trigger_time = datetime.now()
        
        self.logger.info(
            f"转速触发采集 - {self._current_speed:.2f} RPM @ "
            f"{self._last_trigger_time.strftime('%H:%M:%S')}"
        )
        
        # 调用回调函数
        if self.trigger_callback:
            try:
                self.trigger_callback(self._current_speed, self._last_trigger_time)
            except Exception as e:
                self.logger.error(f"触发回调执行失败: {e}", exc_info=True)
        else:
            self.logger.warning("未设置触发回调函数")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取监测器状态
        
        Returns:
            状态字典
        """
        return {
            'enabled': self.enabled,
            'running': self._running,
            'current_speed': self._current_speed,
            'threshold_rpm': self.threshold_rpm,
            'last_trigger_time': self._last_trigger_time.isoformat() if self._last_trigger_time else None,
            'cooldown_seconds': self.cooldown_seconds
        }
    
    def set_callback(self, callback: Callable):
        """设置触发回调函数"""
        self.trigger_callback = callback
        self.logger.info("触发回调函数已设置")
