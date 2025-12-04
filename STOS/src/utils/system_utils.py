import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from src.modelTrain.monitor import Monitor
from src.utils.heartbeat import HeartbeatMonitor
from src.utils.test_data_generator import TestDataManager
from src.dataResource.modbus_collector import ModbusCollector
from src.dataResource.speed_monitor import SpeedMonitor
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class SystemManager:
    """系统管理器"""
    
    def __init__(self, config: Dict[str, Any], logger, project_root: Path):
        self.config = config
        self.logger = logger
        self.project_root = project_root
        
        # 核心组件
        self.monitor: Optional[Monitor] = None
        self.heartbeat_monitor: Optional[HeartbeatMonitor] = None
        self.modbus_collector: Optional[ModbusCollector] = None
        self.speed_monitor: Optional[SpeedMonitor] = None  # 转速监测器
        
        # 测试模式管理器
        self.test_data_manager = TestDataManager(config)
        
        # 简单统计
        self.stats = {
            'start_time': None,
            'processed_count': 0,
            'last_success_time': None,
            'test_mode': self.test_data_manager.is_enabled(),
            'modbus_collected': 0,
            'speed_triggers': 0,  # 转速触发次数
        }
        
        # 文件追踪
        self.processed_files = set()
        
        # Modbus采集计时
        self.last_modbus_time = None
    
    def start(self):
        """启动系统管理器"""
        self.initialize_components()
        
        # 启动心跳监控（可选）
        if self._is_heartbeat_enabled() and HeartbeatMonitor:
            self.heartbeat_monitor = HeartbeatMonitor(self.config, self.logger)
            self.heartbeat_monitor.start()
            self.logger.info("心跳监控已启动")
        
        # 启动转速监测（如果启用）
        if self.speed_monitor:
            self.speed_monitor.start()
            self.logger.info("转速监测已启动")
        
        self.stats['start_time'] = datetime.now()
        self.logger.info("SystemManager 已启动")
    
    def initialize_components(self):
        """初始化组件 - 简单直接，失败就崩溃"""
        # 验证配置
        phm_save_path = self.config.get('phm', {}).get('data_collection', {}).get('save_path')
        if not phm_save_path:
            raise ValueError("配置缺少 phm.data_collection.save_path")
        
        self.logger.info(f"振动数据路径: {phm_save_path}")
        
        # 初始化监控器
        self.monitor = Monitor(config=self.config)
        self.logger.info("监控器初始化完成")
        
        # 打印状态
        status = self.monitor.get_status()
        self.logger.info(f"监控器状态: {status}")
        
        # 初始化转速监测器
        speed_trigger_config = self.config.get('speed_trigger', {})
        if speed_trigger_config.get('enable', False):
            from src.dataResource.db_client import DataHandler
            
            # 创建数据库处理器（用于读取转速）
            db_config = self.config.get('database', {})
            data_handler = DataHandler(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 8086),
                database=db_config.get('database', 'industrial_data'),
                timezone_str=self.config.get('timezone', 'Asia/Shanghai')
            )
            
            # 创建转速监测器，设置回调函数
            self.speed_monitor = SpeedMonitor(
                config=self.config,
                data_handler=data_handler,
                trigger_callback=self._on_speed_trigger
            )
            self.logger.info("转速监测器初始化完成")
        else:
            self.logger.info("转速触发未启用")
        
        # 初始化Modbus采集器（测试模式下也需要，用于生成数据）
        if self.test_data_manager.is_enabled():
            test_config = self.config.get('test_mode', {})
            modbus_sim = test_config.get('modbus_simulation', {})
            
            if modbus_sim.get('enabled', False):
                try:
                    self.modbus_collector = ModbusCollector(self.config)
                    self.logger.info("Modbus采集器已初始化（测试模式）")
                except Exception as e:
                    self.logger.warning(f"Modbus采集器初始化失败: {e}")
    
    def _on_speed_trigger(self, speed: float, trigger_time: datetime):
        """
        转速触发回调函数
        
        当转速超过阈值时被调用，执行振动数据采集和诊断
        """
        self.stats['speed_triggers'] += 1
        
        # 1. 获取采样配置
        vib_config = self.config.get('speed_trigger', {}).get('vibration_acquisition', {})
        duration = vib_config.get('duration_seconds', 2)
        min_valid_ratio = vib_config.get('min_valid_speed_ratio', 0.8)
        
        # 2. 开始采样期间的转速监测
        if self.speed_monitor:
            self.speed_monitor.start_sampling_monitoring(duration)
            # 合并日志：显示采样监测和振动数据生成信息
            mode_tag = "[TEST MODE] " if self.test_data_manager.is_enabled() else ""
            self.logger.info(f"{mode_tag}开始采样监测（{duration}秒）并生成振动数据")
            
        # 3. 执行采集（测试模式：生成数据）
        generated_files = []
        if self.test_data_manager.is_enabled():
            vib_sim = self.config.get('test_mode', {}).get('vibration_simulation', {})
            if vib_sim.get('enabled', False):
                phm_save_path = self.config.get('phm', {}).get('data_collection', {}).get('save_path')
                if phm_save_path:
                    try:
                        generated_files = self.test_data_manager.generate_vibration_data(Path(phm_save_path))
                    except Exception as e:
                        self.logger.error(f"生成振动数据失败: {e}")

        # 4. 等待采样完成（模拟采样过程）
        # 注意：在真实硬件中，这里应该是等待采集卡完成采集
        # 在测试模式下，我们需要模拟这个等待过程，以便SpeedMonitor能收集到转速数据
        time.sleep(duration + 0.2)  # 多等待一点时间确保收集完成
        
        # 5. 验证转速有效性
        if self.speed_monitor:
            if not self.speed_monitor.validate_sampling_speeds(min_valid_ratio):
                self.logger.info("转速验证未通过 - 丢弃振动数据")
                # 删除生成的文件
                for f in generated_files:
                    if f.exists():
                        try:
                            f.unlink()
                            self.logger.info(f"已删除无效数据文件: {f.name}")
                        except Exception as e:
                            self.logger.error(f"删除文件失败: {e}")
                return

            # 6. 嵌入转速数据
            speed_data = self.speed_monitor.get_sampling_speed_data()
            for f in generated_files:
                self._embed_speed_to_vibration_file(f, speed_data)
                self.logger.info(f"[TRIGGERED] 生成有效振动数据: {f.name}")

        # 7. 处理数据（诊断）
        self._process_latest_vibration_data()

    def _embed_speed_to_vibration_file(self, vib_file: Path, speed_data: list):
        """将转速数据插值填充到振动文件"""
        try:
            if not speed_data:
                return

            # 1. 读取振动数据
            df_vib = pd.read_csv(vib_file, header=None)
            vib_length = len(df_vib)
            
            # 2. 提取转速值
            speeds = [d['speed'] for d in speed_data]
            
            if len(speeds) < 2:
                # 数据点太少，直接用均值填充
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                speed_interpolated = np.full(vib_length, avg_speed)
            else:
                # 3. 插值到振动数据长度
                # 原始索引
                x_orig = np.linspace(0, vib_length-1, len(speeds))
                # 目标索引
                x_new = np.arange(vib_length)
                
                # 线性插值
                f = interp1d(x_orig, speeds, kind='linear', fill_value='extrapolate')
                speed_interpolated = f(x_new)
            
            # 4. 添加到振动数据（作为新的一列）
            df_vib['speed'] = speed_interpolated
            
            # 5. 保存（覆盖原文件）
            df_vib.to_csv(vib_file, index=False, header=False)
            
        except Exception as e:
            self.logger.error(f"嵌入转速数据失败: {e}")

    
    def _process_latest_vibration_data(self):
        """
        处理最新的振动数据文件
        """
        if not self.monitor:
            self.logger.error("监控器未初始化")
            return
        
        # 获取最新文件
        latest_file = self.monitor.get_latest_file()
        if not latest_file:
            self.logger.warning("未找到振动数据文件")
            return
        
        # 跳过已处理的文件
        if latest_file in self.processed_files:
            self.logger.debug(f"文件已处理: {latest_file.name}")
            return
        
        # 处理文件
        mode_tag = "[TEST MODE] " if self.test_data_manager.is_enabled() else ""
        self.logger.info(f"{mode_tag}处理文件: {latest_file.name}")
        start_time = time.time()
        
        try:
            result = self.monitor.process_file(latest_file)
            if not result:
                self.logger.info(f"{mode_tag}跳过处理: {latest_file.name}")
                return
            
            # 写入结果
            self.monitor.write_results(result)
            
        except Exception as e:
            # 真正的程序错误
            self.logger.error(f"处理异常: {latest_file.name} - {e}", exc_info=True)
            return
        
        # 更新统计
        self.processed_files.add(latest_file)
        self.stats['processed_count'] += 1
        self.stats['last_success_time'] = datetime.now()
        
        # 记录结果
        duration = time.time() - start_time
        health_score = result.get('health_scores', {}).get('overall', {}).get('score', 0)
        
        self.logger.info(
            f"{mode_tag}诊断完成 - 文件: {latest_file.name}, "
            f"健康度: {health_score:.1f}%, "
            f"耗时: {duration:.2f}s"
        )
    
    def run_diagnosis_cycle(self) -> bool:
        """
        运行一次诊断周期
        
        在转速触发模式下，这个方法主要负责Modbus数据采集
        振动数据处理由转速触发回调完成
        """
        if not self.monitor:
            self.logger.error("监控器未初始化")
            return False
        
        # === 测试模式：生成Modbus数据 ===
        if self.test_data_manager.is_enabled():
            test_config = self.config.get('test_mode', {})
            modbus_sim = test_config.get('modbus_simulation', {})
            
            if modbus_sim.get('enabled', False) and self.modbus_collector:
                # 使用与主循环相同的间隔
                update_interval = self.config.get('system', {}).get('cycle_interval', 0.1)
                current_time = time.time()
                
                if (self.last_modbus_time is None or 
                    (current_time - self.last_modbus_time) >= update_interval):
                    try:
                        count = self.modbus_collector.collect_data()
                        if count > 0:  # 只有实际采集了数据才记录
                            self.stats['modbus_collected'] += count
                            self.logger.debug(f"[TEST MODE] Modbus数据采集: {count} 个信号")
                        self.last_modbus_time = current_time
                    except Exception as e:
                        self.logger.error(f"Modbus数据采集失败: {e}")
        
        return True
    
    def shutdown(self):
        """关闭系统管理器 - 简单清理"""
        self.logger.info("关闭 SystemManager...")
        
        # 停止转速监测
        if self.speed_monitor:
            self.speed_monitor.stop()
        
        if self.heartbeat_monitor:
            self.heartbeat_monitor.stop()
        
        if self.monitor:
            self.monitor.close()
        
        self._log_final_stats()
        self.logger.info("SystemManager 已关闭")
    
    def _log_final_stats(self):
        """记录最终统计"""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            self.logger.info(
                f"最终统计 - 运行时间: {runtime}, "
                f"已处理: {self.stats['processed_count']} 个文件, "
                f"转速触发: {self.stats['speed_triggers']} 次"
            )
    
    def _is_heartbeat_enabled(self) -> bool:
        """检查心跳是否启用"""
        return self.config.get('heartbeat', {}).get('enable_heartbeat', False)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': True,
            'stats': self.stats.copy(),
            'monitor': self.monitor.get_status() if self.monitor else None,
            'components': {
                'monitor': self.monitor is not None,
                'heartbeat_monitor': self.heartbeat_monitor is not None,
            }
        }
