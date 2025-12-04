import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
import time

class VibrationDataGenerator:
    """振动数据生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化振动数据生成器
        
        Args:
            config: 测试模式配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        vib_config = config.get('test_mode', {}).get('vibration_simulation', {})
        self.channels = vib_config.get('channels', ['ch1', 'ch2', 'ch3', 'ch4', 'ch5'])
        self.sampling_rates = vib_config.get('sampling_rates', {})
        self.duration = vib_config.get('duration_seconds', 1)
        
        signal_params = vib_config.get('signal_params', {})
        self.base_freq = signal_params.get('base_frequency', 50)
        self.amplitude = signal_params.get('amplitude', 10.0)
        self.noise_level = signal_params.get('noise_level', 0.1)
        self.health_score = signal_params.get('health_score', 85)
        
        self.logger.info("振动数据生成器已初始化")
    
    def generate_signal(self, channel: str, sampling_rate: int) -> np.ndarray:
        """
        生成单个通道的振动信号
        
        包含: 正弦基频 + FM(转速波动) + AM(负载波动) + 丰富谐波 + 噪声 + 冲击
        """
        n_samples = int(sampling_rate * self.duration)
        t = np.linspace(0, self.duration, n_samples)

        fm_freq = 0.5
        fm_deviation = 0.01 * self.base_freq
        
        phase_carrier = 2 * np.pi * self.base_freq * t
        phase_mod = (fm_deviation / fm_freq) * np.sin(2 * np.pi * fm_freq * t)
        total_phase = phase_carrier + phase_mod
        
        signal = self.amplitude * np.sin(total_phase)

        am_envelope = 1.0 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
        signal *= am_envelope
        
        signal += 0.4 * self.amplitude * np.sin(2 * total_phase)
        signal += 0.2 * self.amplitude * np.sin(3 * total_phase)
        signal += 0.15 * self.amplitude * np.sin(1.5 * total_phase)
        signal += 0.1 * self.amplitude * np.sin(12.5 * total_phase)
        
        if self.health_score < 90:
            fault_factor = (100 - self.health_score) / 100.0
            n_impacts = int(fault_factor * 5)
            for _ in range(n_impacts):
                impact_pos = np.random.randint(0, n_samples)
                impact_width = int(sampling_rate * 0.001)
                if impact_pos + impact_width < n_samples:
                    signal[impact_pos:impact_pos+impact_width] += fault_factor * self.amplitude * 5
        
        noise = np.random.normal(0, self.noise_level * self.amplitude, n_samples)
        signal += noise
        
        return signal
    
    def generate_and_save(self, save_path: Path) -> List[Path]:
        """
        生成振动数据并保存为CSV文件
        
        新格式：所有通道合并到一个文件
        文件名格式: 20251202_102720_25600Hz.csv
        
        Returns:
            保存的文件路径列表
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        sampling_rate = self.sampling_rates.get(self.channels[0], 25600)
        
        data_dict = {}
        for channel in self.channels:
            signal = self.generate_signal(channel, sampling_rate)
            data_dict[channel] = signal
        
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
import time

class VibrationDataGenerator:
    """振动数据生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化振动数据生成器
        
        Args:
            config: 测试模式配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        vib_config = config.get('test_mode', {}).get('vibration_simulation', {})
        self.channels = vib_config.get('channels', ['ch1', 'ch2', 'ch3', 'ch4', 'ch5'])
        self.sampling_rates = vib_config.get('sampling_rates', {})
        self.duration = vib_config.get('duration_seconds', 1)
        
        signal_params = vib_config.get('signal_params', {})
        self.base_freq = signal_params.get('base_frequency', 50)
        self.amplitude = signal_params.get('amplitude', 10.0)
        self.noise_level = signal_params.get('noise_level', 0.1)
        self.health_score = signal_params.get('health_score', 85)
        
        self.logger.info("振动数据生成器已初始化")
    
    def generate_signal(self, channel: str, sampling_rate: int) -> np.ndarray:
        """
        生成单个通道的振动信号
        
        包含: 正弦基频 + FM(转速波动) + AM(负载波动) + 丰富谐波 + 噪声 + 冲击
        """
        n_samples = int(sampling_rate * self.duration)
        t = np.linspace(0, self.duration, n_samples)

        fm_freq = 0.5
        fm_deviation = 0.01 * self.base_freq
        
        phase_carrier = 2 * np.pi * self.base_freq * t
        phase_mod = (fm_deviation / fm_freq) * np.sin(2 * np.pi * fm_freq * t)
        total_phase = phase_carrier + phase_mod
        
        signal = self.amplitude * np.sin(total_phase)

        am_envelope = 1.0 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
        signal *= am_envelope
        
        signal += 0.4 * self.amplitude * np.sin(2 * total_phase)
        signal += 0.2 * self.amplitude * np.sin(3 * total_phase)
        signal += 0.15 * self.amplitude * np.sin(1.5 * total_phase)
        signal += 0.1 * self.amplitude * np.sin(12.5 * total_phase)
        
        if self.health_score < 90:
            fault_factor = (100 - self.health_score) / 100.0
            n_impacts = int(fault_factor * 5)
            for _ in range(n_impacts):
                impact_pos = np.random.randint(0, n_samples)
                impact_width = int(sampling_rate * 0.001)
                if impact_pos + impact_width < n_samples:
                    signal[impact_pos:impact_pos+impact_width] += fault_factor * self.amplitude * 5
        
        noise = np.random.normal(0, self.noise_level * self.amplitude, n_samples)
        signal += noise
        
        return signal
    
    def generate_and_save(self, save_path: Path) -> List[Path]:
        """
        生成振动数据并保存为CSV文件
        
        新格式：所有通道合并到一个文件
        文件名格式: 20251202_102720_25600Hz.csv
        
        Returns:
            保存的文件路径列表
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        sampling_rate = self.sampling_rates.get(self.channels[0], 25600)
        
        data_dict = {}
        for channel in self.channels:
            signal = self.generate_signal(channel, sampling_rate)
            data_dict[channel] = signal
        
        filename = f"{timestamp}_{sampling_rate}Hz.csv"
        file_path = save_path / filename
        
        ordered_data = {ch: data_dict[ch] for ch in self.channels if ch in data_dict}
        ordered_data = {ch: data_dict[ch] for ch in self.channels if ch in data_dict}
        df = pd.DataFrame(ordered_data)
        df.to_csv(file_path, index=False, header=False)
        return [file_path]


class ModbusDataGenerator:
    """Modbus数据生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Modbus数据生成器
        
        Args:
            config: 测试模式配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 尝试从CSV加载信号配置
        from src.utils.project_path import get_project_path
        csv_path = get_project_path("config/signal_config.csv")
        
        self.signals = []
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # 确保必要的列存在
                if all(col in df.columns for col in ['device_name', 'point_name', 'data_type']):
                    self.signals = df.to_dict('records')
                    self.logger.info(f"从CSV加载了 {len(self.signals)} 个信号配置")
                else:
                    self.logger.warning("CSV文件缺少必要列")
            except Exception as e:
                self.logger.error(f"加载信号配置文件失败: {e}")
        
        # 如果CSV加载失败或为空，回退到配置文件的模拟设置
        if not self.signals:
            modbus_config = config.get('test_mode', {}).get('modbus_simulation', {})
            self.signals = modbus_config.get('signals', [])
            self.logger.info(f"从配置文件加载了 {len(self.signals)} 个信号配置")
        
        self.current_values = {}
        # 初始化当前值
        for signal in self.signals:
            point_name = signal['point_name']
            self.current_values[point_name] = self._get_initial_value(signal)
        
        self.speed_cycle_seconds = 60
        self.cycle_start_time = time.time()
        
        self.logger.info("Modbus数据生成器已初始化")

    def _get_initial_value(self, signal: Dict[str, Any]) -> float:
        """获取初始值"""
        point_name = signal['point_name'].lower()
        unit = str(signal.get('unit', '')).lower()
        
        if 'temp' in point_name or '℃' in unit:
            return 40.0 + np.random.uniform(-5, 5)
        elif 'pressure' in point_name or 'bar' in unit:
            return 100.0
        elif 'voltage' in point_name or 'V' == unit:
            return 380.0
        elif 'current' in point_name or 'A' == unit:
            return 50.0
        elif 'speed' in point_name:
            return 0.0
        else:
            return 0.0
    
    def generate_value(self, signal_config: Dict[str, Any]) -> float:
        """
        生成单个信号的值
        """
        point_name = signal_config['point_name']
        unit = str(signal_config.get('unit', ''))
        
        # 根据点名或单位分发生成逻辑
        if 'speed' in point_name.lower():
            return self._generate_speed_value(signal_config)
        elif 'temp' in point_name.lower() or '℃' in unit:
            return self._generate_temperature(point_name)
        elif 'pressure' in point_name.lower() or 'bar' in unit:
            return self._generate_pressure(point_name)
        elif 'run_time' in point_name.lower():
            return self._generate_runtime(point_name)
        elif 'voltage' in point_name.lower():
            return self._generate_voltage(point_name)
        elif 'current' in point_name.lower() or 'power' in point_name.lower() or 'kW' in unit or 'A' == unit:
            return self._generate_load_dependent(point_name)
        else:
            return self._generate_random_walk_value(point_name)
    
    def _generate_speed_value(self, signal_config: Dict[str, Any]) -> float:
        """
        生成周期性的转速值
        """
        # 计算当前在周期中的位置
        elapsed = time.time() - self.cycle_start_time
        position = elapsed % self.speed_cycle_seconds
        
        if position < 20:  # 0-20s: 停机
            speed = np.random.uniform(0, 3)
        elif position < 30:  # 20-30s: 启动
            progress = (position - 20) / 10
            speed = 3 + progress * 12 + np.random.uniform(-1, 1)
        elif position < 50:  # 30-50s: 运行
            speed = np.random.uniform(15, 30)
        elif position < 55:  # 50-55s: 减速
            progress = (position - 50) / 5
            speed = 30 - progress * 20 + np.random.uniform(-1, 1)
        else:  # 55-60s: 停机
            progress = (position - 55) / 5
            speed = 10 - progress * 10 + np.random.uniform(-0.5, 0.5)
        
        return float(max(0, speed))

    def _generate_temperature(self, point_name: str) -> float:
        """生成温度值 (缓慢变化)"""
        current = self.current_values.get(point_name, 40.0)
        # 目标温度范围 40-80
        target = 60.0
        
        # 趋向目标的随机游走
        diff = target - current
        step = np.random.normal(diff * 0.01, 0.1)
        new_val = current + step
        
        self.current_values[point_name] = new_val
        return float(new_val)

    def _generate_pressure(self, point_name: str) -> float:
        """生成压力值"""
        current = self.current_values.get(point_name, 100.0)
        step = np.random.normal(0, 0.5)
        new_val = current + step
        self.current_values[point_name] = new_val
        return float(new_val)

    def _generate_runtime(self, point_name: str) -> float:
        """生成运行时间 (递增)"""
        current = self.current_values.get(point_name, 0.0)
        # 假设每次调用间隔约1秒，或者根据实际情况增加
        # 这里简单每次增加一点点
        new_val = current + 0.01 
        self.current_values[point_name] = new_val
        return float(new_val)

    def _generate_voltage(self, point_name: str) -> float:
        """生成电压 (基本稳定)"""
        base = 380.0
        noise = np.random.normal(0, 2.0)
        return float(base + noise)

    def _generate_load_dependent(self, point_name: str) -> float:
        """生成负载相关值 (与转速相关)"""
        # 简化的负载模型：与周期位置相关
        elapsed = time.time() - self.cycle_start_time
        position = elapsed % self.speed_cycle_seconds
        
        base_load = 0.0
        if 20 <= position < 55:
            base_load = 50.0 # 运行负载
        
        noise = np.random.normal(0, 2.0)
        return float(base_load + noise)
    
    def _generate_random_walk_value(self, point_name: str) -> float:
        """
        使用随机游走生成值
        """
        current = self.current_values.get(point_name, 0.0)
        step = np.random.uniform(-1, 1)
        new_value = current + step
        self.current_values[point_name] = new_value
        return float(new_value)
    
    def generate_all(self) -> List[Dict[str, Any]]:
        """
        生成所有Modbus信号的值
        
        Returns:
            信号值列表
        """
        results = []
        
        for signal_config in self.signals:
            value = self.generate_value(signal_config)
            
            result = {
                'device_name': signal_config['device_name'],
                'point_name': signal_config['point_name'],
                'data_type': signal_config['data_type'],
                'unit': signal_config.get('unit', ''),
                'value': value
            }
            results.append(result)
        
        return results


class TestDataManager:
    """
    测试数据管理器
    
    协调振动数据和Modbus数据的生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化测试数据管理器
        
        Args:
            config: 完整配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        test_config = config.get('test_mode', {})
        self.enabled = test_config.get('enabled', False)
        
        if not self.enabled:
            self.logger.warning("测试模式未启用")
            return
        
        # 初始化生成器
        vib_config = test_config.get('vibration_simulation', {})
        modbus_config = test_config.get('modbus_simulation', {})
        
        self.vib_enabled = vib_config.get('enabled', False)
        self.modbus_enabled = modbus_config.get('enabled', False)
        
        if self.vib_enabled:
            self.vib_generator = VibrationDataGenerator(config)
            self.logger.info("振动数据模拟: 已启用")
        
        if self.modbus_enabled:
            self.modbus_generator = ModbusDataGenerator(config)
            self.logger.info("Modbus数据模拟: 已启用")
        
        self.logger.info("=" * 60)
        self.logger.info("测试模式已启用 - 使用模拟数据")
        self.logger.info("生产环境请设置 test_mode.enabled = false")
        self.logger.info("=" * 60)
    
    def is_enabled(self) -> bool:
        """检查测试模式是否启用"""
        return self.enabled
    
    def generate_vibration_data(self, save_path: Path) -> List[Path]:
        """
        生成振动数据
        
        Returns:
            生成的文件路径列表
        """
        if not self.vib_enabled:
            raise RuntimeError("振动数据模拟未启用")
        
        return self.vib_generator.generate_and_save(save_path)
    
    def generate_modbus_data(self) -> List[Dict[str, Any]]:
        """生成Modbus数据"""
        if not self.modbus_enabled:
            raise RuntimeError("Modbus数据模拟未启用")
        
        return self.modbus_generator.generate_all()