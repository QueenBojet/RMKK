import json
import logging
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pytz
from src.utils.project_path import get_project_path, get_config_path
from src.dataResource.db_client import DataHandler
from src.modelTrain.diagnosis_engine import DiagnosisEngine


class Monitor:

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None,
                 data_config: Optional[Dict[str, Any]] = None,
                 data_config_path: Optional[str] = None):

        self.config = self._load_config(config, config_path)
        self.data_config = self._load_data_config(data_config, data_config_path)
        self.merged_config = {**self.config, **self.data_config}

        phm_save_path = self.config.get('phm', {}).get('data_collection', {}).get('save_path')
        if not phm_save_path:
            raise ValueError("Missing 'phm.data_collection.save_path' in config")

        self.vibration_data_path = Path(phm_save_path)
        self._setup_logging()

        self.data_handler = DataHandler(
            host=self.config['database']['host'],
            port=self.config['database']['port'],
            database=self.config['database']['database'],
            timezone_str=self.config.get('timezone', 'Asia/Shanghai')
        )

        self.enable_features = self.data_config.get('feature_collection', {}).get('enable_feature_extraction', True)
        
        # 初始化诊断引擎
        self.diagnosis_engine = DiagnosisEngine(self.data_config)

        self.logger.info(f"IntegratedMonitor initialized (features: {'ON' if self.enable_features else 'OFF'})")
        self.logger.info(f"Data path: {self.vibration_data_path}")
        self.logger.info(f"Config files: config={'provided' if config or config_path else 'none'}, "
                         f"data_config={'provided' if data_config or data_config_path else 'none'}")

    def _load_config(self, config: Optional[dict], config_path: Optional[str]) -> dict:
        if config:
            return config

        if config_path:
            cfg_file = Path(config_path)
        else:
            cfg_file = get_config_path('config.yaml')
            if not cfg_file.exists():
                cfg_file = get_config_path('config_data.json')

        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")

        if cfg_file.suffix in ['.yaml', '.yml']:
            import yaml
            with open(cfg_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _load_data_config(self, data_config: Optional[dict], data_config_path: Optional[str]) -> dict:
        if data_config:
            return data_config

        if data_config_path:
            cfg_file = Path(data_config_path)
        else:
            cfg_file = get_config_path('data_config.yaml')

        if not cfg_file.exists():
            self.logger.warning(f"Data config file not found: {cfg_file}, using config for all settings")
            return {}

        if cfg_file.suffix in ['.yaml', '.yml']:
            import yaml
            with open(cfg_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _setup_logging(self):
        """使用全局日志系统，不创建单独的日志文件"""
        self.logger = logging.getLogger(self.__class__.__name__)

    def _parse_filename(self, filename: str) -> Optional[Tuple[List[str], datetime, float]]:
        pattern = r'(\d{8})_(\d{6})_(\d+)Hz'
        match = re.search(pattern, filename)
        
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS
            sampling_rate = float(match.group(3))
            
            # 组合时间戳
            datetime_str = f"{date_str}_{time_str}"
            try:
                file_time = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
                
                # 新格式不包含通道信息，默认包含所有5个通道
                channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']
                
                return channels, file_time, sampling_rate
            except ValueError as e:
                self.logger.warning(f"Failed to parse timestamp from {filename}: {e}")
                return None
        
        self.logger.debug(f"Failed to parse filename: {filename}")
        return None

    def get_latest_file(self) -> Optional[Path]:
        if not self.vibration_data_path.exists():
            self.logger.error(f"Data path does not exist: {self.vibration_data_path}")
            return None

        pattern = self.config.get('vibration_data', {}).get('file_pattern', '*.csv')
        files = list(self.vibration_data_path.glob(pattern))

        if not files:
            return None

        files_with_time = []
        for file in files:
            parse_result = self._parse_filename(file.name)
            if parse_result:
                _, file_time, _ = parse_result
                files_with_time.append((file_time, file))

        if not files_with_time:
            return None

        files_with_time.sort(key=lambda x: x[0])
        return files_with_time[-1][1]

    def load_vibration_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        加载振动数据文件
        
        新格式: CSV文件包含5列数据，分别对应 ch1-ch5
        没有表头，按列顺序读取
        """
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        try:
            parse_result = self._parse_filename(file_path.name)
            if not parse_result:
                return None

            channels, file_time, sampling_rate = parse_result

            # 读取CSV文件（无表头）
            df = pd.read_csv(file_path, header=None)
            self.logger.debug(f"Loaded CSV shape: {df.shape}, expected channels: {channels}")
            
            # 验证列数
            expected_columns = len(channels)
            if df.shape[1] < expected_columns:
                self.logger.warning(
                    f"Column mismatch: file has {df.shape[1]} columns, "
                    f"expected {expected_columns} for channels {channels}"
                )
            
            # 构建振动数据字典
            vibration_data = {}
            for i, channel in enumerate(channels):
                if i < df.shape[1]:
                    vibration_data[channel] = df.iloc[:, i].values
                else:
                    self.logger.warning(f"Channel {channel} not found in file (only {df.shape[1]} columns)")

            # 读取转速数据 (第6列，如果存在)
            speed_data = None
            if df.shape[1] > len(channels):
                speed_data = df.iloc[:, len(channels)].values
                self.logger.debug(f"Found speed data in column {len(channels)}")

            return {
                'data': vibration_data,
                'channels': channels,
                'sampling_rate': sampling_rate,
                'timestamp': file_time,
                'filename': file_path.name,
                'speed_data': speed_data
            }

        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def _validate_vibration_data(self, vibration_data: Dict[str, np.ndarray], file_path: Path) -> bool:
        """
        验证振动数据质量
        
        Returns:
            True: 数据有效
            False: 数据无效（应跳过处理）
        """
        for ch, data in vibration_data.items():
            if len(data) == 0:
                self.logger.warning(f"Channel {ch} is empty in {file_path.name}")
                continue
                
            abs_data = np.abs(data)
            rms = np.sqrt(np.mean(data**2))
            mean_abs = np.mean(abs_data)
            
            # 检查信号是否接近零
            if rms < 1e-6 or mean_abs < 1e-6:
                self.logger.warning(
                    f"Invalid data detected in {file_path.name}, channel {ch}: "
                    f"RMS={rms:.6f}, mean_abs={mean_abs:.6f} (near-zero signal)"
                )
                return False
        
        return True
        
    def _get_latest_modbus_data(self) -> Dict[str, float]:
        """
        从数据库获取最新的 Modbus 数据
        
        Returns:
            字典，key 为 point_name，value 为最新值
        """
        try:
            # 读取最近10秒的数据
            query = 'SELECT * FROM signal_data WHERE time > now() - 10s ORDER BY time DESC LIMIT 100'
            df = self.data_handler.client.query(query)
            
            if df and 'signal_data' in df:
                data = df['signal_data']
                # 取每个 point 的最新值
                modbus_data = {}
                for point_name in data['point_name'].unique():
                    point_data = data[data['point_name'] == point_name]
                    if not point_data.empty:
                        modbus_data[point_name] = point_data.iloc[0]['value']
                
                return modbus_data
        except Exception as e:
            self.logger.debug(f"读取 Modbus 数据失败: {e}")
        
        return {}

    def process_file(self, file_path: Path, enable_features: bool = None) -> Optional[Dict[str, Any]]:
        """Process vibration data file - main diagnosis entry point"""
        try:
            # 1. 加载振动数据
            vibration_result = self.load_vibration_data(file_path)
            if not vibration_result:
                return None

            vibration_data = vibration_result['data']
            if not self._validate_vibration_data(vibration_data, file_path):
                self.logger.info(f"Data validation failed for {file_path.name}, deleting file...")
                try:
                    file_path.unlink()  # 删除无效文件
                    self.logger.info(f"Deleted invalid file: {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to delete file {file_path.name}: {e}")
                return None  # 跳过后续处理
            sampling_rate = vibration_result['sampling_rate']
            speed_data = vibration_result.get('speed_data')

            self.logger.info(f"Processing {file_path.name} with {len(vibration_data)} channels at {sampling_rate}Hz")

            # 2. 获取辅助数据（Modbus等）
            modbus_data = self._get_latest_modbus_data()

            # 3. 调用诊断引擎进行诊断
            diagnosis_result = self.diagnosis_engine.diagnose(
                vibration_data=vibration_data,
                sampling_rate=sampling_rate,
                speed_data=speed_data,
                modbus_data=modbus_data
            )
            
            if not diagnosis_result:
                self.logger.info(f"Diagnosis skipped for {file_path.name}")
                return None

            # 4. 添加处理信息
            diagnosis_result['timestamp'] = vibration_result['timestamp']
            diagnosis_result['filename'] = vibration_result['filename']
            diagnosis_result['processing_info'] = {
                'channels': vibration_result['channels'],
                'sampling_rate': sampling_rate,
                'features_enabled': enable_features if enable_features is not None else self.enable_features
            }

            self.logger.info(f"Overall health score: {diagnosis_result['health_scores']['overall']['score']:.1f}")

            return diagnosis_result

        except Exception as e:
            self.logger.error(f"Processing failed for {file_path}: {e}", exc_info=True)
            return None

    def run_diagnosis(self, file_path: Path, enable_features: bool = None) -> Optional[Dict[str, Any]]:
        """Run diagnosis (alias for process_file for compatibility)"""
        return self.process_file(file_path, enable_features)

    def write_results(self, diagnosis_result: Dict[str, Any]):
        """
        写入诊断结果到数据库
        
        按照要求的格式写入4条记录：
        1. 总体健康值 (component = fin_stabilizer)
        2. 机械系统健康值 (algorithm_name = mechanical_system)
        3. 控制系统健康值 (algorithm_name = control_system)
        4. 驱动系统健康值 (algorithm_name = drive_system)
        """
        try:
            timezone_str = self.config.get('timezone', 'Asia/Shanghai')
            tz = pytz.timezone(timezone_str)
            current_time = datetime.now(tz)

            health_data = diagnosis_result.get('health_scores', {})
            component_name = self.data_config.get('monitoring', {}).get('component_name', 'fin_stabilizer')

            if 'overall' not in health_data:
                self.logger.warning("Invalid health data format, missing 'overall' key")
                return

            # 1. 写入总体系统健康值 (使用 component 字段)
            overall_data = health_data['overall']
            self.data_handler.write_health_score(
                components_name=component_name,  # "fin_stabilizer"
                score=overall_data['score'],
                components_score=overall_data['subsystems'],
                components_weight=overall_data['weights'],
                timestamp=current_time
            )

            # 2. 写入各子系统健康值 (使用 component 字段)
            for subsystem_name in ['mechanical_system']:
                if subsystem_name in health_data:
                    subsystem_data = health_data[subsystem_name]
                    
                    # 所有记录都使用 components_name 参数（对应 component tag）
                    self.data_handler.write_health_score(
                        components_name=subsystem_name,  # mechanical_system
                        score=subsystem_data['score'],
                        components_score=subsystem_data['components'],
                        components_weight=subsystem_data['weights'],
                        timestamp=current_time
                    )

            # 写入故障信息
            for fault in diagnosis_result.get('faults', []):
                self.data_handler.write_fault_diagnosis(
                    device_name=fault.get('device_name', 'unknown'),
                    fault_type=fault.get('fault_type', 'unknown'),
                    fault_description=fault.get('fault_description', ''),
                    timestamp=current_time
                )

            self.logger.info("Results written to database successfully")

        except Exception as e:
            self.logger.error(f"Failed to write results: {e}", exc_info=True)

    def close(self):
        """Close monitor and cleanup"""
        try:
            self.data_handler.close()
            self.logger.info("IntegratedMonitor closed")
        except Exception as e:
            self.logger.error(f"Error during close: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'monitor_type': 'IntegratedMonitor',
            'features_enabled': self.enable_features,
            'data_path': str(self.vibration_data_path),
            'data_path_exists': self.vibration_data_path.exists(),
            'database_connected': self.data_handler is not None
        }