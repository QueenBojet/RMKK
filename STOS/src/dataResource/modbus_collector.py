import sys
import time
import logging
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from pymodbus.client import ModbusTcpClient
from pymodbus.constants import Endian
import struct

from src.utils.project_path import get_config_path, get_project_path
from src.dataResource.db_client import DataHandler
from src.utils.test_data_generator import TestDataManager


# 数据类型寄存器数量
DATA_TYPE_REG_COUNT = {
    "int16": 1, "uint16": 1,
    "int32": 2, "uint32": 2,
    "float32": 2,
    "int64": 4, "uint64": 4,
    "float64": 4,
    "bool": 1,
}


class CollectorState(Enum):
    INIT = 0
    CONNECTED = 1
    COLLECTING = 2
    STOPPED = 3


def decode_registers(registers: List[int], data_type: str, wordorder=Endian.LITTLE) -> Any:
    if data_type == 'float32':
        if wordorder == Endian.LITTLE:
            registers = [registers[1], registers[0]]
        byte_data = struct.pack('>HH', *registers)
        return struct.unpack('>f', byte_data)[0]
    
    # Int16
    elif data_type == 'int16':
        return struct.unpack('>h', struct.pack('>H', registers[0]))[0]
    
    # Uint16
    elif data_type == 'uint16':
        return registers[0]
    
    # Int32
    elif data_type == 'int32':
        if wordorder == Endian.LITTLE:
            registers = [registers[1], registers[0]]
        byte_data = struct.pack('>HH', *registers)
        return struct.unpack('>i', byte_data)[0]
    
    # Uint32
    elif data_type == 'uint32':
        if wordorder == Endian.LITTLE:
            registers = [registers[1], registers[0]]
        byte_data = struct.pack('>HH', *registers)
        return struct.unpack('>I', byte_data)[0]
    
    # Bool
    elif data_type == 'bool':
        return bool(registers[0])
    
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def load_config():
    config_path = get_config_path("config.yaml")
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_signals(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"信号文件不存在: {file_path}")
    
    # 读取 CSV
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    # 验证必需列
    required = ['device_name', 'point_name', 'address', 'data_type', 'register_type']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"信号文件缺少列: {missing}")
    
    # 清理数据
    df['unit'] = df['unit'].fillna('')
    df['scale_factor'] = df['scale_factor'].fillna(1.0)
    df['address'] = df['address'].astype(int)
    df['interval'] = df['interval'].fillna(1.0).astype(float)  # 添加interval列读取
    
    # 过滤无效类型
    df = df[df['data_type'].isin(DATA_TYPE_REG_COUNT.keys())]
    
    signals = df.to_dict('records')
    logging.info(f"加载了 {len(signals)} 个信号点")
    
    return signals


class ModbusCollector:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.state = CollectorState.INIT
        self.client: Optional[ModbusTcpClient] = None
        self.db_handler: Optional[DataHandler] = None
        self.signals: List[Dict[str, Any]] = []
        
        # 测试模式管理器
        self.test_data_manager = TestDataManager(config)
        self.test_mode = self.test_data_manager.is_enabled()
        
        # 加载配置
        self._setup_database()

        if not self.test_mode:
            self._setup_modbus_client()
            self._load_signals()
        else:
            self.logger.warning("*** Modbus 测试模式已启用 - 使用模拟数据 ***")
            # 测试模式下使用模拟信号配置
            self._load_test_signals()
        
        self.logger.info("ModbusCollector 初始化完成")
    
    def _setup_database(self):
        """设置数据库连接"""
        db_config = self.config.get('database', {})
        
        self.db_handler = DataHandler(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 8086),
            database=db_config.get('database', 'industrial_data'),
            timezone_str=db_config.get('timezone', 'Asia/Shanghai')
        )
        
        self.logger.info("数据库连接已建立")
    
    def _setup_modbus_client(self):
        """设置 Modbus 客户端"""
        server_config = self.config.get('modbus', {}).get('server', {})
        
        self.client = ModbusTcpClient(
            host=server_config.get('host', '127.0.0.1'),
            port=server_config.get('port', 502),
            timeout=10
        )
        
        self.client.byteorder = Endian.BIG
        self.client.wordorder = Endian.LITTLE
        
        self.logger.info("Modbus 客户端已配置")
    
    def _load_signals(self):
        """加载信号配置"""
        collector_config = self.config.get('modbus', {}).get('collector', {})
        self.poll_interval = collector_config.get('poll_interval_seconds', 5)
        
        signal_file_path = get_project_path(collector_config.get('signal_file_path'))
        self.signals = load_signals(Path(signal_file_path))
        
        self.logger.info(f"加载了 {len(self.signals)} 个信号，轮询间隔 {self.poll_interval}s")
    
    def _load_test_signals(self):
        """加载测试信号配置"""
        test_config = self.config.get('test_mode', {})
        modbus_sim = test_config.get('modbus_simulation', {})
        
        self.poll_interval = modbus_sim.get('update_interval', 5)
        
        # 从测试配置加载信号
        sim_signals = modbus_sim.get('signals', [])
        self.signals = []
        
        # 从配置读取采集间隔
        for sig in sim_signals:
            # 使用配置的interval，不再根据点名推断
            interval = sig.get('interval', 1.0)
            
            self.signals.append({
                'device_name': sig['device_name'],
                'point_name': sig['point_name'],
                'data_type': sig['data_type'],
                'unit': sig.get('unit', ''),
                'scale_factor': 1.0,
                'interval': interval,  # 使用配置的间隔
                'last_collect_time': 0,  # 上次采集时间
                # 测试模式特有：值范围
                'value_range': sig.get('value_range', [0, 100])
            })
        
        self.logger.info(f"[TEST MODE] 加载了 {len(self.signals)} 个测试信号")
    
    def connect(self) -> bool:
        """连接 Modbus 服务器"""
        if self.state == CollectorState.CONNECTED:
            return True
        
        server_config = self.config.get('modbus', {}).get('server', {})
        
        if self.client.connect():
            self.state = CollectorState.CONNECTED
            self.logger.info(f"已连接到 Modbus 服务器 {server_config['host']}:{server_config['port']}")
            return True
        else:
            self.logger.error("Modbus 连接失败")
            return False
    
    def read_signal(self, signal: Dict[str, Any]) -> Optional[float]:
        address = signal['address'] - 1
        data_type = signal['data_type']
        reg_type = signal['register_type']
        count = DATA_TYPE_REG_COUNT.get(data_type, 1)
        
        # 读取
        if reg_type == 'holding_registers':
            result = self.client.read_holding_registers(address=address, count=count)
        elif reg_type == 'input_registers':
            result = self.client.read_input_registers(address=address, count=count)
        elif reg_type == 'coils':
            result = self.client.read_coils(address=address, count=count)
            if not result.isError() and data_type == 'bool':
                return float(result.bits[0])
        elif reg_type == 'discrete_inputs':
            result = self.client.read_discrete_inputs(address=address, count=count)
            if not result.isError() and data_type == 'bool':
                return float(result.bits[0])
        else:
            self.logger.warning(f"不支持的寄存器类型: {reg_type}")
            return None
        
        if result.isError():
            self.logger.error(f"读取失败: {signal['point_name']}")
            return None
        
        # 解码
        try:
            value = decode_registers(result.registers, data_type)
            
            # 应用缩放
            scale_factor = signal.get('scale_factor', 1.0)
            return value * scale_factor
            
        except Exception as e:
            self.logger.error(f"解码失败 {signal['point_name']}: {e}")
            return None
    
    def collect_data(self) -> int:
        success_count = 0
        current_time = time.time()
        
        # === 测试模式：使用生成的数据 ===
        if self.test_mode:
            for signal in self.signals:
                # 检查是否到达采集时间
                interval = signal.get('interval', 1.0)
                last_time = signal.get('last_collect_time', 0)
                
                if current_time - last_time < interval:
                    continue  # 跳过，还没到采集时间
                
                # 生成数据
                data = {
                    'device_name': signal['device_name'],
                    'point_name': signal['point_name'],
                    'data_type': signal['data_type'],
                    'unit': signal['unit'],
                    'value': self.test_data_manager.modbus_generator.generate_value(signal)
                }
                
                try:
                    self.db_handler.write_signal_data(
                        device_name=data['device_name'],
                        point_name=data['point_name'],
                        data_type=data['data_type'],
                        unit=data['unit'],
                        interval_seconds=interval,
                        values=[data['value']]
                    )
                    success_count += 1
                    signal['last_collect_time'] = current_time  # 更新采集时间
                    
                    # 转速信号用DEBUG级别，其他用DEBUG
                    if 'speed' in data['point_name'].lower():
                        self.logger.debug(
                            f"[TEST MODE] 采集: {data['device_name']}.{data['point_name']} = {data['value']:.2f} {data['unit']}"
                        )
                    else:
                        self.logger.debug(
                            f"[TEST MODE] 采集: {data['device_name']}.{data['point_name']} = {data['value']:.2f} {data['unit']}"
                        )
                    
                except Exception as e:
                    self.logger.error(f"写入数据库失败 {data['point_name']}: {e}")
            
            return success_count
        
        # === 生产模式：真实采集 ===
        for signal in self.signals:
            value = self.read_signal(signal)
            
            if value is not None:
                # 写入数据库
                try:
                    self.db_handler.write_signal_data(
                        device_name=signal['device_name'],
                        point_name=signal['point_name'],
                        data_type=signal['data_type'],
                        unit=signal.get('unit', ''),
                        interval_seconds=self.poll_interval,
                        values=[value]
                    )
                    success_count += 1
                    self.logger.debug(f"采集: {signal['device_name']}.{signal['point_name']} = {value:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"写入数据库失败 {signal['point_name']}: {e}")
        
        return success_count
    
    def run(self):
        mode_tag = "[TEST MODE] " if self.test_mode else ""
        self.logger.info(f"{mode_tag}Modbus 采集器启动")
        
        # 连接 Modbus
        if not self.test_mode:
            if not self.connect():
                raise RuntimeError("无法连接到 Modbus 服务器")
        else:
            self.logger.info("[TEST MODE] 跳过 Modbus 连接，使用模拟数据")
        
        self.state = CollectorState.COLLECTING
        
        try:
            while self.state == CollectorState.COLLECTING:
                start_time = time.time()
                
                # 采集数据
                success_count = self.collect_data()
                self.logger.info(f"采集周期完成: {success_count}/{len(self.signals)} 成功")
                
                # 等待下一个周期
                elapsed = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("收到中断信号")
        finally:
            self.shutdown()
    
    def shutdown(self):
        self.logger.info("关闭采集器...")
        self.state = CollectorState.STOPPED
        
        if self.client and self.client.is_socket_open():
            self.client.close()
        
        if self.db_handler:
            self.db_handler.close()
        
        self.logger.info("采集器已关闭")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # 加载配置
        config = load_config()
        
        # 创建并运行采集器
        collector = ModbusCollector(config)
        collector.run()
        
    except KeyboardInterrupt:
        logging.info("用户中断")
    except Exception as e:
        logging.critical(f"程序失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
