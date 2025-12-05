from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# ... (Keep DatabaseConfig, ModbusConfig classes as is) ...
# ==================== Database ====================
class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 8086
    database: str = "industrial_data"
    username: Optional[str] = ""
    password: Optional[str] = ""
    timeout: int = 30
    timezone: str = "Asia/Shanghai"

# ==================== Modbus ====================
class ModbusServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 502

class ModbusCollectorConfig(BaseModel):
    poll_interval_seconds: float = 5.0
    signal_file_path: str = "config/signal_config.csv"

class ModbusConfig(BaseModel):
    server: ModbusServerConfig
    collector: ModbusCollectorConfig

# ==================== PHM / MQTT ====================
class MqttClientConfig(BaseModel):
    client_id: str
    username: str
    password: str

class MqttBrokerConfig(BaseModel):
    ip: str
    port: int

class DataCollectionConfig(BaseModel):
    save_path: str
    sampling_schedules: List[Any] = [] # Keeping generic to avoid validation errors if unused

class PhmConfig(BaseModel):
    server: Dict[str, str]
    mqtt_client: MqttClientConfig
    mqtt_broker: MqttBrokerConfig
    data_collection: DataCollectionConfig

# ==================== Speed Trigger (New) ====================
class ValidationConfig(BaseModel):
    min_valid_ratio: float = 0.8

class SpeedTriggerConfig(BaseModel):
    enable: bool = False
    threshold_rpm: float = 10.0
    cooldown_seconds: float = 1.0
    device_name: str
    point_name: str
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    vibration_acquisition: Dict[str, float] = {}

# ==================== Diagnosis ====================
class HealthAssessmentConfig(BaseModel):
    base_score: float = 100.0
    diagnosis_trigger_threshold: float = 85.0
    enabled_features: List[str]
    feature_weights: Dict[str, float]
    limits: Dict[str, float]

class FaultLibraryItem(BaseModel):
    name: str
    component: str
    target_order: float
    tolerance: float
    amplitude_threshold: float
    severity: str

class FaultDiagnosisConfig(BaseModel):
    max_order: float = 20.0
    fault_library: List[FaultLibraryItem] = []

class DiagnosisConfig(BaseModel):
    health_assessment: HealthAssessmentConfig
    fault_diagnosis: FaultDiagnosisConfig

# ==================== Output & Logging ====================
class LoggingConfig(BaseModel):
    enable_logging: bool = True
    log_path: str = "./logs"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size_mb: int = 100
    backup_count: int = 5

class OutputConfig(BaseModel):
    logging: Optional[LoggingConfig] = None

# ==================== Vibration & Test ====================
class VibrationDataConfig(BaseModel):
    filename_time_format: str = "%Y%m%d_%H%M%S"
    sampling_rates_by_channel: Dict[str, int] = {}

class VibrationSimConfig(BaseModel):
    enabled: bool = False
    channels: List[str] = []
    sampling_rates: Dict[str, int] = {}
    duration_seconds: int = 1
    signal_params: Dict[str, Any] = {}

class ModbusSimConfig(BaseModel):
    enabled: bool = False
    update_interval: float = 1.0

class TestModeConfig(BaseModel):
    enabled: bool = False
    vibration_simulation: VibrationSimConfig = Field(default_factory=VibrationSimConfig)
    modbus_simulation: ModbusSimConfig = Field(default_factory=ModbusSimConfig)

# ==================== Root Config ====================
class AppConfig(BaseModel):
    database: DatabaseConfig
    modbus: Optional[ModbusConfig] = None
    phm: Optional[PhmConfig] = None
    vibration_data: Optional[VibrationDataConfig] = None
    speed_trigger: Optional[SpeedTriggerConfig] = None
    output: Optional[OutputConfig] = None
    diagnosis_config: Optional[DiagnosisConfig] = None
    test_mode: TestModeConfig = Field(default_factory=TestModeConfig)

    class Config:
        extra = "ignore"