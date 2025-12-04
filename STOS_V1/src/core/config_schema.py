from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path


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


class SamplingSchedule(BaseModel):
    name: str
    channels: List[str]
    sampling_rate: int
    interval_minutes: int
    duration_seconds: int
    data_types: List[str]
    merge_channels: bool = False


class DataCollectionConfig(BaseModel):
    save_path: str
    sampling_schedules: List[SamplingSchedule] = []


class PhmConfig(BaseModel):
    server: Dict[str, str]  # Simplified as it's just auth info usually
    mqtt_client: MqttClientConfig
    mqtt_broker: MqttBrokerConfig
    data_collection: DataCollectionConfig


# ==================== Vibration & Speed ====================
class VibrationDataConfig(BaseModel):
    file_pattern: str
    sampling_rates_by_channel: Dict[str, int]
    columns: Dict[str, str]
    filename_time_format: str
    read_latest_only: bool = True


class SpeedTriggerConfig(BaseModel):
    enable: bool = False
    threshold_rpm: float = 10.0
    cooldown_seconds: float = 60.0
    check_interval_seconds: float = 1.0
    device_name: str
    point_name: str
    vibration_acquisition: Dict[str, float] = {}


# ==================== Diagnosis & Algorithms ====================
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
    frequency_resolution: float = 0.1
    max_order: float = 20.0
    fault_library: List[FaultLibraryItem] = []


class DiagnosisConfig(BaseModel):
    health_assessment: HealthAssessmentConfig
    fault_diagnosis: FaultDiagnosisConfig


# ==================== System & Test ====================
class HeartbeatService(BaseModel):
    name: str
    type: str
    check_method: str
    timeout: Optional[int] = None
    threshold: Optional[float] = None
    path: Optional[str] = None


class HeartbeatConfig(BaseModel):
    enable_heartbeat: bool = True
    heartbeat_interval_seconds: int = 60
    services_to_monitor: List[HeartbeatService] = []


class VibrationSimConfig(BaseModel):
    enabled: bool = False
    channels: List[str] = []
    sampling_rates: Dict[str, int] = {}
    duration_seconds: int = 1
    signal_params: Dict[str, Any] = {}


class ModbusSimSignal(BaseModel):
    device_name: str
    point_name: str
    data_type: str
    unit: str
    interval: float
    value_range: List[float]


class ModbusSimConfig(BaseModel):
    enabled: bool = False
    update_interval: float = 1.0
    signals: List[ModbusSimSignal] = []


class TestModeConfig(BaseModel):
    enabled: bool = False
    vibration_simulation: VibrationSimConfig = Field(default_factory=VibrationSimConfig)
    modbus_simulation: ModbusSimConfig = Field(default_factory=ModbusSimConfig)


# ==================== Root Config ====================
class AppConfig(BaseModel):
    """
    The Single Source of Truth for system configuration.
    """
    database: DatabaseConfig
    modbus: Optional[ModbusConfig] = None
    phm: Optional[PhmConfig] = None
    vibration_data: Optional[VibrationDataConfig] = None
    speed_trigger: Optional[SpeedTriggerConfig] = None

    # Diagnosis config might come from data_config.yaml, but we merge it here
    diagnosis_config: Optional[DiagnosisConfig] = None

    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    test_mode: TestModeConfig = Field(default_factory=TestModeConfig)

    # Catch-all for extra fields to prevent validation errors during migration
    # In a strict system, we would remove this.
    class Config:
        extra = "ignore"