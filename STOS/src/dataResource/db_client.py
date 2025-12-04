import pandas as pd
import json
from influxdb import InfluxDBClient
from typing import Dict, List, Optional, Union
import time
import pytz
from datetime import datetime, timedelta

class DataHandler:
    # 类变量，用于防止重复打印初始化信息
    _init_message_printed = False

    def __init__(self, host='localhost', port=8086, database='industrial_data',
                 timezone_str='Asia/Shanghai', database_stores_local_time=True):
        self.client = InfluxDBClient(host=host, port=port)
        self.database = database
        self.client.create_database(database)
        self.client.switch_database(database)

        # 设置时区
        try:
            self.timezone = pytz.timezone(timezone_str)
        except pytz.exceptions.UnknownTimeZoneError:
            print(f"未知时区 {timezone_str}，使用默认时区 Asia/Shanghai")
            self.timezone = pytz.timezone('Asia/Shanghai')

        # 时区处理选项
        self.database_stores_local_time = database_stores_local_time

        if not DataHandler._init_message_printed:
            print(f"数据库客户端初始化完成，使用时区: {self.timezone}")
            print(f"数据库时间处理模式: {'本地时间' if database_stores_local_time else 'UTC时间'}")
            DataHandler._init_message_printed = True

    def _to_configured_timezone(self, dt: Optional[datetime] = None) -> datetime:
        """转换为配置的时区时间"""
        if dt is None:
            # 获取当前时间并转换到配置的时区
            return datetime.now(self.timezone)
        elif dt.tzinfo is None:
            return self.timezone.localize(dt)
        else:
            # 如果已有时区信息，转换到配置的时区
            return dt.astimezone(self.timezone)

    def _to_utc_string(self, dt: datetime) -> str:
        """将时间转换为数据库存储格式"""
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)

        if self.database_stores_local_time:
            # 如果数据库存储本地时间，转换到配置的时区
            local_time = dt.astimezone(self.timezone)
            # 去掉时区信息，作为本地时间存储
            return local_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        else:
            # 如果数据库存储UTC时间，转换到UTC
            utc_time = dt.astimezone(pytz.UTC)
            return utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _parse_influx_time(self, time_str: str) -> datetime:
        """解析InfluxDB返回的时间字符串"""
        if not time_str:
            return None

        # 解析时间字符串
        if 'T' in time_str:
            time_str = time_str.replace('Z', '').replace('+00:00', '')
            if '.' in time_str:
                # 处理微秒
                parts = time_str.split('.')
                dt = datetime.strptime(parts[0] + '.' + parts[1][:6], '%Y-%m-%dT%H:%M:%S.%f')
            else:
                dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        else:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

        if self.database_stores_local_time:
            # 如果数据库存储的是本地时间，直接标记为配置的时区
            return self.timezone.localize(dt)
        else:
            # 如果数据库存储的是UTC时间，需要转换
            utc_time = pytz.UTC.localize(dt)
            return utc_time.astimezone(self.timezone)

    def _write_data(self, data: pd.DataFrame, measurement: str, tags: dict = None,
                    start_time: datetime = None, time_interval_ms: int = 100,
                    batch_size: int = 5000) -> int:
        """内部写入方法"""
        data = data.dropna()
        start_time = self._to_configured_timezone(start_time)
        total = len(data)
        written = 0

        for i in range(0, total, batch_size):
            batch = data.iloc[i:i + batch_size]
            points = []

            for idx, row in batch.iterrows():
                timestamp = start_time + timedelta(milliseconds=(i + idx) * time_interval_ms)

                fields = {}
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        fields[k] = float(v)
                    elif isinstance(v, str) and k not in ['device_name', 'point_name', 'data_type', 'unit']:
                        fields[k] = str(v)
                    elif not isinstance(v, str):
                        fields[k] = str(v)

                point = {
                    "measurement": measurement,
                    "time": self._to_utc_string(timestamp),
                    "fields": fields
                }
                if tags:
                    point["tags"] = tags

                points.append(point)

            self.client.write_points(points)
            written += len(points)
            if batch_size > 1000:
                time.sleep(0.05)

        return written

    def _read_data(self, measurement: str, where: str = None,
                   limit: int = 1000, order: str = 'ASC') -> pd.DataFrame:
        """内部读取方法"""
        query = f"SELECT * FROM {measurement}"
        if where:
            query += f" WHERE {where}"
        query += f" ORDER BY time {order} LIMIT {limit}"

        # 添加调试日志
        # print(f"[DEBUG] InfluxDB Query: {query}")

        result = self.client.query(query)
        points = list(result.get_points())

        #print(f"[DEBUG] Query returned {len(points)} points")

        if points:
            df = pd.DataFrame(points)
            # 转换时间列到配置的时区
            if 'time' in df.columns:
                df['time'] = df['time'].apply(self._parse_influx_time)
            return df
        else:
            return pd.DataFrame()

    # ==================== Signal Data ====================
    def write_signal_data(self, device_name: str, point_name: str, data_type: str,
                          unit: Optional[str], interval_seconds: float,
                          values: Union[List[float], pd.DataFrame],
                          start_time: Optional[datetime] = None) -> int:
        """写入实时信号数据"""
        tags = {
            'device_name': device_name,
            'point_name': point_name,
            'data_type': data_type,
            'unit': unit if unit else '',
            'interval_seconds': str(int(interval_seconds))
        }

        data = pd.DataFrame({'value': values}) if isinstance(values, list) else values
        return self._write_data(data, 'signal_data', tags,
                                self._to_configured_timezone(start_time),
                                int(interval_seconds * 1000))

    def read_signal_data(self, device_name: Optional[str] = None,
                         point_name: Optional[str] = None,
                         data_type: Optional[str] = None,
                         time_range: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 10000,
                         order: str = 'ASC') -> pd.DataFrame:
        """读取实时信号数据"""
        where_conditions = []
        if device_name:
            where_conditions.append(f"device_name = '{device_name}'")
        if point_name:
            where_conditions.append(f"point_name = '{point_name}'")
        if data_type:
            where_conditions.append(f"data_type = '{data_type}'")

        if start_time and end_time:
            # 确保时间有时区信息
            start_time = self._to_configured_timezone(start_time)
            end_time = self._to_configured_timezone(end_time)

            # 转换为UTC字符串
            start_str = self._to_utc_string(start_time).rstrip('Z')
            end_str = self._to_utc_string(end_time).rstrip('Z')

            where_conditions.append(f"time >= '{start_str}Z' AND time <= '{end_str}Z'")
        elif time_range:
            where_conditions.append(f"time > now() - {time_range}")

        where_clause = " AND ".join(where_conditions) if where_conditions else None
        return self._read_data('signal_data', where_clause, limit, order)

    def get_latest_signal_data(self, device_name: str, point_name: str) -> Optional[Dict]:
        """获取指定设备和测点的最新数据"""
        df = self.read_signal_data(device_name=device_name, point_name=point_name, limit=1)
        return df.iloc[-1].to_dict() if not df.empty else None

    # ==================== Fault Diagnosis ====================
    def write_fault_diagnosis(self, device_name: str, fault_type: str,
                              fault_description: str,
                              timestamp: Optional[datetime] = None) -> int:
        """写入故障诊断数据"""
        tags = {'device_name': device_name}
        data = pd.DataFrame({
            'fault_type': [fault_type],
            'fault_description': [fault_description]
        })
        return self._write_data(data, 'fault_diagnosis', tags,
                                self._to_configured_timezone(timestamp), 1000)

    def write_fault_diagnosis_batch(self, faults: List[Dict[str, str]],
                                    start_time: Optional[datetime] = None) -> int:
        """批量写入故障诊断数据"""
        return sum(self.write_fault_diagnosis(f['device_name'], f['fault_type'],
                                              f['fault_description'], start_time)
                   for f in faults)

    def read_fault_diagnosis(self, device_name: Optional[str] = None,
                             fault_type: Optional[str] = None,
                             time_range: Optional[str] = None,
                             limit: int = 1000) -> pd.DataFrame:
        """读取故障诊断数据"""
        where_conditions = []
        if device_name:
            where_conditions.append(f"device_name = '{device_name}'")
        if fault_type:
            where_conditions.append(f"fault_type = '{fault_type}'")
        if time_range:
            where_conditions.append(f"time > now() - {time_range}")

        where_clause = " AND ".join(where_conditions) if where_conditions else None
        return self._read_data('fault_diagnosis', where_clause, limit)

    # ==================== Health Score ====================
    def write_health_score(self, score: float,
                           components_score: Dict[str, float],
                           components_weight: Dict[str, float],
                           timestamp: Optional[datetime] = None,
                           components_name: Optional[str] = None) -> int:
        if not components_name:
            raise ValueError("必须提供 components_name")
        
        tags = {'component': components_name}
        
        data = pd.DataFrame({
            'score': [score],
            'components_score': [json.dumps(components_score)],
            'components_weight': [json.dumps(components_weight)]
        })
        return self._write_data(data, 'health_score', tags,
                                self._to_configured_timezone(timestamp), 600000)

    def read_health_score(self, components_name: Optional[str] = None,
                          time_range: Optional[str] = None,
                          limit: int = 1000) -> pd.DataFrame:
        where_conditions = []
        if components_name:
            where_conditions.append(f"algorithm_name = '{components_name}'")
        if time_range:
            where_conditions.append(f"time > now() - {time_range}")

        where_clause = " AND ".join(where_conditions) if where_conditions else None
        df = self._read_data('health_score', where_clause, limit)

        if not df.empty:
            if 'components_score' in df.columns:
                df['components_score'] = df['components_score'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x)
            if 'components_weight' in df.columns:
                df['components_weight'] = df['components_weight'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x)

        return df

    def get_latest_health_score(self, algorithm_name: str) -> Optional[Dict]:
        df = self.read_health_score(components_name=algorithm_name, limit=1)
        return df.iloc[-1].to_dict() if not df.empty else None

    # ==================== Utility Methods ====================
    def get_all_devices(self) -> List[str]:
        """获取所有设备名称列表"""
        query = "SHOW TAG VALUES FROM signal_data WITH KEY = device_name"
        result = self.client.query(query)
        return [p['value'] for p in result.get_points()] if result else []

    def get_device_points(self, device_name: str) -> List[str]:
        """获取指定设备的所有测点名称"""
        query = f"SHOW TAG VALUES FROM signal_data WITH KEY = point_name WHERE device_name = '{device_name}'"
        result = self.client.query(query)
        return [p['value'] for p in result.get_points()] if result else []

    def close(self):
        """关闭数据库连接"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()