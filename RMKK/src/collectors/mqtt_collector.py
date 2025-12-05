import json
import logging
import base64
import struct
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt

from src.core.interfaces import Collector
from src.core.message_bus import MessageBus
from src.core.types import DataPoint, DataType, SignalSource
from src.core.config_schema import PhmConfig


class MqttCollector(Collector):
    """
    MQTT Data Collector.
    Subscribes to PHM topics, parses payloads, and pushes DataPoints.
    """

    def __init__(self, config: PhmConfig, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger(self.__class__.__name__)

        self._client = mqtt.Client(client_id=self.config.mqtt_client.client_id)
        if self.config.mqtt_client.username:
            self._client.username_pw_set(
                username=self.config.mqtt_client.username,
                password=self.config.mqtt_client.password
            )

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._running = False
        self._stats = {"messages_received": 0, "parse_errors": 0, "connected": False}

    def start(self) -> None:
        if self._running: return

        self.logger.info(f"Connecting to MQTT Broker {self.config.mqtt_broker.ip}:{self.config.mqtt_broker.port}...")
        try:
            self._client.connect(
                self.config.mqtt_broker.ip,
                self.config.mqtt_broker.port,
                keepalive=60
            )
            self._client.loop_start()
            self._running = True
        except Exception as e:
            self.logger.error(f"Failed to start MQTT collector: {e}")

    def stop(self) -> None:
        if not self._running: return
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()
        self.logger.info("MQTT Collector stopped.")

    def get_status(self) -> Dict[str, Any]:
        self._stats["connected"] = self._client.is_connected()
        return self._stats

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Connected to MQTT Broker.")
            self._stats["connected"] = True
            self._subscribe_topics()
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.logger.warning(f"Disconnected from MQTT Broker (rc={rc})")
        self._stats["connected"] = False

    def _subscribe_topics(self):
        """Subscribe based on sampling schedules."""
        # In the new config schema, we iterate over sampling_schedules to find channels
        topics = set()

        # 1. Subscribe based on schedules (Preferred)
        for schedule in self.config.data_collection.sampling_schedules:
            for channel in schedule.channels:
                # Waveform topic
                topics.add(f"PHM/Topics/WaveData/{channel}")
                # Eigen/Scalar topic
                topics.add(f"PHM/Topics/EigenData/{channel}/#")

        # Subscribe
        for t in topics:
            self._client.subscribe(t)
            self.logger.info(f"Subscribed to {t}")

    def _on_message(self, client, userdata, msg):
        """Handle incoming messages."""
        try:
            topic = msg.topic
            payload_str = msg.payload.decode('utf-8')
            payload = json.loads(payload_str)

            # Parse Topic: PHM/Topics/<Type>/<Channel>[/<SubType>]
            parts = topic.split('/')
            if len(parts) < 4:
                return

            data_category = parts[2]  # WaveData or EigenData
            channel = parts[3]

            dp: Optional[DataPoint] = None

            if data_category == "WaveData":
                dp = self._parse_waveform(channel, payload)
            elif data_category == "EigenData":
                sub_type = parts[4] if len(parts) > 4 else "unknown"
                dp = self._parse_eigen(channel, sub_type, payload)

            if dp:
                self.bus.publish(dp)
                self._stats["messages_received"] += 1

        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
            self._stats["parse_errors"] += 1

    def _parse_waveform(self, channel: str, payload: Dict) -> Optional[DataPoint]:
        """Parse Waveform JSON."""
        try:
            # Expected format: {"SampleTime": "...", "Freq": 25600, "Values": "base64..."}
            b64_vals = payload.get("Values")
            if not b64_vals: return None

            # Decode Base64 to floats
            raw_bytes = base64.b64decode(b64_vals)
            # Assuming float32 (4 bytes)
            count = len(raw_bytes) // 4
            values = struct.unpack(f'<{count}f', raw_bytes)

            timestamp_str = payload.get("SampleTime")
            ts = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

            return DataPoint(
                timestamp=ts,
                source=SignalSource.MQTT,
                type=DataType.WAVEFORM,
                device_name="towing_winch",  # Logical grouping
                point_name=channel,
                value=list(values),  # Convert tuple to list/array
                metadata={
                    "sampling_rate": payload.get("Freq", 25600),
                    "unit": "m/s2"  # Default assumption
                }
            )
        except Exception as e:
            self.logger.error(f"Waveform parse error: {e}")
            return None

    def _parse_eigen(self, channel: str, sub_type: str, payload: Dict) -> Optional[DataPoint]:
        """Parse Scalar JSON."""
        try:
            val = payload.get("Value")
            if val is None: return None

            timestamp_str = payload.get("SampleTime")
            ts = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

            return DataPoint(
                timestamp=ts,
                source=SignalSource.MQTT,
                type=DataType.SCALAR,
                device_name="towing_winch",
                point_name=f"{channel}_{sub_type}",
                value=float(val),
                metadata={
                    "feature_type": sub_type
                }
            )
        except Exception as e:
            self.logger.error(f"Eigen parse error: {e}")
            return None