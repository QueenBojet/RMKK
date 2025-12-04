import os
import json
import base64
import struct
import logging
import datetime as dt
import threading
from pathlib import Path
from collections import defaultdict


class mqttHandler:
    def __init__(self, save_path: str, sampling_schedules=None):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"数据将保存在目录: {self.save_path.absolute()}")
        self.sampling_schedules = sampling_schedules or []
        self.data_buffers = defaultdict(lambda: defaultdict(list))
        self.active_collections = {}
        self.timers = {}
        self.lock = threading.Lock()
        self.channel_to_schedule = self._build_channel_schedule_map()
        self._start_sampling_schedules()

    def _build_channel_schedule_map(self):
        """建立通道到采样计划的映射"""
        mapping = {}
        for schedule in self.sampling_schedules:
            for channel in schedule['channels']:
                mapping[channel] = schedule['name']
        return mapping

    def _schedule_next_collection(self, schedule):
        """为指定采样计划调度下一次采集"""
        schedule_name = schedule['name']
        interval_seconds = schedule['interval_minutes'] * 60

        def start_collection():
            self._start_collection_window(schedule)
            timer = threading.Timer(interval_seconds, start_collection)
            timer.daemon = True
            timer.start()
            self.timers[f"{schedule_name}_next"] = timer

        timer = threading.Timer(0, start_collection)
        timer.daemon = True
        timer.start()
        self.timers[f"{schedule_name}_next"] = timer

    def _start_sampling_schedules(self):
        """启动所有采样调度"""
        for schedule in self.sampling_schedules:
            self._schedule_next_collection(schedule)

    def _start_collection_window(self, schedule):
        """开始采集窗口"""
        schedule_name = schedule['name']

        with self.lock:
            # 清空该schedule的所有通道缓冲区
            self.data_buffers[schedule_name] = defaultdict(list)

            self.active_collections[schedule_name] = {
                'schedule': schedule,
                'start_time': dt.datetime.now(),
                'expected_end_time': dt.datetime.now() + dt.timedelta(seconds=schedule['duration_seconds'])
            }

            logging.info(
                f"开始采集计划 '{schedule_name}' (通道: {schedule['channels']})，持续 {schedule['duration_seconds']} 秒")

        end_timer = threading.Timer(
            schedule['duration_seconds'],
            self._end_collection_window,
            args=[schedule_name]
        )
        end_timer.daemon = True
        end_timer.start()
        self.timers[f"{schedule_name}_end"] = end_timer

    def _end_collection_window(self, schedule_name):
        """结束采集窗口并保存数据"""
        with self.lock:
            if schedule_name in self.active_collections:
                collection_info = self.active_collections[schedule_name]
                schedule = collection_info['schedule']
                buffered_data = dict(self.data_buffers[schedule_name])

                del self.active_collections[schedule_name]

                # 检查是否需要合并通道
                if schedule.get('merge_channels', False) and buffered_data:
                    self._save_merged_channel_data(schedule_name, collection_info, buffered_data)
                elif buffered_data:
                    for channel, data_points in buffered_data.items():
                        if data_points:
                            self._save_single_channel_data(channel, collection_info, data_points)
                else:
                    logging.warning(f"采样计划 '{schedule_name}' 采集窗口内未收到任何数据")

                self.data_buffers[schedule_name].clear()

    def _save_merged_channel_data(self, schedule_name, collection_info, buffered_data):
        """保存合并的多通道数据到单个文件"""
        try:
            schedule = collection_info['schedule']
            start_time = collection_info['start_time']
            channels = schedule['channels']

            # 生成文件名：包含所有通道信息
            time_for_filename = start_time.strftime('%Y%m%d_%H%M%S_%f')
            duration = schedule['duration_seconds']
            sampling_rate = schedule.get('sampling_rate', 25600)
            channel_str = '-'.join([f"ch{ch}" for ch in channels])
            filename = f"{channel_str}_{sampling_rate}Hz_{duration}s_{time_for_filename}.csv"
            filepath = self.save_path / filename
            merged_data = self._merge_channel_data(buffered_data, channels)

            if not merged_data:
                logging.warning(f"采样计划 '{schedule_name}' 没有可合并的数据")
                return

            # 写入CSV文件
            with open(filepath, 'w', newline='') as f:
                # 写入表头
                header = ['timestamp'] + [f'ch{ch}' for ch in channels]
                f.write(','.join(header) + '\n')

                # 写入数据行
                for row in merged_data:
                    f.write(','.join(str(v) for v in row) + '\n')

            total_points = len(merged_data)
            logging.info(f"已保存采样计划 '{schedule_name}' 的合并数据至 {filepath} (共 {total_points} 个时间点)")

        except Exception as e:
            logging.error(f"保存采样计划 '{schedule_name}' 合并数据时出错: {e}", exc_info=True)

    def _merge_channel_data(self, buffered_data, channels):
        """合并多个通道的数据，按时间戳对齐"""
        # 收集所有唯一时间戳
        all_timestamps = set()
        for channel_data in buffered_data.values():
            for point in channel_data:
                all_timestamps.add(point['timestamp'])

        if not all_timestamps:
            return []

        # 排序时间戳
        sorted_timestamps = sorted(all_timestamps)

        # 为每个通道建立时间戳到值的映射
        channel_maps = {}
        for channel in channels:
            channel_maps[channel] = {}
            if channel in buffered_data:
                for point in buffered_data[channel]:
                    channel_maps[channel][point['timestamp']] = point['value']

        # 构建合并数据
        merged_data = []
        for ts in sorted_timestamps:
            row = [ts.isoformat()]
            for channel in channels:
                value = channel_maps[channel].get(ts, '')
                row.append(value)
            merged_data.append(row)

        return merged_data

    def _save_single_channel_data(self, channel, collection_info, data_points):
        """保存单个通道数据（保持原有逻辑）"""
        try:
            start_time = collection_info['start_time']
            time_for_filename = start_time.strftime('%Y%m%d_%H%M%S_%f')
            duration = collection_info['schedule']['duration_seconds']
            filename = f"ch{channel}_{duration}s_time_{time_for_filename}.csv"
            filepath = self.save_path / filename

            data_points.sort(key=lambda x: x['timestamp'])

            with open(filepath, 'w', newline='') as f:
                f.write("timestamp,value\n")
                for point in data_points:
                    f.write(f"{point['timestamp'].isoformat()},{point['value']}\n")

            logging.info(f"已保存通道 {channel} 的数据至 {filepath}")

        except Exception as e:
            logging.error(f"保存通道 {channel} 数据时出错: {e}", exc_info=True)

    def _handle_waveform(self, external_code, payload):
        """处理波形数据"""
        try:
            # 查找该通道所属的schedule
            schedule_name = self.channel_to_schedule.get(external_code)
            if not schedule_name:
                logging.debug(f"通道 {external_code} 未配置采样计划，忽略数据")
                return

            with self.lock:
                # 检查该schedule是否在活跃采集窗口内
                if schedule_name not in self.active_collections:
                    return

                # 解析数据
                start_time_str = payload.get("SampleTime")
                freq = payload.get("Freq")
                base64_values = payload.get("Values")

                if not all([start_time_str, freq, base64_values]):
                    logging.error("波形数据负载缺少必要字段。")
                    return

                decoded_bytes = base64.b64decode(base64_values)
                num_floats = len(decoded_bytes) // 4
                values = struct.unpack(f'<{num_floats}f', decoded_bytes)

                start_time_obj = dt.datetime.fromisoformat(start_time_str)
                time_delta_sec = 1.0 / freq

                # 添加到对应schedule和通道的缓冲区
                for i, value in enumerate(values):
                    point_time = start_time_obj + dt.timedelta(seconds=i * time_delta_sec)
                    self.data_buffers[schedule_name][external_code].append({
                        'timestamp': point_time,
                        'value': value
                    })

                logging.debug(f"已缓冲通道 {external_code} (计划: {schedule_name}) 的 {len(values)} 个数据点")

        except Exception as e:
            logging.error(f"处理通道 {external_code} 的波形数据时出错: {e}", exc_info=True)

    def _save_buffered_waveform_data(self, channel, collection_info, data_points):
        """保存缓冲的波形数据为连续时间序列"""
        try:
            start_time = collection_info['start_time']

            # 生成文件名
            time_for_filename = start_time.strftime('%Y%m%d_%H%M%S_%f')
            duration = collection_info['schedule']['duration_seconds']
            filename = f"ch{channel}_{duration}s_time_{time_for_filename}.csv"
            filepath = self.save_path / filename

            # 将所有数据点按时间排序并写入文件
            data_points.sort(key=lambda x: x['timestamp'])

            with open(filepath, 'w', newline='') as f:
                f.write("timestamp,value\n")
                for point in data_points:
                    f.write(f"{point['timestamp'].isoformat()},{point['value']}\n")

            logging.info(f"已保存通道 {channel} 的连续数据至 {filepath}")

        except Exception as e:
            logging.error(f"保存通道 {channel} 缓冲数据时出错: {e}")

    def process_message(self, topic, payload_str):
        try:
            payload = json.loads(payload_str)
            topic_parts = topic.split('/')

            if len(topic_parts) >= 4:
                data_type = topic_parts[2]
                external_code = topic_parts[3]

                if data_type == "WaveData":
                    self._handle_waveform(external_code, payload)
                elif data_type == "EigenData":
                    eigen_type = topic_parts[4] if len(topic_parts) > 4 else "unknown"
                    self._handle_eigen_value(external_code, eigen_type, payload)
                else:
                    logging.warning(f"数据类型 '{data_type}' 的处理程序未实现。保存为原始JSON。")
                    self._save_raw_json(data_type, external_code, payload)
            else:
                logging.warning(f"在意外的主题格式上收到消息: {topic}")

        except json.JSONDecodeError:
            logging.error(f"无法从主题 {topic} 的负载中解析 JSON")
        except Exception as e:
            logging.error(f"在 process_message 中发生错误: {e}")

    def _handle_eigen_value(self, external_code, eigen_type, payload):
        """处理特征值数据"""
        try:
            sample_time = payload.get("SampleTime")
            value = payload.get("Value")

            if sample_time is None or value is None:
                logging.error("特征值数据负载缺少必要字段。")
                return

            filename = f"eigen_{external_code}_{eigen_type}.csv"
            filepath = os.path.join(self.save_path, filename)

            if not os.path.exists(filepath):
                with open(filepath, 'w', newline='') as f:
                    f.write("SampleTime,Value\n")

            with open(filepath, 'a', newline='') as f:
                f.write(f"{sample_time},{value}\n")

            logging.debug(f"已将特征值追加到 {filepath}")

        except Exception as e:
            logging.error(f"处理点位 {external_code} 的特征值数据时出错: {e}")

    def _save_raw_json(self, data_type, external_code, payload):
        """将未处理的消息负载保存为原始JSON文件。"""
        now = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{data_type}_{external_code}_{now}.json"
        filepath = os.path.join(self.save_path, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(payload, f, indent=4)
            logging.info(f"已将原始JSON负载保存至 {filepath}")
        except Exception as e:
            logging.error(f"保存原始JSON负载失败: {e}")

    def shutdown(self):
        """清理资源"""
        with self.lock:
            # 取消所有定时器
            for timer in self.timers.values():
                timer.cancel()
            self.timers.clear()

            # 保存剩余缓冲数据
            for channel, collection_info in self.active_collections.items():
                if self.data_buffers[channel]:
                    self._save_buffered_waveform_data(channel, collection_info, self.data_buffers[channel])

            self.active_collections.clear()
            self.data_buffers.clear()