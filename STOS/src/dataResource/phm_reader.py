import yaml
import logging
import paho.mqtt.client as mqtt
from src.dataResource.mqtt_handler import mqttHandler
from pathlib import Path
from src.utils.project_path import get_config_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)


def load_config(config_name):
    try:
        config_path = get_config_path(config_name)
        
        logging.info(f"正在从标准路径 '{config_path}' 加载配置...")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("配置加载成功。")
        return config
    except FileNotFoundError:
        logging.error(f"配置文件未找到，期望路径为: {config_path}")
        logging.error("请确保您的目录结构为 project_root/config/config.yaml")
        return None
    except yaml.YAMLError as e:
        logging.error(f"解析YAML文件时出错: {e}")
        return None


def on_connect(client, userdata, flags, rc):
    rc_meanings = {
        0: "连接成功",
        1: "连接被拒绝 - 不正确的协议版本",
        2: "连接被拒绝 - 无效的客户端标识符 (Client ID)",
        3: "连接被拒绝 - 服务器不可用",
        4: "连接被拒绝 - 错误的用户名或密码",
        5: "连接被拒绝 - 未经授权"
    }

    if rc == 0:
        logging.info(f"成功连接到 MQTT Broker! ({rc_meanings[rc]})")

        # 连接成功后，进行订阅
        logging.info("开始订阅主题...")

        phm_config = userdata['config'].get('phm', {})
        data_collection_config = phm_config.get('data_collection', {})
        sampling_schedules = data_collection_config.get('sampling_schedules', [])

        # 优先使用采样调度配置进行订阅
        if sampling_schedules:
            subscribed_channels = set()
            for schedule in sampling_schedules:
                for channel in schedule['channels']:
                    if channel not in subscribed_channels:
                        for data_type in schedule['data_types']:
                            topic = None
                            if data_type == 'waveform':
                                topic = f"PHM/Topics/WaveData/{channel}"
                            elif data_type == 'eigen_value':
                                topic = f"PHM/Topics/EigenData/{channel}/#"

                            if topic:
                                client.subscribe(topic)
                                logging.info(f"已订阅主题: {topic}")
                                subscribed_channels.add(channel)

        # 如果没有采样调度配置，回退到传统订阅配置
        else:
            subscriptions = data_collection_config.get('subscriptions', [])
            if not subscriptions:
                logging.warning("配置文件中既没有 'sampling_schedules' 也没有 'subscriptions'，将不会收到任何数据。")
                return

            for sub_item in subscriptions:
                code = sub_item['external_code']
                for data_type in sub_item['data_types']:
                    topic = None
                    if data_type == 'waveform':
                        topic = f"PHM/Topics/WaveData/{code}"
                    elif data_type == 'eigen_value':
                        topic = f"PHM/Topics/EigenData/{code}/#"

                    if topic:
                        client.subscribe(topic)
                        logging.info(f"已订阅主题: {topic}")
                    else:
                        logging.warning(f"配置文件中存在未知的 data_type: {data_type}，未进行订阅。")
    else:
        logging.error(f"MQTT Broker 连接失败! 返回码: {rc} ({rc_meanings.get(rc, '未知错误')})")
        logging.error("请重点检查:")
        logging.error(
            "1. `config.yaml` 中 `phm.mqtt_client` 的 username 和 password 是否与 PHM 平台后台配置的 MQTT 用户一致。")
        logging.error("2. `client_id` 是否被其他程序占用。")
        logging.error("3. 防火墙是否允许访问 Broker 的 IP 和端口。")

def on_message(client, userdata, msg):
    """当从订阅的主题接收到消息时的回调函数"""
    logging.debug(f"从主题 '{msg.topic}' 收到原始消息 (payload大小: {len(msg.payload)} bytes)")

    handler = userdata['handler']
    try:
        payload_str = msg.payload.decode('utf-8')
        handler.process_message(msg.topic, payload_str)
    except UnicodeDecodeError:
        logging.error(f"无法使用UTF-8解码来自主题 '{msg.topic}' 的消息，消息可能不是文本格式。")
    except Exception as e:
        logging.error(f"处理来自主题 '{msg.topic}' 的消息时发生未知错误: {e}")


def on_disconnect(client, userdata, rc):
    """当连接断开时的回调"""
    if rc != 0:
        logging.warning(f"与 MQTT Broker 的连接意外断开。返回码: {rc}。程序将会自动尝试重连。")
    else:
        logging.info("MQTT 连接正常断开。")


def main():
    config = load_config('config.yaml')
    if not config:
        logging.critical("无法加载配置，程序退出。")
        return

    phm_config = config.get('phm', {})
    if not phm_config:
        logging.critical("配置文件中未找到 'phm' 配置部分，程序退出。")
        return

    # 从 phm 配置块中加载 MQTT 服务器信息 ---
    logging.info("--- 阶段 1: 正在从配置文件获取 MQTT 服务器信息 ---")

    mqtt_broker_config = phm_config.get('mqtt_broker', {})
    if 'ip' not in mqtt_broker_config or 'port' not in mqtt_broker_config:
        logging.critical("配置文件 'phm.mqtt_broker'部分缺少 'ip' 或 'port'。")
        return

    logging.info("--- 阶段 2: 正在设置 MQTT 客户端 ---")

    data_collection_config = phm_config.get('data_collection', {})
    sampling_schedules = data_collection_config.get('sampling_schedules', [])

    if sampling_schedules:
        logging.info(f"加载了 {len(sampling_schedules)} 个采样调度配置")
        for schedule in sampling_schedules:
            logging.info(f"调度 '{schedule['name']}': 通道 {schedule['channels']}, "
                         f"间隔 {schedule['interval_minutes']} 分钟, "
                         f"持续 {schedule['duration_seconds']} 秒")

    handler = mqttHandler(
        save_path=data_collection_config.get('save_path'),
        sampling_schedules=sampling_schedules
    )

    userdata = {'config': config, 'handler': handler}

    mqtt_client_config = phm_config.get('mqtt_client', {})
    client_id = mqtt_client_config.get('client_id', f"phm_reader_{Path.cwd().name}")
    logging.info(f"创建 MQTT 客户端，Client ID: '{client_id}'")
    client = mqtt.Client(client_id=client_id)

    # 设置认证信息
    mqtt_username = mqtt_client_config.get('username')
    mqtt_password = mqtt_client_config.get('password')
    logging.info(f"设置 MQTT 用户名: '{mqtt_username}'")
    client.username_pw_set(username=mqtt_username, password=mqtt_password)

    # 关联 userdata 和回调函数
    client.user_data_set(userdata)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    broker_ip = mqtt_broker_config['ip']
    broker_port = mqtt_broker_config['port']
    logging.info(f"--- 阶段 3: 正在连接到 MQTT Broker ({broker_ip}:{broker_port}) ---")

    try:
        client.connect(broker_ip, broker_port, 60)
        client.loop_forever()

    except ConnectionRefusedError:
        logging.critical("MQTT 连接被拒绝。这通常是 Broker 端的配置问题。")
    except OSError as e:
        logging.critical(f"MQTT 连接失败，发生操作系统级错误 (例如网络不可达): {e}")
    except KeyboardInterrupt:
        logging.info("接收到手动中断信号 (Ctrl+C)，正在优雅退出...")
    except Exception as e:
        logging.critical(f"发生未预料的严重错误: {e}")
    finally:
        logging.info("程序结束，清理资源...")
        handler.shutdown()
        client.disconnect()


if __name__ == "__main__":
    main()