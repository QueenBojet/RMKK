import os
import time
import psutil
import threading
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

class HeartbeatMonitor:
    """System heartbeat monitor"""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.full_config = config
        self.heartbeat_config = config.get('heartbeat', {})
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Configuration
        self.enable_heartbeat = self.heartbeat_config.get('enable_heartbeat', True)
        self.heartbeat_interval = self.heartbeat_config.get('heartbeat_interval_seconds', 30)
        self.heartbeat_timeout = self.heartbeat_config.get('heartbeat_timeout_seconds', 120)
        self.services_to_monitor = self.heartbeat_config.get('services_to_monitor', [])
        self.failure_actions = self.heartbeat_config.get('failure_actions', {})

        # Process service configurations
        self._preprocess_service_configs()

        # Runtime state
        self.running = False
        self.monitor_thread = None
        self.last_heartbeat_time = None
        self.service_status = {}

        # For data file monitoring
        self.last_file_check_time = {}
        self.last_known_files = {}

        # Statistics
        self.stats = {
            'total_checks': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'service_failures': {},
            'last_check_time': None,
            'uptime_start': None,
            'restart_count': 0
        }

        # Initialize service status
        for service in self.services_to_monitor:
            service_name = service['name']
            self.service_status[service_name] = {
                'status': 'unknown',
                'last_check': None,
                'consecutive_failures': 0,
                'total_failures': 0
            }
            self.stats['service_failures'][service_name] = 0

            # Initialize data file monitoring state
            if service['check_method'] == 'data_files':
                self.last_file_check_time[service_name] = datetime.now()
                self.last_known_files[service_name] = set()

    def _preprocess_service_configs(self):
        """Preprocess service configs to resolve path references"""
        try:
            for service in self.services_to_monitor:
                if 'path' in service:
                    path_value = service['path']

                    # Support path references:
                    # "$phm.save_path", "auto", None, "" - use PHM path
                    if path_value in ['$phm.save_path', 'auto', None, ''] or \
                            (isinstance(path_value, str) and path_value.startswith('$phm')):
                        phm_path = self._get_phm_save_path()
                        if phm_path:
                            service['path'] = phm_path
                            self.logger.info(f"Service '{service['name']}' using PHM data path: {phm_path}")
                        else:
                            self.logger.warning(f"Service '{service['name']}' cannot get PHM path, using default")
                            service['path'] = '.'

        except Exception as e:
            self.logger.error(f"Error preprocessing service configs: {e}", exc_info=True)

    def _get_phm_save_path(self) -> Optional[str]:
        """Get PHM data save path from configuration"""
        try:
            # Try multiple possible config paths
            possible_paths = [
                ['phm', 'data_collection', 'save_path'],  # Standard path
                ['phm', 'save_path'],  # Simplified path
                ['vibration_data', 'path'],  # Backup path
            ]

            for path_keys in possible_paths:
                try:
                    value = self.full_config
                    for key in path_keys:
                        value = value.get(key, {})
                        if not isinstance(value, dict) and value:
                            self.logger.debug(f"Found path from config {'.'.join(path_keys)}: {value}")
                            return str(value)
                except Exception:
                    continue

            self.logger.warning("Could not find PHM data save path in config")
            self.logger.debug(f"Available config keys: {list(self.full_config.keys())}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting PHM save path: {e}")
            return None

    def start(self):
        """Start heartbeat monitoring"""
        if not self.enable_heartbeat or self.running:
            return

        try:
            self.running = True
            self.stats['uptime_start'] = datetime.now()
            self.last_heartbeat_time = datetime.now()

            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="HeartbeatMonitor",
                daemon=True
            )
            self.monitor_thread.start()

            self.logger.info("Heartbeat monitoring started")

        except Exception as e:
            self.logger.error(f"Failed to start heartbeat monitoring: {e}")
            self.running = False

    def stop(self):
        """Stop heartbeat monitoring"""
        if not self.running:
            return

        try:
            self.running = False

            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)

            self._print_monitoring_summary()
            self.logger.info("Heartbeat monitoring stopped")

        except Exception as e:
            self.logger.error(f"Error stopping heartbeat monitoring: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.last_heartbeat_time = datetime.now()
                self._perform_health_check()
                time.sleep(self.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Heartbeat monitoring loop error: {e}")
                time.sleep(self.heartbeat_interval)

    def _perform_health_check(self):
        """Perform health check on all services"""
        try:
            self.stats['total_checks'] += 1
            self.stats['last_check_time'] = datetime.now()

            check_results = []
            for service in self.services_to_monitor:
                result = self._check_service(service)
                check_results.append(result)

            successful_count = sum(1 for r in check_results if r)
            failed_count = len(check_results) - successful_count

            if failed_count == 0:
                self.stats['successful_checks'] += 1
            else:
                self.stats['failed_checks'] += 1
                self._handle_failures()

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def _check_service(self, service: Dict[str, Any]) -> bool:
        """Check individual service health"""
        service_name = service['name']
        check_method = service['check_method']

        try:
            if check_method == 'ping':
                result = self._check_database_connection(service)
            elif check_method == 'file_access':
                result = self._check_file_access(service)
            elif check_method == 'data_files':
                result = self._check_data_files(service)
            elif check_method == 'memory_usage':
                result = self._check_memory_usage(service)
            elif check_method == 'disk_usage':
                result = self._check_disk_usage(service)
            else:
                self.logger.warning(f"Unknown check method: {check_method}")
                result = False

            self._update_service_status(service_name, result)
            return result

        except Exception as e:
            self.logger.error(f"Error checking service {service_name}: {e}")
            self._update_service_status(service_name, False)
            return False

    def _check_database_connection(self, service: Dict[str, Any]) -> bool:
        """Check database connection (ping)"""
        try:
            # Implement actual database ping here
            # For now, return True as placeholder
            return True
        except Exception:
            return False

    def _check_file_access(self, service: Dict[str, Any]) -> bool:
        """Check file access permissions"""
        try:
            file_path = service.get('path', '../threadMgt')

            if not os.path.exists(file_path):
                return False

            if not os.access(file_path, os.R_OK):
                return False

            if not os.access(file_path, os.W_OK):
                return False

            return True

        except Exception:
            return False

    def _check_data_files(self, service: Dict[str, Any]) -> bool:
        """Check data file generation"""
        try:
            service_name = service['name']
            file_path = Path(service.get('path', '../threadMgt'))
            file_pattern = service.get('file_pattern', '*.csv')
            max_age_minutes = service.get('max_file_age_minutes', 5)

            if not file_path.exists():
                self.logger.warning(f"Data file path does not exist: {file_path}")
                return False

            # Get current file list
            try:
                current_files = set(glob.glob(str(file_path / file_pattern)))
            except Exception as e:
                self.logger.error(f"Error scanning data files: {e}")
                return False

            current_time = datetime.now()

            # Check for new files
            if service_name in self.last_known_files:
                new_files = current_files - self.last_known_files[service_name]
                if new_files:
                    # New files detected
                    self.last_file_check_time[service_name] = current_time
                    self.last_known_files[service_name] = current_files
                    self.logger.debug(f"Detected {len(new_files)} new data files")
                    return True
            else:
                # First check, record current state
                self.last_known_files[service_name] = current_files
                self.last_file_check_time[service_name] = current_time
                return True

            # Check if most recent file is too old
            if current_files:
                newest_file_time = 0
                newest_file_path = None

                for file_path_str in current_files:
                    try:
                        file_mtime = os.path.getmtime(file_path_str)
                        if file_mtime > newest_file_time:
                            newest_file_time = file_mtime
                            newest_file_path = file_path_str
                    except OSError:
                        continue

                if newest_file_time > 0:
                    newest_file_dt = datetime.fromtimestamp(newest_file_time)
                    age_minutes = (current_time - newest_file_dt).total_seconds() / 60

                    if age_minutes <= max_age_minutes:
                        self.last_file_check_time[service_name] = current_time
                        self.last_known_files[service_name] = current_files
                        return True
                    else:
                        self.logger.warning(
                            f"Newest data file too old: {newest_file_path}, "
                            f"age: {age_minutes:.1f}min > {max_age_minutes}min"
                        )
                        return False
                else:
                    self.logger.warning("Cannot get file modification time")
                    return False
            else:
                self.logger.warning(f"No matching files found in data directory: {file_pattern}")
                return False

        except Exception as e:
            self.logger.error(f"Error checking data files: {e}")
            return False

    def _check_memory_usage(self, service: Dict[str, Any]) -> bool:
        """Check memory usage"""
        try:
            threshold = service.get('threshold', 80)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > threshold:
                self.logger.warning(f"Memory usage above threshold: {memory_percent:.1f}%")
                return False

            return True

        except Exception:
            return False

    def _check_disk_usage(self, service: Dict[str, Any]) -> bool:
        """Check disk usage"""
        try:
            threshold = service.get('threshold', 90)
            path = service.get('path', '/')

            disk = psutil.disk_usage(path)
            disk_percent = disk.percent

            if disk_percent > threshold:
                self.logger.warning(f"Disk usage above threshold: {disk_percent:.1f}%")
                return False

            return True

        except Exception:
            return False

    def _update_service_status(self, service_name: str, success: bool):
        """Update service status tracking"""
        if service_name not in self.service_status:
            self.service_status[service_name] = {
                'status': 'unknown',
                'last_check': None,
                'consecutive_failures': 0,
                'total_failures': 0
            }

        status = self.service_status[service_name]
        status['last_check'] = datetime.now()

        if success:
            status['status'] = 'healthy'
            status['consecutive_failures'] = 0
        else:
            status['status'] = 'unhealthy'
            status['consecutive_failures'] += 1
            status['total_failures'] += 1
            self.stats['service_failures'][service_name] = status['total_failures']

    def _handle_failures(self):
        """Handle service failures"""
        try:
            critical_services = []

            for service_name, status in self.service_status.items():
                if status['status'] == 'unhealthy' and status['consecutive_failures'] >= 3:
                    critical_services.append(service_name)

            if not critical_services:
                return

            self.logger.error(f"Critical service failures detected: {critical_services}")

            if self.failure_actions.get('notification', False):
                self._send_notification(critical_services)

        except Exception as e:
            self.logger.error(f"Failure handling failed: {e}")

    def _send_notification(self, failed_services: List[str]):
        """Send failure notification"""
        try:
            notification_message = f"Failed services: {', '.join(failed_services)}"
            self.logger.warning(f"Sending failure notification: {notification_message}")

        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")

    def is_alive(self) -> bool:
        """Check if monitor is alive"""
        if not self.enable_heartbeat:
            return True

        if not self.running:
            return False

        if self.last_heartbeat_time is None:
            return False

        time_since_heartbeat = (datetime.now() - self.last_heartbeat_time).total_seconds()
        return time_since_heartbeat < self.heartbeat_timeout

    def get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total_gb': memory.total / (1024 ** 3),
                    'available_gb': memory.available / (1024 ** 3),
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024 ** 3),
                    'free_gb': disk.free / (1024 ** 3),
                    'percent': disk.percent
                },
                'process': {
                    'memory_mb': process_memory.rss / (1024 ** 2),
                    'threads': process.num_threads()
                }
            }

        except Exception:
            return {}

    def _print_monitoring_summary(self):
        """Print monitoring summary"""
        if self.stats['uptime_start']:
            uptime = datetime.now() - self.stats['uptime_start']
            self.logger.info(f"Monitoring uptime: {uptime}")
            self.logger.info(f"Total checks: {self.stats['total_checks']}")
            self.logger.info(f"Successful checks: {self.stats['successful_checks']}")
            self.logger.info(f"Failed checks: {self.stats['failed_checks']}")