import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from typing import Any, Dict, Optional, List, Tuple

class DiagnosisEngine:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.diag_config = self.config.get('diagnosis_config', {})
        self.health_config = self.diag_config.get('health_assessment', {})
        self.fault_config = self.diag_config.get('fault_diagnosis', {})
        
        self.logger.info("DiagnosisEngine initialized: Computed Order Tracking (COT) Enabled")

    def diagnose(self,
                 vibration_data: Dict[str, np.ndarray],
                 sampling_rate: float,
                 speed_data: Optional[np.ndarray] = None,
                 modbus_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        try:
            # 1. 基础校验
            if not vibration_data:
                return None

            # 简单的转速均值检查，用于快速判断设备状态
            mean_rpm = np.mean(speed_data) if (speed_data is not None and len(speed_data) > 0) else 0.0
            if mean_rpm < 10.0:
                self.logger.info(f"Equipment low speed ({mean_rpm:.1f} RPM). Skipping diagnosis.")
                return None

            # 2. 计算健康指标 (使用域不变特征，不受转速波动影响)
            channel_health = self._calculate_domain_invariant_health(vibration_data)
            system_score = self._aggregate_health_score(channel_health)
            
            # 3. 故障诊断 (仅在健康分低时触发)
            faults = []
            trigger_threshold = self.health_config.get('diagnosis_trigger_threshold', 85.0)
            
            if system_score < trigger_threshold:
                self.logger.warning(f"Health score ({system_score:.1f}) < {trigger_threshold}. Triggering COT Diagnosis.")
                
                # --- 核心变化：传入原始转速数组进行重采样 ---
                faults = self._diagnose_faults_with_cot(
                    vibration_data, 
                    sampling_rate, 
                    speed_data,  # 传入转速数组
                    channel_health
                )

            result = {
                'health_scores': {
                    'overall': {
                        'score': system_score,
                        'subsystems': {'mechanical_system': system_score},
                        'weights': {'mechanical_system': 1.0}
                    },
                    'mechanical_system': {
                        'score': system_score,
                        'components': {ch: data['score'] for ch, data in channel_health.items()},
                        'weights': {ch: 1.0/len(channel_health) if len(channel_health) > 0 else 0 for ch in channel_health}
                    },
                    'channels': channel_health
                },
                'faults': faults,
                'meta': {'rpm': mean_rpm}
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Diagnosis failed: {e}", exc_info=True)
            return None

    def _calculate_domain_invariant_health(self, vibration_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        scores = {}
        features_cfg = self.health_config.get('enabled_features', [])
        weights = self.health_config.get('feature_weights', {})
        limits = self.health_config.get('limits', {})
        
        for ch, data in vibration_data.items():
            if len(data) == 0: continue
            
            abs_data = np.abs(data)
            max_val = np.max(abs_data)
            rms = np.sqrt(np.mean(data**2))
            mean_abs = np.mean(abs_data)
            
            # 计算无量纲特征
            feats = {}
            if 'kurtosis' in features_cfg:
                mean_val = np.mean(data)
                std_val = np.std(data)
                feats['kurtosis'] = np.mean(((data - mean_val)/std_val)**4) if std_val > 0 else 3.0
            
            if 'crest_factor' in features_cfg:
                feats['crest_factor'] = max_val / rms
                
            if 'shape_factor' in features_cfg:
                feats['shape_factor'] = rms / mean_abs
                
            if 'impulse_factor' in features_cfg:
                feats['impulse_factor'] = max_val / mean_abs
                
            if 'margin_factor' in features_cfg:
                sq_mean = np.mean(np.sqrt(abs_data))**2
                feats['margin_factor'] = max_val / sq_mean if sq_mean > 0 else 0.0

            # 计算扣分
            total_deduction = 0.0
            for fname, fval in feats.items():
                w = weights.get(fname, 0.2)
                limit = limits.get(fname, 10.0)
                baseline = 3.0 if fname == 'kurtosis' else 1.0
                if fval > baseline:
                    ratio = min(1.0, (fval - baseline) / (limit - baseline))
                    deduction = ratio * w * 100.0
                    total_deduction += deduction
                    if deduction > 0.1:  # 只记录有意义的扣分
                        self.logger.debug(f"  {ch}.{fname}: {fval:.3f} -> deduction={deduction:.2f}")
            
            final_score = max(0.0, 100.0 - total_deduction)
            scores[ch] = {
                'score': final_score,
                'features': feats,
                'rms': rms
            }
            
            # 记录详细信息（改为DEBUG级别，只在需要时显示）
            self.logger.debug(f"Channel {ch}: RMS={rms:.4f}, Score={final_score:.1f}, Total Deduction={total_deduction:.1f}")
            self.logger.debug(f"  Features: {feats}")
            
        return scores

    def _resample_to_angle_domain(self, 
                                  vib_data: np.ndarray, 
                                  fs: float, 
                                  speed_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        核心算法：计算阶次跟踪 (Computed Order Tracking)
        将时域振动信号重采样到角度域。
        
        Args:
            vib_data: 振动原始数据 (N points)
            fs: 振动采样率
            speed_data: 转速原始数据 (M points, e.g., 5Hz)
        
        Returns:
            angle_domain_data: 角度域重采样后的数据
            points_per_rev: 每转采样点数 (用于阶次谱归一化)
        """
        n_vib = len(vib_data)
        n_speed = len(speed_data)
        
        duration = n_vib / fs
        t_vib = np.linspace(0, duration, n_vib)
        t_speed = np.linspace(0, duration, n_speed)
        
        try:
            speed_interpolator = interp1d(t_speed, speed_data, kind='cubic', fill_value="extrapolate")
            inst_speed_rpm = speed_interpolator(t_vib) 
        except Exception:
            speed_interpolator = interp1d(t_speed, speed_data, kind='linear', fill_value="extrapolate")
            inst_speed_rpm = speed_interpolator(t_vib)

        inst_speed_rpm = np.maximum(inst_speed_rpm, 1.0) 
        
        inst_freq_hz = inst_speed_rpm / 60.0
        dt = 1.0 / fs
        phase = np.cumsum(inst_freq_hz) * dt
        
        total_revolutions = phase[-1]
        
        even_phase = np.linspace(0, total_revolutions, n_vib)
        
        vib_interpolator = interp1d(phase, vib_data, kind='linear', bounds_error=False, fill_value=0.0)
        angle_domain_data = vib_interpolator(even_phase)
        
        points_per_rev = n_vib / total_revolutions
        
        return angle_domain_data, points_per_rev

    def _diagnose_faults_with_cot(self, 
                                  vibration_data: Dict[str, np.ndarray], 
                                  fs: float, 
                                  speed_data: np.ndarray,
                                  channel_health: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        使用阶次谱进行故障匹配
        """
        detected_faults = []
        fault_library = self.fault_config.get('fault_library', [])
        
        # 获取阶次分析的最大阶次设置 (e.g., 20)
        max_order_cfg = self.fault_config.get('max_order', 50.0)

        for ch, data in vibration_data.items():
            if channel_health[ch]['score'] > 90: continue
            
            try:
                # --- Step 1: 执行 COT 重采样 ---
                angle_data, points_per_rev = self._resample_to_angle_domain(data, fs, speed_data)
                
                # --- Step 2: 阶次谱计算 (FFT) ---
                # 在角度域做FFT，横坐标即为阶次 (Order)
                N = len(angle_data)
                yf = fft(angle_data)
                # 阶次轴构建
                # 频率 = k * Fs / N -> 阶次 = k * (Points_Per_Rev) / N
                xf_order = fftfreq(N, 1/points_per_rev)[:N//2] * points_per_rev 
                amplitude_spec = 2.0/N * np.abs(yf[0:N//2])
                
                # --- Step 3: 故障库匹配 ---
                for fault_def in fault_library:
                    target_order = fault_def['target_order']
                    tol = fault_def['tolerance']
                    threshold = fault_def['amplitude_threshold']
                    
                    # 仅在关心的阶次范围内查找
                    if target_order > max_order_cfg: continue

                    # 查找目标阶次附近的峰值
                    mask = (xf_order >= target_order - tol) & (xf_order <= target_order + tol)
                    
                    if np.any(mask):
                        measured_amp = np.max(amplitude_spec[mask])
                        
                        if measured_amp >= threshold:
                            detected_faults.append({
                                'device_name': fault_def.get('component', 'unknown'),
                                'fault_type': fault_def['name'],
                                'severity': fault_def['severity'],
                                'fault_description': (
                                    f"COT Fault on {ch}: {fault_def['name']} "
                                    f"(Order {target_order}X, Amp {measured_amp:.3f})"
                                ),
                                'timestamp': None
                            })
                            
            except Exception as e:
                self.logger.warning(f"COT calculation failed for channel {ch}: {e}")
                continue

        return detected_faults

    def _aggregate_health_score(self, channel_health):
        if not channel_health: return 100.0
        return min([v['score'] for v in channel_health.values()])