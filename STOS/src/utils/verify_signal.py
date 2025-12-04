import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.test_data_generator import VibrationDataGenerator

def verify():
    config = {
        'test_mode': {
            'vibration_simulation': {
                'channels': ['ch1'],
                'sampling_rates': {'ch1': 25600},
                'duration_seconds': 1,
                'signal_params': {
                    'base_frequency': 50,
                    'amplitude': 10.0,
                    'noise_level': 0.1,
                    'health_score': 85
                }
            }
        }
    }
    
    generator = VibrationDataGenerator(config)
    signal = generator.generate_signal('ch1', 25600)
    
    rms = np.sqrt(np.mean(signal**2))
    abs_mean = np.mean(np.abs(signal))
    
    print(f"RMS: {rms:.4f}")
    print(f"Absolute Mean: {abs_mean:.4f}")
    
    if rms < 1.0 or abs_mean < 1.0:
        print("FAIL: Signal too weak")
        sys.exit(1)
    else:
        print("PASS: Signal strength adequate")

if __name__ == "__main__":
    verify()
