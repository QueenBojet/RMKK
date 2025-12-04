import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Literal
from src.functions.angular_resampling import angular_resampling

def order_mfcc(vibration_signal: np.ndarray,
               rpm_signal: np.ndarray,
               fs: float,
               order_range: Tuple[float, float] = (0.1, 10),
               num_coeffs: int = 13,
               num_filters: int = 20,
               rectification: Literal['log', 'cubic-root'] = 'log',
               log_energy: Literal['ignore', 'append', 'replace'] = 'append',
               delta_window_length: int = 9) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Extract MFCC coefficients from vibration signal in order domain

    Parameters:
    -----------
    vibration_signal : np.ndarray
        Input vibration signal (time domain)
    rpm_signal : np.ndarray
        RPM signal corresponding to vibration signal
    fs : float
        Sampling frequency (Hz)
    order_range : tuple, optional
        Order range for analysis [minOrder, maxOrder]. Default is (0.1, 10).
    num_coeffs : int, optional
        Number of coefficients returned. Default is 13.
    fft_length : int, optional
        FFT length for order domain analysis. Default is 2048.
    num_filters : int, optional
        Number of mel filters. Default is 20.
    rectification : str, optional
        Type of non-linear rectification ('log' or 'cubic-root'). Default is 'log'.
    log_energy : str, optional
        How the log energy is used ('append', 'replace', 'ignore'). Default is 'append'.
    delta_window_length : int, optional
        Delta and delta-delta window length. Must be odd. Default is 9.

    Returns:
    --------
    coeffs : np.ndarray
        MFCC coefficients
    delta : np.ndarray (optional)
        Delta coefficients (if requested via multiple returns)
    delta_delta : np.ndarray (optional)
        Delta-delta coefficients (if requested via multiple returns)
    """

    # Validate inputs
    if len(vibration_signal) != len(rpm_signal):
        raise ValueError('Vibration signal and RPM signal must have the same length.')

    if order_range[0] >= order_range[1]:
        raise ValueError('OrderRange must be [minOrder, maxOrder] where minOrder < maxOrder.')

    if delta_window_length % 2 == 0:
        raise ValueError(f'Parameter delta_window_length must be odd, got {delta_window_length}.')

    if rectification not in ['log', 'cubic-root']:
        raise ValueError(f"Invalid rectification: {rectification}. Must be 'log' or 'cubic-root'.")

    if log_energy not in ['ignore', 'append', 'replace']:
        raise ValueError(f"Invalid log_energy: {log_energy}. Must be 'ignore', 'append', or 'replace'.")

    # Convert input signals to 1D arrays
    vibration_signal = np.asarray(vibration_signal).flatten()
    rpm_signal = np.asarray(rpm_signal).flatten()

    min_order = order_range[0]
    max_order = order_range[1]

    # Convert time domain signal to order domain using the previously converted function
    _, _, order_spectrum, order_axis, _ = angular_resampling(
        vibration_signal, rpm_signal, fs, max_order=max_order
    )

    # Design mel filter bank for order domain
    # Create order-based band edges (similar to frequency band edges but in order domain)
    order_min = 2595 * np.log10(min_order * 700 + 1)
    order_max = 2595 * np.log10(max_order * 700 + 1)
    order1 = np.linspace(order_min, order_max, num_filters + 2)
    order_band_edges = (10 ** (order1 / 2595) - 1) / 700

    # Validate that we have enough bands for the requested coefficients
    num_valid_bands = max(len(order_band_edges) - 2, 0)
    if num_valid_bands < num_coeffs:
        raise ValueError(f'Not enough filter bands ({num_valid_bands}) for requested coefficients ({num_coeffs}). '
                         f'Reduce num_coeffs or increase order_range.')

    # Design mel-style filter bank for order domain
    filter_bank = design_order_mel_filter_bank(order_axis, order_band_edges, num_filters)

    # Calculate log energy if needed
    if log_energy != 'ignore':
        E = np.sum(np.abs(order_spectrum) ** 2)
        E = np.maximum(E, np.finfo(float).eps)  # Avoid log(0)
        log_E = np.log(E)

    # Calculate cepstral coefficients
    coeffs = cepstral_coefficients(filter_bank.T @ order_spectrum,
                                   num_coeffs=num_coeffs,
                                   rectification=rectification)

    # Handle log energy according to options
    if log_energy == 'append':
        log_E = np.array(log_E).reshape(1, 1)
        coeffs = np.column_stack([log_E, coeffs.reshape(1, -1)])
    elif log_energy == 'replace':
        log_E = np.array(log_E).reshape(1, 1)
        coeffs = np.column_stack([log_E, coeffs[:, 1:]])

    # Calculate delta and delta-delta if needed
    delta = calculate_delta(coeffs, delta_window_length)
    delta_delta = calculate_delta(delta, delta_window_length)

    return coeffs, delta, delta_delta


def order_gtcc(vibration_signal: np.ndarray,
               rpm_signal: np.ndarray,
               fs: float,
               order_range: Tuple[float, float] = (0.1, 10),
               num_coeffs: int = 13,
               fft_length: int = 2048,
               num_filters: int = 20,
               n: int = 120,  # Gammatone filter order
               rectification: Literal['log', 'cubic-root'] = 'log',
               log_energy: Literal['ignore', 'append', 'replace'] = 'append',
               delta_window_length: int = 9) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Extract GTCC coefficients from vibration signal in order domain

    Parameters:
    -----------
    vibration_signal : np.ndarray
        Input vibration signal (time domain)
    rpm_signal : np.ndarray
        RPM signal corresponding to vibration signal
    fs : float
        Sampling frequency (Hz)
    order_range : tuple, optional
        Order range for analysis [minOrder, maxOrder]. Default is (0.1, 10).
    num_coeffs : int, optional
        Number of coefficients returned. Default is 13.
    fft_length : int, optional
        FFT length for order domain analysis. Default is 2048.
    num_filters : int, optional
        Number of gammatone filters. Default is 20.
    n : int, optional
        Gammatone filter order. Default is 120.
    rectification : str, optional
        Type of non-linear rectification ('log' or 'cubic-root'). Default is 'log'.
    log_energy : str, optional
        How the log energy is used ('append', 'replace', 'ignore'). Default is 'append'.
    delta_window_length : int, optional
        Delta and delta-delta window length. Must be odd. Default is 9.

    Returns:
    --------
    coeffs : np.ndarray
        GTCC coefficients
    delta : np.ndarray (optional)
        Delta coefficients (if requested via multiple returns)
    delta_delta : np.ndarray (optional)
        Delta-delta coefficients (if requested via multiple returns)
    """

    # Validate inputs
    if len(vibration_signal) != len(rpm_signal):
        raise ValueError('Vibration signal and RPM signal must have the same length.')

    if order_range[0] >= order_range[1]:
        raise ValueError('OrderRange must be [minOrder, maxOrder] where minOrder < maxOrder.')

    if delta_window_length % 2 == 0:
        raise ValueError(f'Parameter delta_window_length must be odd, got {delta_window_length}.')

    if rectification not in ['log', 'cubic-root']:
        raise ValueError(f"Invalid rectification: {rectification}. Must be 'log' or 'cubic-root'.")

    if log_energy not in ['ignore', 'append', 'replace']:
        raise ValueError(f"Invalid log_energy: {log_energy}. Must be 'ignore', 'append', or 'replace'.")

    # Convert input signals to 1D arrays
    vibration_signal = np.asarray(vibration_signal).flatten()
    rpm_signal = np.asarray(rpm_signal).flatten()

    min_order = order_range[0]
    max_order = order_range[1]

    # Convert time domain signal to order domain using the previously converted function
    _, _, order_spectrum, order_axis, _ = angular_resampling(
        vibration_signal, rpm_signal, fs, max_order=max_order
    )

    # Create order-based band edges using ERB scale
    order_min = 2595 * np.log10(min_order * 700 + 1)
    order_max = 2595 * np.log10(max_order * 700 + 1)
    order1 = np.linspace(order_min, order_max, num_filters + 2)
    order_band_edges = (10 ** (order1 / 2595) - 1) / 700

    # Validate that we have enough bands for the requested coefficients
    num_valid_bands = max(len(order_band_edges) - 2, 0)
    if num_valid_bands < num_coeffs:
        raise ValueError(f'Not enough filter bands ({num_valid_bands}) for requested coefficients ({num_coeffs}). '
                         f'Reduce num_coeffs or increase order_range.')

    # Design ERB filter bank for order domain
    filter_bank = design_order_erb_filter_bank(order_axis, order_band_edges, num_filters, n)

    # Calculate log energy if needed
    if log_energy != 'ignore':
        E = np.sum(np.abs(order_spectrum) ** 2)
        E = np.maximum(E, np.finfo(float).eps)  # Avoid log(0)
        log_E = np.log(E)

    # Calculate cepstral coefficients
    coeffs = cepstral_coefficients(filter_bank.T @ order_spectrum,
                                   num_coeffs=num_coeffs,
                                   rectification=rectification)

    # Handle log energy according to options
    if log_energy == 'append':
        log_E = np.array(log_E).reshape(1, 1)
        coeffs = np.column_stack([log_E, coeffs.reshape(1, -1)])
    elif log_energy == 'replace':
        log_E = np.array(log_E).reshape(1, 1)
        coeffs = np.column_stack([log_E, coeffs[:, 1:]])

    # Calculate delta and delta-delta if needed
    delta = calculate_delta(coeffs, delta_window_length)
    delta_delta = calculate_delta(delta, delta_window_length)

    return coeffs, delta, delta_delta


def design_order_erb_filter_bank(order_axis: np.ndarray,
                                 order_band_edges: np.ndarray,
                                 num_filters: int,
                                 n: int = 4) -> np.ndarray:
    """
    Design ERB-based Gammatone filter bank for order domain

    Parameters:
    -----------
    order_axis : np.ndarray
        Order axis values
    order_band_edges : np.ndarray
        Band edge frequencies in order domain
    num_filters : int
        Number of filters
    n : int
        Filter order (typically 4 for Gammatone filters)

    Returns:
    --------
    filter_bank : np.ndarray
        Filter bank matrix (num_bins x num_filters)
    """
    num_bins = len(order_axis)
    filter_bank = np.zeros((num_bins, num_filters))
    center_freqs = order_band_edges[1:-1]  # Center frequencies

    for i in range(num_filters):
        fc = center_freqs[i]  # Center frequency

        # ERB bandwidth calculation
        ERB = 24.7 * (4.37 * fc / 1000 + 1)
        b = 1.019 * ERB  # Bandwidth parameter

        # Calculate frequency response for each bin
        for j in range(num_bins):
            f = order_axis[j]

            if f > 0:
                # Gammatone filter frequency response
                # Using the magnitude response of a Gammatone filter
                # |H(f)| = 1 / sqrt(1 + ((f - fc) / (b/2))^(2n))

                # Normalized frequency difference
                delta_f = (f - fc) / (b / 2)

                # Gammatone magnitude response
                filter_bank[j, i] = 1 / np.sqrt(1 + delta_f ** (2 * n))

        # Normalize each filter
        max_val = np.max(filter_bank[:, i])
        if max_val > 0:
            filter_bank[:, i] = filter_bank[:, i] / max_val

    return filter_bank


def hz2erb(freq_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert frequency in Hz to ERB scale

    Parameters:
    -----------
    freq_hz : float or np.ndarray
        Frequency in Hz

    Returns:
    --------
    erb : float or np.ndarray
        Frequency in ERB scale
    """
    return 21.4 * np.log10(0.00437 * freq_hz + 1)

def erb2hz(erb: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert ERB scale to frequency in Hz

    Parameters:
    -----------
    erb : float or np.ndarray
        Frequency in ERB scale

    Returns:
    --------
    freq_hz : float or np.ndarray
        Frequency in Hz
    """
    return (10 ** (erb / 21.4) - 1) / 0.00437

def design_order_mel_filter_bank(order_axis: np.ndarray,
                                 order_band_edges: np.ndarray,
                                 num_filters: int) -> np.ndarray:
    """
    Design mel-style filter bank for order domain

    Parameters:
    -----------
    order_axis : np.ndarray
        Order axis values
    order_band_edges : np.ndarray
        Band edge frequencies in order domain
    num_filters : int
        Number of filters

    Returns:
    --------
    filter_bank : np.ndarray
        Filter bank matrix (num_bins x num_filters)
    """
    num_bins = len(order_axis)
    filter_bank = np.zeros((num_bins, num_filters))

    for i in range(num_filters):
        # Triangle filter key points
        left = order_band_edges[i]
        center = order_band_edges[i + 1]
        right = order_band_edges[i + 2]

        # Build triangular filter
        for j in range(num_bins):
            if left <= order_axis[j] <= center:
                # Rising edge
                filter_bank[j, i] = (order_axis[j] - left) / (center - left)
            elif center <= order_axis[j] <= right:
                # Falling edge
                filter_bank[j, i] = (right - order_axis[j]) / (right - center)

        # Normalize
        max_val = np.max(filter_bank[:, i])
        if max_val > 0:
            filter_bank[:, i] = filter_bank[:, i] / max_val

    return filter_bank


def cepstral_coefficients(S: np.ndarray,
                          num_coeffs: int = 13,
                          rectification: str = 'log') -> np.ndarray:
    """
    Extract cepstral coefficients

    Parameters:
    -----------
    S : np.ndarray
        Input spectrogram (L x M) or (L x M x N)
        L - Number of frequency bands
        M - Number of frames
        N - Number of channels (optional)
    num_coeffs : int, optional
        Number of coefficients returned per frame. Default is 13.
    rectification : str, optional
        Type of nonlinear rectification ('log', 'cubic-root', or 'none'). Default is 'log'.

    Returns:
    --------
    coeffs : np.ndarray
        Cepstral coefficients (M x B) or (M x B x N)
        M - Number of frames
        B - Number of coefficients
        N - Number of channels (if input is 3D)
    """
    # Handle different input dimensions
    if S.ndim == 1:
        S = S.reshape(-1, 1)  # Convert to column vector
        squeeze_output = True
    else:
        squeeze_output = False

    original_shape = S.shape
    if S.ndim == 3:
        L, M, N = S.shape
        S = S.reshape(L, M * N)
    else:
        L, M = S.shape
        N = 1

    # Apply rectification
    if rectification == 'log':
        S = np.log10(np.maximum(S, np.finfo(float).eps))
    elif rectification == 'cubic-root':
        S = np.cbrt(S)
    # 'none' means no rectification

    # Create DCT matrix
    dct_matrix = create_dct_matrix(num_coeffs, L)

    # Apply DCT
    coeffs = dct_matrix @ S

    # Reshape back to original dimensions
    if N > 1:
        coeffs = coeffs.reshape(num_coeffs, M, N)
        coeffs = np.transpose(coeffs, (1, 0, 2))  # Put time along first dimension
    else:
        coeffs = coeffs.T  # M x num_coeffs
        if squeeze_output:
            coeffs = coeffs.squeeze()

    return coeffs


def create_dct_matrix(num_coeffs: int, num_bands: int) -> np.ndarray:
    """
    Create DCT matrix for cepstral coefficient calculation

    Parameters:
    -----------
    num_coeffs : int
        Number of coefficients to compute
    num_bands : int
        Number of frequency bands

    Returns:
    --------
    dct_matrix : np.ndarray
        DCT matrix (num_coeffs x num_bands)
    """
    # Create DCT-II matrix
    n = np.arange(num_bands)
    k = np.arange(num_coeffs).reshape(-1, 1)

    dct_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * num_bands))

    # Normalization
    dct_matrix[0, :] *= np.sqrt(1 / num_bands)
    dct_matrix[1:, :] *= np.sqrt(2 / num_bands)

    return dct_matrix


def calculate_delta(coeffs: np.ndarray, window_length: int) -> np.ndarray:
    """
    Calculate delta (first derivative) of coefficients

    Parameters:
    -----------
    coeffs : np.ndarray
        Input coefficients
    window_length : int
        Window length for delta calculation (must be odd)

    Returns:
    --------
    delta : np.ndarray
        Delta coefficients
    """
    if coeffs.shape[1] < window_length:
        raise ValueError('Not enough data points for delta calculation.')

    delta = np.zeros_like(coeffs)
    half_window = window_length // 2

    for i in range(coeffs.shape[0]):
        if i < half_window:
            # Use forward difference for beginning
            delta[i] = coeffs[min(i + 1, len(coeffs) - 1)] - coeffs[i]
        elif i >= coeffs.shape[0] - half_window:
            # Use backward difference for end
            delta[i] = coeffs[i] - coeffs[max(i - 1, 0)]
        else:
            # Use central difference for middle
            numerator = 0
            denominator = 0
            for j in range(1, half_window + 1):
                numerator += j * (coeffs[i + j] - coeffs[i - j])
                denominator += j ** 2
            delta[i] = numerator / (2 * denominator)

    return delta


def plot_order_mfcc(coeffs: np.ndarray,
                    log_energy_option: str,
                    min_order: float,
                    max_order: float):
    """
    Plot order domain MFCC coefficients

    Parameters:
    -----------
    coeffs : np.ndarray
        MFCC coefficients
    log_energy_option : str
        Log energy option ('append', 'replace', 'ignore')
    min_order : float
        Minimum order
    max_order : float
        Maximum order
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(coeffs.T, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Coefficient Value')
    plt.xlabel('Time Frame')
    plt.ylabel('MFCC Coefficient Index')
    plt.title('Order Domain MFCC Coefficients')

    # Adjust y-axis labels based on log energy option
    if log_energy_option == 'append':
        ytick_labels = ['logE'] + [str(i) for i in range(coeffs.shape[1] - 1)]
    elif log_energy_option == 'replace':
        ytick_labels = ['logE'] + [str(i) for i in range(1, coeffs.shape[1])]
    else:  # 'ignore'
        ytick_labels = [str(i) for i in range(coeffs.shape[1])]

    plt.yticks(range(len(ytick_labels)), ytick_labels)

    # Add order range information
    plt.text(0.02, 0.98, f'Order Range: {min_order:.1f} - {max_order:.1f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    plt.tight_layout()
    plt.show()


def plot_order_gtcc(coeffs: np.ndarray,
                    log_energy_option: str,
                    order_axis: np.ndarray,
                    min_order: float,
                    max_order: float):
    """
    Plot order domain GTCC coefficients

    Parameters:
    -----------
    coeffs : np.ndarray
        GTCC coefficients
    log_energy_option : str
        Log energy option ('append', 'replace', 'ignore')
    order_axis : np.ndarray
        Order axis values
    min_order : float
        Minimum order
    max_order : float
        Maximum order
    """
    plt.figure(figsize=(12, 6))

    # Plot coefficients
    plt.subplot(1, 2, 1)
    plt.imshow(coeffs.T, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Coefficient Value')
    plt.xlabel('Time Frame')
    plt.ylabel('GTCC Coefficient Index')
    plt.title('Order Domain GTCC Coefficients')

    # Adjust y-axis labels based on log energy option
    if log_energy_option == 'append':
        ytick_labels = ['logE'] + [str(i) for i in range(coeffs.shape[1] - 1)]
    elif log_energy_option == 'replace':
        ytick_labels = ['logE'] + [str(i) for i in range(1, coeffs.shape[1])]
    else:  # 'ignore'
        ytick_labels = [str(i) for i in range(coeffs.shape[1])]

    plt.yticks(range(len(ytick_labels)), ytick_labels)

    # Add order range information
    plt.text(0.02, 0.98, f'Order Range: {min_order:.1f} - {max_order:.1f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Plot coefficient distribution
    plt.subplot(1, 2, 2)
    mean_coeffs = np.mean(coeffs, axis=0)
    std_coeffs = np.std(coeffs, axis=0)
    x = np.arange(len(mean_coeffs))

    plt.errorbar(x, mean_coeffs, yerr=std_coeffs, fmt='o-', capsize=5)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Mean ± Std')
    plt.title('GTCC Coefficient Statistics')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, ytick_labels, rotation=45 if len(ytick_labels) > 10 else 0)

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate test signals
    fs = 10000  # Sampling frequency Hz
    t = np.arange(0, 10, 1 / fs)  # 10 second signal

    # Variable speed signal
    rpm_signal = 1800 + 300 * np.sin(2 * np.pi * 0.1 * t)  # RPM
    speed = rpm_signal / 60  # Convert to Hz

    # Generate vibration signal with multiple order components
    phase = np.cumsum(speed) / fs
    vibration_signal = (np.sin(2 * np.pi * 1 * phase) +  # 1st order
                        0.5 * np.sin(2 * np.pi * 2 * phase) +  # 2nd order
                        0.3 * np.sin(2 * np.pi * 3.5 * phase) +  # 3.5th order
                        0.1 * np.random.randn(len(t)))  # Noise

    # Extract order domain MFCC
    coeffs, delta, delta_delta = order_mfcc(
        vibration_signal,
        speed,  # Use Hz instead of RPM
        fs,
        order_range=(0.5, 10),
        num_coeffs=13,
        num_filters=20,
        rectification='log',
        log_energy='append',
        delta_window_length=3
    )

    print(f"MFCC shape: {coeffs.shape}")
    print(f"Delta shape: {delta.shape}")
    print(f"Delta-Delta shape: {delta_delta.shape}")

    # Plot results
    plot_order_mfcc(coeffs, 'append', 0.5, 10)

    # Extract order domain GTCC
    coeffs, delta, delta_delta = order_gtcc(
        vibration_signal,
        speed,  # Use Hz instead of RPM
        fs,
        order_range=(0.5, 10),
        num_coeffs=13,
        num_filters=20,
        n=4,  # Gammatone filter order
        rectification='log',
        log_energy='append',
        delta_window_length=9
    )

    print(f"GTCC shape: {coeffs.shape}")
    print(f"Delta shape: {delta.shape}")
    print(f"Delta-Delta shape: {delta_delta.shape}")

    # Calculate and display statistics
    print(f"\nGTCC Statistics:")
    print(f"Mean coefficients: {np.mean(coeffs, axis=0)}")
    print(f"Std coefficients: {np.std(coeffs, axis=0)}")

    # Plot results
    order_axis = np.linspace(0.5, 10, 100)  # Dummy order axis for plotting
    plot_order_gtcc(coeffs, 'append', order_axis, 0.5, 10)