import numpy as np

from custom_behavior.optical_flow.signals.signal_buffer import SignalBuffer
from custom_behavior.optical_flow.models.signal_model import (
    SignalMetricsNames
)
from custom_behavior.optical_flow.signals.signal_filter import SignalFilter

class SignalEvaluator:

    def __init__(self, buffer):
        self.buffer = buffer
        self.signal_filter = SignalFilter()
        print("SignalEvaluator::init")

    # --------------------------------------------------
    # RMS шума (отклонение от EMA)
    # 
    # ema_window controls locality
    # 
    # Larger window → slower trend → higher RMS
    # 
    # Smaller window → more aggressive jitter detection
    # --------------------------------------------------
    #     
    def prepare_noise_std(self, window: int = 10):

        def noise_std(signal):
            # semantic filter (по enum)
            if not self.signal_filter.allows(signal.name, SignalMetricsNames.NOISE_STD):
                raise Exception(f'NOISE_STD doesn\'t support {signal.name}')

            self.buffer.update(signal)
            values = np.array(self.buffer.values(signal.name))

            if len(values) < window:
                return None

            # медленный тренд (EMA / SMA)
            trend = np.convolve(
                values,
                np.ones(window) / window,
                mode="same"
            )

            # локальный шум
            noise = values - trend

            return float(np.std(noise))

        return noise_std


    def prepare_noise_rms(self, window: int = 10):

        def noise_rms(signal):
            if not self.signal_filter.allows(signal.name, SignalMetricsNames.NOISE_RMS):
                raise Exception(f'NOISE_RMS doesn\'t support {signal.name}')

            self.buffer.update(signal)
            values = np.array(self.buffer.values(signal.name))

            if len(values) < window:
                return None

            return float(np.sqrt(np.mean(values ** 2)))

        return noise_rms


    # --------------------------------------------------
    # Спектральная плотность (HF энергия)
    # --------------------------------------------------
    def prepare_spectral_density(self):

        def spectral(signal):
            self.buffer.update(signal)
            values = np.array(self.buffer.values(signal.name))

            if len(values) < 8:
                return None

            fft = np.fft.rfft(values - np.mean(values))
            power = np.abs(fft) ** 2

            # доля энергии в ВЧ
            hf = power[len(power)//2:]
            return float(np.mean(hf))

        return spectral

    # --------------------------------------------------
    # Dropout rate (пропуски)
    # --------------------------------------------------
    # | fps видео | expected_dt |
    # | --------- | ----------- |
    # | 30        | 33 ms       |
    # | 60        | 16.6 ms     |
    # | 120       | 8.3 ms      |

    def prepare_dropout_rate(self, expected_dt):

        def dropout(signal):
            self.buffer.update(signal)
            ts = self.buffer.timestamps(signal.name)

            if len(ts) < 2:
                return 0.0

            gaps = np.diff(ts)
            dropouts = gaps > (1.5 * expected_dt)

            return float(np.mean(dropouts))

        return dropout

    # --------------------------------------------------
    # Устойчивость знака
    # --------------------------------------------------
    def prepare_sign_stability(self):

        def stability(signal):
            self.buffer.update(signal)
            values = np.array(self.buffer.values(signal.name))

            if len(values) < 3:
                return None

            signs = np.sign(values)
            signs = signs[signs != 0]

            if len(signs) == 0:
                return 0.0

            return float(abs(np.sum(signs)) / len(signs))

        return stability

    # --------------------------------------------------
    # Латентность (запаздывание реакции)
    # --------------------------------------------------
    def prepare_latency(self):

        def latency(signal):
            self.buffer.update(signal)
            values = np.array(self.buffer.values(signal.name))
            ts = np.array(self.buffer.timestamps(signal.name))

            if len(values) < 5:
                return None

            dv = np.diff(values)
            idx = np.argmax(np.abs(dv))

            if idx <= 0 or idx >= len(ts):
                return None

            return float(ts[-1] - ts[idx])

        return latency

    # --------------------------------------------------
    # Монотонность (пригодность для управления)
    # --------------------------------------------------
    def prepare_monotonic_coefficient(self):

        def monotonic(signal):
            self.buffer.update(signal)
            values = np.array(self.buffer.values(signal.name))

            if len(values) < 3:
                return None

            diffs = np.diff(values)
            signs = np.sign(diffs)
            non_zero = signs[signs != 0]

            if len(non_zero) == 0:
                return 0.0

            # 1.0 → строго монотонно
            # ≈0 → шум / колебания

            return float(abs(np.sum(non_zero)) / len(non_zero))

        return monotonic