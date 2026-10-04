"""Portable mono PCM I/O."""

import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_pcm(path, sample_rate=16000):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = audio.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Input audio must be nonempty and finite")
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        wave = resample_poly(wave, sample_rate // divisor, rate // divisor)
    return np.rint(np.clip(wave * 32768, -32768, 32767)).astype("<i2").tobytes()


def save_pcm(path, pcm, sample_rate=24000):
    sf.write(path, np.frombuffer(pcm, dtype="<i2"), sample_rate, subtype="PCM_16")
