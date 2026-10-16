"""Numerische Hilfsfunktionen (NumPy) für Filter, die Pillow nicht direkt anbietet."""

from __future__ import annotations

import math

import numpy as np


def box_mean(x: np.ndarray, r: int) -> np.ndarray:
    """Mittelwert über ein (2r+1)²-Fenster mit Randfortsetzung (Summen-Tabellen, O(n))."""
    k = 2 * r + 1
    p = np.pad(x.astype(np.float64), ((r + 1, r), (r + 1, r)), mode="edge")
    c = p.cumsum(axis=0).cumsum(axis=1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def guided_filter(p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Kantenerhaltende Glättung (He et al., selbstgeführt). ``p`` liegt in [0, 1]."""
    mean_p = box_mean(p, r)
    var_p = box_mean(p * p, r) - mean_p * mean_p
    a = var_p / (var_p + eps)
    b = (1.0 - a) * mean_p
    return box_mean(a, r) * p + box_mean(b, r)


def _gaussian_transfer(shape: tuple[int, int], sigma: float) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    return np.exp(-2.0 * (math.pi**2) * sigma**2 * (fx**2 + fy**2)).astype(np.float32)


def _tv_divergence(u: np.ndarray) -> np.ndarray:
    """Divergenz des normierten Gradienten – Ableitung der Totalvariation."""
    ux = np.diff(u, axis=1, append=u[:, -1:])
    uy = np.diff(u, axis=0, append=u[-1:, :])
    norm = np.sqrt(ux * ux + uy * uy + 1e-4)
    nx, ny = ux / norm, uy / norm
    zeros_x = np.zeros_like(nx[:, :1])
    zeros_y = np.zeros_like(ny[:1, :])
    return (nx - np.concatenate([zeros_x, nx[:, :-1]], axis=1)) + (ny - np.concatenate([zeros_y, ny[:-1, :]], axis=0))


def richardson_lucy(channel: np.ndarray, sigma: float, iterations: int, tv: float = 0.0) -> np.ndarray:
    """Richardson-Lucy-Dekonvolution mit gaußförmiger Unschärfe.

    ``channel`` ist ein 2D-Array mit Werten in [0, 1]. Die gaußsche Punktspreizfunktion
    ist symmetrisch, daher genügt dieselbe Faltung für Vorwärts- und Rückprojektion.
    ``tv`` > 0 regularisiert mit Totalvariation (Dey et al.) und verhindert, dass
    Bildrauschen mitverstärkt wird.
    """
    pad = int(math.ceil(3 * sigma)) + 4
    pad_y = min(pad, channel.shape[0] - 1)
    pad_x = min(pad, channel.shape[1] - 1)
    observed = np.pad(channel.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    observed = np.maximum(observed, 0.0) + 1e-3  # Positivität für die multiplikative Iteration
    transfer = _gaussian_transfer(observed.shape, sigma)

    def blur(x: np.ndarray) -> np.ndarray:
        return np.fft.irfft2(np.fft.rfft2(x) * transfer, s=x.shape).astype(np.float32)

    estimate = observed.copy()
    for _ in range(iterations):
        correction = blur(observed / np.maximum(blur(estimate), 1e-4))
        if tv > 0:
            correction /= np.maximum(1.0 - tv * _tv_divergence(estimate), 0.2)
        estimate = np.clip(estimate * correction, 0.0, 2.0)

    estimate -= 1e-3
    return estimate[pad_y : pad_y + channel.shape[0], pad_x : pad_x + channel.shape[1]]


def rotated_crop_size(width: int, height: int, angle_deg: float) -> tuple[int, int]:
    """Größtes achsenparalleles Rechteck innerhalb eines um ``angle_deg`` gedrehten Bildes."""
    if width <= 0 or height <= 0:
        return 0, 0
    angle = math.radians(angle_deg)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (width * cos_a - height * sin_a) / cos_2a
        hr = (height * cos_a - width * sin_a) / cos_2a
    return max(1, int(wr + 1e-6)), max(1, int(hr + 1e-6))
