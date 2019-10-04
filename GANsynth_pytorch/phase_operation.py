import librosa
import numpy as np
from intervaltree import Interval, IntervalTree
import torch


def select(x: np.ndarray, axis: int, index: int, keepdims: bool = True
           ) -> np.ndarray:
    """Indexing over arbitrary dimension

    Equivalent to torch.Tensor().select

    Arguments:
        x (np.ndarray)
        axis (int):
            The dimension over which to index
        index (int)
        keepdims (bool):
            if False, the indexed dimension is removed in the return value
    Returns:
        np.ndarray:
            The selected sub-array
    """
    x_permuted = np.swapaxes(x, 0, axis)
    selected_permuted = x_permuted[0]
    if keepdims:
        selected = np.swapaxes(np.expand_dims(selected_permuted,
                                              axis=0),
                               0, axis)
    else:
        selected = np.swapaxes(selected_permuted, 0, axis-1)

    return selected


def diff(x: np.ndarray, axis: int) -> np.ndarray:
    """Take the finite difference of a tensor along an axis.
    Arguments:
        x (np.ndarray):
            Input tensor of any dimension.
        axis (int):
            Axis on which to take the finite difference.
    Returns:
        d (np.ndarray):
            Tensor with size less than x by 1 along the difference dimension.
    Raises:
        ValueError: Axis out of range for tensor.
    """
    # bring dimension for finite difference to first dimension
    x_permuted = np.swapaxes(x, 0, axis)

    # compute finite difference over the first dimension
    d_permuted = x_permuted[1:] - x_permuted[:-1]

    # transpose the dimensions back
    d = np.swapaxes(d_permuted, 0, axis)
    return d


def unwrap(p: np.ndarray, discont: float = np.pi, axis: int = -1
           ) -> np.ndarray:
    """Unwrap a cyclical phase tensor.

    Arguments:
        p (np.ndarray):
            Phase tensor.
        discont (float):
            size of the cyclic discontinuity.
        axis (int):
            Axis over which to unwrap.
    Returns:
        unwrapped (np.ndarray):
            Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    ddmod = np.mod(dd + np.pi, 2.0*np.pi) - np.pi

    idx = np.logical_and(np.equal(ddmod, -np.pi), np.greater(dd, 0))
    ddmod = np.where(idx, np.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd

    idx = np.less(np.abs(dd), discont)
    ddmod = np.where(idx, np.zeros_like(ddmod), dd)
    ph_cumsum = np.cumsum(ph_correct, axis=axis)

    shape = np.array(p.shape)
    shape[axis] = 1
    ph_cumsum = np.concatenate([np.zeros(shape, dtype=p.dtype), ph_cumsum],
                               axis=axis)
    unwrapped = p + ph_cumsum
    return unwrapped


def instantaneous_frequency(phase_angle: np.ndarray, time_axis: int
                            ) -> np.ndarray:
    """Transform a fft tensor from phase angle to instantaneous frequency.

    Unwrap and take the finite difference of the phase.
    Pad with initial phase to keep the tensor the same size.

    Arguments:
        phase_angle:
            Tensor of angles in radians. [Batch, Time, Freqs]
        time_axis:
            Axis over which to unwrap and take finite difference.
    Returns:
        dphase:
            Instantaneous frequency (derivative of phase). Same size as input.
    """
    phase_unwrapped = unwrap(phase_angle, axis=time_axis)

    dphase = diff(phase_unwrapped, axis=time_axis)

    # Add an initial phase to dphase by duplicating the first phase
    initial_phase = select(phase_unwrapped, time_axis, 0)
    dphase = np.concatenate([initial_phase, dphase], axis=time_axis)

    # rescale to [-1, 1]
    dphase = dphase / np.pi
    return dphase


def polar2rect(mag: np.array, phase_angle: np.array) -> np.array:
    """Convert polar-form complex number to its rectangular form."""
    temp_phase = np.zeros(mag.shape, dtype=np.complex_)

    mag_complex = mag.astype(np.complex_)

    for i, time in enumerate(phase_angle):
        for j, _ in enumerate(time):
            temp_phase[i, j] = np.complex(np.cos(phase_angle[i, j]),
                                          np.sin(phase_angle[i, j]))

    return mag_complex * temp_phase
