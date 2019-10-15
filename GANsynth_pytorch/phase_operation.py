import librosa
import numpy as np
from intervaltree import Interval, IntervalTree
import torch


def diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Take the finite difference of a tensor along an axis.
    Arguments:
        x (torch.Tensor):
            Input tensor of any dimension.
        axis (int):
            Axis on which to take the finite difference.
    Returns:
        d (torch.Tensor):
            Tensor with size less than x by 1 along the difference dimension.
    Raises:
        ValueError: Axis out of range for tensor.
    """
    length = x.shape[dim]
    return x.narrow(dim, 1, length-1) - x.narrow(dim, 0, length-1)


def unwrap(p: torch.Tensor, discont: float = np.pi, dim: int = -1
           ) -> torch.Tensor:
    """Unwrap a cyclical phase tensor.

    Arguments:
        p (torch.Tensor):
            Phase tensor.
        discont (float):
            size of the cyclic discontinuity.
        axis (int):
            Axis over which to unwrap.
    Returns:
        unwrapped (torch.Tensor):
            Unwrapped tensor of same size as input.
    """
    dd = diff(p, dim=dim)
    ddmod = torch.remainder(dd + np.pi, 2.0*np.pi) - np.pi

    idx = torch.eq(ddmod, -np.pi) * torch.gt(dd, 0)
    ddmod = torch.where(idx, torch.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd

    idx = torch.lt(torch.abs(dd), discont)
    ddmod = torch.where(idx, torch.zeros_like(ddmod), dd)
    ph_cumsum = torch.cumsum(ph_correct, dim=dim)

    # Add initial zero value to the cumulative sum
    # shape = torch.Tensor(p.shape)
    # shape[axis] = 1
    initial_zeros = torch.zeros_like(p).select(dim=dim, index=0).unsqueeze(dim)
    ph_cumsum = torch.cat([initial_zeros, ph_cumsum],
                          dim=dim)
    unwrapped = p + ph_cumsum
    return unwrapped


def instantaneous_frequency(phase_angle: torch.Tensor, time_axis: int
                            ) -> torch.Tensor:
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
    phase_unwrapped = unwrap(phase_angle, dim=time_axis)

    dphase = diff(phase_unwrapped, dim=time_axis)

    # Add an initial phase to dphase by duplicating the first phase
    initial_phase = phase_unwrapped.select(time_axis, 0).unsqueeze(time_axis)
    dphase = torch.cat([initial_phase, dphase], dim=time_axis)

    # rescale to [-1, 1]
    dphase = dphase / np.pi
    return dphase


def polar2rect(mag: torch.Tensor, phase_angle: torch.Tensor) -> torch.Tensor:
    """Convert polar-form complex number to its rectangular form

    Complex tensors are shaped like [..., 2] in torchaudio's framework
    with the last two dimensions being the real and imaginary parts"""
    # add dummy complex dimension at dimension 0 for broadcasting to operate
    phase_angle = phase_angle.unsqueeze(0)
    phase_complex_form = torch.cat([torch.cos(phase_angle),
                                    torch.sin(phase_angle)], dim=0)

    # broadcasting magnitude over dimension 0
    rectangular = (mag * phase_complex_form)
    rectangular_complex_form = rectangular.permute(1, 2, 3, 0)
    return rectangular_complex_form
