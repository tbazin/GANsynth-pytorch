
import numpy as np 

import GANsynth_pytorch.spec_ops as spec_ops
import GANsynth_pytorch.phase_operation as phase_op


def _linear_to_mel_matrix():
    """Get the mel transformation matrix."""
    _sample_rate = 16000
    _mel_downscale = 1
    num_freq_bins = 2048 // 2
    lower_edge_hertz = 0.0
    upper_edge_hertz = 16000 / 2.0
    num_mel_bins = num_freq_bins // _mel_downscale
    return spec_ops.linear_to_mel_weight_matrix(
        num_mel_bins, num_freq_bins, _sample_rate, lower_edge_hertz,
        upper_edge_hertz)


def _mel_to_linear_matrix():
    """Get the inverse mel transformation matrix."""
    m = _linear_to_mel_matrix()
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def melspecgrams_to_specgrams(logmelmag2, mel_p):
    """Converts melspecgrams to specgrams.
    Args:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    """
    time_axis = 1

    logmelmag2 = logmelmag2.T
    mel_p = mel_p.T
    time_axis = 0

    logmelmag2 = np.array([logmelmag2])
    mel_p = np.array([mel_p])
    time_axis = 1
  
    mel2l = _mel_to_linear_matrix()  
    mag2 = np.tensordot(np.exp(logmelmag2), mel2l, 1)
    logmag = 0.5 * np.log(mag2+1e-6)
    mel_phase_angle = np.cumsum(mel_p * np.pi, axis=time_axis)
    phase_angle = np.tensordot(mel_phase_angle, mel2l, axes=time_axis)
    p = phase_op.instantaneous_frequency(phase_angle, time_axis=time_axis)
    return logmag[0].T, p[0].T


def specgrams_to_melspecgrams(magnitude, IF):
    """Converts specgrams to melspecgrams.

    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    """
    # assumed initial dimensions
    (freq_axis, time_axis) = (0, 1)

    # transpose to [time, freq] shape
    logmag = magnitude.T
    p = IF.T

    # dimensions after transpose
    (freq_axis, time_axis) = (1, 0)

    mag2 = np.exp(2.0 * logmag)
    phase_angle = np.cumsum(p * np.pi, axis=time_axis)

    # reshape to [batch, time, freq] for mel-frequency conversion
    mag2 = np.array([mag2])
    phase_angle = np.array([phase_angle])
    # dimensions after batching
    (freq_axis, time_axis) = (2, 1)

    l2mel = _linear_to_mel_matrix()
    logmelmag2 = np.log(np.matmul(mag2, l2mel) + 1e-6)
    mel_phase_angle = np.matmul(phase_angle, l2mel)
    mel_p = phase_op.instantaneous_frequency(mel_phase_angle,
                                             time_axis=time_axis)

    # unbatch and transpose
    logmelmag2, mel_p = (matrix[0].T
                         for matrix in [logmelmag2, mel_p])

    # we're back to the initial dimensions
    (freq_axis, time_axis) = (0, 1)

    return logmelmag2, mel_p

