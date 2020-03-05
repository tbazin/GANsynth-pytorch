
import numpy as np
import pathlib
import torch
import torchaudio
from typing import Tuple, Optional
import scipy.signal
from torchvision import transforms

import GANsynth_pytorch.spec_ops as spec_ops
import GANsynth_pytorch.phase_operation as phase_op


SPEC_THRESHOLD = 1e-6


def _linear_to_mel_matrix_np() -> np.array:
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


def _linear_to_mel_matrix() -> torch.Tensor:
    return torch.from_numpy(_linear_to_mel_matrix_np())


# old version in numpy, for reference only
# def _mel_to_linear_matrix_np():
#     """Get the inverse mel transformation matrix."""
#     m = _linear_to_mel_matrix_np()
#     m_t = m.T
#     p = np.matmul(m, m_t)
#     d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
#     return np.matmul(m_t, np.diag(d))


def _mel_to_linear_matrix() -> torch.Tensor:
    """Get the inverse mel transformation matrix."""
    m = _linear_to_mel_matrix()
    return _linear_to_mel_matrix_to_mel_to_linear_matrix(m)


def _linear_to_mel_matrix_to_mel_to_linear_matrix(m: torch.Tensor
                                                  ) -> torch.Tensor:
    """Get the inverse mel transformation matrix."""
    m_t = m.T
    p = torch.matmul(m, m_t)
    sums = torch.sum(p, dim=0)
    d = torch.where(torch.abs(sums).gt(1.e-8), 1.0/sums, sums)
    return torch.matmul(m_t, torch.diag(d))


def expand(t: torch.Tensor) -> torch.Tensor:
    """"Repeat the last column of the input matrix twice"""
    # FIXME uses a hardcoded value valid only
    # for the duration of the NSynth samples
    expand_vec = t.select(dim=2, index=-1).unsqueeze(2)
    expanded = torch.cat([t, expand_vec, expand_vec], dim=2)
    return expanded


def get_spectrogram_and_IF(sample: torch.Tensor, n_fft: int = 2048,
                           hop_length: int = 512,
                           window_length: Optional[int] = None,
                           use_mel_scale: bool = True,
                           lower_edge_hertz: float = 0.0,
                           upper_edge_hertz: float = 16000 / 2.0,
                           mel_break_frequency_hertz: float = (
                               spec_ops._MEL_BREAK_FREQUENCY_HERTZ),
                           mel_downscale: int = 1,
                           sample_rate: int = 16000,
                           linear_to_mel: Optional[torch.Tensor] = None
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    if sample.ndim == 1:
        sample = sample.unsqueeze(0)

    if window_length is None:
        window_length = n_fft
    window = torch.hann_window(window_length, periodic=False,
                               device=sample.device, dtype=sample.dtype)
    spec = torch.stft(sample, n_fft=n_fft, hop_length=hop_length,
                      window=window, win_length=window_length)

    # trim-off Nyquist frequency as advocated by GANSynth
    spec = spec[:, :-1]

    magnitude, angle = torchaudio.functional.magphase(spec)

    logmagnitude = torch.log(magnitude + 1.0e-6)

    IF = phase_op.instantaneous_frequency(
        angle, time_axis=2)

    logmagnitude = expand(logmagnitude)
    IF = expand(IF)
    # return logmagnitude, IF

    if use_mel_scale:
        mel_logmagnitude, mel_IF = specgrams_to_melspecgrams(
            logmagnitude, IF,
            sample_rate=sample_rate,
            num_freq_bins=n_fft // 2,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            mel_break_frequency_hertz=mel_break_frequency_hertz,
            mel_downscale=mel_downscale,
            linear_to_mel=linear_to_mel)
        return mel_logmagnitude, mel_IF
    else:
        return logmagnitude, IF


def melspecgrams_to_specgrams(logmelmag2: torch.Tensor, mel_IF: torch.Tensor,
                              sample_rate: int = 16000,
                              mel_downscale: int = 1,
                              num_freq_bins: int = 2048 // 2,
                              lower_edge_hertz: float = 0.0,
                              upper_edge_hertz: float = 16000 / 2.0,
                              num_mel_bins: Optional[int] = None,
                              mel_break_frequency_hertz: float = (
                                    spec_ops._MEL_BREAK_FREQUENCY_HERTZ),
                              linear_to_mel: Optional[torch.Tensor] = None,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts melspecgrams to specgrams.
    Args:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, freq, time], mel scaling of frequencies.
    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, freq, time].
    """

    batch_dim, freq_dim, time_dim = (0, 1, 2)

    # transpose time and frequency dimensions for mel-conversion
    logmelmag2 = logmelmag2.transpose(freq_dim, time_dim)
    mel_IF = mel_IF.transpose(freq_dim, time_dim)
    batch_dim, freq_dim, time_dim = (0, 2, 1)

    # retrieve conversion matrix
    if linear_to_mel is None:
        if num_mel_bins is None:
            num_mel_bins = num_freq_bins // mel_downscale
        linear_to_mel_np = spec_ops.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins, num_spectrogram_bins=num_freq_bins,
            sample_rate=sample_rate, lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            mel_break_frequency_hertz=mel_break_frequency_hertz)
        linear_to_mel = torch.from_numpy(linear_to_mel_np)

    linear_to_mel = linear_to_mel.to(logmelmag2.device).to(logmelmag2.dtype)
    mel_to_linear = _linear_to_mel_matrix_to_mel_to_linear_matrix(
        linear_to_mel)

    magnitude2 = torch.tensordot(torch.exp(logmelmag2), mel_to_linear, dims=1)
    logmagnitude = 0.5 * torch.log(magnitude2 + SPEC_THRESHOLD)
    mel_phase_angle = torch.cumsum(mel_IF * np.pi, dim=time_dim)
    phase_angle = torch.tensordot(mel_phase_angle, mel_to_linear,
                                  dims=time_dim)
    IF = phase_op.instantaneous_frequency(phase_angle, time_axis=time_dim)

    # transpose back to original time and frequency dimensions
    logmagnitude = logmagnitude.transpose(freq_dim, time_dim)
    IF = IF.transpose(freq_dim, time_dim)
    batch_dim, freq_dim, time_dim = (0, 1, 2)
    return logmagnitude, IF


def specgrams_to_melspecgrams(magnitude: torch.Tensor, IF: torch.Tensor,
                              sample_rate: int = 16000,
                              mel_downscale: int = 1,
                              num_freq_bins: int = 2048 // 2,
                              lower_edge_hertz: float = 0.0,
                              upper_edge_hertz: float = 16000 / 2.0,
                              num_mel_bins: Optional[int] = None,
                              mel_break_frequency_hertz: float = (
                                    spec_ops._MEL_BREAK_FREQUENCY_HERTZ),
                              linear_to_mel: Optional[torch.Tensor] = None,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts specgrams to melspecgrams.

    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, freq, time].
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [batch, freq, time], mel scaling of frequencies.
    """
    # assumed initial dimensions
    batch_dim, freq_dim, time_dim = (0, 1, 2)

    # transpose to [batch, time, freq] shape
    logmag, p = [t.transpose(time_dim, freq_dim)
                 for t in [magnitude, IF]]
    # dimensions after transpose
    batch_dim, freq_dim, time_dim = (0, 2, 1)

    mag2 = torch.exp(2.0 * logmag)
    phase_angle = torch.cumsum(p * np.pi, dim=time_dim)

    if linear_to_mel is None:
        if num_mel_bins is None:
            num_mel_bins = num_freq_bins // mel_downscale
        linear_to_mel_np = spec_ops.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins, num_spectrogram_bins=num_freq_bins,
            sample_rate=sample_rate, lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            mel_break_frequency_hertz=mel_break_frequency_hertz)
        linear_to_mel = torch.from_numpy(linear_to_mel_np)

    linear_to_mel = linear_to_mel.to(mag2.device).to(mag2.dtype)

    logmelmag2 = torch.log(torch.matmul(mag2, linear_to_mel) + SPEC_THRESHOLD)
    mel_phase_angle = torch.matmul(phase_angle, linear_to_mel)
    mel_p = phase_op.instantaneous_frequency(mel_phase_angle,
                                             time_axis=time_dim)

    # transpose-back to [batch, freq, time] shape
    logmelmag2, mel_p = (t.transpose(time_dim, freq_dim)
                         for t in [logmelmag2, mel_p])

    # we're back to the initial dimensions
    batch_dim, freq_dim, time_dim = (0, 1, 2)

    return logmelmag2, mel_p


def logmag_and_IF_to_audio(logmag: torch.Tensor, IF: torch.Tensor,
                           hop_length: int, n_fft: int,
                           window_length: Optional[int] = None
                           ) -> torch.Tensor:
    batch_dim, freq_dim, time_dim = 0, 1, 2

    # replicate the last frequency band
    # emulates the previously dropped Nyquist frequency, as advocated
    # by GANSynth
    logmag = torch.cat([logmag, logmag.select(dim=freq_dim, index=-1
                                              ).unsqueeze(freq_dim)],
                       dim=freq_dim)
    IF = torch.cat([IF, IF.select(dim=freq_dim, index=-1
                                  ).unsqueeze(freq_dim)],
                   dim=freq_dim)

    mag = torch.exp(logmag) - SPEC_THRESHOLD
    reconstructed_magnitude = torch.abs(mag)

    reconstructed_phase_angle = torch.cumsum(IF * np.pi, dim=time_dim)

    stft = phase_op.polar2rect(reconstructed_magnitude,
                               reconstructed_phase_angle)

    if window_length is None:
        window_length = n_fft
    window = torch.hann_window(window_length, periodic=False,
                               device=stft.device, dtype=stft.dtype)
    audio = torchaudio.functional.istft(stft,
                                        hop_length=hop_length, n_fft=n_fft,
                                        window=window,
                                        win_length=window_length)

    return audio


def mel_logmag_and_IF_to_audio(mel_logmag: torch.Tensor, mel_IF: torch.Tensor,
                               hop_length: int, n_fft: int,
                               window_length: Optional[int] = None,
                               sample_rate: int = 16000,
                               mel_downscale: int = 1,
                               num_freq_bins: int = 2048 // 2,
                               lower_edge_hertz: float = 0.0,
                               upper_edge_hertz: float = 16000 / 2.0,
                               mel_break_frequency_hertz: float = (
                                     spec_ops._MEL_BREAK_FREQUENCY_HERTZ),
                               linear_to_mel: Optional[torch.Tensor] = None,
                               ) -> torch.Tensor:
    batch_dim, freq_dim, time_dim = 0, 1, 2

    logmag, IF = melspecgrams_to_specgrams(
        mel_logmag, mel_IF,
        sample_rate=sample_rate,
        mel_downscale=mel_downscale,
        num_freq_bins=num_freq_bins,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        mel_break_frequency_hertz=mel_break_frequency_hertz,
        linear_to_mel=linear_to_mel)

    audio = logmag_and_IF_to_audio(logmag, IF, n_fft=n_fft,
                                   hop_length=hop_length,
                                   window_length=window_length)
    return audio


def wavfile_to_spec_and_IF(audio_path: pathlib.Path,
                           target_fs_hz: int = 16000,
                           duration_s: float = 4,
                           to_mono: bool = True,
                           zero_pad: bool = True,
                           use_mel_scale: bool = True
                           ) -> torch.Tensor:
    """Load and convert a single audio file"""
    sample_audio, fs_hz = torchaudio.load_wav(audio_path,
                                              channels_first=True)

    # resample to target sampling frequency
    if not fs_hz == target_fs_hz:
        resampler = torchaudio.transforms.Resample(
            orig_freq=fs_hz, new_freq=target_fs_hz)
        sample_audio = resampler(sample_audio.cuda())

    if to_mono:
        sample_audio = sample_audio.sum(0)

    # trim from beginning for the selected duration
    duration_n = int(duration_s * target_fs_hz)

    if zero_pad:
        sample_duration_n = sample_audio.shape[-1]
        padding_amount_n = max(duration_n - sample_duration_n, 0)
        sample_audio = torch.nn.functional.pad(sample_audio,
                                               (0, padding_amount_n))

    sample_audio = sample_audio[:duration_n]

    toFloat = transforms.Lambda(lambda x: (x / np.iinfo(np.int16).max))
    sample_audio = toFloat(sample_audio)
    spec, IF = get_spectrogram_and_IF(
        sample_audio, use_mel_scale=use_mel_scale)
    channel_dim = 1
    spec = spec.unsqueeze(channel_dim)
    IF = IF.unsqueeze(channel_dim)
    spec_and_IF = torch.cat([spec,
                             IF],
                            dim=channel_dim)
    return spec_and_IF
