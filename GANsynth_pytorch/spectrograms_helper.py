
import numpy as np
import pathlib
import torch
from torch import nn
import torchaudio
from typing import Tuple, Optional
from torchvision import transforms

import GANsynth_pytorch.spec_ops as spec_ops
import GANsynth_pytorch.phase_operation as phase_op


# SPEC_THRESHOLD = 1e-6


class SpectrogramsHelper(nn.Module):
    def __init__(self, fs_hz: int = 16000,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 window_length: Optional[int] = None,
                 safelog_eps: float = 1e-6
                 ):
        super().__init__()
        self.fs_hz = fs_hz
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length or self.n_fft

        self.window = nn.Parameter(
            torch.hann_window(self.window_length, periodic=False),
            requires_grad=False)

        self.safelog_eps = safelog_eps

    @property
    def device(self):
        return self.window.device

    def _expand(self, t: torch.Tensor) -> torch.Tensor:
        """"Repeat the last column of the input matrix twice"""
        # FIXME uses a hardcoded value valid only
        # for the duration of the NSynth samples
        expand_vec = t.select(dim=2, index=-1).unsqueeze(2)
        expanded = torch.cat([t, expand_vec, expand_vec], dim=2)
        return expanded

    @staticmethod
    def _split_channels(magnitude_and_IF: torch.Tensor) -> Tuple[torch.Tensor,
                                                                 torch.Tensor]:
        """Split magnitude and Instantaneous Frequency tensors in image shape

        The converse of SpectrogramsHelper._merge_channels()
        """
        assert magnitude_and_IF.ndim == 4, (
            "Input is expected to be batched")
        channel_dim = 1
        magnitude = magnitude_and_IF.select(channel_dim, 0)
        IF = magnitude_and_IF.select(channel_dim, 1)
        return magnitude, IF

    @staticmethod
    def _merge_channels(magnitude: torch.Tensor, IF: torch.Tensor
                        ) -> torch.Tensor:
        """Join magnitude and Instantaneous Frequency tensors for use as images

        Joins the two input time-frequency representation along dimension
        1 to be considered as channels of a single 2D image by nn.Conv2D
        """
        assert magnitude.ndim == IF.ndim == 3, (
            "Input is expected to be batched")
        channel_dim = 1
        magnitude = magnitude.unsqueeze(channel_dim)
        IF = IF.unsqueeze(channel_dim)
        return torch.cat([magnitude, IF], dim=1)

    def _stft(self, sample: torch.Tensor) -> torch.Tensor:
        """Helper function for computing the STFT

        Arguments:
            sampler, torch.Tensor, shape [batch, duration]:
                a batch of audio samples
        Return:
            torch.Tensor, shape [batch, self.n_fft//2 + 1, num_windows, 2]:
                the magnitude and Instantaneous Frequency of the input
        """
        return torch.stft(sample, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, win_length=self.window_length)

    def _istft(self, stft: torch.Tensor) -> torch.Tensor:
        """Helper function for computing the iSTFT

        Argument and output are the converse of self._stft()
        """
        return torch.istft(
            stft, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, win_length=self.window_length)

    def to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # shape [batch, self.n_fft//2 + 1, num_windows, 2]
        spec = self._stft(audio)

        # trim-off Nyquist frequency as advocated by GANSynth
        spec = spec[:, :-1]

        magnitude, angle = torchaudio.functional.magphase(spec)

        logmagnitude = torch.log(magnitude + self.safelog_eps)

        IF = phase_op.instantaneous_frequency(angle, time_axis=2)

        logmagnitude = self._expand(logmagnitude)
        IF = self._expand(IF)
        return self._merge_channels(logmagnitude, IF)

    def to_audio(self, spectrogram: torch.Tensor) -> torch.Tensor:
        channel_dim: Optional[int]
        batch_dim, channel_dim, freq_dim, time_dim = 0, 1, 2, 3

        # replicate the last frequency band
        # emulates the previously dropped Nyquist frequency, as advocated
        # by GANSynth
        spectrogram = torch.cat([spectrogram,
                                 spectrogram.select(
                                     dim=freq_dim, index=-1
                                     ).unsqueeze(freq_dim)
                                 ],
                                dim=freq_dim)

        logmag, IF = self._split_channels(spectrogram)
        channel_dim = None
        freq_dim, time_dim = freq_dim - 1, time_dim - 1

        mag = torch.exp(logmag) - self.safelog_eps
        reconstructed_magnitude = torch.abs(mag)

        reconstructed_phase_angle = torch.cumsum(IF * np.pi, dim=time_dim)

        stft = phase_op.polar2rect(reconstructed_magnitude,
                                   reconstructed_phase_angle)
        audio = self._istft(stft)
        return audio

    def from_wavfile(self, audio_path: pathlib.Path,
                     duration_n: Optional[int] = None,
                     to_mono: bool = True,
                     zero_pad: bool = True,
                     ) -> torch.Tensor:
        """Load and convert a single audio file"""
        audio, fs_hz = torchaudio.load(audio_path, channels_first=True)
        audio = audio.to(self.window.device)

        # resample to target sampling frequency
        if not fs_hz == self.fs_hz:
            resampler = torchaudio.transforms.Resample(orig_freq=fs_hz,
                                                       new_freq=self.fs_hz)
            audio = resampler(audio)

        if to_mono:
            audio = audio.sum(0)

        # trim from beginning for the selected duration
        if duration_n is not None:
            if zero_pad:
                sample_duration_n = audio.shape[-1]
                padding_amount_n = max(duration_n - sample_duration_n, 0)
                audio = torch.nn.functional.pad(audio, (0, padding_amount_n))

            audio = audio[:duration_n]

        return self.to_spectrogram(audio)


class MelSpectrogramsHelper(SpectrogramsHelper):
    def __init__(self, lower_edge_hertz: float = 0.0,
                 upper_edge_hertz: float = 16000 / 2.0,
                 mel_break_frequency_hertz: float = (
                     spec_ops._MEL_BREAK_FREQUENCY_HERTZ),
                 mel_downscale: int = 1,
                 mel_bin_width_threshold_factor: float = 1.5,
                 num_freq_bins: Optional[int] = None,
                 num_mel_bins: Optional[int] = None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.mel_lower_edge_hertz = lower_edge_hertz
        self.mel_upper_edge_hertz = upper_edge_hertz
        self.mel_break_frequency_hertz = mel_break_frequency_hertz
        self.mel_downscale = mel_downscale
        self.mel_bin_width_threshold_factor = mel_bin_width_threshold_factor
        self.num_freq_bins = num_freq_bins or self.n_fft // 2
        self.num_mel_bins = num_mel_bins or self.num_freq_bins // mel_downscale

        # initialize the linear-to-mel conversion matrix
        self._linear_to_mel_matrix = self.precompute_linear_to_mel()

    def precompute_linear_to_mel(self) -> torch.Tensor:
        linear_to_mel_np = spec_ops.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=self.num_freq_bins,
            sample_rate=self.fs_hz,
            lower_edge_hertz=self.mel_lower_edge_hertz,
            upper_edge_hertz=self.mel_upper_edge_hertz,
            mel_break_frequency_hertz=self.mel_break_frequency_hertz,
            expand_resolution_factor=self.mel_bin_width_threshold_factor)
        return torch.from_numpy(linear_to_mel_np)

    @property
    def linear_to_mel_matrix(self) -> torch.Tensor:
        return self._linear_to_mel_matrix

    @property
    def mel_to_linear_matrix(self) -> torch.Tensor:
        """Matrix used for linear mel-to-linear conversion à-la-Magenta"""
        m_t = self.linear_to_mel_matrix.T
        p = torch.matmul(self.linear_to_mel_matrix, m_t)
        sums = torch.sum(p, dim=0)
        d = torch.where(torch.abs(sums).gt(1.e-8),
                        torch.div(torch.ones_like(sums), sums),
                        sums)
        return torch.matmul(m_t, torch.diag(d))

    def to_spectrogram(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        spectrogram = super().to_spectrogram(audio_tensor)
        return self._linear_to_mel(spectrogram)

    def to_audio(self, mel_spectrogram: torch.Tensor
                 ) -> torch.Tensor:
        spectrogram = self._mel_to_linear(mel_spectrogram)
        return super().to_audio(spectrogram)

    def _linear_to_mel(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Converts specgrams to melspecgrams.

        Args:
        specgrams: Tensor of log magnitudes and instantaneous frequencies,
            shape [batch, freq, time].
        Returns:
        melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
            shape [batch, freq, time], mel scaling of frequencies.
        """
        # assumed initial dimensions
        channel_dim: Optional[int]
        batch_dim, channel_dim, freq_dim, time_dim = (0, 1, 2, 3)

        # transpose to [batch, time, freq] shape
        spectrogram = spectrogram.transpose(time_dim, freq_dim)
        freq_dim, time_dim = time_dim, freq_dim

        logmag, IF = self._split_channels(spectrogram)
        freq_dim, time_dim = freq_dim-1, time_dim-1
        channel_dim = None

        mag2 = torch.exp(2.0 * logmag)

        # retrieve the phase angle for conversion, unroll IF
        phase_angle = torch.cumsum(IF * np.pi, dim=time_dim)

        linear_to_mel_matrix = (self.linear_to_mel_matrix
                                .to(mag2.device).to(mag2.dtype))

        logmelmag2 = torch.log(
            torch.matmul(mag2, linear_to_mel_matrix)
            + self.safelog_eps)
        mel_phase_angle = torch.matmul(phase_angle, linear_to_mel_matrix)
        mel_IF = phase_op.instantaneous_frequency(mel_phase_angle,
                                                  time_axis=time_dim)

        mel_spectrogram = self._merge_channels(logmelmag2, mel_IF)
        freq_dim, time_dim = freq_dim+1, time_dim+1
        channel_dim = 1

        # transpose-back to [batch, freq, time] shape
        mel_spectrogram = mel_spectrogram.transpose(time_dim, freq_dim)
        freq_dim, time_dim = time_dim, freq_dim
        return mel_spectrogram

    def _mel_to_linear(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Converts melspecgrams to specgrams.
        Args:
        melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
            shape [batch, freq, time], mel scaling of frequencies.
        Returns:
        specgrams: Tensor of log magnitudes and instantaneous frequencies,
            shape [batch, freq, time].
        """
        channel_dim: Optional[int]
        batch_dim, channel_dim, freq_dim, time_dim = (0, 1, 2, 3)

        # transpose time and frequency dimensions for mel-conversion
        mel_spectrogram = mel_spectrogram.transpose(freq_dim, time_dim)
        freq_dim, time_dim = time_dim, freq_dim

        logmelmag2, mel_IF = self._split_channels(mel_spectrogram)
        channel_dim = None
        freq_dim, time_dim = freq_dim-1, time_dim-1

        mel_to_linear_matrix = (self.mel_to_linear_matrix
                                .to(logmelmag2.device)
                                .to(logmelmag2.dtype))

        mag2 = torch.tensordot(torch.exp(logmelmag2) - self.safelog_eps,
                               mel_to_linear_matrix,
                               dims=1)
        logmag = 0.5 * torch.log(mag2
                                 + self.safelog_eps)
        mel_phase_angle = torch.cumsum(mel_IF * np.pi, dim=time_dim)
        phase_angle = torch.tensordot(mel_phase_angle, mel_to_linear_matrix,
                                      dims=time_dim)
        IF = phase_op.instantaneous_frequency(phase_angle, time_axis=time_dim)

        spectrogram = self._merge_channels(logmag, IF)
        freq_dim, time_dim = freq_dim+1, time_dim+1
        channel_dim = 1

        # transpose back to original time and frequency dimensions
        spectrogram = spectrogram.transpose(freq_dim, time_dim)
        freq_dim, time_dim = time_dim, freq_dim
        return spectrogram
