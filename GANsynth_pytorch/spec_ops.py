from typing import Iterable, Union
import warnings
import numpy as np

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def get_mel_high_frequency_q(mel_break_frequency_hertz: float) -> float:
    if mel_break_frequency_hertz != _MEL_BREAK_FREQUENCY_HERTZ:
        # by convention, 1000mels should be equal to 1000Hz
        return 1000 / np.log(1 + 1000/mel_break_frequency_hertz)
    else:
        # for consistency with the GANSynth codebase
        return _MEL_HIGH_FREQUENCY_Q


def mel_to_hertz(mel_values: Union[float, Iterable[float]],
                 mel_break_frequency_hertz: float = _MEL_BREAK_FREQUENCY_HERTZ
                 ) -> np.array:
    """Converts frequencies in `mel_values` from the mel scale to linear scale
    """
    mel_high_frequency_q = get_mel_high_frequency_q(mel_break_frequency_hertz)
    return mel_break_frequency_hertz * (
        np.exp(np.array(mel_values) / mel_high_frequency_q) - 1.0)


def hertz_to_mel(frequencies_hertz: Union[float, Iterable[float]],
                 mel_break_frequency_hertz: float = _MEL_BREAK_FREQUENCY_HERTZ
                 ) -> np.array:
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale
    """
    mel_high_frequency_q = get_mel_high_frequency_q(mel_break_frequency_hertz)
    return mel_high_frequency_q * np.log(
        1.0 + (np.array(frequencies_hertz) / mel_break_frequency_hertz))


def linear_to_mel_weight_matrix(num_mel_bins: int = 20,
                                num_spectrogram_bins: int = 129,
                                sample_rate: int = 16000,
                                lower_edge_hertz: float = 125.0,
                                upper_edge_hertz: float = 3800.0,
                                mel_break_frequency_hertz: float = (
                                    _MEL_BREAK_FREQUENCY_HERTZ),
                                expand_resolution_factor: float = 1.5,
                                ) -> np.array:
    """Returns a matrix to warp linear scale spectrograms to the mel scale.
    Adapted from tf.contrib.signal.linear_to_mel_weight_matrix with a minimum
    band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
    we compute the matrix at float64 precision and then cast to `dtype`
    at the end. This function can be constant folded by graph optimization
    since there are no Tensor inputs.
    Args:
        num_mel_bins: Int, number of output frequency dimensions.
        num_spectrogram_bins: Int, number of input frequency dimensions.
        sample_rate: Int, sample rate of the audio.
        lower_edge_hertz: Float, lowest frequency to consider.
        upper_edge_hertz: Float, highest frequency to consider.
    Returns:
        Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].
    Raises:
        ValueError: Input argument in the wrong range.
    """
    def mel_to_hertz_partial(mel_values: Union[float, Iterable[float]]
                             ) -> np.array:
        return mel_to_hertz(mel_values, mel_break_frequency_hertz)

    def hertz_to_mel_partial(frequencies_hertz: Union[float, Iterable[float]]
                             ) -> np.array:
        return hertz_to_mel(frequencies_hertz, mel_break_frequency_hertz)

    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError('num_mel_bins must be positive. Got: %s'
                         % num_mel_bins)
    if num_spectrogram_bins <= 0:
        raise ValueError(
            'num_spectrogram_bins must be positive. Got: %s'
            % num_spectrogram_bins)
    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive. Got: %s'
                         % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            'lower_edge_hertz must be non-negative. Got: %s'
            % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                         (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError(
            'upper_edge_hertz must not be larger than the Nyquist '
            'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
            % (upper_edge_hertz, sample_rate))
    if expand_resolution_factor < 1.:
        raise ValueError('Mel bin resolution factor %s should be >= 1.'
                         % (expand_resolution_factor))

    mel_high_frequency_q = get_mel_high_frequency_q(mel_break_frequency_hertz)

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:, np.newaxis]
    spectrogram_bins_mel = hertz_to_mel_partial(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = np.linspace(
        hertz_to_mel_partial(lower_edge_hertz),
        hertz_to_mel_partial(upper_edge_hertz),
        num_mel_bins + 2)

    lower_edge_mel = band_edges_mel[0:-2].copy()
    center_mel = band_edges_mel[1:-1].copy()
    upper_edge_mel = band_edges_mel[2:].copy()

    if expand_resolution_factor > 1.:
        freq_res = nyquist_hertz / float(num_spectrogram_bins)
        freq_th = expand_resolution_factor * freq_res
        for i in range(0, num_mel_bins):
            center_hz = mel_to_hertz_partial(center_mel[i])
            lower_hz = mel_to_hertz_partial(lower_edge_mel[i])
            upper_hz = mel_to_hertz_partial(upper_edge_mel[i])
            if upper_hz - lower_hz < freq_th:
                rhs = 0.5 * freq_th / (center_hz + mel_break_frequency_hertz)
                dm = mel_high_frequency_q * np.log(rhs + np.sqrt(1.0 + rhs**2))
                lower_edge_mel[i] = center_mel[i] - dm
                upper_edge_mel[i] = center_mel[i] + dm

        lower_edge_hz = mel_to_hertz_partial(lower_edge_mel)[np.newaxis, :]
        center_hz = mel_to_hertz_partial(center_mel)[np.newaxis, :]
        upper_edge_hz = mel_to_hertz_partial(upper_edge_mel)[np.newaxis, :]

        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the mel domain, not Hertz.
        lower_slopes = (linear_frequencies - lower_edge_hz) / (
            center_hz - lower_edge_hz)
        upper_slopes = (upper_edge_hz - linear_frequencies) / (
            upper_edge_hz - center_hz)
    else:
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the mel domain, not Hertz.
        lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
            center_mel - lower_edge_mel)
        upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
            upper_edge_mel - center_mel)

    # check if all filters are non-empty
    if not np.all(upper_edge_mel - lower_edge_mel > 0):
        warnings.warn("Some filters are empty")

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0,
                                    np.minimum(lower_slopes,
                                               upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = np.pad(mel_weights_matrix,
                                [[bands_to_zero, 0], [0, 0]],
                                'constant')
    return mel_weights_matrix
