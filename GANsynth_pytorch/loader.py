import numpy as np
from typing import Tuple, Callable, Any, Iterable
import functools

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .spectrograms_helper import SpectrogramsHelper

DatasetElement = Tuple[torch.Tensor, Iterable[Any]]


class WavToSpectrogramDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dataset,
                 spectrogramsHelper: SpectrogramsHelper,
                 transform=transforms.Lambda(lambda x: x),
                 **kwargs,
                 ):
        super().__init__(dataset, **kwargs)
        self.spectrogramsHelper = spectrogramsHelper
        self.transform = transform

        self._transforms = transforms.Compose(
            [self._collated_to_spectrogram_transform,
             self._make_collated_transform(self.transform)]
        )

    @staticmethod
    def _make_collated_transform(transform: Callable[[torch.Tensor],
                                                     torch.Tensor]
                                 ):
        """Extend data transform to operate on elements of a Dataset with targets
        """
        def collated_transform(x_and_targets: DatasetElement):
            x = x_and_targets[0]
            x = transform(x)
            targets = x_and_targets[1:]
            return [x] + targets

        return collated_transform

    @property
    def _collated_to_spectrogram_transform(self) -> transforms.Compose:
        """Efficient audio-to-spectrogram conversion using CUDA if available"""
        def to_spectrogram_ondevice(audio: torch.Tensor):
            return self.spectrogramsHelper.to_spectrogram(audio.to(
                self.spectrogramsHelper.device))

        return self._make_collated_transform(to_spectrogram_ondevice)

    def __iter__(self):
        wavforms_iterator = super().__iter__()
        return map(self._transforms, wavforms_iterator)


def mask_phase(spectrogram: torch.Tensor, min_magnitude: float):
    """Set IF to 0 where

    2020/01/17(theis): threshold set at -13~~log(2e-6) since the
    spectrograms returned by the NSynth dataset have minimum amplitude:
        spec_helpers.SPEC_THRESHOLD = log(1e-6)
    """
    if spectrogram.ndim == 3:
        channel_dim = 0
    elif spectrogram.ndim == 4:
        channel_dim = 1
    else:
        raise ValueError(
            f"Incorrect shape {spectrogram.shape} for parameter spec_and_IF")
    logmag = spectrogram.select(channel_dim, 0)
    IF = spectrogram.select(channel_dim, 1)

    log_threshold = np.log(2 * min_magnitude)
    mask = logmag < log_threshold

    logmag_fill_value = np.log(min_magnitude)
    logmag.masked_fill_(mask, logmag_fill_value)
    IF.masked_fill_(mask, 0)
    return spectrogram


def make_masked_phase_transform(min_magnitude: float):
    return transforms.Lambda(functools.partial(
        mask_phase, min_magnitude=min_magnitude))


class MaskedPhaseWavToSpectrogramDataLoader(WavToSpectrogramDataLoader):
    def __init__(self, dataset: Dataset,
                 spectrogramsHelper: SpectrogramsHelper,
                 **kwargs,
                 ):
        threshold_phase_transform = make_masked_phase_transform(
            spectrogramsHelper.safelog_eps)

        super().__init__(dataset, spectrogramsHelper,
                         transform=threshold_phase_transform,
                         **kwargs)
