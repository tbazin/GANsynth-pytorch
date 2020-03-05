"""
File: nsynth.py
Author: Kwon-Young Choi
Email: kwon-young.choi@hotmail.fr
Date: 2018-11-13
Description: Load NSynth dataset using pytorch Dataset.
If you want to modify the output of the dataset, use the transform
and target_transform callbacks as ususal.
"""
import os
import pathlib
import json
import glob
import numpy as np
import scipy.signal

import torch
import torchaudio
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from .. import phase_operation
from .. import spectrograms_helper as spec_helper
from .. import spec_ops
import functools

from typing import Tuple, Optional, List, Union, Iterable, Callable


class NSynth(data.Dataset):
    """Pytorch dataset for NSynth dataset
    args:
        root: root dir containing examples.json and audio directory with
            wav files.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        blacklist_pattern: list of string used to blacklist dataset element.
            If one of the string is present in the audio filename, this sample
            together with its metadata is removed from the dataset.
        categorical_field_list: list of string. Each string is a key like
            instrument_family that will be used as a classification target.
            Each field value will be encoding as an integer using sklearn
            LabelEncoder.
    """
    def __init__(self, audio_directory_paths: Union[Iterable[str], str],
                 json_data_path: str,
                 transform=None, target_transform=None,
                 blacklist_pattern=[],
                 categorical_field_list=["instrument_family"],
                 valid_pitch_range: Optional[Tuple[int, int]] = None,
                 convert_to_float: bool = True,
                 squeeze_mono_channel: bool = True):
        """Constructor"""
        assert(isinstance(blacklist_pattern, list))
        assert(isinstance(categorical_field_list, list))

        self.filenames: List[str] = []
        # ensure audio_directory_paths is an iterable
        try:
            _ = audio_directory_paths[0]
        except TypeError:
            audio_directory_paths = [audio_directory_paths]
        for audio_directory_path in audio_directory_paths:
            self.filenames.extend(sorted(
                glob.glob(os.path.join(audio_directory_path, "*.wav"))
            ))
        with open(json_data_path, "r") as f:
            self.json_data = json.load(f)

        # only keep filenames corresponding to files present in the split-describing metadata file
        self._filter_filenames_in_json_data()

        # filter-out invalid pitches
        self.valid_pitch_range = valid_pitch_range
        if self.valid_pitch_range is not None:
            print("Filter out invalid pitches")
            self.filenames, self.json_data = self._filter_pitches_()

        for pattern in blacklist_pattern:
            self.filenames, self.json_data = self.blacklist(
                self.filenames, self.json_data, pattern)

        self.categorical_field_list = categorical_field_list
        self.label_encoders = {}
        for field in self.categorical_field_list:
            self.label_encoders[field] = LabelEncoder()
            field_values = [value[field] for value in self.json_data.values()]
            self.label_encoders[field].fit(field_values)

        self.squeeze_mono_channel = squeeze_mono_channel
        self.transform = transform or transforms.Lambda(lambda x: x)
        self.convert_to_float = convert_to_float
        if self.convert_to_float:
            # audio samples are loaded as an int16 numpy array
            # rescale intensity range as float [-1, 1]
            toFloat = transforms.Lambda(lambda x: (
                x / np.iinfo(np.int16).max))
            self.transform = transforms.Compose([toFloat,
                                                 self.transform])
        self.target_transform = target_transform

    def blacklist(self, filenames, json_data, pattern):
        filenames = [filename for filename in filenames
                     if pattern not in filename]
        json_data = {
            key: value for key, value in json_data.items()
            if pattern not in key
        }
        return filenames, json_data

    def _filter_filenames_in_json_data(self):
        """Removes filenames of files not present in the json_data"""
        valid_filenames = set(self.json_data.keys())
        self.filenames = [filename for filename in self.filenames
                          if pathlib.Path(filename).stem in valid_filenames
                          ]

    def _get_metadata(self, filename: str) -> int:
        note_str = pathlib.Path(filename).stem
        return self.json_data[note_str]

    def _filter_pitches_(self):
        valid_pitches_filenames = []
        valid_pitches_json_data = {}
        for filename in self.filenames:
            metadata = self._get_metadata(filename)
            pitch = metadata['pitch']
            if (self.valid_pitch_range[0] <= pitch <= self.valid_pitch_range[1]):
                valid_pitches_filenames.append(filename)
                valid_pitches_json_data[metadata['note_str']] = metadata
        return valid_pitches_filenames, valid_pitches_json_data

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (audio sample, *categorical targets, json_data)
        """
        name = self.filenames[index]
        sample, sample_rate = torchaudio.load_wav(name, channels_first=True)
        if self.squeeze_mono_channel:
            sample = sample.squeeze(0)
        metadata = self._get_metadata(name)
        categorical_target = [
            self.label_encoders[field].transform([metadata[field]])[0]
            for field in self.categorical_field_list]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            metadata = self.target_transform(metadata)
        if sample.ndim == 4:
            sample = sample.squeeze(0)
        return [sample, *categorical_target, metadata]


def channels_to_image(channels: Iterable[torch.Tensor]):
    """Reshape data into nn.Conv2D-compatible image shape"""
    channel_dimension = 1
    return torch.cat([channel.float().unsqueeze(channel_dimension)
                      for channel in channels],
                     dim=channel_dimension)


def to_spec_and_IF_image(sample: torch.Tensor, n_fft: int = 2048,
                         hop_length: int = 512,
                         window_length: int = 2048,
                         use_mel_scale: bool = True,
                         lower_edge_hertz: float = 0.0,
                         upper_edge_hertz: float = 16000 / 2.0,
                         mel_break_frequency_hertz: float = (
                             spec_ops._MEL_BREAK_FREQUENCY_HERTZ),
                         mel_downscale: int = 1,
                         sample_rate: int = 16000,
                         linear_to_mel: Optional[torch.Tensor] = None
                         ) -> torch.Tensor:
    """Transforms wav samples to image-like mel-spectrograms [magnitude, IF]"""
    spec_and_IF = spec_helper.get_spectrogram_and_IF(
        sample, hop_length=hop_length, n_fft=n_fft,
        sample_rate=sample_rate,
        use_mel_scale=use_mel_scale,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        mel_break_frequency_hertz=mel_break_frequency_hertz,
        linear_to_mel=linear_to_mel
        )
    spec_and_IF_as_image_tensor = channels_to_image(spec_and_IF)
    return spec_and_IF_as_image_tensor


def make_to_spec_and_IF_image_transform(n_fft: int = 2048,
                                        hop_length: int = 512,
                                        use_mel_scale: bool = True):
    to_image_transform = functools.partial(to_spec_and_IF_image,
                                           n_fft=n_fft, hop_length=hop_length,
                                           use_mel_scale=use_mel_scale)
    return transforms.Lambda(to_image_transform)


class WavToSpectrogramDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset,
                 batch_size: int = 1, shuffle: bool = False,
                 sampler: Sampler = None,
                 batch_sampler: Sampler = None,
                 num_workers: int = 0, collate_fn=None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0,
                 worker_init_fn: Callable[[int], None] = None,
                 multiprocessing_context=None,
                 n_fft: int = 2048,
                 window_length: Optional[int] = None,
                 hop_length: int = 512,
                 device: str = 'cpu',
                 transform: Optional[object] = None,
                 fs_hz: int = 16000,
                 use_mel_scale: bool = True,
                 lower_edge_hertz: float = 0.0,
                 upper_edge_hertz: float = 16000 / 2.0,
                 mel_break_frequency_hertz: float = (
                     spec_ops._MEL_BREAK_FREQUENCY_HERTZ),
                 mel_downscale: int = 1,
                 expand_resolution_factor: float = 1.5,
                 ):
        super().__init__(dataset, batch_size=batch_size,
                         shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn,
                         pin_memory=pin_memory, drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context)
        self.n_fft = n_fft
        self.fs_hz = fs_hz
        self.hop_length = hop_length
        window_length = window_length
        if window_length is None:
            window_length = n_fft
        self.window_length = window_length
        self.device = device
        self.transform = transform
        self.use_mel_scale = use_mel_scale

        self.lower_edge_hertz = None
        self.upper_edge_hertz = None
        self.mel_break_frequency_hertz = None
        self.mel_downscale = None
        self.linear_to_mel = None
        if self.use_mel_scale:
            self.lower_edge_hertz = lower_edge_hertz
            self.upper_edge_hertz = upper_edge_hertz
            self.mel_break_frequency_hertz = mel_break_frequency_hertz
            self.mel_downscale = mel_downscale
            self.expand_resolution_factor = expand_resolution_factor
            self.linear_to_mel = self.precompute_linear_to_mel()

    def precompute_linear_to_mel(self) -> torch.Tensor:
        assert self.use_mel_scale
        assert isinstance(self.mel_downscale, int)
        assert isinstance(self.mel_break_frequency_hertz, (int, float))
        assert isinstance(self.lower_edge_hertz, (int, float))
        assert isinstance(self.upper_edge_hertz, (int, float))

        num_freq_bins = self.n_fft // 2
        num_mel_bins = num_freq_bins // self.mel_downscale

        linear_to_mel_np = spec_ops.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins, num_spectrogram_bins=num_freq_bins,
            sample_rate=self.fs_hz,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz,
            mel_break_frequency_hertz=self.mel_break_frequency_hertz,
            expand_resolution_factor=self.expand_resolution_factor)
        return torch.from_numpy(linear_to_mel_np).to(self.device)

    def to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        return to_spec_and_IF_image(
            audio,
            n_fft=self.n_fft, hop_length=self.hop_length,
            window_length=self.window_length,
            use_mel_scale=self.use_mel_scale,
            linear_to_mel=self.linear_to_mel)

    def to_audio(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if self.use_mel_scale:
            mel_logmag, mel_IF = self.split_spectrogram(spectrogram)
            return spec_helper.mel_logmag_and_IF_to_audio(
                mel_logmag, mel_IF,
                n_fft=self.n_fft, hop_length=self.hop_length,
                window_length=self.window_length,
                linear_to_mel=self.linear_to_mel
            )
        else:
            logmag, IF = self.split_spectrogram(spectrogram)
            return spec_helper.logmag_and_IF_to_audio(
                logmag, IF,
                n_fft=self.n_fft, hop_length=self.hop_length,
                window_length=self.window_length
            )

    @property
    def to_spec_and_IF_image_transform(self) -> transforms.Compose:
        """Return a Transform to use for efficient data generation"""
        my_transforms = []

        if self.device is not None:
            def to_device_collated(wav_and_targets):
                wav = wav_and_targets[0].to(self.device)
                targets = wav_and_targets[1:]
                return [wav] + targets
            my_transforms.append(to_device_collated)

        to_image = self.to_spectrogram

        def make_collated_transform(transform):
            def collated_transform(data_and_targets):
                transformed_data = transform(data_and_targets[0])
                targets = data_and_targets[1:]
                return [transformed_data] + targets
            return collated_transform

        my_transforms.append(make_collated_transform(to_image))

        if self.transform is not None:
            my_transforms.append(make_collated_transform(self.transform))

        return transforms.Compose(my_transforms)

    def __iter__(self):
        wavforms_iterator = super().__iter__()
        return map(self.to_spec_and_IF_image_transform,
                   wavforms_iterator)

    @staticmethod
    def split_spectrogram(spec_and_IF: torch.Tensor) -> Tuple[torch.Tensor,
                                                              torch.Tensor]:
        assert spec_and_IF.ndim == 4, "input is expected to be batched"
        channel_dim = 1
        logmag = spec_and_IF.select(channel_dim, 0)
        IF = spec_and_IF.select(channel_dim, 1)
        return logmag, IF

    @staticmethod
    def merge_spec_and_IF(logmag: torch.Tensor, IF: torch.Tensor
                          ) -> torch.Tensor:
        assert logmag.ndim == IF.ndim == 3, "input is expected to be batched"
        channel_dim = 1
        logmag = logmag.unsqueeze(channel_dim)
        IF = IF.unsqueeze(channel_dim)
        return torch.cat([logmag, IF], dim=1)


def mask_phase(spec_and_IF: torch.Tensor,
               threshold: float = -13,  # TODO(theis) define a proper threshold
               min_value_spec: float = spec_helper.SPEC_THRESHOLD):
    """
    2020/01/17(theis): threshold set at -13~~log(2e-6) since the
    spectrograms returned by the NSynth dataset have minimum amplitude:
        spec_helpers.SPEC_THRESHOLD = log(1e-6)
    """
    if spec_and_IF.ndim == 3:
        channel_dim = 0
    elif spec_and_IF.ndim == 4:
        channel_dim = 1
    else:
        raise ValueError(
            f"Incorrect shape {spec_and_IF.shape} for parameter spec_and_IF")
    spec = spec_and_IF.select(channel_dim, 0)
    IF = spec_and_IF.select(channel_dim, 1)
    mask = spec < threshold

    spec_fill_value = np.log(min_value_spec)
    spec.masked_fill_(mask, spec_fill_value)
    IF.masked_fill_(mask, 0)
    return spec_and_IF


def make_masked_phase_transform(threshold: float = -13,  # TODO(theis) define a proper threshold
                                min_value_spec: float = spec_helper.SPEC_THRESHOLD):
    """
    20200117(theis): threshold set at -13~~log(2e-6) since the
    spectrograms returned by the NSynth dataset have minimum amplitude
    spec_helpers.SPEC_THRESHOLD = log(1e-6)
    """
    partial_mask_phase = functools.partial(
        mask_phase,
        threshold=threshold, min_value_spec=min_value_spec)
    return transforms.Lambda(partial_mask_phase)
