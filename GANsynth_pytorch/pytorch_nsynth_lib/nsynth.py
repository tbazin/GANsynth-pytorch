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
from tqdm import tqdm
import numpy as np
import scipy.signal

import torch
import torchaudio
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import librosa
from .. import phase_operation
from .. import spectrograms_helper as spec_helper
import numpy as np
import functools

from typing import Tuple, Optional, List


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

    def __init__(self, root, transform=None, target_transform=None,
                 blacklist_pattern=[],
                 categorical_field_list=["instrument_family"],
                 valid_pitch_range: Optional[Tuple[int, int]]=None,
                 convert_to_float: bool = True,
                 squeeze_mono_channel: bool = True):
        """Constructor"""
        assert(isinstance(root, str))
        assert(isinstance(blacklist_pattern, list))
        assert(isinstance(categorical_field_list, list))
        self.root = root
        self.filenames = glob.glob(os.path.join(root, "audio/*.wav"))
        with open(os.path.join(root, "examples.json"), "r") as f:
            self.json_data = json.load(f)
        self.valid_pitch_range = valid_pitch_range
        if self.valid_pitch_range is not None:
            print("Filter out invalid pitches")
            self.filenames, self.json_data = self._filter_pitches_()
        for pattern in blacklist_pattern:
            self.filenames, self.json_data = self.blacklist(
                self.filenames, self.json_data, pattern)

        self.categorical_field_list = categorical_field_list
        self.le = []
        for i, field in enumerate(self.categorical_field_list):
            self.le.append(LabelEncoder())
            field_values = [value[field] for value in self.json_data.values()]
            self.le[i].fit(field_values)

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

    def _filter_pitches_(self):
        valid_pitches_filenames = []
        json_data = {}
        for sample_name, sample_details in tqdm(self.json_data.items()):
            pitch = sample_details['pitch']
            if (self.valid_pitch_range[0] <= pitch
                    and pitch <= self.valid_pitch_range[1]):
                filename = os.path.join(self.root, f"audio/{sample_name}.wav")
                valid_pitches_filenames.append(filename)
                json_data[sample_name] = sample_details
        filenames = list(set(valid_pitches_filenames) & set(self.filenames))
        return filenames, json_data

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
        target = self.json_data[os.path.splitext(os.path.basename(name))[0]]
        categorical_target = [
            le.transform([target[field]])[0]
            for field, le in zip(self.categorical_field_list, self.le)]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if sample.ndim == 4:
            sample = sample.squeeze(0)
        return [sample, *categorical_target, target]


def expand(t: torch.Tensor) -> torch.Tensor:
    """"Repeat the last column of the input matrix twice"""
    # FIXME uses a hardcoded value valid only
    # for the duration of the NSynth samples
    expand_vec = t.select(dim=2, index=125).unsqueeze(2)
    expanded = torch.cat([t, expand_vec, expand_vec], dim=2)
    return expanded


def get_spectrogram_and_IF(sample: torch.Tensor, n_fft: int = 2048,
                           hop_length: int = 512
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    if sample.ndim == 1:
        sample = sample.unsqueeze(0)
    window_np = scipy.signal.get_window('hann', n_fft)
    window = torch.as_tensor(window_np).to(sample.device).to(sample.dtype)
    spec = torch.stft(sample, n_fft=n_fft, hop_length=hop_length,
                      window=window)

    # trim-off Nyquist frequency as advocated by GANSynth
    spec = spec[:, :-1]

    magnitude, angle = torchaudio.functional.magphase(spec)

    logmagnitude = torch.log(magnitude + 1.0e-6)

    IF = phase_operation.instantaneous_frequency(
        angle, time_axis=2)

    logmagnitude = expand(logmagnitude)
    IF = expand(IF)
    return logmagnitude, IF


def get_mel_spectrogram_and_IF(sample: torch.Tensor, n_fft: int = 2048,
                               hop_length: int = 512) -> torch.Tensor:
    magnitude_float64, IF_float64 = get_spectrogram_and_IF(
        sample, n_fft=n_fft, hop_length=hop_length)

    logmelmag2_float64, mel_p_float64 = spec_helper.specgrams_to_melspecgrams(
        magnitude_float64, IF_float64)
    return logmelmag2_float64, mel_p_float64


def channels_to_image(channels: List[torch.Tensor]):
    """Reshape data into nn.Conv2D-compatible image shape"""
    channel_dimension = 1
    return torch.cat([channel.float().unsqueeze(channel_dimension)
                      for channel in channels],
                     dim=channel_dimension)


def to_mel_spec_and_IF_image(sample: torch.Tensor, n_fft: int = 2048,
                             hop_length: int = 512) -> torch.Tensor:
    """Transforms wav samples to image-like mel-spectrograms [magnitude, IF]"""
    mel_spec_and_IF = get_mel_spectrogram_and_IF(
        sample, hop_length=hop_length, n_fft=n_fft)
    mel_spec_and_IF_as_image_tensor = channels_to_image(mel_spec_and_IF)
    return mel_spec_and_IF_as_image_tensor


def make_to_mel_spec_and_IF_image_transform(n_fft: int = 2048,
                                            hop_length: int = 512):
    to_image_transform = functools.partial(to_mel_spec_and_IF_image,
                                           n_fft=n_fft, hop_length=hop_length)
    return transforms.Lambda(to_image_transform)


class WavToSpectrogramDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 n_fft: int = 2048, hop_length: int = 512,
                 device: str = 'cpu'
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
        self.hop_length = hop_length
        self.device = device

    @property
    def to_mel_spec_and_IF_image_transform(self) -> transforms.Compose:
        """Return a Transform to use for efficient data generation"""
        my_transforms = []

        if self.device is not None:
            def to_device_collated(wav_and_targets):
                wav = wav_and_targets[0].to(self.device)
                targets = wav_and_targets[1:]
                return [wav] + targets
            my_transforms.append(to_device_collated)

        to_image = functools.partial(to_mel_spec_and_IF_image,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_length)

        def to_image_collated(wav_and_targets):
            spec = to_image(wav_and_targets[0])
            targets = wav_and_targets[1:]
            return [spec] + targets
        my_transforms.append(to_image_collated)

        return transforms.Compose(my_transforms)

    def __iter__(self):
        wavforms_iterator = super().__iter__()
        return map(self.to_mel_spec_and_IF_image_transform,
                   wavforms_iterator)


def wavfile_to_melspec_and_IF(audio_path: pathlib.Path
                              ) -> torch.Tensor:
    """Load and convert a single audio file"""
    sample_audio, fs_hz = torchaudio.load_wav(audio_path,
                                              channels_first=True)
    toFloat = transforms.Lambda(lambda x: (x / np.iinfo(np.int16).max))
    sample_audio = toFloat(sample_audio)
    mel_spec, mel_IF = get_mel_spectrogram_and_IF(
        sample_audio)
    channel_dim = 1
    mel_spec = mel_spec.unsqueeze(channel_dim)
    mel_IF = mel_IF.unsqueeze(channel_dim)
    mel_spec_and_IF = torch.cat([mel_spec,
                                 mel_IF],
                                dim=channel_dim)
    return mel_spec_and_IF


if __name__ == "__main__":
    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
    # use instrument_family and instrument_source as classification targets
    dataset = NSynth(
        "../nsynth-test",
        transform=toFloat,
        blacklist_pattern=["string"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    for samples, instrument_family_target, instrument_source_target, targets \
            in loader:
        print(samples.shape, instrument_family_target.shape,
              instrument_source_target.shape)
        print(torch.min(samples), torch.max(samples))
