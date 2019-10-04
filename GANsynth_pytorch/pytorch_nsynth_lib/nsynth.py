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
import json
import glob
from tqdm import tqdm
import numpy as np
import scipy.io.wavfile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import librosa
from .. import phase_operation
from .. import spectrograms_helper as spec_helper
import numpy as np
import functools

from typing import Tuple, Optional


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
                 convert_to_float: bool = True):
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

        self.transform = transform
        self.convert_to_float = convert_to_float
        if self.convert_to_float:
            # audio samples are loaded as an int16 numpy array
            # rescale intensity range as float [-1, 1]
            toFloat = transforms.Lambda(lambda x: (
                x / np.iinfo(np.int16).max).astype(np.float32))
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
        _, sample = scipy.io.wavfile.read(name)
        target = self.json_data[os.path.splitext(os.path.basename(name))[0]]
        categorical_target = [
            le.transform([target[field]])[0]
            for field, le in zip(self.categorical_field_list, self.le)]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return [sample, *categorical_target, target]


def expand(mat):
    """"Repeat the last column of the input matrix twice"""
    expand_vec = np.expand_dims(mat[:, 125], axis=1)
    expanded = np.hstack((mat, expand_vec, expand_vec))
    return expanded


def get_spectrogram_and_IF(sample, n_fft: int = 2048, hop_length: int = 512):
    sample = sample.squeeze()
    spec_float64 = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)

    magnitude_float64 = np.log(np.abs(spec_float64) + 1.0e-6)[:1024]
    angle_float64 = np.angle(spec_float64)

    # magnitude = magnitude_float64.astype(np.float32)
    # angle = angle_float64.astype(np.float32)

    IF_float64 = phase_operation.instantaneous_frequency(
        angle_float64, time_axis=1)[:1024]

    magnitude = expand(magnitude_float64)
    IF = expand(IF_float64)
    return magnitude, IF


def get_mel_spectrogram_and_IF(sample, n_fft: int = 2048,
                               hop_length: int = 512):
    magnitude_float64, IF_float64 = get_spectrogram_and_IF(
        sample, n_fft=n_fft, hop_length=hop_length)
    logmelmag2_float64, mel_p_float64 = spec_helper.specgrams_to_melspecgrams(
        magnitude_float64, IF_float64)
    return logmelmag2_float64, mel_p_float64


def channels_to_image(channels_np: List[np.ndarray]):
    """Reshape data into nn.Conv2D-compatible image shape"""
    channel_dimension = 0
    channels = []
    for data_array in channels_np:
        data_tensor = torch.as_tensor(data_array, dtype=torch.float32)
        data_tensor_as_image_channel = data_tensor.unsqueeze(
            channel_dimension)
        channels.append(data_tensor_as_image_channel)

    return torch.cat(channels, channel_dimension)


def to_mel_spec_and_IF_image(sample, n_fft: int = 2048, hop_length: int = 512):
    """Transforms wav samples to image-like mel-spectrograms [magnitude, IF]"""
    mel_spec, mel_IF = get_mel_spectrogram_and_IF(
        sample, hop_length=hop_length, n_fft=n_fft)
    mel_spec_and_IF_as_image_tensor = channels_to_image(
        [a.astype(np.float32) for a in [mel_spec, mel_IF]])
    return mel_spec_and_IF_as_image_tensor


def make_to_mel_spec_and_IF_image_transform(n_fft: int = 2048,
                                            hop_length: int = 512):
    my_transform = functools.partials(to_mel_spec_and_IF_image,
                                      n_fft=n_fft, hop_length=hop_length)
    return transforms.Lambda(my_transform)


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
