"""
File: nsynth.py
Authors: Théis Bazin, Kwon-Young Choi
Email: tbazin@github.com, kwon-young.choi@hotmail.fr
Date: 2019-2020, 2018-11-13
Description: Load NSynth dataset using pytorch Dataset.
If you want to modify the output of the dataset, use the transform
and target_transform callbacks as ususal.
"""
import os
import pathlib
import json
import glob
import numpy as np

import torchaudio
import torch.utils.data as data
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from typing import Tuple, Optional, List, Union, Iterable, Dict


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
                 squeeze_mono_channel: bool = True,
                 return_full_metadata: bool = True,
                 label_encode_categorical_data: bool = True):
        """Constructor"""
        assert(isinstance(blacklist_pattern, list))
        assert(isinstance(categorical_field_list, list))

        self.filenames: List[str] = []
        # ensure audio_directory_paths is an iterable
        try:
            _ = audio_directory_paths[0]  # type: ignore
        except TypeError:
            audio_directory_paths = [audio_directory_paths]  # type: ignore
        for audio_directory_path in audio_directory_paths:
            self.filenames.extend(sorted(
                glob.glob(os.path.join(audio_directory_path, "*.wav"))
            ))
        with open(json_data_path, "r") as f:
            self.json_data = json.load(f)

        # only keep filenames corresponding to files present in the
        # split-describing metadata file
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
        self.label_encode_categorical_data = label_encode_categorical_data
        self.label_encoders: Dict[str, LabelEncoder] = {}
        if self.label_encode_categorical_data:
            for field in self.categorical_field_list:
                self.label_encoders[field] = LabelEncoder()
                field_values = [value[field]
                                for value in self.json_data.values()]
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
        self.return_full_metadata = return_full_metadata

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
        if self.label_encode_categorical_data:
            categorical_target = [
                self.label_encoders[field].transform([metadata[field]])[0]
                for field in self.categorical_field_list]
        else:
            categorical_target = [metadata[field]
                                  for field in self.categorical_field_list]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            metadata = self.target_transform(metadata)
        if sample.ndim == 4:
            sample = sample.squeeze(0)

        if self.return_full_metadata:
            return [sample, *categorical_target, metadata]
        else:
            return [sample, *categorical_target]
