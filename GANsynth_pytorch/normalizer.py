from typing import Optional, Union, Iterable
import pathlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class DataNormalizerStatistics(object):
    s_a: float
    s_b: float
    p_a: float
    p_b: float

    def __init__(self, s_a: float, s_b: float, p_a: float, p_b: float):
        self.s_a = s_a
        self.s_b = s_b
        self.p_a = p_a
        self.p_b = p_b


class DataNormalizer(object):
    def __init__(self, statistics: Optional[DataNormalizerStatistics] = None,
                 dataloader: Optional[DataLoader] = None):
        if statistics is not None:
            self.statistics = statistics
        elif dataloader is not None:
            print("Computing normalization statistics over the whole dataset")
            self._init_range_normalizer(dataloader,
                                        magnitude_margin=0.8, IF_margin=1.0)
        else:
            raise ValueError("Must either provide example dataset"
                             "or pre-computed statistics")

        print("Statistics:", self.statistics.__dict__)

        # pre-compute torch tensors
        a = np.asarray([self.statistics.s_a, self.statistics.p_a])[
            None, :, None, None]
        b = np.asarray([self.statistics.s_b, self.statistics.p_b])[
            None, :, None, None]
        self.a = torch.as_tensor(a).float()
        self.b = torch.as_tensor(b).float()

    def _init_range_normalizer(self, dataloader: DataLoader,
                               magnitude_margin: float, IF_margin: float):
        min_spec = np.inf
        max_spec = -np.inf
        min_IF = np.inf
        max_IF = -np.inf

        for batch_idx, (img, *_) in enumerate(tqdm(dataloader)):
            spec = img.select(1, 0)
            IF = img.select(1, 1)

            if spec.min() < min_spec:
                min_spec = spec.min().cpu().item()
            if spec.max() > max_spec:
                max_spec = spec.max().cpu().item()

            if IF.min() < min_IF:
                min_IF = IF.min().cpu().item()
            if IF.max() > max_IF:
                max_IF = IF.max().cpu().item()

        s_a = magnitude_margin * (2.0
                                  / (max_spec - min_spec))
        s_b = magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec)
                                  - 1.0)

        p_a = IF_margin * (2.0 / (max_IF - min_IF))
        p_b = IF_margin * (-2.0 * min_IF / (max_IF - min_IF) - 1.0)

        self.statistics = DataNormalizerStatistics(s_a, s_b, p_a, p_b)

    def normalize(self, spec_and_IF: torch.Tensor):
        device = spec_and_IF.device
        a = self.a.to(device)
        b = self.b.to(device)

        spec_and_IF = spec_and_IF*a + b

        return spec_and_IF

    def denormalize(self, spec_and_IF: torch.Tensor):
        device = spec_and_IF.device
        a = self.a.to(device)
        b = self.b.to(device)

        spec_and_IF = (spec_and_IF - b) / a
        # spec = (spec - self.s_b) / self.s_a
        # IF = (IF - self.p_b) / self.p_a
        return spec_and_IF

    def dump_statistics(self, path: pathlib.Path):
        with path.open('w') as f:
            json.dump(self.statistics.__dict__, f, indent=4)

    @classmethod
    def load_statistics(cls, path: pathlib.Path):
        with path.open('r') as f:
            statistics_dict = json.load(f)
            statistics = DataNormalizerStatistics(**statistics_dict)
        return cls(statistics=statistics)
