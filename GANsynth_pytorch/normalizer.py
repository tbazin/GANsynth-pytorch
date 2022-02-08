from typing import Optional, Union, Iterable
import pathlib
import json
from typing_extensions import Final
import numpy as np
import torch
from torch import nn
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


@torch.jit.interface
class DataNormalizerInterface(nn.Module):
    def normalize(self, spec_and_IF: torch.Tensor) -> torch.Tensor:
        pass

    def denormalize(self, spec_and_IF: torch.Tensor) -> torch.Tensor:
        pass

class DataNormalizer(nn.Module):
    def __init__(self, statistics: Optional[DataNormalizerStatistics] = None,
                 dataloader: Optional[DataLoader] = None):
        super().__init__()

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
        a_numpy = np.asarray([self.statistics.s_a, self.statistics.p_a])[
            None, :, None, None]
        b_numpy = np.asarray([self.statistics.s_b, self.statistics.p_b])[
            None, :, None, None]
        self.a = nn.Parameter(torch.as_tensor(a_numpy).float(),
                              requires_grad=False)
        self.b = nn.Parameter(torch.as_tensor(b_numpy).float(),
                              requires_grad=False)

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

    @torch.jit.export
    def normalize(self, spec_and_IF: torch.Tensor):
        return spec_and_IF * self.a + self.b

    @torch.jit.export
    def denormalize(self, spec_and_IF: torch.Tensor):
        return (spec_and_IF - self.b) / self.a

    def dump_statistics(self, path: pathlib.Path):
        with path.open('w') as f:
            json.dump(self.statistics.__dict__, f, indent=4)

    @classmethod
    def load_statistics(cls, path: pathlib.Path):
        with path.open('r') as f:
            statistics_dict = json.load(f)
            statistics = DataNormalizerStatistics(**statistics_dict)
        return cls(statistics=statistics)
