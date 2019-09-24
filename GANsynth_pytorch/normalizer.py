from typing import Optional
import pathlib
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader


class DataNormalizer(object):
    def __init__(self, dataloader: Optional[DataLoader] = None,
                 s_a: Optional[float] = None,
                 s_b: Optional[float] = None,
                 p_a: Optional[float] = None,
                 p_b: Optional[float] = None,
                 ):
        if dataloader is not None:
            self.dataloader = dataloader

            self._init_range_normalizer(magnitude_margin=0.8, IF_margin=1.0)
        elif all([x is not None for x in [s_a, s_b, p_a, p_b]]):
            self.s_a = s_a
            self.s_b = s_b
            self.p_a = p_a
            self.p_b = p_b
        else:
            raise ValueError("Must either provide example dataset"
                             "or pre-computed statistics")

        print("s_a:", self.s_a)
        print("s_b:", self.s_b)
        print("p_a:", self.p_a)
        print("p_b:", self.p_b)

    def _init_range_normalizer(self, magnitude_margin, IF_margin):
        min_spec = 10000
        max_spec = -10000
        min_IF = 10000
        max_IF = -10000

        for batch_idx, ((img, pitch), target) in enumerate(self.dataloader): 
            spec = img.select(1, 0)
            IF = img.select(1, 1)
            
            if spec.min() < min_spec: min_spec=spec.min()
            if spec.max() > max_spec: max_spec=spec.max()

            if IF.min() < min_IF: min_IF=IF.min()
            if IF.max() > max_IF: max_IF=IF.max()
    
        self.s_a = magnitude_margin * (2.0 / (max_spec - min_spec))
        self.s_b = magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)
        
        # min_IF = -np.pi
        # max_IF = np.pi
        
        self.p_a = IF_margin * (2.0 / (max_IF - min_IF))
        self.p_b = IF_margin * (-2.0 * min_IF / (max_IF - min_IF) - 1.0)

    def normalize(self, feature_map):
        a = np.asarray([self.s_a, self.p_a])[None, :, None, None]
        b = np.asarray([self.s_b, self.p_b])[None, :, None, None]
        a = torch.FloatTensor(a).cuda()
        b = torch.FloatTensor(b).cuda()
        
        feature_map = feature_map*a + b

        return feature_map

    def denormalize(self, spec, IF):
        spec = (spec - self.s_b) / self.s_a
        IF = (IF-self.p_b) / self.p_a
        return spec, IF
    
    def dump_statistics(self, path: pathlib.Path):
        statistics = {
            attr: getattr(self, attr)
            for attr in ['s_a', 's_b', 'p_a', 'p_b']
        }
        pickle.dump(statistics, path.open('wb'))