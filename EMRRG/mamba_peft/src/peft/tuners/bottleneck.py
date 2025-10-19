import torch
from dataclasses import dataclass
from ..utils import PeftConfig, PeftType

@dataclass
class BottleneckConfig(PeftConfig):
    """
    NotImpremented
    """
    def __post_init__(self):
        self.peft_type = PeftType.BOTTLENECK


class BottleneckModel(torch.nn.Module):
    """
    NotImpremented
    """
    def __init__(self, config, model):
        super().__init__()
        raise NotImplementedError