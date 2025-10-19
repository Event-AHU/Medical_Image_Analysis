import torch
from dataclasses import dataclass
from ..utils import PeftType, PromptLearningConfig

@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    NotImpremented
    """
    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_TUNING


class PrefixEncoder(torch.nn.Module):
    """
    NotImpremented
    """
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError
