
import enum
import torch
from dataclasses import dataclass
from ..utils import PeftType, PromptLearningConfig

class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"


@dataclass
class PromptEncoderConfig(PromptLearningConfig):
    """
    NotImpremented
    """
    def __post_init__(self):
        self.peft_type = PeftType.P_TUNING


class PromptEncoder(torch.nn.Module):
    """
    NotImpremented
    """
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError
