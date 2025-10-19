import enum
import torch
from dataclasses import dataclass
from ..utils import PeftType, PromptLearningConfig


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    NotImpremented
    """
    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING


class PromptEmbedding(torch.nn.Module):
    """
    NotImpremented
    """
    def __init__(self, config, word_embeddings):
        super().__init__()
        raise NotImplementedError