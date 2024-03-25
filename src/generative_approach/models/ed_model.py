from abc import abstractmethod, ABC
from typing import Optional

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig


class EDModel(torch.nn.Module):
    device: torch.device
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    config: PretrainedConfig
    max_encoder_position_embeddings: int
    max_decoder_position_embeddings: int

    def __init__(self, device: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    # @property
    # @abstractmethod
    # def tokenizer(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def model(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def config(self):
    #     pass