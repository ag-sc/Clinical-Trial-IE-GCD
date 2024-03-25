import json
import os
import re
from itertools import chain
from typing import Sequence, Set, Iterable

import torch

from generative_approach.models.flan_t5 import FlanT5
from generative_approach.models.longformer import Longformer


DEBUG = False

def get_model_class(model: str):
    match model:
        case "allenai/led-base-16384":
            return Longformer
        case "google/flan-t5-small":
            return FlanT5
        case "google/flan-t5-base":
            return FlanT5
        case "google/flan-t5-large":
            return FlanT5
        case "google/flan-t5-xl":
            return FlanT5
        case "google/flan-t5-xxl":
            return FlanT5
        case _:
            raise RuntimeError("Model not recognized!")


def create_start_token(string: str) -> str:
    return f'[start:{string}]'


def create_end_token(string: str) -> str:
    return f'[end:{string}]'


def decode_start_token(token: str) -> str:
    return token.replace('[start:', '').replace(']', '')


def is_start_token(token: str):
    return re.match(r'\[start:[A-Za-z_]+]$', token) is not None


def is_end_token(token: str):
    return re.match(r'\[end:[A-Za-z_]+]$', token) is not None


def decode_end_token(token: str) -> str:
    return token.replace('[end:', '').replace(']', '')


def import_task_config(filename: str):
    with open(filename) as fp:
        return json.load(fp)


def save_model(model, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    model.save_pretrained(dirname, from_pt=True)


class DocumentSubsequenceManager:
    def __init__(self, sentences: Sequence[Sequence[str]]):
        self._sentences = sentences
        self._vocabulary = set(chain(*sentences))

    def get_vocabulary(self) -> Set[str]:
        return self._vocabulary

    def get_next_possible_tokens(self, prefix: Sequence[str] = None) -> Set[str]:
        if prefix is None or len(prefix) == 0:
            return self.get_vocabulary()

        subseq_length = len(prefix) + 1  # +1 for next possible token
        next_possible_tokens = set()

        for sentence in self._sentences:
            if len(sentence) < subseq_length:
                continue

            for i in range(0, len(sentence) - subseq_length + 1):
                subseq = sentence[i:i + subseq_length]
                if subseq[:-1] == prefix:
                    next_possible_tokens.add(subseq[-1])

        return next_possible_tokens


class IdentityTokenizer:
    def tokenize(self, pretokens: Sequence[str], skip_special_tokens: bool = False):
        return pretokens


class TemplateGenerationTokenizer:
    def __init__(
            self,
            tokenizer,
            json_filename: str = None,
            template_names: Iterable[str] = None,
            slot_names: Iterable[str] = None,
            start_of_sentence_token: str = None,
            end_of_sentence_token: str = None,
            filler_sep_token: str = None,
            control_tokens_start_id: int = None
    ):
        self.tokenizer = tokenizer

        if json_filename is not None:
            fp = open(json_filename)
            [self.control_token_ids, special_token_names] = json.load(fp)
            self.start_of_sentence_token = special_token_names['sos']
            self.end_of_sentence_token = special_token_names['eos']
            self.pad_token = special_token_names['pad_token'] if 'pad_token' in special_token_names else tokenizer.pad_token
            self.filler_sep_token = special_token_names['filler_sep_token']

            self.control_token_ids_reverse = {token_id: token for token, token_id in self.control_token_ids.items()}
            return
        else:
            self.control_token_ids = dict()
            self.start_of_sentence_token = start_of_sentence_token
            self.end_of_sentence_token = end_of_sentence_token
            self.pad_token = tokenizer.pad_token
            self.filler_sep_token = filler_sep_token

        if start_of_sentence_token is not None:
            self.control_token_ids[start_of_sentence_token] = self.tokenizer.convert_tokens_to_ids(
                [start_of_sentence_token]
            )[0]

        if end_of_sentence_token is not None:
            self.control_token_ids[end_of_sentence_token] = self.tokenizer.convert_tokens_to_ids(
                [end_of_sentence_token]
            )[0]

        if self.pad_token is not None:
            self.control_token_ids[self.pad_token] = self.tokenizer.convert_tokens_to_ids(
                [self.pad_token]
            )[0]

        self.control_token_ids[filler_sep_token] = control_tokens_start_id
        control_tokens_start_id += 1

        for name in chain(template_names, slot_names):
            self.control_token_ids[create_start_token(name)] = control_tokens_start_id
            control_tokens_start_id += 1

            self.control_token_ids[create_end_token(name)] = control_tokens_start_id
            control_tokens_start_id += 1

        # create dict reverse dict for control token ids
        self.control_token_ids_reverse = {token_id: token for token, token_id in self.control_token_ids.items()}

    def tokenize(self, pretokens: Sequence[str], skip_special_tokens: bool = False):
        is_split = (not isinstance(pretokens, str))
        token_ids = self.tokenizer.tokenize(pretokens, is_split_into_words=is_split)
        if skip_special_tokens and len(token_ids) > 0 and token_ids[0] == self.tokenizer.bos_token_id:
            token_ids = token_ids[1:]
        if skip_special_tokens and len(token_ids) > 0 and token_ids[-1] == self.tokenizer.eos_token_id:
            token_ids = token_ids[:-1]
        return token_ids#self.convert_ids_to_tokens(token_ids)

    def convert_tokens_to_ids(self, tokens: Sequence[str]):
        token_ids = list()

        for token in tokens:
            if token in self.control_token_ids:
                token_ids.append(self.control_token_ids[token])
            else:
                token_ids.append(self.tokenizer.convert_tokens_to_ids([token])[0])

        return token_ids

    def convert_ids_to_tokens(self, token_ids: Sequence[int]):
        token_list = list()

        for token_id in token_ids:
            if token_id in self.control_token_ids_reverse:
                token_list.append(self.control_token_ids_reverse[token_id])
            else:
                token_list.append(
                    self.tokenizer.convert_ids_to_tokens([token_id])[0]#.replace("\u2581", "")# T5 prepends \u2581 for some reason...
                )

        return token_list

    def is_control_token(self, token: str):
        return token in self.control_token_ids

    def to_json(self, filename):
        # dict which stores names of sos, eos and filler sep token
        special_token_names = dict()
        special_token_names['sos'] = self.start_of_sentence_token
        special_token_names['eos'] = self.end_of_sentence_token
        special_token_names['filler_sep_token'] = self.filler_sep_token
        special_token_names['filler_sep_token'] = self.filler_sep_token
        special_token_names['pad_token'] = self.pad_token

        # control token data is saved as pair [control_token_ids, special_token_names]
        fp = open(filename, 'w')
        json.dump([self.control_token_ids, special_token_names], fp)
        fp.close()

    def get_vocab_size(self):
        # -2 since <s> and </s> are part of led tokenizer
        return self.tokenizer.vocab_size + len(self.control_token_ids) - 2


def pt_stack_lists(lists, padding_number):
    max_len = max([len(l) for l in lists])

    padded_tensors = list()
    for l in lists:
        num_padding_numbers = max_len - len(l)
        padded_tensors.append(torch.tensor(l + [padding_number] * num_padding_numbers))

    return torch.stack(padded_tensors, dim=0)


'''
model_name = 'allenai/led-base-16384'
tokenizer = TemplateGenerationTokenizer(
    led_tokenizer=LEDTokenizer.from_pretrained(model_name),
    template_names=['template1', 'template2'],
    slot_names=['Slot_name1', 'slot_name2'],
    start_of_sentence_token='<s>',
    end_of_sentence_token='</s>',
    filler_sep_token='[FILLER_SEP]',
    control_tokens_start_id=100
)

tokenizer.to_json(filename='special_tokens.json')
new_tokenizer = TemplateGenerationTokenizer(
    led_tokenizer=LEDTokenizer.from_pretrained(model_name),
    json_filename='special_tokens.json'
)

print(tokenizer.control_token_ids)
print('------------------------')
print(new_tokenizer.control_token_ids)
'''
