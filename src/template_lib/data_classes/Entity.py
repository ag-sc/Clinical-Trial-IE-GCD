import copy
import json
from typing import Dict, Tuple, List

import torch


class Entity:
    def __init__(self, entity_index=None, sentence_index=None, start_pos=None,
                 end_pos=None, label=None, tokens=None):
        # list of tokens representing entity
        self._tokens = tokens

        # global document-level entity index, 0-based
        self._global_entity_index = entity_index

        # index of sentence  which contains entity; space of sentence indices is per document, 0-based
        self._sentence_index = sentence_index

        # in sentence token offset of first entity token
        self._start_pos = start_pos

        # in sentence token offset of last entity token
        self._end_pos = end_pos

        # entity class label (e.g. most general superclass)
        self._label = label

        # set of slot names by which entity is referenced, if any
        self._referencing_slot_names = set()

        # set of template ids indicating referencing templates
        self._referencing_template_ids = set()

        # vector representation of entity
        self.vector_representation = None

        # chunk index when DocumentChunking is used
        self._chunk_index = None

    def to_dict(self, simplified=False):
        if simplified:
            return ' '.join(self.get_tokens())
        else:
            return {
                "Entity": {
                    "tokens": self._tokens,
                    "global_entity_index": self._global_entity_index,
                    "sentence_index": self._sentence_index,
                    "start_pos": self._start_pos,
                    "end_pos": self._end_pos,
                    "label": self._label,
                }
            }

    def to_json(self, simplified=False):
        return json.dumps(self.to_dict(simplified=simplified), indent=2)

    @classmethod
    def from_dict(cls, d: Dict | Tuple[str, List[str]], simplified: bool = False):
        if simplified:
            assert isinstance(d, Tuple)
            sn, sf = d
            return cls(
                label=sn,
                tokens=sf,
            )
        else:
            assert isinstance(d, Dict) and len(d) == 1 and "Entity" in d
            d = d["Entity"]
            return cls(
                entity_index=d["global_entity_index"],
                sentence_index=d["sentence_index"],
                start_pos=d["start_pos"],
                end_pos=d["end_pos"],
                label=d["label"],
                tokens=d["tokens"],
            )

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    def copy(self):
        return copy.copy(self)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            raise TypeError()

        for k, v in self.__dict__.items():
            if k == 'vector_representation':
                continue
            if v != other.__dict__[k]:
                return False

        return True

    def __neq__(self, other):
        return not self == other

    def __hash__(self):
        if self.get_sentence_index() is None:
            raise Exception('Hash of entity without sentence index is undefined')

        return hash((
            self.get_sentence_index(),
            self.get_start_pos(),
            self.get_end_pos()
        ))

    def __repr__(self):
        return str(self.get_tokens())

    def set_tokens(self, tokens):
        self._tokens = []

        for token in tokens:
            self._tokens.append(token)

    def get_tokens(self):
        return self._tokens

    def set_global_entity_index(self, global_entity_index):
        self._global_entity_index = global_entity_index

    def get_global_entity_index(self):
        return self._global_entity_index

    def get_index(self):
        return self._global_entity_index

    def set_sentence_index(self, sentence_index):
        self._sentence_index = sentence_index

    def get_sentence_index(self):
        return self._sentence_index

    def get_start_pos(self):
        return self._start_pos

    def set_start_pos(self, start_pos):
        self._start_pos = start_pos

    def get_end_pos(self):
        return self._end_pos

    def set_end_pos(self, end_pos):
        self._end_pos = end_pos

    def set_label(self, label):
        self._label = label

    def get_label(self):
        return self._label

    def get_referencing_slot_names(self):
        return self._referencing_slot_names

    def add_referencing_slot_name(self, slot_name):
        self._referencing_slot_names.add(slot_name)

    def get_referencing_template_ids(self):
        return self._referencing_template_ids

    def add_referencing_template_id(self, template_id):
        self._referencing_template_ids.add(template_id)

    def get_vector_representation(self):
        return self.vector_representation

    def set_vector_representation(self, vector):
        self.vector_representation = vector

    def get_chunk_index(self):
        return self._chunk_index

    def set_chunk_index(self, chunk_index):
        self._chunk_index = chunk_index

    def print_out(self):
        print('--- Entity start ---')

        print('global entity index:', self.get_global_entity_index())
        print('sentence index:', self.get_sentence_index())
        print('chunk index:', self.get_chunk_index())
        print('entity tokens:', self.get_tokens())
        print('start pos:', self.get_start_pos())
        print('end pos:', self.get_end_pos())
        print('entity label:', self.get_label())
        print('referencing slots:', self.get_referencing_slot_names())
        print('referencing templates:', self.get_referencing_template_ids())

        print('--- Entity end ---')

    def cpu(self):
        for name, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                self.__dict__[name] = val.cpu()
        return self


def sort_entities_by_pos(entities):
    return sorted(entities, key=lambda entity: (entity.get_sentence_index(), entity.get_start_pos()))
