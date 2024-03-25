from template_lib.data_classes.Entity import *
from template_lib.santo.SantoToken import SantoToken


class Sentence:
    def __init__(self, tokens=None, index=None, santo_tokens=None):
        if santo_tokens is None:
            santo_tokens = []
        if tokens is None:
            tokens = []
        self._tokens = tokens
        self._santo_tokens = santo_tokens
        self._index = index
        self._entities = []

    def to_dict(self, simplified=False):
        return {
            "Sentence": {
                "tokens": self._tokens,
                "santo_tokens": [t.to_dict(simplified=simplified) for t in self._santo_tokens],
                "index": self._index,
                "entities": [e.to_dict(simplified=simplified) for e in self._entities],
            }
        }

    def to_json(self, simplified=False):
        return json.dumps(self.to_dict(simplified=simplified), indent=2)

    @classmethod
    def from_dict(cls, d):
        assert len(d) == 1 and "Sentence" in d
        d = d["Sentence"]
        s = cls(
            tokens=d["tokens"],
            santo_tokens=[SantoToken.from_dict(t) for t in d["santo_tokens"]],
            index=d["index"],
        )
        s._entities = [Entity.from_dict(e) for e in d["entities"]]
        return s

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    def clear(self):
        self._tokens = []
        self._santo_tokens = []
        self._index = None
        self._entities = []

    def __eq__(self, other):
        if not isinstance(other, Sentence):
            raise TypeError()

        if self.get_tokens() != other.get_tokens():
            return False
        if self.get_index() != other.get_index():
            return False
        if self.get_entities() != other.get_entities():
            return False

        return True

    def __neq__(self, other):
        return not self == other

    def __len__(self):
        return len(self._tokens)

    def get_tokens(self):
        return self._tokens.copy()

    def get_santo_tokens(self):
        return self._santo_tokens.copy()

    def get_index(self):
        return self._index

    def set_index(self, index):
        self._index = index

    def get_entities(self):
        return self._entities.copy()

    def add_entity(self, entity):
        self._entities.append(entity)

    def get_num_entities(self):
        return len(self._entities)

    def print_out(self):
        # print sentence index and tokens
        print('Sentence index:', self.get_index())
        print('Sentence string:', ' '.join(self.get_tokens()))

        # print entities
        for entity in self._entities:
            entity.print_out()

    def test_entity_range(self):
        tokens = self.get_tokens()

        for entity in self.get_entities():
            print('entity tokens: ' + ' '.join(entity.get_tokens()))

            tokens_from_range = tokens[entity.get_start_pos():entity.get_end_pos() + 1]
            print('entity tokens from range: ' + ' '.join(tokens_from_range))
            print('######')

    def set_from_annotated_sentence(self, annotated_sentence):
        sentence_index = annotated_sentence.sentence_id - 1
        self._index = sentence_index
        self._tokens = annotated_sentence.tokens.copy()
        self._entities = []

        # extract entities from AnnotatedSentence object
        for entity_index, annotation in enumerate(annotated_sentence.annotations):
            entity = Entity()
            start_index, end_index = annotated_sentence.entity_boundaries[entity_index]

            # set sentence index of entity
            entity.set_sentence_index(sentence_index)

            # set globbal index of entity
            entity.set_global_entity_index(annotation.global_index)

            # set in sentence boundaries of entity
            entity.set_start_pos(start_index)
            entity.set_end_pos(end_index)

            # extract tokens
            tokens = []

            for token_index in range(start_index, end_index + 1):
                tokens.append(annotated_sentence.tokens[token_index])

            entity.set_tokens(tokens)

            # most general superclass of entity
            entity.set_label(annotation.most_general_superclass)

            # estimate referencing slot names
            for (group_id, slot_name) in annotation.referencing_slots:
                entity.add_referencing_slot_name(slot_name)
                entity.add_referencing_template_ids(group_id)

            # add entity to collection
            self.add_entity(entity)
