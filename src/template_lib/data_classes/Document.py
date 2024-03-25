from template_lib.data_classes.AnnotatedSentence import extract_annotated_sentences_from_abstract
from template_lib.data_classes.Sentence import *


def get_doc_onset_token_pairs_from_abstract(abstract):
    # dict containing (doc_char_onset, token) pairs for each sentence
    # key: sentence id
    # value: list of (doc_char_onset, token) pairs
    onset_token_pairs = dict()

    for (sentence_id, doc_char_onset, token) in abstract.tokenization.tokens:
        if sentence_id not in onset_token_pairs:
            onset_token_pairs[sentence_id] = []

        onset_token_pairs[sentence_id].append((doc_char_onset, token))

    return onset_token_pairs


def get_sentence_strings_from_abstract(abstract):
    onset_token_pairs = get_doc_onset_token_pairs_from_abstract(abstract)

    # estimate sentence strings
    sentence_ids = sorted(onset_token_pairs.keys())
    sentences = [[token for onset, token in onset_token_pairs[sentence_id]] for sentence_id in sentence_ids]

    return [' '.join(sentence_tokens) for sentence_tokens in sentences]


class DocumentPosition:
    def __init__(self, sentence_index=None, in_sentence_token_offset=None):
        # index of sentence in document; 0-based
        self.sentence_index = sentence_index

        self.in_sentence_token_offset = in_sentence_token_offset


class Document:
    def __init__(self, sentences=[]):
        # ctro abstract
        self._abstract = None

        # list of Sentence objects
        self._sentences = sentences

        # document id, e.g. pubmed id
        self._id = None

    def to_dict(self, simplified=False):
        return {
            "Document": {
                "abstract": self._abstract,
                "sentences": [s.to_dict(simplified=simplified) for s in self._sentences],
                "id": self._id,
            }
        }

    def to_json(self, simplified=False):
        return json.dumps(self.to_dict(simplified=simplified), indent=2)

    @classmethod
    def from_dict(cls, d):
        assert len(d) == 1 and "Document" in d
        d = d["Document"]
        doc = cls(
            sentences=[Sentence.from_dict(s) for s in d["sentences"]],
        )
        doc._abstract = d["abstract"]
        doc._id = d["id"]
        return doc

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    def get_abstract(self):
        return self._abstract

    def get_sentences(self):
        return self._sentences

    def get_id(self):
        return self._id

    def set_id(self, _id):
        self._id = _id

    def __iter__(self):
        return iter(self._sentences)

    def get_sentence_by_index(self, sentence_index):
        for sentence in self.get_sentences():
            if sentence.get_index() == sentence_index:
                return sentence

        raise IndexError('Sentence not found')

    def get_num_sentences(self):
        return len(self._sentences)

    def append_sentence(self, sentence):
        self._sentences.append(sentence)

    def set_from_abstract(self, abstract, tokenizer):
        self._abstract = abstract
        self._sentences = []

        # extract all annotated sentences from abstract
        annotated_sentences = extract_annotated_sentences_from_abstract(abstract, tokenizer)

        # convert AnnotatedSentence objects to Sentence objects
        for annotated_sentence in annotated_sentences:
            sentence = Sentence()
            sentence.set_from_annotated_sentence(annotated_sentence)
            self._sentences.append(sentence)

    def split_sentence_seq_for_chunking(self, max_chunk_size):
        # list of lists; inner list contains Sentence objects
        sentence_blocks = []

        # list of sentence for current block
        current_block = []

        left_chunk_size = max_chunk_size - 1  # -1 since first token is [CLS]

        # split sentences
        for sentence in self.get_sentences():
            num_tokens = len(sentence)

            # check if sentence fits into chunk at all
            if num_tokens + 2 > max_chunk_size:
                raise Exception('Sentence does not fit into chunk')

            # does sentence fit into current chunk?
            if left_chunk_size >= num_tokens + 1:  # + 1 because of [SEP] token

                # add sentence to current block
                current_block.append(sentence)

                # decrease left chunk size by number of tokens used by current sentence
                left_chunk_size -= (num_tokens + 1)
            else:  # sentence does not fit into chunk; create new one

                # save current block representing current chunk
                sentence_blocks.append(current_block)

                # new list for current block
                current_block = [sentence]

                # initial left chunk size of new block
                left_chunk_size = max_chunk_size - (num_tokens + 2)  # +2 for CLS token and SEP token

        # check if there is an unsaved block
        if len(current_block) > 0:
            # save last block
            sentence_blocks.append(current_block)

        return sentence_blocks

    def get_entities(self):
        entities = []

        for sentence in self.get_sentences():
            entities.extend(sentence.get_entities())

        # return sorted(entities, key=lambda entity: entity.get_global_entity_index())
        return entities

    def set_entity_tokens(self, entities):
        for entity in entities:
            sentence = self.get_sentence_by_index(entity.get_sentence_index())
            entity_tokens = sentence.get_tokens()[entity.get_start_pos():entity.get_end_pos() + 1]
            entity.set_tokens(entity_tokens)
