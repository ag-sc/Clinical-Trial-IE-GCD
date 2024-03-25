import numpy as np

from template_lib.data_classes.Chunk import *
from .DocumentChunking import *
from .data_classes.Sentence import Sentence


class DocumentEncoder:
    def __init__(self, tokenizer, label_indices, no_label, use_slot_labels=True,
                 cls_token='[CLS]', sep_token='[SEP]'):
        self._tokenizer = tokenizer
        self._label_indices = label_indices
        self._no_label = no_label
        self._use_slot_labels = use_slot_labels

        # special token ids
        [self._cls_token_id] = tokenizer.convert_tokens_to_ids([cls_token])
        [self._sep_token_id] = tokenizer.convert_tokens_to_ids([sep_token])

    def get_cls_token_id(self):
        return self._cls_token_id

    def get_sep_token_id(self):
        return self._sep_token_id

    def encode_tokens_sentence(self, sentence):
        if not isinstance(sentence, Sentence):
            raise TypeError('Sentence object expected')

        tokens = sentence.get_tokens()
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def encode_tokens_chunk(self, chunk, max_chunk_size):
        if not isinstance(chunk, Chunk):
            raise TypeError('Chunk object expected')

        chunk_token_ids = [self._cls_token_id]

        # encode tokens of each sentence in chunk
        for sentence in chunk.get_sentences():
            sentence_token_ids = self.encode_tokens_sentence(sentence)
            chunk_token_ids.extend(sentence_token_ids)

            # [SEP] token after each sentence
            chunk_token_ids.append(self._sep_token_id)

        # check if number of tokens in chunk is valid
        assert len(chunk_token_ids) <= max_chunk_size

        # convert to numpy array
        chunk_token_ids = np.array(chunk_token_ids)

        # padding
        num_unused_positions = max_chunk_size - len(chunk_token_ids)

        if num_unused_positions > 0:
            chunk_token_ids = np.pad(chunk_token_ids, (0, num_unused_positions), 'constant')

        return chunk_token_ids

    def encode_tokens_document_chunking(self, document_chunking):
        if not isinstance(document_chunking, DocumentChunking):
            raise TypeError('DocumentChunking object expected')

        # list for encoded tokens of each chunk in document chunking
        encoded_chunk_tokens = []

        # encode tokens of each chunk
        for chunk in document_chunking.get_chunks():
            encoded_chunk_tokens.append(self.encode_tokens_chunk(chunk, document_chunking.get_max_chunk_size()))

        return np.vstack(encoded_chunk_tokens)

    def encode_entity_positions_sentence(self, sentence):
        no_label_index = self._label_indices[self._no_label]
        num_tokens = sentence.get_num_tokens()

        if self._use_slot_labels:
            # get entities which are referenced by a slot
            entities = [entity for entity in sentence.get_entities() if len(entity.get_referencing_slot_names()) > 0]
        else:
            entities = sentence.get_entities()

        # check if sentence contains entities
        if len(entities) == 0:
            return [no_label_index] * num_tokens, [no_label_index] * num_tokens

        # create numpy arrays for encoded positions
        encoded_start_positions = np.full(shape=(num_tokens,), fill_value=no_label_index)
        encoded_end_positions = np.full(shape=(num_tokens,), fill_value=no_label_index)

        # encode start and end positions
        for entity in entities:
            # estimate label index
            if self._use_slot_labels:
                label = list(entity.get_referencing_slot_names())[0]
            else:
                label = entity.get_label()

            if label in self._label_indices:
                label_index = self._label_indices[label]
            else:
                label_index = no_label_index

            encoded_start_positions[entity.get_start_pos()] = label_index
            encoded_end_positions[entity.get_end_pos()] = label_index

        return encoded_start_positions.tolist(), encoded_end_positions.tolist()

    def encode_entity_positions_chunk(self, chunk, dummy_value=0):
        # lists for encoded start and end positions of entities
        # first pos is CLS token and hence no entity
        encoded_start_positions_chunk = [dummy_value]
        encoded_end_positions_chunk = [dummy_value]

        for sentence in chunk.get_sentences():
            encoded_start_positions_sentence, encoded_end_positions_sentence = self.encode_entity_positions_sentence(
                sentence)

            encoded_start_positions_chunk.extend(encoded_start_positions_sentence)
            encoded_end_positions_chunk.extend(encoded_end_positions_sentence)

            # add dummy encoding for SEP token
            encoded_start_positions_chunk.append(dummy_value)
            encoded_end_positions_chunk.append(dummy_value)

        return np.array(encoded_start_positions_chunk), np.array(encoded_end_positions_chunk)

    def encode_entity_positions_document_chunking(self, document_chunking, dummy_value=0):
        max_chunk_size = document_chunking.get_max_chunk_size()

        # list of numpy arrays containing encoded start/end positions of entities
        # for each chunk
        encoded_start_positions_document = []
        encoded_end_positions_document = []

        for chunk in document_chunking.get_chunks():
            encoded_start_positions_chunk, encoded_end_positions_chunk = self.encode_entity_positions_chunk(chunk,
                                                                                                            dummy_value)

            # validate length of encoding
            assert len(encoded_start_positions_chunk) == len(encoded_end_positions_chunk)
            assert len(encoded_start_positions_chunk) <= max_chunk_size

            # padding
            num_unused_tokens = max_chunk_size - len(encoded_start_positions_chunk)

            if num_unused_tokens > 0:
                encoded_start_positions_chunk = np.pad(encoded_start_positions_chunk, (0, num_unused_tokens),
                                                       'constant')
                encoded_end_positions_chunk = np.pad(encoded_end_positions_chunk, (0, num_unused_tokens), 'constant')

            encoded_start_positions_document.append(encoded_start_positions_chunk)
            encoded_end_positions_document.append(encoded_end_positions_chunk)

        # stack encodings of chunks
        return np.vstack(encoded_start_positions_document), np.vstack(encoded_end_positions_document)

    def create_token_mask_chunk(self, chunk, max_chunk_size):
        num_tokens_chunk = 1  # CLS token

        for sentence in chunk.get_sentences():
            num_tokens_chunk += sentence.get_num_tokens() + 1  # + 1 for SEP token

        tokens_mask = np.zeros((max_chunk_size,))

        for i in range(num_tokens_chunk):
            tokens_mask[i] = 1

        return tokens_mask

    def create_token_masks_document_chunking(self, document_chunking):
        token_masks = []
        max_chunk_size = document_chunking.get_max_chunk_size()

        # create token mask for each chunk
        for chunk in document_chunking.get_chunks():
            token_masks.append(self.create_token_mask_chunk(chunk, max_chunk_size))

        return np.vstack(token_masks)
