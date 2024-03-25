import numpy as np

from template_lib.data_classes.Entity import *


class EntityDecoder:
    def __init__(self, label_indices, no_label, use_slot_labels=True):
        self._label_indices_reverse = {index: label for label, index in label_indices.items()}
        self._no_label_index = label_indices[no_label]
        self._use_slot_labels = use_slot_labels

    def decode_sentence(self, encoded_start_positions, encoded_end_positions, sentence_index):
        label_indices_reverse = self._label_indices_reverse
        no_label_index = self._no_label_index

        # check if length is label arrays is value
        assert len(encoded_start_positions) == len(encoded_end_positions)

        # list of Entity objects representing decoded entities
        decoded_entities = []

        # list of pairs (pos,label_index) for start/end positions of entities
        start_positions = [(pos, label_index) for pos, label_index in enumerate(encoded_start_positions) if
                           label_index != no_label_index]
        end_positions = [(pos, label_index) for pos, label_index in enumerate(encoded_end_positions) if
                         label_index != no_label_index]

        # match extracted start and end positions
        for start_pos, start_label_index in start_positions:
            # filter end positions by label and constraint end_pos >= start_pos
            filtered_end_positions = [(end_pos, end_label_index) for end_pos, end_label_index in end_positions
                                      if end_pos >= start_pos and start_label_index == end_label_index]

            # if matching end pos was found, create Entity object
            if len(filtered_end_positions) > 0:
                (end_pos, end_label_index) = filtered_end_positions[0]
                assert start_label_index == end_label_index
                label = label_indices_reverse[start_label_index]

                entity = Entity()
                entity.set_start_pos(start_pos)
                entity.set_end_pos(end_pos)
                entity.set_sentence_index(sentence_index)

                if self._use_slot_labels:
                    entity.add_referencing_slot_name(label)
                else:
                    entity.set_label(label)

                decoded_entities.append(entity)

        return decoded_entities

    def decode_chunk(self, encoded_start_positions_chunk, encoded_end_positions_chunk, chunk):
        # list for decoded entities
        decoded_entities = []

        # decode entities of each sentence given by current chunk
        for sentence_index in chunk.get_sentence_indices():
            # get encoded start/end positions of current sentence
            encoded_start_positions_sentence = chunk.extract_sentence_subarray(encoded_start_positions_chunk,
                                                                               sentence_index)
            encoded_end_positions_sentence = chunk.extract_sentence_subarray(encoded_end_positions_chunk,
                                                                             sentence_index)

            # decode entities of current sentence
            entities = self.decode_sentence(encoded_start_positions_sentence, encoded_end_positions_sentence,
                                            sentence_index)
            decoded_entities.extend(entities)

        return decoded_entities

    def decode_document_chunking(self, encoded_start_positions, encoded_end_positions, doc_chunking):
        num_chunks = doc_chunking.get_num_chunks()
        assert np.shape(encoded_start_positions)[0] == num_chunks
        assert np.shape(encoded_end_positions)[0] == num_chunks
        decoded_entities = []
        print(np.shape(encoded_start_positions))
        # get arrays of each chunk
        encoded_start_positions = np.vsplit(encoded_start_positions, num_chunks)
        encoded_start_positions = np.squeeze(encoded_start_positions)

        encoded_end_positions = np.vsplit(encoded_end_positions, num_chunks)
        encoded_end_positions = np.squeeze(encoded_end_positions)

        # decode entities of each chunk
        for i, chunk in enumerate(doc_chunking.get_chunks()):
            entities_chunk = self.decode_chunk(encoded_start_positions[i], encoded_end_positions[i], chunk)
            decoded_entities.extend(entities_chunk)

        return decoded_entities
