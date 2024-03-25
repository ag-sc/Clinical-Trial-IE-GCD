from template_lib.data_classes.Chunk import Chunk


class DocumentChunking:
    def __init__(self, max_chunk_size, document=None):
        # max tokens of each chunk
        self._max_chunk_size = max_chunk_size

        # list of chunks
        self._chunks = []

        # if document is given, set from document
        if document is not None:
            self.set_from_document(document)

    def append_chunk(self, chunk):
        self._chunks.append(chunk)

    def __iter__(self):
        return iter(self._chunks)

    def set_from_document(self, document):
        self._chunks = []

        # split sentences of document into blocks which
        # fit into chunks
        sentence_blocks = document.split_sentence_seq_for_chunking(self.get_max_chunk_size())

        # create chunks from sentence blocks
        current_chunk_index = 0

        for sentence_block in sentence_blocks:
            chunk = Chunk(sentence_block, current_chunk_index)
            self.append_chunk(chunk)
            current_chunk_index += 1

    def get_max_chunk_size(self):
        return self._max_chunk_size

    def __len__(self):
        return len(self._chunks)

    def get_chunks(self):
        return self._chunks

    def get_chunk_by_index(self, chunk_index):
        chunks_dict = {chunk: chunk.get_chunk_index() for chunk in self.get_chunks()}

        # check if chunk index is valid
        if chunk_index not in chunks_dict:
            raise IndexError('Invalid chunk index')

        return chunks_dict[chunk_index]

    def get_chunk_by_sentence_index(self, sentence_index):
        for chunk in iter(self):
            if sentence_index in chunk.get_sentence_indices():
                return chunk

        # no matching chunk found
        raise IndexError('no chunk found for sentence index ' + str(sentence_index))

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError('int required for chunk index')
        if index < 0 or index >= len(self):
            raise IndexError('index {} out of bounds'.format(index))

        return self._chunks[index]

    def append_chunk(self, chunk):
        self._chunks.append(chunk)

    def set_entity_chunk_indices(self, entities):
        for entity in entities:
            chunk_found = False
            entity_sentence_index = entity.get_sentence_index()

            for chunk in self.get_chunks():
                if entity_sentence_index in chunk.get_sentence_indices():
                    chunk_found = True
                    entity.set_chunk_index(chunk.get_chunk_index())
                    break

            # check if a chunk for current entity was found
            if not chunk_found:
                raise Exception('ERROR: no chunk found for entity')

    def get_num_tokens(self):
        return sum([chunk.get_num_tokens() for chunk in self])

    def get_chunk_offset(self, chunk_index):
        chunk_offset = 0

        for chunk in self:
            if chunk.get_index() == chunk_index:
                return chunk_offset
            else:
                chunk_offset = chunk.get_num_tokens()

        raise Exception('no chunk found with index ' + str(chunk_index))

    def get_absolute_sentence_offset(self, sentence_index):
        # get chunk containing query sentence
        chunk = self.get_chunk_by_sentence_index(sentence_index)

        # compute globbal offset of query sentence
        return self.get_chunk_offset(chunk.get_index()) + chunk.get_sentence_offset(sentence_index)

    def absolute_to_relative_pos(self, absolute_pos):
        # estimate chunk which contains absolute pos
        for chunk in self:
            chunk_offset = self.get_chunk_offset(chunk.get_index())

            if chunk_offset <= absolute_pos and chunk_offset + chunk.get_num_tokens() > absolute_pos:  # chunk found
                # estimate sentence in chunk which contains absolute pos
                for sentence in chunk:
                    sentence_index = sentence.get_index()
                    sentence_offset = chunk.get_sentence_offset(sentence_index)

                    if chunk_offset + sentence_offset <= absolute_pos and chunk_offset + sentence_offset + len(
                            sentence) > absolute_pos:  # sentence found
                        in_sentence_offset = absolute_pos - chunk_offset - sentence_offset
                        return sentence_index, in_sentence_offset

        raise Exception('absolute pos not found in document chunking: ' + str(absolute_pos))

    def get_entity_start_end_indices(self, entities):
        # lists of [chunk_index, in_chunk_offset] lists representing indices
        # into doc chunking of entity start/end positions
        entity_start_indices = []
        entity_end_indices = []

        for entity in entities:
            chunk_index = None
            entity_sentence_index = entity.get_sentence_index()

            # get chunk containing entity ################################
            for chunk in self.get_chunks():
                if entity_sentence_index in chunk.get_sentence_indices():
                    chunk_index = chunk.get_chunk_index()
                    break

            # check if a chunk for current entity was found
            if chunk_index is None:
                raise Exception('ERROR: no chunk found for entity')

            # compute start/end positions of entity in chunk
            in_chunk_sentence_offset = chunk.get_sentence_offset(entity_sentence_index)
            in_chunk_start_pos = in_chunk_sentence_offset + entity.get_start_pos()
            in_chunk_end_pos = in_chunk_sentence_offset + entity.get_end_pos()

            # add computed indices to list
            entity_start_indices.append([chunk_index, in_chunk_start_pos])
            entity_end_indices.append([chunk_index, in_chunk_end_pos])

        return entity_start_indices, entity_end_indices
