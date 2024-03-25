# class for handling sequences of (token, offset) pairs
class TokenSequence:
    def __init__(self, token_offset_pairs=None):
        if token_offset_pairs is None:
            self._token_offset_pairs = []
        else:
            self._token_offset_pairs = [(token, offset) for token, offset in token_offset_pairs]

    def get_token_index_pairs(self, offset_start, offset_end):
        if len(self._token_offset_pairs) == 0:
            return []

        return [(index, token) for index, (token, offset) in enumerate(self._token_offset_pairs)
                if offset >= offset_start and offset <= offset_end]

    def print_out(self):
        print('number of tokens: ' + str(len(self._token_offset_pairs)))
        print('token offset pairs:')

        for token, offset in self._token_offset_pairs:
            print(token, offset)
