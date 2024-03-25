import csv
import itertools

from template_lib.data_classes.Sentence import *
from template_lib.data_classes.TokenSequence import *
from template_lib.santo.SantoToken import *

# constants for column indices of csv file
COL_INDEX_SENTENCE_NUMBER = 1
COL_INDEX_DOC_CHAR_ONSET = 6
COL_INDEX_DOC_CHAR_OFFSET = 7
COL_INDEX_TOKEN = 8


class SantoTokenizationFile:
    def __init__(self, filename):
        # list for SantoToken objects
        santo_tokens = []

        # open csv file containing tokenization
        with open(filename) as f:
            lines = f.readlines()

        # remove whitespace at beginning/end of each line
        lines = [line.strip() for line in lines]

        # remove comments, i.e., lines starting with '#'
        lines = [line for line in lines if not line.startswith('#')]

        # create csv reader
        csv_reader = csv.reader(lines, delimiter=',', quotechar='"', skipinitialspace=True)

        # process each line
        for cols in csv_reader:
            # create new SantoToken object
            santo_token = SantoToken()
            santo_token.sentence_number = int(cols[COL_INDEX_SENTENCE_NUMBER].strip())
            santo_token.doc_char_onset = int(cols[COL_INDEX_DOC_CHAR_ONSET].strip())
            santo_token.doc_char_offset = int(cols[COL_INDEX_DOC_CHAR_OFFSET].strip())
            santo_token.token = cols[COL_INDEX_TOKEN].strip()

            # add new santo token object to list
            santo_tokens.append(santo_token)

        # save complete list of created SantoToken objects
        self._santo_tokens = santo_tokens

    def to_text(self):
        tokens = [santo_token.token for santo_token in self._santo_tokens]
        return ' '.join(tokens)

    def __len__(self):
        return len(self._santo_tokens)

    def __iter__(self):
        return iter(self._santo_tokens)

    def print_out(self):
        for santo_token in self._santo_tokens:
            print('sentence_number:', santo_token.sentence_number)
            print('doc char onset:', santo_token.doc_char_onset)
            print('doc char offset:', santo_token.doc_char_offset)
            print('token:', santo_token.token)
            print('-------------------------------')

        print('number of tokens:', len(self._santo_tokens))

    def get_sentence_numbers(self):
        return {santo_token.sentence_number for santo_token in iter(self)}

    def get_santo_tokens_of_sentence(self, sentence_number):
        if sentence_number not in self.get_sentence_numbers():
            raise Exception('sentence number not found in tokenization file: ' + str(sentence_number))

        return [santo_token for santo_token in self._santo_tokens if santo_token.sentence_number == sentence_number]

    def extract_token_sequence(self, sentence_number, tokenizer=None):
        # santo tokens of query sentence
        santo_tokens = self.get_santo_tokens_of_sentence(sentence_number)

        # list for token,offset pairs of query sentence
        token_offset_pairs = []

        # if a tokenizer is provided, use it to subtokenize santo tokens
        if tokenizer is not None:
            for santo_token in santo_tokens:
                # tokenize token into subtokens
                subtokens = tokenizer.tokenize(santo_token.token)

                # add subtokens to token offset pairs list
                for subtoken in subtokens:
                    token_offset_pairs.append((subtoken, santo_token.doc_char_onset))
        else:
            for santo_token in santo_tokens:
                token_offset_pairs.append((santo_token.token, santo_token.doc_char_onset))

        return TokenSequence(token_offset_pairs)

    def get_sentence_onset_range(self, sentence_number):
        santo_tokens = self.get_santo_tokens_of_sentence(sentence_number)
        santo_tokens = sorted(santo_tokens, key=lambda santo_token: santo_token.doc_char_onset)

        return santo_tokens[0].doc_char_onset, santo_tokens[-1].doc_char_offset - 1

    def extract_sentence(self, sentence_number, tokenizer=None):
        santo_tokens = self.get_santo_tokens_of_sentence(sentence_number)
        tokens = [santo_token.token for santo_token in santo_tokens]

        # if tokenizer is provided, subtokenize tokens of sentence
        if tokenizer is not None:
            # subtokenization
            tokens = [tokenizer.tokenize(santo_token.token, skip_special_tokens=True) for santo_token in santo_tokens]

            # if any(["<unk>" in tks for tks in tokens]):
            #     tokens = [tokenizer.tokenize(santo_token.token) for santo_token in santo_tokens]

            # join lists of tokens
            tokens = list(itertools.chain(*tokens))

        return Sentence(tokens=tokens, index=sentence_number - 1, santo_tokens=santo_tokens)  # -1 because of id to index conversion

    def extract_all_sentences(self, tokenizer=None):
        sentence_numbers = self.get_sentence_numbers()
        return [self.extract_sentence(sentence_number, tokenizer) for sentence_number in sentence_numbers]


'''
tokenization_file = SantoTokenizationFile('/home/cwitte/annotation/data3/dm2 11315821_export.csv')
#print(tokenization_file.print_out())
char_tokenizer = CharTokenizer()
sentences = tokenization_file.extract_all_sentences()
'''
