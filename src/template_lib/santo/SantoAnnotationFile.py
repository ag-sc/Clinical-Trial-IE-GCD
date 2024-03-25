import csv

from template_lib.data_classes.Entity import Entity
from template_lib.santo.SantoAnnotation import *

# indices of columns in csv file
COL_INDEX_CLASS_TYPE = 1
COL_INDEX_DOC_CHAR_ONSET = 2
COL_INDEX_DOC_CHAR_OFFSET = 3
COL_INDEX_TEXT = 4
COL_INDEX_INSTANCES = 6


class SantoAnnotationFile:
    def __init__(self, filename, quotechar='"', escapechar='\\'):
        self.quotechar = quotechar
        self.escapechar = escapechar

        # list containing SanntoAnnotation instances
        self._santo_annotations = []

        # open csv file containing triples
        with open(filename) as f:
            lines = f.readlines()

        # remove whitespace at beginning/end of each line
        lines = [line.strip() for line in lines]

        # remove empty lines and comments, i.e., lines starting with '#'
        lines = [line for line in lines if not line.startswith('#') and len(line) > 0]

        # csv reader for annotations
        csv_reader = csv.reader(lines, delimiter=',', quotechar=self.quotechar, escapechar=escapechar,
                                skipinitialspace=True)

        # process each annotation
        for annotation_record in csv_reader:
            annotation = SantoAnnotation(quotechar)

            annotation.class_type = annotation_record[COL_INDEX_CLASS_TYPE].strip()
            annotation.doc_char_onset = int(annotation_record[COL_INDEX_DOC_CHAR_ONSET].strip())
            annotation.doc_char_offset = int(annotation_record[COL_INDEX_DOC_CHAR_OFFSET].strip())
            annotation.text = annotation_record[COL_INDEX_TEXT].strip()
            annotation.instances = annotation_record[COL_INDEX_INSTANCES].strip()

            self._santo_annotations.append(annotation)

    def __len__(self):
        return len(self._santo_annotations)

    def __iter__(self):
        return iter(self._santo_annotations)

    def __getitem__(self, index):
        # check if index value is valid
        if index < 0 or index >= len(self._santo_annotations):
            raise IndexError('Invalid index of annotation: ' + str(index))

        return self._santo_annotations[index]

    def get_santo_annotations(self, onset_start, onset_end):
        return [annotation for annotation in iter(self) if annotation.doc_char_onset >= onset_start
                and annotation.doc_char_onset <= onset_end]

    def extract_entities(self, sentence_number, tokenization_file, tokenizer=None, template_collection=None):
        token_seq = tokenization_file.extract_token_sequence(sentence_number, tokenizer)

        # get santo annotations of sentence
        onset_start, onset_end = tokenization_file.get_sentence_onset_range(sentence_number)
        annotations = self.get_santo_annotations(onset_start, onset_end)

        # create Entity object from SantoAnnotation objects
        entities = []
        for annotation in annotations:
            entity = Entity()
            entity.set_sentence_index(sentence_number - 1)  # -1 because id to index conversion

            # validate onset/offset value
            if annotation.doc_char_offset < annotation.doc_char_onset:
                print('ERROR')
                continue

            # entity tokens
            token_index_pairs = token_seq.get_token_index_pairs(annotation.doc_char_onset, annotation.doc_char_offset)
            token_indices, tokens = tuple(zip(*token_index_pairs))
            #token_indices = tuple([p[0] for p in token_index_pairs])
            #tokens = tuple([p[1] for p in token_index_pairs])
            entity.set_tokens(tokens)

            # start/end pos
            token_indices = sorted(token_indices)
            entity.set_start_pos(token_indices[0])
            entity.set_end_pos(token_indices[-1])

            # entity label
            entity.set_label(annotation.class_type)

            # referencing slots
            for template_id, slot_name in annotation.get_references():
                entity.add_referencing_template_id(template_id)
                entity.add_referencing_slot_name(slot_name)

                # if template collection is provided, add entity to corresponding slot
                if template_collection is not None:
                    template = template_collection.get_template_by_id(template_id)
                    template.add_slot_filler(slot_name, entity)

            entities.append(entity)

        return entities


'''
annotation_file = SantoAnnotationFile('/home/cwitte/annotation/data3/dm2 11315821_admin.annodb')
tokenization_file = SantoTokenizationFile('/home/cwitte/annotation/data3/dm2 11315821_export.csv')

sentence_number = 7
tokenizer = CharTokenizer()
sentence = tokenization_file.extract_sentence(sentence_number,tokenizer)

entities = annotation_file.extract_entities(sentence_number, tokenization_file, tokenizer)
for entity in entities:
    sentence.add_entity(entity)
    
sentence.test_entity_range()
'''
