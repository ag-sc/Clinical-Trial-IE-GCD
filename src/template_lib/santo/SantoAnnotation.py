from template_lib.rdf_strings import *


def split_instance_string(instances):
    substrings = []

    while len(instances) > 0:
        # remove leading whitespace
        instances = instances.lstrip()

        # dot separates rdf triples -> ignore
        if instances[0] == '.':
            instances = instances[1:]
            continue

        # get next substring, either <...> string or "..." string
        if instances[0] == '<':
            separator_index = instances.index('>') + 1
        elif instances[0] == '"':
            separator_index = instances.index('"', 1) + 1
        else:
            raise Exception('unexpected character in instances string')

        # extract next substrng
        substrings.append(instances[:separator_index])

        # prune instances string
        instances = instances[separator_index + 1:]

    assert len(substrings) % 3 == 0
    return substrings


class SantoAnnotation:
    def __init__(self, quotechar='"'):
        self.class_type = None
        self.doc_char_onset = None
        self.doc_char_offset = None
        self.text = None
        self.instances = None

        self.quotechar = quotechar

    def get_insatnce_tripes(self):
        if len(self.instances) == 0:
            # no references -> return empty list
            return []

        # list of strings representing instance records
        # each string in turn represents an rdf triple
        instance_triples = []

        instance_substrings = split_instance_string(self.instances)

        # extract instance triples from instance substrings
        while len(instance_substrings) > 0:
            instance_triples.append(instance_substrings[:3])
            instance_substrings = instance_substrings[3:]

        return instance_triples

    def get_references(self):
        # extract instance triples
        instance_triples = self.get_insatnce_tripes()

        # list containing string pairs (template_id, slot_name)
        # each one describes a reference to the respective annotation
        annotation_references = []

        # extract annotation references from extracted rdf insatnce triples
        for triple in instance_triples:
            # check if triple containe valid ctro predicate
            if not is_ctro_data_string(triple[1]):
                continue

            # extract reference pair from triple
            template_id = extract_rdf_identifier(triple[0])
            slot_name = extract_rdf_identifier(triple[1])

            annotation_references.append((template_id, slot_name))

        return annotation_references

    def print_out(self):
        print('class type:', self.class_type)
        print('doc char onset:', self.doc_char_onset)
        print('doc char offset:', self.doc_char_offset)
        print('text:', self.text)
        print('instances:', self.instances)


'''                        
santo_annotation = SantoAnnotation()
santo_annotation.instances = '<http://ctro/data#Publication_24249> <http://ctro/data#hasJournal> "Diabetes Care .".'
instance_triples = santo_annotation.get_insatnce_tripes()
'''
