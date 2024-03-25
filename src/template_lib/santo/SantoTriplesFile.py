import csv

from template_lib.data_classes.TemplateCollection import *
from template_lib.rdf_strings import *


class SantoTriplesFile:
    def __init__(self, filename):
        # list for triples
        self._triples = []

        # open csv file containing triples
        with open(filename) as f:
            lines = f.readlines()

        # remove whitespace at beginning/end of each line
        lines = [line.strip() for line in lines]

        # remove comments, i.e., lines starting with '#'
        lines = [line for line in lines if not line.startswith('#') and len(line) > 0]

        # extract triples from each line, i.e., split in subject, predicate, object
        csv_reader = csv.reader(lines, delimiter=' ', quotechar='"', skipinitialspace=True)
        for cols in csv_reader:
            # first three columns represent triple
            self._triples.append(tuple(cols[:3]))

    def __len__(self):
        return len(self._triples)

    def __iter__(self):
        return iter(self._triples)

    def count_subject_references(self):
        counts_dict = dict()

        for subject, predicate, obejct in self._triples:
            if is_rdf_type_predicate(predicate) or is_rdf_label_predicate(predicate):
                continue

            # extract raw template identifier string from subject string
            template_id = extract_rdf_identifier(subject)

            # increment counter for template id
            if template_id not in counts_dict:
                counts_dict[template_id] = 1
            else:
                counts_dict[template_id] += 1

        return counts_dict

    def extract_template_ids(self, min_ref_count=1):
        # count how often template is referenced as subject role
        ref_counts_dict = self.count_subject_references()

        # filter template ids by min ref count
        return [template_id for template_id, ref_count in ref_counts_dict.items() if ref_count >= min_ref_count]

    def print_out(self):
        print('number of triples:', len(self._triples), '--------------')
        print('triples: =====================')

        for triple in self._triples:
            print(triple)

    # use template ids from triples file to create template collection consisting of empty template
    # i.e., only template type and id is provided        
    def extract_initial_template_collection(self):
        template_collection = TemplateCollection()

        # iterate template ids of triples file and create empty template
        for template_id in self.extract_template_ids():
            # santo template ids follow schema TemplateType_number
            template_type, _ = template_id.split('_')
            template_collection.add_empty_template(template_type, template_id)

        return template_collection


'''            
triples_file = SantoTriplesFile('/home/cwitte/annotation/data3/dm2 11315821_admin.n-triples')
for triple in triples_file:
    print(triple)
'''
