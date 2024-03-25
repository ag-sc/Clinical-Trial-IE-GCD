import json
import os
import sys
from collections import defaultdict
from pprint import pprint

import jsonpickle

from template_lib import consts
from template_lib.data_classes.Template import Template
from template_lib.santo.SantoDataset import SantoDataset, import_trial_ids

if __name__ == "__main__":
    # 'gl' for Glaucoma and 'dm2' for dianetes
    disease_prefix = 'dm2'

    # directory path of dataset
    dataset_path = consts.data_path

    # filenames containing ids of train/test files
    #rel_train_ids_filename = 'dataset_splits/glaucoma_train_ids.txt'
    #rel_test_ids_filename = 'dataset_splits/glaucoma_test_ids.txt'
    rel_train_ids_filename = 'dataset_splits/dm2_train_ids.txt'
    rel_test_ids_filename = 'dataset_splits/dm2_test_ids.txt'

    train_ids_filename = os.path.join(dataset_path, rel_train_ids_filename)
    test_ids_filename = os.path.join(dataset_path, rel_test_ids_filename)

    # import file ids
    train_ids = import_trial_ids(train_ids_filename)
    test_ids = import_trial_ids(test_ids_filename)

    # import dataset
    train_dataset = SantoDataset(dataset_path, train_ids, disease_prefix, 'admin')
    test_dataset = SantoDataset(dataset_path, test_ids, disease_prefix, 'admin')

    journals = defaultdict(int)

    firstpub = None

    for document, template_collection in train_dataset:
        for t in template_collection.grouped_by_type["Publication"]:
            #td = t.to_dict()
            #t2 = Template.from_dict(td)
            if firstpub is None:
                firstpub = t.to_json(simplified=False)

            for j in t.get_slot_fillers_as_strings("hasJournal"):
                journals[j] += 1
            print(t.get_slot_fillers_as_strings("hasJournal"))

    for document, template_collection in test_dataset:
        for t in template_collection.grouped_by_type["Publication"]:
            for j in t.get_slot_fillers_as_strings("hasJournal"):
                journals[j] += 1
            print(t.get_slot_fillers_as_strings("hasJournal"))

    print(train_dataset.to_json(simplified=False))
    print(firstpub)

    #pprint(journals)
        # if document.get_id() == "30019498":
        #     for s in document.get_sentences():
        #         print(s.get_tokens())
            #template_collection.get_templates_by_type()
            #print(json.dumps(json.loads(jsonpickle.encode(document)), indent=2))
            #print(json.dumps(json.loads(jsonpickle.encode(template_collection)), indent=2))

    #d = [(document, template_collection) for document, template_collection in test_dataset]

    #print(json.dumps(json.loads(jsonpickle.encode(d)), indent=2))
