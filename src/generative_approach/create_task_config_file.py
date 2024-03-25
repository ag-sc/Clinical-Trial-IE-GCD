import collections
import json
import os
import sys
from argparse import ArgumentParser

from template_lib import ctro, consts
from template_lib.santo.SantoDataset import SantoDataset, import_trial_ids
from utils import IdentityTokenizer

if __name__ == "__main__":
    # process arguments ###################################################
    argparser = ArgumentParser()
    argparser.add_argument('--host', required=True)
    argparser.add_argument('--disease-prefix', required=True)
    argparser.add_argument('--min-slot-freq', required=True)
    argparser.add_argument('--filename', required=True)
    argparser.add_argument('--model', required=False, default="allenai/led-base-16384")
    # argparser.add_argument('--model-dim', required=False, default=None)
    argparser.add_argument('--batch-size', required=False, default=None)
    argparser.add_argument('--lr', required=False, default=None)
    argparser.add_argument('--ld', required=False, default=None)
    argparser.add_argument('--epochs', required=False, default=None)
    argparser.add_argument('--trials', required=False, default=30)
    argparser.add_argument('--grammar', required=False, default=None)
    argparser.add_argument('--ptrmodel', action='store_true')
    argparser.add_argument('--ptrdecoding', action='store_true')
    argparser.add_argument('--traindecoding', action='store_true')
    argparser.add_argument('--noevaldecoding', action='store_true')
    arguments = argparser.parse_args()

    host = arguments.host
    disease_prefix = arguments.disease_prefix
    min_slot_freq = int(arguments.min_slot_freq)
    task_config_filename = arguments.filename
    model_name = arguments.model
    grammar = arguments.grammar
    batch_size = 1
    if arguments.batch_size is not None:
        batch_size = int(arguments.batch_size)
    epochs = 30
    if arguments.epochs is not None:
        epochs = int(arguments.epochs)

    trials = int(arguments.trials)
    lr = None
    if arguments.lr is not None:
        lr = float(arguments.lr)
    ld = None
    if arguments.ld is not None:
        ld = float(arguments.ld)

    ptr_model = arguments.ptrmodel
    ptr_decoding = arguments.ptrdecoding
    train_decoding = arguments.traindecoding
    no_eval_decoding = arguments.noevaldecoding

    print('host:', host)
    print('disease prefix:', disease_prefix)
    print('min slot freq:', min_slot_freq)
    print('task config filename:', task_config_filename)
    print('model name:', model_name)
    # print('model dim:', model_dim)
    print('batch size:', batch_size)
    print('epochs:', epochs)

    if grammar is None:
        if disease_prefix == 'dm2':
            grammar = os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", "grammar_gl.txt")
        elif disease_prefix == 'gl':
            grammar = os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", "grammar_dm2.txt")
        else:
            raise Exception('Unknown disease prefix and no grammar file given: ' + disease_prefix)


    # host = sys.argv[1]
    # disease_prefix = sys.argv[2]
    # min_slot_freq = int(sys.argv[3])
    # task_config_filename = sys.argv[4]
    # model_name = 'allenai/led-base-16384'
    #
    # if len(sys.argv) > 5:
    #     model_name = sys.argv[5]
    #
    # print('host:', host)
    # print('disease prefix:', disease_prefix)
    # print('min slot freq:', min_slot_freq)
    # print('task config filename:', task_config_filename)

    ontology = ctro.CtroOntology(consts.template_lib_path)

    task_config_dict = {
        'disease_prefix': disease_prefix,
        'grammar_file': grammar,

        # dataset paths
        'dataset_path': None,
        'top_level_templates': ['Publication'],

        # set below from Ctro ontology
        'slots_ordering_dict': None,

        # set below
        'used_slots': None,

        # list of slot names which contain template instances as slot fillers
        'slots_containing_templates': {
            'describes': 'ClinicalTrial',
            'hasArm': 'Arm',
            'hasPopulation': 'Population',
            'hasDiffBetweenGroups': 'DiffBetweenGroups',
            'hasOutcome1': 'Outcome',
            'hasOutcome2': 'Outcome',
            'hasOutcome': 'Outcome',
            'hasAdverseEffect': 'Outcome',
            'hasIntervention': 'Intervention',
            'hasMedication': 'Medication',
            'hasEndpoint': 'Endpoint',
        },

        # huggingface model names
        'model_name': model_name,

        # training configuration
        'batch_size': batch_size,# if model_name == 'allenai/led-base-16384' else 1,
        'num_epochs': epochs,# if model_name == 'allenai/led-base-16384' else 50,
        'trials': trials,
        #'initial_learning_rate': 1e-5 if model_name == 'allenai/led-base-16384' else 5e-5
        "ptr_model": ptr_model,
        "ptr_decoding": ptr_decoding,
        "train_decoding": train_decoding,
        "no_eval_decoding": no_eval_decoding,
    }

    # dataset path ######################################
    if host == 'local':
        task_config_dict['dataset_path'] = consts.data_path
    elif host == 'rational':
        task_config_dict['dataset_path'] = consts.data_path#TODO: replace
    else:
        raise Exception('unknown host: ' + host)

    if lr is not None:
        task_config_dict['lr'] = lr

    if ld is not None:
        task_config_dict['ld'] = ld

    # set id filenames ####################################
    if disease_prefix == 'dm2':
        task_config_dict['rel_train_ids_filename'] = 'dataset_splits/dm2_train_ids.txt'
        task_config_dict['rel_validation_ids_filename'] = 'dataset_splits/dm2_validation_ids.txt'
        task_config_dict['rel_test_ids_filename'] = 'dataset_splits/dm2_test_ids.txt'
    elif disease_prefix == 'gl':
        task_config_dict['rel_train_ids_filename'] = 'dataset_splits/glaucoma_train_ids.txt'
        task_config_dict['rel_validation_ids_filename'] = 'dataset_splits/glaucoma_validation_ids.txt'
        task_config_dict['rel_test_ids_filename'] = 'dataset_splits/glaucoma_test_ids.txt'
    else:
        raise Exception('unknown disease prefix: ' + disease_prefix)

    # slots ordering dict; key: template name; values: list of slots for respective template ##########
    slots_ordering_dict = dict()
    for template_type in ontology.used_group_names:
        slots_ordering_dict[template_type] = sorted(ontology.group_slots[template_type])
    task_config_dict['slots_ordering_dict'] = slots_ordering_dict

    # set used slots ###################################################
    dataset_path = consts.data_path #'/home/cwitte/annotation/data3'
    rel_train_ids_filename = task_config_dict['rel_train_ids_filename']

    train_ids_filename = os.path.join(dataset_path, rel_train_ids_filename)
    train_ids = import_trial_ids(train_ids_filename)

    tokenizer = IdentityTokenizer()
    dataset = SantoDataset(dataset_path, train_ids, disease_prefix, 'admin', tokenizer)

    # count how often slots are used
    slot_names = list()
    for doc, temp_collection in dataset:
        for template in temp_collection:
            for slot_name, slot_fillers in template:
                slot_names.append(slot_name)

    slot_freqs = collections.Counter(slot_names)
    print('========')
    print('slot frequencies: ', slot_freqs)
    print('========')
    used_slots = [slot_name for slot_name, freq in slot_freqs.items() if freq >= min_slot_freq]

    # remove slots which links DiffBetweenGroups and Outcome templates to avoid cycles
    used_slots.remove('hasOutcome1')
    used_slots.remove('hasOutcome2')
    print('used slots: ', used_slots)
    task_config_dict['used_slots'] = used_slots

    # save task config dict as json file #################################
    fp = open(task_config_filename, 'w')
    json.dump(task_config_dict, fp)
    fp.close()
