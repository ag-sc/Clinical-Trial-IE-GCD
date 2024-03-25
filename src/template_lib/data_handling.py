import os

from template_lib.santo.SantoDataset import import_trial_ids, SantoDataset


def import_train_test_data_from_task_config(task_config_dict, tokenizer):
    dataset_path = task_config_dict['dataset_path']
    rel_train_ids_filename = task_config_dict['rel_train_ids_filename']
    rel_test_ids_filename = task_config_dict['rel_test_ids_filename']

    train_ids_filename = os.path.join(dataset_path, rel_train_ids_filename)
    test_ids_filename = os.path.join(dataset_path, rel_test_ids_filename)

    train_ids = import_trial_ids(train_ids_filename)
    test_ids = import_trial_ids(test_ids_filename)

    train_dataset = SantoDataset(dataset_path, train_ids, task_config_dict['disease_prefix'], 'admin', tokenizer)
    test_dataset = SantoDataset(dataset_path, test_ids, task_config_dict['disease_prefix'], 'admin', tokenizer)

    for doc, tc in train_dataset:
        tc.use_slot_names_as_entity_labels()

    for doc, tc in test_dataset:
        tc.use_slot_names_as_entity_labels()

    return train_dataset, test_dataset


def import_train_test_valid_data_from_task_config(task_config_dict, tokenizer):
    dataset_path = task_config_dict['dataset_path']
    rel_train_ids_filename = task_config_dict['rel_train_ids_filename']
    rel_validation_ids_filename = task_config_dict['rel_validation_ids_filename']
    rel_test_ids_filename = task_config_dict['rel_test_ids_filename']

    train_ids_filename = os.path.join(dataset_path, rel_train_ids_filename)
    validation_ids_filename = os.path.join(dataset_path, rel_validation_ids_filename)
    test_ids_filename = os.path.join(dataset_path, rel_test_ids_filename)

    train_ids = import_trial_ids(train_ids_filename)
    validation_ids = import_trial_ids(validation_ids_filename)
    test_ids = import_trial_ids(test_ids_filename)

    train_dataset = SantoDataset(dataset_path, train_ids, task_config_dict['disease_prefix'], 'admin', tokenizer)
    validation_dataset = SantoDataset(dataset_path, validation_ids, task_config_dict['disease_prefix'], 'admin', tokenizer)
    test_dataset = SantoDataset(dataset_path, test_ids, task_config_dict['disease_prefix'], 'admin', tokenizer)

    for doc, tc in train_dataset:
        tc.use_slot_names_as_entity_labels()

    for doc, tc in validation_dataset:
        tc.use_slot_names_as_entity_labels()

    for doc, tc in test_dataset:
        tc.use_slot_names_as_entity_labels()

    return train_dataset, test_dataset, validation_dataset
