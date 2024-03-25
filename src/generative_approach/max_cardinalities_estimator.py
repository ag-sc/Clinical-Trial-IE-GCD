import json
import sys
from typing import Dict

from generative_approach.utils import IdentityTokenizer
from template_lib.data_handling import import_train_test_data_from_task_config
from template_lib.max_cardinalities_estimator import gen_max_slot_cardinalities

if __name__ == '__main__':
    with open(sys.argv[1]) as fp:
        task_config_dict = json.load(fp)

        # import training data
        train_dataset, _ = import_train_test_data_from_task_config(
            task_config_dict,
            IdentityTokenizer()
        )

        max_slots_cardinalities: Dict[str, int] = gen_max_slot_cardinalities(train_dataset)

        for slot_name in sorted(max_slots_cardinalities):
            print(f'{slot_name}: {max_slots_cardinalities[slot_name]}')

        with open(sys.argv[2], 'w') as fpout:
            json.dump(max_slots_cardinalities, fpout)