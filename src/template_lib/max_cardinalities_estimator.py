from typing import Dict

def gen_max_slot_cardinalities(dataset):
    num_slot_fillers_dict = dict()

    for _, template_collection in dataset:
        for template in template_collection:
            for slot_name, slot_fillers in template:
                num_slot_fillers_dict.setdefault(slot_name, []).append(len(slot_fillers))

    max_slots_cardinalities: Dict[str, int] = {slot_name: max(num_slot_fillers_list)
                                               for slot_name, num_slot_fillers_list in num_slot_fillers_dict.items()}

    return max_slots_cardinalities
