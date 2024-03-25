import itertools
import json
from collections import defaultdict
from typing import Dict

from template_lib.data_classes.Entity import *


class Template:
    def __init__(self, _type, _id):
        # string identifier of template instance
        self._id = _id

        # type of template e.g. Medication
        self._type = _type

        # 0-based index of template instance
        self._index = None

        # dict for slots
        # key: string, slot name
        # value: list, can contain instances of Entity or Template
        self._slots = dict()

    def get_id(self):
        return self._id

    def set_id(self, _id):
        self._id = _id

    def get_type(self):
        return self._type

    def set_type(self, _type):
        self._type = _type

    def get_index(self):
        return self._index

    def to_dict(self, simplified=False):
        if simplified:
            return {
                self._type: {
                    sn: [sf.to_dict(simplified=simplified) for sf in sfs]
                    for sn, sfs in self._slots.items()
                }
            }
        else:
            return {
                self._type: {
                    sn: [sf.to_dict(simplified=simplified) for sf in sfs]
                    for sn, sfs in self._slots.items()
                },
                "id": self._id,
            }

    def to_json(self, simplified=False):
        return json.dumps(self.to_dict(simplified=simplified), indent=2)

    @classmethod
    def from_dict(cls, d: Dict, simplified: bool = False, next_id: int = 0):
        if simplified:

            def get_next_id():
                nonlocal next_id
                next_id += 1
                return next_id

            #if not (len(d) == 1):
            #    pass
            #assert len(d) == 1

            ttype = list(d.keys())[0]
            tid = ttype+"_"+str(get_next_id())
            t = cls(ttype, tid)

            t._slots = defaultdict(list)
            #for ld in d[t._type].items():

            for slots in d[t._type]:
                if isinstance(slots, List) or isinstance(slots, str):
                    pass
                for sn, sfs in slots.items():
                    t._slots[sn].extend(
                        [Entity.from_dict((sn, sf), simplified=simplified)
                         if isinstance(sf, List)# and all([isinstance(sfistr, str) for sfi in sf for sfistr in sfi])
                         else Template.from_dict(sf, simplified=simplified, next_id=get_next_id())
                         for sf in sfs]
                    )


            # {
            #     sn: [Entity.from_dict((sn, sf), simplified=simplified)
            #          if isinstance(sf, List) and all([isinstance(sfi, str) for sfi in sf])
            #          else Template.from_dict(sf, simplified=simplified, next_id=get_next_id())
            #          for sf in sfs]
            #     for ld in d[t._type] for sn, sfs in ld.items()
            # }
            return t
        else:
            if not (len(d) == 2 and "id" in d.keys()):
                pass
            assert len(d) == 2 and "id" in d.keys()
            ttype = [k for k in d.keys() if k != "id"][0]
            tid = d["id"]
            t = cls(ttype, tid)
            t._slots = {
                sn: [Entity.from_dict(sf) if "Entity" in sf.keys() else Template.from_dict(sf) for sf in sfs]
                for sn, sfs in d[t._type].items()
            }
            return t

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    def add_slot_filler(self, slot_name, slot_value, verify_constraints=False):
        # if slot  does not yet exist, add 
        if slot_name not in self._slots:
            self._slots[slot_name] = []

        # slot value has to be instance of Entity or Template
        if not isinstance(slot_value, Entity) and not isinstance(slot_value, Template):
            raise TypeError('slot value has to be instance of Entity or Template: ' + str(type(slot_value)))

        # add slot value
        self._slots[slot_name].append(slot_value)

    def get_slot_fillers(self, slot_name):
        if slot_name in self._slots:
            return self._slots[slot_name]
        else:
            return []

    def get_slots(self):
        return self._slots

    def __getitem__(self, key):
        return self.get_slot_fillers(key)

    def get_slot_fillers_as_strings(self, slot_name):
        string_representations = list()

        for slot_filler in self.get_slot_fillers(slot_name):
            if isinstance(slot_filler, Entity):
                string_representations.append(' '.join(slot_filler.get_tokens()))
            elif isinstance(slot_filler, Template):
                string_representations.append(slot_filler.get_id())
            else:
                raise TypeError('unknown slot-filler type:' + str(type(slot_filler)))

        return string_representations

    def get_assigned_entities(self):
        # create list of all slot fillers
        slot_fillers = itertools.chain(*[filler_list for _, filler_list in self])

        # only keep entities
        return [sf for sf in slot_fillers if isinstance(sf, Entity)]

    def get_assigned_templates(self):
        # create list of all slot fillers
        slot_fillers = itertools.chain(*[filler_list for _, filler_list in self])

        # only keep templates
        return [sf for sf in slot_fillers if isinstance(sf, Template)]

    def get_filled_slot_names(self):
        return set(self._slots.keys())

    def is_empty(self):
        return len(self._slots) == 0

    def clear_slot(self, slot_name):
        if slot_name in self._slots:
            del self._slots[slot_name]

    def trim_entity_slots(self, sort_slot_fillers=True):
        # trim slots
        for slot_name in self.get_filled_slot_names():
            slot_fillers = self.get_slot_fillers(slot_name)

            # skip slot if it contains template instances
            if isinstance(slot_fillers[0], Template):
                continue

            # sort slot fillers
            if sort_slot_fillers:
                slot_fillers = sort_entities_by_pos(slot_fillers)

            # trim slot
            self.clear_slot(slot_name)
            self.add_slot_filler(slot_name, slot_fillers[0])

    def __iter__(self):
        return iter([(slot_name, slot_filler) for slot_name, slot_filler in self._slots.items()])

    def __eq__(self, other):
        if not isinstance(other, Template):
            raise TypeError(str(type(other)))

        return self.__dict__ == other.__dict__

    def __hash__(self):
        if self.get_id() is None:
            raise Exception('hash of template without id is undefined')

        return hash(self.get_id())

    def __repr__(self):
        return self.get_id()

    def get_all_slot_fillers(self):
        return sum([
            slot_filler.get_all_slot_fillers() if isinstance(slot_filler, Template)
            else [slot_filler]
            for _, slot_filler_list in iter(self) for slot_filler in slot_filler_list
        ], [])

    def print_out(self):
        print('template id:', self._id)
        print('template type:', self._type)
        print('template index:', self._index)

        # slot fillers
        for slot_name, slot_fillers in iter(self):
            print(slot_name)

            for slot_filler in slot_fillers:
                slot_filler.print_out()

    def get_entity_representative(self):
        assigned_entities = self.get_assigned_entities()
        if len(assigned_entities) > 0:
            return sort_entities_by_pos(assigned_entities)[0]

        assigned_templates = self.get_assigned_templates()
        next_layer_entities = [template.get_assigned_entities() for template in assigned_templates]

        next_layer_entities = list(itertools.chain(*next_layer_entities))
        if len(next_layer_entities) > 0:
            return sort_entities_by_pos(next_layer_entities)[0]

        next_layer_representatives = list()
        for template in assigned_templates:
            entity = template.get_entity_representative()
            if entity is not None:
                next_layer_representatives.append(entity)

        if len(next_layer_representatives) > 0:
            return sort_entities_by_pos(next_layer_representatives)[0]
        else:
            return None

    def cpu(self):
        for slot_name, slot_fillers in iter(self):
            for slot_filler in slot_fillers:
                slot_filler.cpu()


def sort_templates(templates):
    template_triples = list()

    for template in templates:
        entity = template.get_entity_representative()

        if entity is None:
            triple = (template, 0, 0)
        else:
            triple = (template, entity.get_sentence_index(), entity.get_start_pos())

        template_triples.append(triple)

    template_triples = sorted(template_triples, key=lambda t: (t[1], t[2]))
    return [triple[0] for triple in template_triples]
