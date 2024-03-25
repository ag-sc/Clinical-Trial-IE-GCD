import itertools
from collections import defaultdict
from typing import Set

from .Document import *
from .Template import *


def depth_first_search(start_template: Template):
    template_list = [start_template]
    for template in start_template.get_assigned_templates():
        template_list.extend(depth_first_search(template))
    return template_list


class TemplateCollection:
    def __init__(self, initial_templates=None):
        # collection of template instances is represented as python dict
        # key: template id
        # value: template
        self._templates = dict()

        if initial_templates is not None:
            for template in initial_templates:
                self.add_template(template)

    def to_dict(self, simplified=False):
        if simplified:
            return [t.to_dict(simplified=simplified) for t in self.grouped_by_type["Publication"]]
        else:
            return {
                "TemplateCollection": {
                    "tid": t.to_dict(simplified=simplified)
                    for tid, t in self._templates.items()
                }
            }

    def to_json(self, simplified=False):
        return json.dumps(self.to_dict(simplified=simplified), indent=2)

    @classmethod
    def from_dict(cls, d):
        assert len(d) == 1 and "TemplateCollection" in d
        d = d["TemplateCollection"]
        tc = cls()
        tc._templates = {tid: Template.from_dict(t) for tid, t in d.items()}
        return tc

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    def __len__(self):
        return len(self._templates)

    def __iter__(self):
        return iter(self._templates.values())

    def add_template(self, template):
        if not isinstance(template, Template):
            raise TypeError('Only instances of class Template can be added to TemplateCollection')

        template_id = template.get_id()

        # check if template id is already used
        if template_id in self._templates:
            raise KeyError('template id already in use: ' + str(template_id))

        # add new template
        self._templates[template_id] = template

    def add_empty_template(self, template_type, template_id):
        # create new template instance without any slot fillers
        template = Template(template_type, template_id)

        # add template to collection
        self.add_template(template)

    def clear(self):
        self._templates.clear()

    def get_template_by_id(self, _id):
        if _id not in self._templates:
            raise KeyError('no template found with id ' + _id)

        return self._templates[_id]

    def __getitem__(self, template_id):
        return self.get_template_by_id(template_id)

    def __contains__(self, template: Template):
        return template.get_id() in self._templates

    def get_template_by_index(self, index):
        for template in self:
            if template.get_index() == index:
                return template

        raise IndexError('no template found with index ' + str(index))

    def get_templates_by_type(self, _type):
        return [template for template in self if template.get_type() == _type]

    def remove_empty_templates(self):
        self._templates = {templ.get_id(): templ for templ in self if len(list(templ)) > 0}

    # assign new 0-based indices to all template of collection
    def assign_new_indices(self):
        for i, template in enumerate(self):
            template.set_index(i)

    # assign entities to templates;
    # uses referencing_template_ids and referencing_slot_names attributes of entity objects
    def assign_entities(self, entities):
        for entity in entities:
            for templyte_id in entity.get_referencing_template_ids():
                slot_name = list(entity.get_referencing_slot_names())[0]
                template = self.get_template_by_id(templyte_id)
                template.add_slot_filler(slot_name, entity)

    def print_out(self):
        for template in iter(self):
            template.print_out()
            print('----------')

    def get_template_types(self):
        return {template.get_type() for template in self}

    def use_slot_names_as_entity_labels(self):
        for template in self:
            for slot_name, slot_fillers in template:
                for slot_filler in slot_fillers:
                    if isinstance(slot_filler, Entity):
                        slot_filler.set_label(slot_name)

    def get_filled_slot_names(self):
        slot_names = [template.get_filled_slot_names() for template in self]
        return set().union(*slot_names)

    def set_entity_tokens(self, document):
        entities = list(itertools.chain(*[temp.get_assigned_entities() for temp in self]))
        document.set_entity_tokens(entities)

    def count_template_references(self, template_ids=None):
        template_ref_counts = {template.get_id(): 0 for template in self}

        for template in self:
            for slot_name, slot_fillers in template:
                for slot_filler in slot_fillers:
                    if isinstance(slot_filler, Template):
                        template_ref_counts[slot_filler.get_id()] += 1

        return template_ref_counts

    def remove(self, template_id):
        del self._templates[template_id]

    def get_all_entities(self) -> Set[Entity]:
        return {entity for template in self for entity in template.get_assigned_entities()}

    def get_connected_subbcollection(self, start_template: Template):
        template_list = depth_first_search(start_template)
        template_collection = TemplateCollection()

        for template in template_list:
            if template not in template_collection:
                template_collection.add_template(template)

        return template_collection

    @property
    def grouped_by_type(self):
        grouped = defaultdict(list)
        for t in self:
            grouped[t.get_type()].append(t)
        return grouped

    def cpu(self):
        for template in self:
            template.cpu()
        return self
