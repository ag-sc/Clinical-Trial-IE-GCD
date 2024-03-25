from typing import Dict, List

from template_lib.data_classes.Entity import Entity, sort_entities_by_pos
from template_lib.data_classes.Template import Template, sort_templates
from template_lib.data_classes.TemplateCollection import TemplateCollection
from generative_approach.utils import *


class LinearTemplateEncoder:
    def __init__(
            self,
            top_level_templates: Sequence[str],
            slots_ordering: Dict[str, Sequence[str]],
            templates_ordering: Sequence[str],
            used_slots: Iterable[str],
            slots_containing_templates: Dict[str, str],
            filler_sep_token: str,
            start_of_document_token: str,
            end_of_document_token: str
    ):
        self._top_level_templates = top_level_templates
        self._slots_ordering = slots_ordering
        self._templates_ordering = templates_ordering
        self._used_slots = used_slots
        self._slots_containing_templates = slots_containing_templates
        self._filler_sep_token = filler_sep_token
        self._start_of_document_token = start_of_document_token
        self._end_of_document_token = end_of_document_token

    def encode_slot_filler(self, slot_filler):
        if isinstance(slot_filler, Entity):
            return slot_filler.get_tokens()
        else:
            return self.encode_template(slot_filler)

    def encode_slot(
            self, slot_name: str,
            slot_fillers: List
    ) -> List[str]:
        if len(slot_fillers) == 0:
            return list()

        # list for slot encoding
        token_list = list()

        # sort slot fillers by position and encode
        if isinstance(slot_fillers[0], Entity):
            slot_fillers = sort_entities_by_pos(slot_fillers)
        else:
            slot_fillers = sort_templates(slot_fillers)

        # convert list of encoded slot fillers to string
        for filler in slot_fillers:
            token_list.extend(
                [create_start_token(slot_name)]
                + self.encode_slot_filler(filler)
                + [create_end_token(slot_name)]
            )

        return token_list

    def encode_template(
            self,
            template: Template,
            encode_flat: bool = False,
            sentence_index: int = None
    ) -> List[str]:
        if sentence_index is not None:
            if not encode_flat:
                raise ValueError('When filtering slot fillers by sentence index, only flat encoding is possible')

        slots_to_consider = set(template.get_filled_slot_names()) & set(self._used_slots)
        if encode_flat:
            slots_to_consider -= set(self._slots_containing_templates.keys())

        # if no slots are filled, nothing needs to be encoded
        # TODO check if template is empty after filtering fillers of ALL slots by sentence index
        if len(slots_to_consider) == 0:
            return list()

        # list for final token seq representing encoded template
        template_type = template.get_type()
        result_token_list = [create_start_token(template_type)]

        # encode slots of template
        for slot_name in self._slots_ordering[template_type]:
            if slot_name not in slots_to_consider:
                continue

            slot_fillers = template.get_slot_fillers(slot_name)

            # filter slot fillers by sentence index if given
            if sentence_index is not None:
                slot_fillers = list(filter(
                    lambda
                        entity: entity.get_sentence_index() is not None and entity.get_sentence_index() == sentence_index,
                    slot_fillers
                ))
                if len(slot_fillers) == 0:
                    continue

            # encode slot fillers
            result_token_list.extend(self.encode_slot(slot_name, slot_fillers))

        result_token_list.append(create_end_token(template_type))
        return result_token_list

    def encode_template_collection(
            self,
            template_collection: TemplateCollection,
            encode_flat: bool = False,
            sentence_index: int = None
    ) -> List[str]:
        result_token_list = [self._start_of_document_token] if self._start_of_document_token is not None else []

        if encode_flat:
            for template_type in self._templates_ordering:
                templates = template_collection.get_templates_by_type(template_type)
                if len(templates) == 0:
                    continue

                templates = sort_templates(templates)
                for template in templates:
                    result_token_list.extend(self.encode_template(
                        template=template,
                        encode_flat=True,
                        sentence_index=sentence_index
                    ))
        else:
            for top_level_temp_type in self._top_level_templates:
                templates = template_collection.get_templates_by_type(top_level_temp_type)
                for template in sort_templates(templates):
                    result_token_list.extend(self.encode_template(template))

        if self._end_of_document_token is not None:
            result_token_list.append(self._end_of_document_token)
        return result_token_list


class SequenceDecoder:
    def __init__(
            self,
            tokenizer: TemplateGenerationTokenizer,
            subsequence_manager: DocumentSubsequenceManager,
            slots_of_templates: Dict[str, Iterable[str]],
            slots_containing_templates: Dict[str, str],
            top_level_templates: Iterable[str],
            max_slot_fillers=15
    ):
        self._tokenizer = tokenizer
        self._subsequence_manager = subsequence_manager
        self._slots_of_templates = slots_of_templates
        self._slots_containing_templates = slots_containing_templates
        self._top_level_templates = top_level_templates
        self._max_slot_fillers = max_slot_fillers

        self._template_collection = TemplateCollection()
        self._token_sequence = list()
        self._next_possible_control_token_ids: Set[int] = set()
        self._next_possible_text_token_ids: Set[int] = set()

        # stack of objects which are currently decoded
        # could either be instances of Template, tuple(pair) or list;
        # in case of tuple, the pair (slot_name, list(slot_filler))
        # contains the currently decoded slot
        # in case of list: contains tokens of the currently decoded slot
        self._decoding_path = list()

        # counter for template id generation
        self._template_id_counter = 0

        self._update_next_possible_text_token_ids()
        self._update_next_possible_control_token_ids()

    def reset(self):
        self._template_collection = TemplateCollection()
        self._decoding_path = list()
        self._token_sequence = list()
        self._next_possible_control_token_ids = set()
        self._next_possible_text_token_ids = set()

    def get_token_sequence(self):
        return self._token_sequence

    def get_decoding_stack_size(self):
        return len(self._decoding_path)

    def get_next_possible_control_token_ids(self) -> Set[int]:
        return self._next_possible_control_token_ids

    def get_next_possible_text_token_ids(self) -> Set[int]:
        return self._next_possible_text_token_ids

    def _update_next_possible_control_token_ids(self):
        next_possible_control_tokens = list()

        # next token is start token of top-level template or end-of-sentence token
        if len(self._decoding_path) == 0:
            if len(self._template_collection.get_filled_slot_names()) > 0:
                next_possible_control_tokens.append(self._tokenizer.end_of_sentence_token)

            for temp_type in self._top_level_templates:
                next_possible_control_tokens.append(create_start_token(temp_type))

        elif isinstance(self._decoding_path[-1], Template):
            template: Template = self._decoding_path[-1]
            next_possible_control_tokens.append(create_end_token(template))

            for slot_name in self._slots_of_templates[template.get_type()]:
                if len(template.get_slot_fillers(slot_name)) < self._max_slot_fillers:
                    next_possible_control_tokens.append(create_start_token(slot_name))

        elif isinstance(self._decoding_path[-1], tuple):
            slot_name, slot_fillers = self._decoding_path[-1]
            next_possible_control_tokens.append(create_end_token(slot_name))

            if slot_name in self._slots_containing_templates:
                if len(slot_fillers) < self._max_slot_fillers:
                    temp_type = self._slots_containing_templates[slot_name]
                    next_possible_control_tokens.append(create_start_token(temp_type))

        elif isinstance(self._decoding_path[-1], list):
            slot_name, slot_fillers = self._decoding_path[-2]
            next_possible_control_tokens.append(create_end_token(slot_name))
            next_possible_control_tokens.append(self._tokenizer.filler_sep_token)

        else:
            raise Exception('unknown type of decoding object')

        next_control_token_ids = self._tokenizer.convert_tokens_to_ids(next_possible_control_tokens)
        self._next_possible_control_token_ids = set(next_control_token_ids)

    def _update_next_possible_text_token_ids(self):
        if len(self._decoding_path) == 0 or not isinstance(self._decoding_path[-1], list):
            self._next_possible_text_token_ids = set()
        else:
            prefix = self._decoding_path[-1]
            prefix = self._tokenizer.convert_tokens_to_ids(prefix)
            self._next_possible_text_token_ids = self._subsequence_manager.get_next_possible_tokens(prefix=prefix)

    def get_next_possible_token_ids(self):
        return self._next_possible_control_token_ids | self._next_possible_text_token_ids

    def _close_last_decoding_object(self):
        if len(self._decoding_path) == 0:
            return
        last_decoding_object = self._decoding_path.pop(-1)

        if isinstance(last_decoding_object, list):  # last decoding object is slot filler
            if len(last_decoding_object) == 0:
                return
            slot_name, slot_fillers = self._decoding_path[-1]

            entity = Entity()
            entity.set_label(slot_name)
            entity.set_tokens(last_decoding_object)
            slot_fillers.append(entity)

        elif isinstance(last_decoding_object, tuple):
            slot_name, slot_fillers = last_decoding_object
            template = self._decoding_path[-1]
            assert isinstance(template, Template)
            for slot_filler in slot_fillers:
                template.add_slot_filler(slot_name, slot_filler)

        elif isinstance(last_decoding_object, Template):
            # if template is not top-level template, add it as slot filler to next level template
            if len(self._decoding_path) > 0:
                assert isinstance(self._decoding_path[-1], tuple)
                slot_name, slot_fillers = self._decoding_path[-1]
                slot_fillers.append(last_decoding_object)

    def append_token_id(self, token_id: int):
        token = self._tokenizer.convert_ids_to_tokens([token_id])[0]
        if token_id not in self.get_next_possible_text_token_ids() | self.get_next_possible_control_token_ids():
            raise ValueError('invalid token while decoding: ' + token)

        self._token_sequence.append(token)

        # update decoding path
        if not self._tokenizer.is_control_token(token):
            # check if new list for slot filler has to be created
            if not isinstance(self._decoding_path[-1], list):
                self._decoding_path.append(list())

            self._decoding_path[-1].append(token)
        elif token == self._tokenizer.filler_sep_token or is_end_token(token):
            self._close_last_decoding_object()
        elif is_start_token(token):
            start_token = decode_start_token(token)

            # new decoding object for template
            if start_token in self._slots_of_templates:
                self._template_id_counter += 1
                template = Template(start_token, start_token + '_' + str(self._template_id_counter))
                self._template_collection.add_template(template)
                self._decoding_path.append(template)

            # new decoding object for slot
            elif start_token in self._slots_of_templates.values():
                slot_name = start_token
                self._decoding_path.append((slot_name, list()))

                # if slot fillers have to be textual, append list to decoding path for tokens
                if slot_name not in self._slots_containing_templates:
                    self._decoding_path.append(list())

        self._update_next_possible_text_token_ids()
        self._update_next_possible_control_token_ids()
