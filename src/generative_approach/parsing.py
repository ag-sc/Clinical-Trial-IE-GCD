import sys
from collections import deque
from typing import List

from transformers import LEDTokenizer

from template_lib.data_handling import import_train_test_data_from_task_config
from generative_approach.models.longformer import Longformer
from template_lib.SlotFillingEvaluation import SlotFillingEvaluation
from template_lib.TemplateAlignment import TemplateAlignment
from template_lib.data_classes.Entity import Entity
from template_lib.data_classes.Template import Template
from template_lib.data_classes.TemplateCollection import TemplateCollection
from generative_approach.utils import *


class TokenNotFoundException(Exception):
    pass


def seek_deque(
        token_deque: deque,
        termination_token: str,
        include_termination_token: bool = False,
        raise_token_not_found_exception: bool = True
):
    extracted_tokens = list()

    while len(token_deque) > 0:
        token = token_deque.popleft()

        # check if termination token was found
        if token == termination_token:
            if include_termination_token:
                extracted_tokens.append(token)

            return extracted_tokens
        else:
            extracted_tokens.append(token)

    # unexpected end of deque
    if raise_token_not_found_exception:
        raise TokenNotFoundException()
    else:
        return extracted_tokens


class Parser:
    def __init__(self, task_config_dict):
        self._used_slots = set(task_config_dict['used_slots'])
        self._used_templates = set(task_config_dict['slots_ordering_dict'].keys())
        self._slots_containing_templates = task_config_dict['slots_containing_templates'].copy()
        self._template_instance_counter = 0
        self._template_collection: TemplateCollection = None

    def parse_document_level(self, output_tokens: List[str]):
        self._template_instance_counter = 0
        self._template_collection = TemplateCollection()
        tokens_deque = deque(output_tokens)

        self.parse_template(tokens_deque)
        return self._template_collection

    def parse_template(self, tokens_deque):
        current_token = tokens_deque.popleft()
        assert is_start_token(current_token)
        template_name = decode_start_token(current_token)

        if template_name not in self._used_templates:
            raise Exception('Unused template type found: ' + template_name)

        # create template instance and add to template collection
        template = Template(template_name, str(self._template_instance_counter))
        self._template_collection.add_template(template)
        self._template_instance_counter += 1

        # parse slots of template
        while len(tokens_deque) > 0:
            # check if end delimiter of template is reached
            current_token = tokens_deque[0]
            if is_end_token(current_token):
                if decode_end_token(current_token) == template_name:
                    # remove end delimiter token from deque
                    tokens_deque.popleft()
                    break

            # parse next slot
            self.parse_slot(tokens_deque, template)

        return template

    def parse_slot(self, tokens_deque, template):
        current_token = tokens_deque.popleft()
        assert is_start_token(current_token)
        slot_name = decode_start_token(current_token)

        if slot_name not in self._used_slots:
            raise Exception('Unused slot type found: ' + slot_name)

        if slot_name in self._slots_containing_templates:
            slot_filler_template = self.parse_template(tokens_deque)
            template.add_slot_filler(slot_name, slot_filler_template)

            # remove end delimiter of slot
            if len(tokens_deque) > 0:
                current_token = tokens_deque.popleft()
                assert is_end_token(current_token)
                assert decode_end_token(current_token) == slot_name
        else:
            slot_filler_tokens = seek_deque(
                token_deque=tokens_deque,
                termination_token=create_end_token(slot_name),
                include_termination_token=False,
                raise_token_not_found_exception=False
            )

            entity = Entity()
            entity.set_label(slot_name)
            entity.set_tokens(slot_filler_tokens)
            template.add_slot_filler(slot_name, entity)

    def parse_slot_flat(self, tokens_deque):
        current_token = tokens_deque[0]
        assert is_start_token(current_token)
        slot_name = decode_start_token(current_token)

        if slot_name not in self._used_slots:
            raise Exception('unused slot fond: ' + slot_name)

        slot_tokens = seek_deque(tokens_deque, create_end_token(slot_name), include_termination_token=True)

        # remove boundary tokens of slot-filler
        slot_tokens = slot_tokens[1:-1]

        # remove slot boundary tokens
        return slot_name, slot_tokens

    def parse_template_flat(self, tokens_deque):
        current_token = tokens_deque.popleft()
        assert is_start_token(current_token)
        template_name = decode_start_token(current_token)

        if template_name not in self._used_templates:
            raise Exception('Unused template type found: ' + template_name)

        template = Template(template_name, str(self._template_instance_counter))
        self._template_instance_counter += 1

        while len(tokens_deque) > 0:
            # check if end delimiter of template is reached
            current_token = tokens_deque[0]
            if is_end_token(current_token) and decode_end_token(current_token) == template_name:
                tokens_deque.popleft()
                break

            slot_name, slot_filler_tokens = self.parse_slot_flat(tokens_deque)
            entity = Entity()
            entity.set_label(slot_name)
            entity.set_tokens(slot_filler_tokens)
            template.add_slot_filler(slot_name, entity)

        return template

    def parse_sentence_flat(self, tokens: Iterable[str]):
        # remove start of sentence and end of sentence tokens
        tokens = list(tokens)[1:-1]
        if len(tokens) == 0:
            return []

        print(tokens)
        tokens_deque = deque(tokens)
        templates = list()

        while len(tokens_deque) > 0:
            templates.append(self.parse_template_flat(tokens_deque))

        return templates

    def parse_document_sentences(self, sentences: List[List[str]]) -> TemplateCollection:
        template_collection = TemplateCollection()

        for sentence_tokens in sentences:
            templates = self.parse_sentence_flat(sentence_tokens)

            for template in templates:
                template_collection.add_template(template)

        return template_collection


if __name__ == "__main__":
    task_config_dict = import_task_config(sys.argv[1])
    model_name = task_config_dict['model_name']
    model_class = get_model_class(model_name)
    model = model_class(model_name=model_name)

    temp_generation_tokenizer = TemplateGenerationTokenizer(
        tokenizer=model.tokenizer,
        template_names=task_config_dict['slots_ordering_dict'].keys(),
        slot_names=task_config_dict['used_slots'],
        start_of_sentence_token='<s>',
        end_of_sentence_token='</s>',
        filler_sep_token='[FILLER_SEP]',
        control_tokens_start_id=model.tokenizer.vocab_size
    )

    # prepare evaluation
    training_set, test_dataset = import_train_test_data_from_task_config(task_config_dict, temp_generation_tokenizer)
    parser = Parser(task_config_dict)
    slot_filling_evaluation = SlotFillingEvaluation()
    used_slots = set(task_config_dict['used_slots']) - set(task_config_dict['slots_containing_templates'].keys())

    dataset_to_use = test_dataset

    # parse output strings and evaluate
    with open(sys.argv[2]) as fp:
        outputs_dict = json.load(fp)

    for doc_id, output_tokens in outputs_dict.items():
        # remove start of sentence and end of sentence tokens
        output_tokens = output_tokens[1:-1]
        try:
            predicted_template_collection = parser.parse_document_level(output_tokens)
            _, gt_template_collection = dataset_to_use[doc_id]

            template_alignment = TemplateAlignment(
                gt_temp_collection=gt_template_collection,
                predicted_temp_collection=predicted_template_collection,
                used_slots=used_slots
            )

            template_alignment.update_evaluation(slot_filling_evaluation)
            slot_filling_evaluation.update_instance_counts(
                gt_template_collection=gt_template_collection,
                predicted_template_collection=predicted_template_collection
            )

        except Exception as e:
            print(e)

    slot_filling_evaluation.print_out()

    '''
    # doc id -> List[List[sentence tokens]]
    documents_sentences = dict()
    
    # import output sequences
    with open(sys.argv[2]) as fp:
        data = json.load(fp)
    
        for sentence in data:
            doc_id = sentence['document_id']
            doc_sentences_list = documents_sentences.setdefault(doc_id, [])
            doc_sentences_list.append(sentence['output_tokens'])
    
    
    # create template generation tokenizer ################################################
    task_config_dict = import_task_config(sys.argv[1])
    led_tokenizer = LEDTokenizer.from_pretrained(task_config_dict['model_name'])
    
    temp_generation_tokenizer = TemplateGenerationTokenizer(
        led_tokenizer=led_tokenizer,
        template_names=task_config_dict['slots_ordering_dict'].keys(),
        slot_names=task_config_dict['used_slots'],
        start_of_sentence_token='<s>',
        end_of_sentence_token='</s>',
        filler_sep_token='[FILLER_SEP]',
        control_tokens_start_id=led_tokenizer.vocab_size
    )
    
    # parse output sequences and evaluate ###############################################
    _, test_dataset = import_train_test_data_from_task_config(task_config_dict, temp_generation_tokenizer)
    parser = Parser(task_config_dict)
    slot_filling_evaluation = SlotFillingEvaluation()
    used_slots = set(task_config_dict['used_slots']) - set(task_config_dict['slots_containing_templates'].keys())
    print(used_slots)
    
    for doc_id in documents_sentences:
        predicted_template_collection = parser.parse_document_sentences(documents_sentences[doc_id])
        _, gt_template_collection = test_dataset[doc_id]
        template_alignment = TemplateAlignment(
            gt_temp_collection=gt_template_collection,
            predicted_temp_collection=predicted_template_collection,
            used_slots=used_slots
        )
    
        template_alignment.update_evaluation(slot_filling_evaluation)
    
    slot_filling_evaluation.print_out()
    '''
