from typing import Dict, List

from generative_approach.template_coding import LinearTemplateEncoder
from template_lib.data_classes.TemplateCollection import *
from template_lib.santo.SantoDataset import SantoDataset
from generative_approach.utils import *


def estimate_dominant_sentence_of_template(
        template: Template,
        document: Document
) -> int:
    sentence_distribution = {sentence.get_index(): 0 for sentence in document.get_sentences()}

    for entity in template.get_assigned_entities():
        if entity.get_sentence_index() is not None:
            sentence_distribution[entity.get_sentence_index()] += 1

    return sorted(sentence_distribution.items(), key=lambda x: x[1], reverse=True)[0][0]


def assign_templates_to_sentence(
        template_collection: TemplateCollection,
        document: Document
) -> Dict[int, List[Template]]:
    templates_of_sentence_dict = {sentence.get_index(): list() for sentence in document.get_sentences()}

    # get templates which have at least one entity as slot filler
    templates = [template for template in template_collection if len(list(template.get_assigned_entities())) > 0]

    # assign each template to exactly one sentence by estimating dominant sentence of template
    for template in templates:
        sentence_index = estimate_dominant_sentence_of_template(template, document)
        templates_of_sentence_dict[sentence_index].append(template)

    return templates_of_sentence_dict


class DataElement:
    def __init__(
            self,
            document: Document,
            template_collection: TemplateCollection,
            tokenizer: TemplateGenerationTokenizer,
            template_encoder,
            max_input_seq_len: int,
            max_output_seq_len: int,
            encode_flat: bool = False,
            sentence: Sentence = None
    ):
        self.document = document
        self.template_collection = template_collection
        self.sentence = sentence

        # create input token ids ####################################################
        input_tokens = list()
        sos = [tokenizer.start_of_sentence_token] if tokenizer.start_of_sentence_token is not None else []
        eos = [tokenizer.end_of_sentence_token] if tokenizer.end_of_sentence_token is not None else []

        if sentence is None:
            for sent in document:
                input_tokens.extend(sos + sent.get_tokens() + eos)
        else:
            input_tokens.extend(sos + sentence.get_tokens() + eos)

        if len(input_tokens) > max_input_seq_len:
            print('WARNING: number of input tokens exceeds max input length')
            input_tokens = input_tokens[:max_input_seq_len]

        self.input_tokens = input_tokens
        self.input_token_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        # create output token ids ######################################################
        output_tokens = template_encoder.encode_template_collection(
            template_collection=template_collection,
            sentence_index=sentence.get_index() if sentence is not None else None,
            encode_flat=encode_flat
        )

        if len(output_tokens) > max_output_seq_len:
            print('WARNING: number of output tokens exceeds max output length')
            output_tokens = output_tokens[:max_output_seq_len]

        self.output_tokens = output_tokens
        self.output_token_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        #print(self.input_token_ids)
        #print('---')
        #print("Final:", document.get_id(), self.output_token_ids)
        #print("Final2:", document.get_id(), self.output_tokens)
        #print('===')


def get_data_element_by_doc_id(data_elements: List[DataElement], doc_id: str):
    for data_element in data_elements:
        if data_element.document.get_id() == doc_id:
            return data_element

    raise Exception('data element not found')


def extract_data_elements_from_dataset(
        dataset: SantoDataset,
        tokenizer: TemplateGenerationTokenizer,
        template_encoder: LinearTemplateEncoder,
        max_input_seq_len: int,
        max_output_seq_len: int,
        encode_flat: bool = False,
        per_sentence: bool = False
):
    data_elements = list()

    for document, template_collection in dataset:
        if per_sentence:
            template_to_sentence_assignment = assign_templates_to_sentence(template_collection, document)

            for sentence in document:
                temp_collection_for_sentence = TemplateCollection(template_to_sentence_assignment[sentence.get_index()])

                data_elements.append(
                    DataElement(document=document,
                                template_collection=temp_collection_for_sentence,
                                tokenizer=tokenizer,
                                template_encoder=template_encoder,
                                max_input_seq_len=max_input_seq_len,
                                max_output_seq_len=max_output_seq_len,
                                encode_flat=True,
                                sentence=sentence
                                )
                )
        else:
            data_elements.append(
                DataElement(document=document,
                            template_collection=template_collection,
                            tokenizer=tokenizer,
                            template_encoder=template_encoder,
                            max_input_seq_len=max_input_seq_len,
                            max_output_seq_len=max_output_seq_len,
                            encode_flat=encode_flat
                            )
            )
    return data_elements
