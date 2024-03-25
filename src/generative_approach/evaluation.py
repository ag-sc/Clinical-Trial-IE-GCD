import cProfile
import sys
import traceback
from argparse import ArgumentParser
from typing import Dict, Optional

from torch_scatter import scatter
from transformers import LEDForConditionalGeneration, LEDConfig, LEDTokenizer

from generative_approach.data_handling import DataElement, extract_data_elements_from_dataset
from generative_approach.decoding import Decoder, EDPredictionProxy
from generative_approach.grammar import Grammar
from generative_approach.lark_decoding import LarkDecoder
from generative_approach.lark_parser import LarkParser
from generative_approach.models.ed_model import EDModel
from generative_approach.models.longformer import Longformer
from generative_approach.parsing import Parser
from generative_approach.template_coding import LinearTemplateEncoder, SequenceDecoder
from template_lib.SlotFillingEvaluation import SlotFillingEvaluation
from template_lib.TemplateAlignment import TemplateAlignment
from template_lib.data_classes.TemplateCollection import TemplateCollection
from template_lib.data_handling import import_train_test_valid_data_from_task_config
from template_lib.max_cardinalities_estimator import gen_max_slot_cardinalities
from template_lib.santo.SantoDataset import SantoDataset, import_trial_ids
from generative_approach.utils import *

def evaluate_slot_filling(
        dataset: SantoDataset,
        model: EDModel,
        temp_generation_tokenizer: TemplateGenerationTokenizer,
        task_config_dict: Dict,
        max_slots_cardinalities: Optional[Dict[str, int]],
        device=torch.device("cpu"),
        ptr_decoding=False,
        ptr_model=False,
        no_eval_decoding=False
):
    model.model.to(device)
    slot_filling_evaluation = SlotFillingEvaluation()
    used_slots = set(task_config_dict['used_slots']) - set(task_config_dict['slots_containing_templates'].keys())

    parser = LarkParser(tokenizer=temp_generation_tokenizer.tokenizer,
                    grammar_file=os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", 'grammar_full.lark'))#Parser(task_config_dict)
    decoder = LarkDecoder(
        edmodel=model,
        tgtokenizer=temp_generation_tokenizer,
        grammar_file=os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", 'grammar_full.lark'),
        ptr_decoding=ptr_decoding,
        ptr_model=ptr_model,
    )

    # decode documents in test set
    # outputs_dict = dict()
    for document, gt_template_collection in dataset:
        # if DEBUG:
        #     print(f'decoding document {document.get_id()}')


        if DEBUG:
            predicted_template_collection = single_run(document=document,
                                                       model=model,
                                                       temp_generation_tokenizer=temp_generation_tokenizer,
                                                       task_config_dict=task_config_dict,
                                                       max_slots_cardinalities=max_slots_cardinalities,
                                                       device=device,
                                                       ptr_decoding=ptr_decoding,
                                                       ptr_model=ptr_model,
                                                       no_eval_decoding=no_eval_decoding,
                                                       parser=parser,
                                                       decoder=decoder)
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
        else:
            try:
                predicted_template_collection = single_run(document=document,
                                                           model=model,
                                                           temp_generation_tokenizer=temp_generation_tokenizer,
                                                           task_config_dict=task_config_dict,
                                                           max_slots_cardinalities=max_slots_cardinalities,
                                                           device=device,
                                                           ptr_decoding=ptr_decoding,
                                                           ptr_model=ptr_model,
                                                           no_eval_decoding=no_eval_decoding,
                                                           parser=parser,
                                                           decoder=decoder)
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
              print("Error in eval:", e)
              print(traceback.format_exc())
           #print(sys.exc_info()[2])

    # slot_filling_evaluation.print_out()
    return slot_filling_evaluation

# def single_run_raw(document,
#                    model: EDModel,
#                    tokenizer: TemplateGenerationTokenizer,
#                    device=torch.device("cpu"),
#                    ptr_model=False):
#     sentences_token_ids = [tokenizer.convert_tokens_to_ids(sentence.get_tokens())
#                            for sentence in document.get_sentences()]
#     start_of_sentence_token_id = None
#     end_of_sentence_token_id = None
#     pad_token_id = None
#     if tokenizer.start_of_sentence_token is not None:
#         start_of_sentence_token_id = tokenizer.convert_tokens_to_ids(
#             tokens=[tokenizer.start_of_sentence_token]
#         )[0]
#     if tokenizer.end_of_sentence_token is not None:
#         end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(
#             tokens=[tokenizer.end_of_sentence_token]
#         )[0]
#     if tokenizer.pad_token is not None:
#         pad_token_id = tokenizer.convert_tokens_to_ids(
#             tokens=[tokenizer.pad_token]
#         )[0]
#
#     sos = [start_of_sentence_token_id] if start_of_sentence_token_id is not None else []
#     eos = [end_of_sentence_token_id] if end_of_sentence_token_id is not None else []
#     input_ids = [sos + sentence_token_ids + eos
#                  for sentence_token_ids in sentences_token_ids]
#
#     input_ids = list(chain(*input_ids))
#     tinput_ids = input_ids
#
#     input_ids_tensor_gen = torch.tensor(tinput_ids, dtype=torch.long, device=device).unsqueeze(0)
#     #tinput_ids = pt_stack_lists([input_ids], pad_token_id).to(device)
#     attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.long, device=device).unsqueeze(0)
#     #global_attention_mask = torch.zeros_like(tinput_ids).to(device)
#     #global_attention_mask[:, 0] = 1
#
#     decoder_input_ids = []
#
#     if start_of_sentence_token_id is not None:
#         decoder_input_ids.append(start_of_sentence_token_id)
#     elif isinstance(model, FlanT5):
#         decoder_input_ids.append(pad_token_id)
#
#     outputs = model(input_ids_tensor_gen,
#                     #decoder_input_ids=input_ids_tensor#pt_stack_lists([decoder_input_ids], pad_token_id).to(device),
#                     attention_mask=attention_mask,
#                     #global_attention_mask=global_attention_mask,
#                     output_hidden_states=True,
#                     output_attentions=True,
#                     return_dict=True)
#     generative_dist = outputs["logits"].view(-1, model.model.config.vocab_size).softmax(dim=1)
#
#     final_dist = generative_dist
#
#     if ptr_model:
#         p_gens = model.p_gen(outputs["decoder_hidden_states"][-1]).squeeze(0)
#
#         generative_dist = generative_dist * p_gens
#
#         cross_attentions = torch.mean(outputs["cross_attentions"][-1], dim=1)
#
#         input_ids_tensor = input_ids_tensor.repeat(1, cross_attentions.shape[1], 1)
#         ptr_logits = scatter(cross_attentions, input_ids_tensor, dim_size=model.model.config.vocab_size,
#                              reduce="max")  # .flatten().softmax(dim=-1)
#         ptr_dist = ptr_logits.view(-1, model.model.config.vocab_size) * (1.0 - p_gens)
#
#         print("p_gens", p_gens.max().item(), p_gens.min().item(), p_gens.mean().item())
#         final_dist = (generative_dist + ptr_dist)


def single_run(document,
               model: EDModel,
               temp_generation_tokenizer: TemplateGenerationTokenizer,
               task_config_dict,
               max_slots_cardinalities,
               device=torch.device("cpu"),
               ptr_decoding=False,
               ptr_model=False,
               no_eval_decoding=False,
               parser: Optional[LarkParser] = None,
               decoder: Optional[LarkDecoder] = None):
    model.model.to(device)
    #grammar = Grammar(task_config_dict['grammar_file'])
    if parser is None:
        parser = LarkParser(tokenizer=temp_generation_tokenizer.tokenizer,
                        grammar_file=os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", 'grammar_full.lark'))#Parser(task_config_dict)
    if decoder is None:
        decoder = LarkDecoder(
            edmodel=model,
            tgtokenizer=temp_generation_tokenizer,
            grammar_file=os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", 'grammar_full.lark'),
            ptr_decoding=ptr_decoding,
            ptr_model=ptr_model,
        )
    #     grammar=grammar,
    #     tokenizer=temp_generation_tokenizer,
    #     prediction_proxy=EDPredictionProxy(model),
    #     template_names=list(task_config_dict['slots_ordering_dict'].keys()),
    #     ptr_decoding=ptr_decoding,
    #     ptr_model=ptr_model
    # )

    output_tokens, output_ids, output_logits = None, None, None

    if no_eval_decoding:
        output_tokens, output_ids, output_logits = decoder.decode_document_greedy(
            document=document,
            masked_vocab=False
        )
    else:
        output_tokens, output_ids, output_logits = decoder.decode_document_greedy(
            document=document,
            # start_symbol='#PUBLICATION_HEAD',
            # max_len=1020,
            # max_slots_cardinalities=max_slots_cardinalities,  # None
            # used_slots=task_config_dict['used_slots'] + list(task_config_dict['slots_ordering_dict'].keys())
        )

    # remove start of sentence and end of sentence tokens
    if output_tokens[0] == temp_generation_tokenizer.start_of_sentence_token or output_tokens[0] == temp_generation_tokenizer.pad_token:
        output_tokens = output_tokens[1:]
    if output_tokens[-1] == temp_generation_tokenizer.end_of_sentence_token:
        output_tokens = output_tokens[:-1]
    #output_tokens = output_tokens[1:-1]
    try:
        return parser.parse("".join(output_tokens))
    except Exception as e:
        print("Error parsing tokens:", output_tokens)
        print("".join(output_tokens))
        print(e)
        return TemplateCollection()

def cli_eval(task_config: str,
             model_path: str,
             tokenizer_path: Optional[str] = None,
             use_validation: bool = False,
             ptr_decoding: Optional[bool] = None,
             ptr_model: Optional[bool] = None,
             no_eval_decoding: Optional[bool] = None,
             ):
    task_config_dict = import_task_config(task_config)

    if ptr_decoding is None:
        ptr_decoding = task_config_dict["ptr_decoding"]
    if ptr_model is None:
        ptr_model = task_config_dict["ptr_model"]
    if no_eval_decoding is None:
        no_eval_decoding = task_config_dict["no_eval_decoding"]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    # device = "cpu" #torch.device("cpu")

    # load pretrained led model and create instantiate tokenizers #########################
    model_name = task_config_dict['model_name']
    model_class = get_model_class(model_name)
    model = model_class(model_path=model_path, model_name=model_name, device=device_str)

    # led_model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    # led_tokenizer = LEDTokenizer.from_pretrained(model_name)
    # led_config = LEDConfig.from_pretrained(model_name)

    # create template generation tokenizer ################################################
    temp_generation_tokenizer: TemplateGenerationTokenizer
    temp_generation_tokenizer = TemplateGenerationTokenizer(
        tokenizer=model.tokenizer,
        json_filename=tokenizer_path,
        template_names=task_config_dict['slots_ordering_dict'].keys(),
        slot_names=task_config_dict['used_slots'],
        start_of_sentence_token=model.tokenizer.bos_token,
        end_of_sentence_token=model.tokenizer.eos_token,
        filler_sep_token='[FILLER_SEP]',
        control_tokens_start_id=model.tokenizer.vocab_size
    )
    # temp_generation_tokenizer.to_json(sys.argv[2])

    # add embeddings for control tokens
    model.model.resize_token_embeddings(
        model.tokenizer.vocab_size + len(temp_generation_tokenizer.control_token_ids)
        - (1 if model.tokenizer.bos_token is not None else 0)
        - (1 if model.tokenizer.eos_token is not None else 0)
        - (1 if model.tokenizer.pad_token is not None else 0))
    # print(led_model.led.shared.num_embeddings)

    start_token = temp_generation_tokenizer.start_of_sentence_token
    if temp_generation_tokenizer.start_of_sentence_token is None:
        if temp_generation_tokenizer.pad_token is not None and isinstance(model, FlanT5):
            start_token = temp_generation_tokenizer.pad_token

    template_encoder = LinearTemplateEncoder(
        top_level_templates=task_config_dict['top_level_templates'],
        slots_ordering=task_config_dict['slots_ordering_dict'],
        templates_ordering=sorted(task_config_dict['slots_ordering_dict'].keys()),
        used_slots=task_config_dict['used_slots'],
        slots_containing_templates=task_config_dict['slots_containing_templates'],
        filler_sep_token=temp_generation_tokenizer.filler_sep_token,
        start_of_document_token=start_token,
        end_of_document_token=temp_generation_tokenizer.end_of_sentence_token
    )

    # load training data ###############################################
    training_dataset, test_dataset, validation_dataset = import_train_test_valid_data_from_task_config(task_config_dict,
                                                                                                       temp_generation_tokenizer)

    max_slots_cardinalities = gen_max_slot_cardinalities(training_dataset)

    slot_filling_evaluation = evaluate_slot_filling(
        dataset=test_dataset if not use_validation else validation_dataset,
        model=model,
        temp_generation_tokenizer=temp_generation_tokenizer,
        task_config_dict=task_config_dict,
        max_slots_cardinalities=max_slots_cardinalities,
        device=device,
        ptr_decoding=ptr_decoding,
        ptr_model=ptr_model,
        no_eval_decoding=no_eval_decoding
    )

    slot_filling_evaluation.print_out()
    print(slot_filling_evaluation.compute_micro_stats().f1())
    return slot_filling_evaluation

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--task_config', required=True)
    argparser.add_argument('--model', required=True)
    argparser.add_argument('--tokenizer', required=False)
    argparser.add_argument('--profile', action='store_true')
    argparser.add_argument('--nodecoding', action='store_true')
    argparser.add_argument('--validation', action='store_true')
    arguments = argparser.parse_args()

    profiler = cProfile.Profile()
    if arguments.profile:
        profiler.enable()

    cli_eval(task_config=arguments.task_config, model_path=arguments.model, tokenizer_path=arguments.tokenizer, use_validation=arguments.validation, no_eval_decoding=arguments.nodecoding if arguments.nodecoding else None)

    if arguments.profile:
        profiler.disable()
        profiler.dump_stats('gen-eval.prof')

    # task_config_dict = import_task_config(sys.argv[1])
    # device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device_str)
    # # device = "cpu" #torch.device("cpu")
    #
    # model_name = task_config_dict['model_name']
    # model_class = get_model_class(model_name)
    # model = model_class(model_path=sys.argv[3], model_name=model_name, device=device_str)
    #
    # temp_generation_tokenizer = TemplateGenerationTokenizer(
    #     tokenizer=model.tokenizer,
    #     json_filename=sys.argv[2]
    # )
    #
    # # import test data #############################################################################################
    # print('importing test data ...')
    # dataset_path = task_config_dict['dataset_path']
    # rel_test_ids_filename = task_config_dict['rel_test_ids_filename']
    #
    # test_ids_filename = os.path.join(dataset_path, rel_test_ids_filename)
    # test_ids = import_trial_ids(test_ids_filename)
    # dataset = SantoDataset(dataset_path, test_ids, task_config_dict['disease_prefix'], 'admin', model.tokenizer)
    #
    # start_token = temp_generation_tokenizer.start_of_sentence_token
    # if temp_generation_tokenizer.start_of_sentence_token is None:
    #     if temp_generation_tokenizer.pad_token is not None and isinstance(model, FlanT5):
    #         start_token = temp_generation_tokenizer.pad_token
    #
    # # create test data elements
    # template_encoder = LinearTemplateEncoder(
    #     top_level_templates=task_config_dict['top_level_templates'],
    #     slots_ordering=task_config_dict['slots_ordering_dict'],
    #     used_slots=task_config_dict['used_slots'],
    #     slots_containing_templates=task_config_dict['slots_containing_templates'],
    #     filler_sep_token=temp_generation_tokenizer.filler_sep_token,
    #     start_of_document_token=start_token,
    #     end_of_document_token=temp_generation_tokenizer.end_of_sentence_token
    # )
    #
    # test_data_elements = list()
    #
    # for document, template_collection in dataset:
    #     data_element = DataElement(
    #         document=document,
    #         template_collection=template_collection,
    #         tokenizer=temp_generation_tokenizer,
    #         template_encoder=template_encoder,
    #         max_input_seq_len=model.max_encoder_position_embeddings,
    #         max_output_seq_len=model.max_decoder_position_embeddings
    #     )
    #     test_data_elements.append(data_element)
    #
    # print('number of test documents: ' + str(len(test_data_elements)))
    #
    # # prediction ##################################################################
    # data_element = test_data_elements[0]
    # print('doc id: ' + data_element.document.get_id())
    # print('input tokens:')
    # print(data_element.input_tokens)
    # print('-----------------------------')
    # print(data_element.output_tokens)
    # print('-----------------------------')
    #
    # predicted_token_ids = temp_generation_tokenizer.convert_tokens_to_ids(
    #     [temp_generation_tokenizer.start_of_sentence_token])
    # predicted_tokens = [temp_generation_tokenizer.start_of_sentence_token]
    # input_ids = torch.tensor([data_element.input_token_ids])
    #
    # global_attention_mask = torch.zeros_like(input_ids)
    # global_attention_mask[:, 0] = 1
    #
    # subsequence_manager = DocumentSubsequenceManager(
    #     sentences=[temp_generation_tokenizer.convert_tokens_to_ids(sentence.get_tokens()) for sentence in
    #                data_element.document.get_sentences()]
    # )
    #
    # seq_decoder = SequenceDecoder(
    #     tokenizer=temp_generation_tokenizer,
    #     subsequence_manager=subsequence_manager,
    #     slots_of_templates=task_config_dict['slots_ordering_dict'],
    #     slots_containing_templates=task_config_dict['slots_containing_templates'],
    #     top_level_templates=task_config_dict['top_level_templates'],
    #     max_slot_fillers=15
    # )
    #
    # for i in range(800):
    #     with torch.no_grad():
    #         outputs = model.model(
    #             input_ids=input_ids,
    #             decoder_input_ids=torch.tensor([predicted_token_ids]),
    #             global_attention_mask=global_attention_mask,
    #             return_dict=True
    #         )
    #
    #     # create token mask
    #     '''
    #     next_possible_token_ids = seq_decoder.get_next_possible_token_ids()
    #     next_possible_tokens = temp_generation_tokenizer.convert_ids_to_tokens(list(next_possible_token_ids))
    #     print('next possible tokens: -----')
    #     print(next_possible_tokens)
    #     print('--------')
    #     mask = torch.ones(temp_generation_tokenizer.get_vocab_size()) * (-1e9)
    #     for token_id in next_possible_token_ids:
    #         mask[token_id] = 0
    #     '''
    #
    #     logits = outputs.token_logits[0, i, :]  # + mask
    #
    #     predicted_token_id = logits.argmax(-1).item()
    #     predicted_token = temp_generation_tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
    #     print('predicted token: ' + predicted_token)
    #     print('-----')
    #     predicted_token_ids.append(predicted_token_id)
    #     # seq_decoder.append_token_id(predicted_token_id)
    #     predicted_tokens.append(predicted_token)
    #
    #     if predicted_token == temp_generation_tokenizer.end_of_sentence_token:
    #         break
    #
    # print('predicted token sequence: -----------------')
    # print(predicted_tokens)
