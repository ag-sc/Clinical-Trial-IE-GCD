import math
import sys
from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch
from torch_scatter import scatter

from generative_approach.data_handling import *
from generative_approach.grammar import Grammar
from template_lib.data_handling import import_train_test_data_from_task_config




def create_token_mask(logits: torch.Tensor, token_ids: Iterable[int]):
    mask = np.ones(logits.shape)
    # mask = torch.ones_like(logits)

    # for token_id in token_ids:
    #    mask[token_id] = 0
    mask[list(token_ids)] = 0

    mask = mask * (-1e9)

    return torch.from_numpy(mask).to(logits.device)


class SlotCardinalityTracker:
    def __init__(self, template_names):
        self._template_names = template_names
        self._cardinality_dicts_stack = list()

    def reset(self):
        self._cardinality_dicts_stack = list()

    def get_stack_size(self):
        return len(self._cardinality_dicts_stack)

    def add_token(self, token):
        if is_start_token(token):
            decoded_token = decode_start_token(token)

            if decoded_token in self._template_names:
                self._cardinality_dicts_stack.append(dict())
            else:
                top_dict = self._cardinality_dicts_stack[-1]
                cardinality = top_dict.setdefault(decoded_token, 0)
                top_dict[decoded_token] = cardinality + 1
        elif is_end_token(token):
            decoded_token = decode_end_token(token)

            if decoded_token in self._template_names:
                self._cardinality_dicts_stack.pop(-1)

    def get_saturated_slots(
            self,
            max_slots_cardinalities: Dict[str, int],
            functional_only: bool = True
    ) -> List[str]:
        if len(self._cardinality_dicts_stack) == 0:
            return list()

        cardinality_dict = self._cardinality_dicts_stack[-1]
        saturated_slots = list()

        for slot_name in cardinality_dict:
            if slot_name not in max_slots_cardinalities:
                continue
            if functional_only and max_slots_cardinalities[slot_name] > 1:
                continue

            if cardinality_dict[slot_name] >= max_slots_cardinalities[slot_name]:
                saturated_slots.append(slot_name)

        return saturated_slots

    def print_stack(self):
        print('-------')
        for i, cardinality_dict in enumerate(self._cardinality_dicts_stack):
            for key, value in cardinality_dict.items():
                print('\t' * (i) + f'{key}: {value}')


class DecodingState:
    def __init__(self, grammar: Grammar, template_names: List[str]):
        self.grammar = grammar
        self.slot_cardinality_tracker = SlotCardinalityTracker(template_names)
        self.grammar_rules_stack = list()
        self.token_ids = list()
        self.logits = list()
        self.logits_sum = 0.0
        self.free_text_tokens = list()
        self.past_key_values = None

    def print_rules_stack(self):
        if len(self.grammar_rules_stack) == 0:
            print('empty')
        else:
            for i, symbols_deque in enumerate(self.grammar_rules_stack):
                print('\t' * (i + 1) + str(list(symbols_deque)))

    def add_token_prediction(
            self,
            token_id: int,
            token: str,
            logits: torch.Tensor,
            ptr_token_id: Optional[int] = None,
            ptr_token: Optional[str] = None,
            ptr_logits: Optional[torch.Tensor] = None,
            past_key_values=None
    ):
        current_symbols_deque = self.grammar_rules_stack[-1]
        current_symbol = current_symbols_deque[0]
        pointer_termination_symbol = current_symbols_deque[1] if len(current_symbols_deque) > 1 else None

        # if currently in free text and not predicted special slot end token yet, use pointer network for next token
        if (ptr_token_id is not None and ptr_token is not None and ptr_logits is not None
                and current_symbol == self.grammar.pointer_symbol and token != pointer_termination_symbol):
            token_id = ptr_token_id
            token = ptr_token
            logits = ptr_logits

        logit = logits.max(-1).values.item()

        self.token_ids.append(token_id)
        self.logits.append(logit)
        self.logits_sum += logit
        self.past_key_values = past_key_values

        self.slot_cardinality_tracker.add_token(token)

        # update grammar rules stack
        if current_symbol == self.grammar.pointer_symbol:
            self.free_text_tokens.append(token)

            if token == pointer_termination_symbol:
                # remove pointer symbol
                self.remove_grammar_symbol_from_stack_top()

                # remove pointer termination symbol
                self.remove_grammar_symbol_from_stack_top()

                # clear free text prediction list
                self.free_text_tokens.clear()
        elif current_symbol in self.grammar.get_nonterminal_symbols():
            next_grammar_rule = self.grammar.get_rule_option_by_start_symbol(current_symbol, token)

            # remove current nonterminal
            self.remove_grammar_symbol_from_stack_top()

            self.grammar_rules_stack.append(deque(next_grammar_rule.get_symbols()))

            # remove first symbol of new grammar rule
            self.remove_grammar_symbol_from_stack_top()
        else:
            self.remove_grammar_symbol_from_stack_top()

        if not logits.requires_grad:
            logits.requires_grad = True
        logits = logits.softmax(dim=-1)
        return logits

    def remove_empty_deques_from_stack(self):
        while len(self.grammar_rules_stack) > 0:
            top_deque = self.grammar_rules_stack[-1]

            if len(top_deque) == 0:
                del self.grammar_rules_stack[-1]
                continue
            else:
                return

    def get_current_grammar_symbol(self):
        if len(self.grammar_rules_stack) == 0:
            raise IndexError('No grammar symbols left')

        return self.grammar_rules_stack[-1][0]

    def remove_grammar_symbol_from_stack_top(self):
        self.remove_empty_deques_from_stack()

        if len(self.grammar_rules_stack) == 0:
            raise IndexError('No grammar symbols left')

        self.grammar_rules_stack[-1].popleft()
        self.remove_empty_deques_from_stack()

    def get_next_possible_tokens(
            self,
            subseq_manager: DocumentSubsequenceManager = None,
            max_slots_cardinalities: Dict[str, int] = None,
            constrain_only_functional_slots: bool = False,
            used_slots: List[str] = None,
    ) -> Tuple[Set[str], bool]:
        # nothing to decode case
        if len(self.grammar_rules_stack) == 0:
            return set(), False

        current_symbols_deque = self.grammar_rules_stack[-1]
        current_symbol = current_symbols_deque[0]

        is_free_text = False

        if current_symbol == self.grammar.pointer_symbol:
            # there has to be another symbol right next to pointer symbol
            assert len(current_symbols_deque) > 1
            next_possible_tokens = {current_symbols_deque[1]}

            # free text tokens
            next_possible_tokens.update(subseq_manager.get_next_possible_tokens(prefix=self.free_text_tokens))
            is_free_text = True
        elif current_symbol in self.grammar.get_nonterminal_symbols():
            next_possible_tokens = set(self.grammar.get_rule_start_symbols(current_symbol))
        else:
            next_possible_tokens = {current_symbols_deque[0]}

        # incorporate max slots cardinalities
        if max_slots_cardinalities is not None:
            saturated_slots = self.slot_cardinality_tracker.get_saturated_slots(
                max_slots_cardinalities=max_slots_cardinalities,
                functional_only=constrain_only_functional_slots
            )
            next_possible_tokens -= {create_start_token(slot_name) for slot_name in saturated_slots}

        if used_slots is not None and current_symbol != self.grammar.pointer_symbol:
            valid_names = {create_start_token(slot_name) for slot_name in used_slots}.union(
                {create_end_token(slot_name) for slot_name in used_slots})
            # if len(next_possible_tokens - valid_names) > 0:
            #     print("!!!", next_possible_tokens - valid_names)
            next_possible_tokens.intersection_update(valid_names)

        return next_possible_tokens, is_free_text


# class DecodingStateSuccessor:
#     def __init__(
#             self,
#             decoding_state: DecodingState,
#             token_id: int = None,
#             token: str = None,
#             token_logit: float = None,
#             start_pos_logit: float = None,
#             end_pos_logit: float = None,
#             past_key_values=None
#     ):
#         self.decoding_state = decoding_state
#         self.token_id = token_id
#         self.token = token
#         self.token_logit = token_logit
#         self.start_pos_logit = start_pos_logit
#         self.end_pos_logit = end_pos_logit
#         self.past_key_values = past_key_values
#
#     def compute_token_logits_score(self) -> float:
#         unnormalized_score = self.decoding_state.logits_sum
#         normalization_factor = len(self.decoding_state.logits)
#
#         # if decoding has not finished, incorporate most recent prediction
#         if self.token is not None:
#             unnormalized_score += self.token_logit
#             normalization_factor += 1
#
#         return unnormalized_score / normalization_factor
#
#     def compute_pointer_network_logits_score(self) -> float:
#         pass
#
#     def compute_score(self, include_pointer_network=False) -> float:
#         score = self.compute_token_logits_score()
#         if include_pointer_network:
#             score += self.compute_pointer_network_logits_score()
#         return score
#
#     def next_decoiding_state(self) -> DecodingState:
#         if self.token is not None:
#             self.decoding_state.add_token_prediction(
#                 token_id=self.token_id,
#                 token=self.token,
#                 logit=self.token_logit,
#                 past_key_values=self.past_key_values
#             )
#         return self.decoding_state


class EncoderDecoderPrediction:
    def __init__(self):
        self.token_logits: torch.Tensor = None
        self.encoder_last_hidden_state: torch.Tensor = None
        self.decoder_last_hidden_state: torch.Tensor = None
        self.start_pos_logits: torch.Tensor = None
        self.end_pos_logits: torch.Tensor = None
        self.past_key_values = None
        self.last_cross_attention = None


class EDPredictionProxy:
    def __init__(self, led_model):
        self._model = led_model

    def predict(
            self,
            input_ids: List[int],
            output_ids: List[int],
            encoder_last_hidden_state: torch.Tensor = None,
            past_key_values=None
    ) -> EncoderDecoderPrediction:
        torch.cuda.empty_cache()
        assert len(input_ids) > 0 and len(output_ids) > 0
        device = self._model.device

        # prepare inputs
        input_ids_tensor = torch.tensor([input_ids])
        if past_key_values is not None:
            output_ids = [output_ids[-1]]

        pos = len(output_ids) - 1
        global_attention_mask = torch.zeros_like(input_ids_tensor)
        global_attention_mask[:, 0] = 1#TODO: this might be bad, only first token marked??

        if encoder_last_hidden_state is None:
            input_ids = input_ids_tensor.to(device)
            encoder_outputs = None
        else:
            input_ids = None
            encoder_outputs = (encoder_last_hidden_state.to(device),)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                decoder_input_ids=torch.tensor([output_ids]).to(device),
                global_attention_mask=global_attention_mask.to(device),
                encoder_outputs=encoder_outputs,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=True
            )

        encoder_decoder_prediction = EncoderDecoderPrediction()
        encoder_decoder_prediction.token_logits = outputs.logits[0, pos, :]
        encoder_decoder_prediction.encoder_last_hidden_state = outputs.encoder_last_hidden_state
        encoder_decoder_prediction.decoder_last_hidden_state = outputs.decoder_hidden_states[-1]
        encoder_decoder_prediction.past_key_values = outputs.past_key_values
        if outputs.cross_attentions is not None:
            encoder_decoder_prediction.last_cross_attention = outputs.cross_attentions[-1][:, :, pos:pos+1, :]
        else:
            print("Warning: No cross attentions found!")

        return encoder_decoder_prediction


class Decoder:
    def __init__(
            self,
            grammar: Grammar,
            tokenizer: TemplateGenerationTokenizer,
            prediction_proxy,
            template_names: List[str],
            ptr_decoding=False,
            ptr_model=False
    ):
        if ptr_model and ptr_decoding:
            print("Error: Pointer model and pointer decoding are both enabled!", flush=True)
        assert not (ptr_model and ptr_decoding)
        self._grammar = grammar
        self._tokenizer = tokenizer
        self._prediction_proxy = prediction_proxy
        self._template_names = template_names
        self.ptr_decoding = ptr_decoding
        self.ptr_model = ptr_model

        self._input_ids: List[int] = None
        self._encoder_last_hidden_state: torch.Tensor = None

        self._start_of_sentence_token_id = None
        self._end_of_sentence_token_id = None
        self._pad_token_id = None
        if tokenizer.start_of_sentence_token is not None:
            self._start_of_sentence_token_id = tokenizer.convert_tokens_to_ids(
                tokens=[tokenizer.start_of_sentence_token]
            )[0]
        if tokenizer.end_of_sentence_token is not None:
            self._end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(
                tokens=[tokenizer.end_of_sentence_token]
            )[0]
        if tokenizer.pad_token is not None:
            self._pad_token_id = tokenizer.convert_tokens_to_ids(
                tokens=[tokenizer.pad_token]
            )[0]

    def decode_document_greedy(
            self,
            document: Document,
            start_symbol: str,
            max_len=1020,
            max_slots_cardinalities: Dict[str, int] = None,
            constrain_only_functional_slots=False,
            used_slots: List[str] = None,
    ):
        device = self._prediction_proxy._model.device
        if used_slots is None:
            used_slots = []
        # create list of token ids for each sentence in document
        sentences_token_ids = [self._tokenizer.convert_tokens_to_ids(sentence.get_tokens())
                               for sentence in document.get_sentences()]
        sos = [self._start_of_sentence_token_id] if self._start_of_sentence_token_id is not None else []
        eos = [self._end_of_sentence_token_id] if self._end_of_sentence_token_id is not None else []
        input_ids = [sos + sentence_token_ids + eos
                     for sentence_token_ids in sentences_token_ids]

        input_ids = list(chain(*input_ids))

        # save last encode hidden state for better decoding speed
        encoder_last_hidden_state = None

        # create subseq manager
        subseq_manager = DocumentSubsequenceManager([sentence.get_tokens() for sentence in document.get_sentences()])

        # initial decoding sate
        decoding_state = DecodingState(grammar=self._grammar, template_names=self._template_names)
        if self._start_of_sentence_token_id is not None:
            decoding_state.token_ids.append(self._start_of_sentence_token_id)
        elif isinstance(self._prediction_proxy._model, FlanT5):
            decoding_state.token_ids.append(self._pad_token_id)
        decoding_state.grammar_rules_stack.append(deque([start_symbol]))

        sequence_logits = []

        # predict tokens
        while len(decoding_state.token_ids) < max_len and len(decoding_state.grammar_rules_stack) > 0:
            if DEBUG:
                print('--- grammar rules stack ---')
                decoding_state.print_rules_stack()
                print()

            # predict next token
            encoder_decoder_prediction = self._prediction_proxy.predict(
                input_ids=input_ids,
                output_ids=decoding_state.token_ids,
                encoder_last_hidden_state=encoder_last_hidden_state,
                past_key_values=decoding_state.past_key_values
            )
            encoder_last_hidden_state = encoder_decoder_prediction.encoder_last_hidden_state

            if DEBUG:
                print('--- next possible tokens ---')
            next_possible_tokens, is_free_text = decoding_state.get_next_possible_tokens(
                subseq_manager=subseq_manager,
                max_slots_cardinalities=max_slots_cardinalities,
                constrain_only_functional_slots=constrain_only_functional_slots,
                used_slots=used_slots
            )

            next_possible_token_ids = self._tokenizer.convert_tokens_to_ids(list(next_possible_tokens))
            if DEBUG:
                print(next_possible_tokens)
                print()

            # mask invalid tokens
            mask = create_token_mask(
                logits=encoder_decoder_prediction.token_logits,
                token_ids=next_possible_token_ids
            )

            generative_dist = encoder_decoder_prediction.token_logits.softmax(dim=-1)

            final_dist = generative_dist

            if self.ptr_model:
                p_gens = self._prediction_proxy._model.p_gen(
                    encoder_decoder_prediction.decoder_last_hidden_state).squeeze(0)

                generative_dist = generative_dist * p_gens

                cross_attentions = torch.mean(encoder_decoder_prediction.last_cross_attention, dim=1)
                tinput_ids = input_ids

                input_ids_tensor = torch.tensor(tinput_ids, dtype=torch.long, device=device)
                input_ids_tensor = input_ids_tensor.repeat(1, cross_attentions.shape[1], 1)
                ptr_logits = scatter(cross_attentions, input_ids_tensor, dim_size=generative_dist.shape[1],
                                     reduce="sum")  # .flatten().softmax(dim=-1)
                ptr_dist = ptr_logits.view(-1, generative_dist.shape[1]) * (1.0 - p_gens)
                # self._prediction_proxy._model.config.vocab_size is without special tokens

                print("p_gens", p_gens.max().item(), p_gens.min().item(), p_gens.mean().item())
                final_dist = (generative_dist + ptr_dist)

            # times softmax not working because in "all 0" case arbitrary token like pad can be generated
            logits = final_dist + mask#* mask.softmax(dim=-1)  # + mask

            # estimate predicted token by argmax
            token_id = logits.argmax(-1).item()
            [token] = self._tokenizer.convert_ids_to_tokens([token_id])

            # copy words if is_free_text
            ptr_token_id = None
            ptr_token = None
            # ptr_logit = None
            ptr_logits = None
            if self.ptr_decoding and is_free_text:
                attention_dist = torch.mean(encoder_decoder_prediction.last_cross_attention, dim=1)
                tinput_ids = input_ids
                input_ids_tensor = torch.tensor(tinput_ids, dtype=torch.long, device=attention_dist.device)
                try:
                    ptr_logits = (scatter(attention_dist, input_ids_tensor, dim_size=len(logits),
                                          reduce="sum").flatten() + mask).softmax(dim=-1)
                except RuntimeError:
                    tr = self._prediction_proxy.predict(
                        input_ids=input_ids,
                        output_ids=decoding_state.token_ids,
                        encoder_last_hidden_state=encoder_last_hidden_state,
                        past_key_values=decoding_state.past_key_values
                    )
                    pass

                ptr_token_id = ptr_logits.argmax(-1).item()  # tinput_ids[attention_dist.argmax(-1).item()]
                [ptr_token] = self._tokenizer.convert_ids_to_tokens([ptr_token_id])

            # add token prediction
            if DEBUG:
                print(logits.max(-1).values.item())
            chosen_logits = decoding_state.add_token_prediction(
                token_id=token_id,
                token=token,
                logits=logits,  # .softmax(dim=-1)#.max(-1).values.item(),
                ptr_token_id=ptr_token_id,
                ptr_token=ptr_token,
                ptr_logits=ptr_logits,
                past_key_values=encoder_decoder_prediction.past_key_values
            )
            sequence_logits.append(chosen_logits)
            if DEBUG:
                print('-------------------------')

        if DEBUG:
            print("Final:", document.get_id(), decoding_state.token_ids + eos)
            print("Final2:", document.get_id(), self._tokenizer.convert_ids_to_tokens(decoding_state.token_ids + eos))
        # convert predicted token ids to tokens and return
        return self._tokenizer.convert_ids_to_tokens(
            decoding_state.token_ids + eos), decoding_state.token_ids + eos, sequence_logits

    def decode_document_greedy_raw(
            self,
            document: Document,
            max_len=1020
    ):
        device = self._prediction_proxy._model.device
        # create list of token ids for each sentence in document
        sentences_token_ids = [self._tokenizer.convert_tokens_to_ids(sentence.get_tokens())
                               for sentence in document.get_sentences()]
        sos = [self._start_of_sentence_token_id] if self._start_of_sentence_token_id is not None else []
        eos = [self._end_of_sentence_token_id] if self._end_of_sentence_token_id is not None else []
        input_ids = [sos + sentence_token_ids + eos
                     for sentence_token_ids in sentences_token_ids]

        input_ids = list(chain(*input_ids))

        # save last encode hidden state for better decoding speed
        encoder_last_hidden_state = None

        # initial decoding sate
        token_ids = list()
        if self._start_of_sentence_token_id is not None:
            token_ids.append(self._start_of_sentence_token_id)
        elif isinstance(self._prediction_proxy._model, FlanT5):
            token_ids.append(self._pad_token_id)

        past_key_values = None

        sequence_logits = []

        # predict tokens
        while (len(token_ids) < max_len and (len(token_ids) == 0 or token_ids[-1] != self._end_of_sentence_token_id)):
            # predict next token
            encoder_decoder_prediction = self._prediction_proxy.predict(
                input_ids=input_ids,
                output_ids=token_ids,
                encoder_last_hidden_state=encoder_last_hidden_state,
                past_key_values=past_key_values
            )
            encoder_last_hidden_state = encoder_decoder_prediction.encoder_last_hidden_state

            generative_dist = encoder_decoder_prediction.token_logits.softmax(dim=-1)

            final_dist = generative_dist

            if self.ptr_model:
                p_gens = self._prediction_proxy._model.p_gen(
                    encoder_decoder_prediction.decoder_last_hidden_state).squeeze(0)

                generative_dist = generative_dist * p_gens

                cross_attentions = torch.mean(encoder_decoder_prediction.last_cross_attention, dim=1)
                tinput_ids = input_ids

                input_ids_tensor = torch.tensor(tinput_ids, dtype=torch.long, device=device)
                input_ids_tensor = input_ids_tensor.repeat(1, cross_attentions.shape[1], 1)
                ptr_logits = scatter(cross_attentions, input_ids_tensor, dim_size=generative_dist.shape[1],
                                     reduce="sum")  # .flatten().softmax(dim=-1)
                ptr_dist = ptr_logits.view(-1, generative_dist.shape[1]) * (1.0 - p_gens)
                # self._prediction_proxy._model.config.vocab_size is without special tokens

                print("p_gens", p_gens.max().item(), p_gens.min().item(), p_gens.mean().item())
                final_dist = (generative_dist + ptr_dist)

            logits = final_dist
            if not logits.requires_grad:
                logits.requires_grad = True

            # estimate predicted token by argmax
            token_id = logits.argmax(-1).item()
            [token] = self._tokenizer.convert_ids_to_tokens([token_id])

            token_ids.append(token_id)
            past_key_values = encoder_decoder_prediction.past_key_values
            sequence_logits.append(logits)

        # convert predicted token ids to tokens and return
        return self._tokenizer.convert_ids_to_tokens(token_ids + eos), token_ids + eos, sequence_logits

    # def decode_document(
    #         self,
    #         document: Document,
    #         start_symbol: str,
    #         beam_size: int,
    #         max_len=1020,
    #         max_slots_cardinalities: Dict[str, int] = None,
    #         constrain_only_functional_slots=False
    # ):
    #     # clear caching values of prev document
    #     self._encoder_last_hidden_state = None
    #
    #     # set constraints of decoding process
    #     self._max_slots_cardinalities = max_slots_cardinalities
    #     self._constrain_only_functional_slots = constrain_only_functional_slots


# arg1: task config file name
# arg2: special tokens json file name
# arg3: directory of trained model
# arg4: grammar file
# arg5: json filename for decoded strings
# arg6: max cardinalities json file
if __name__ == '__main__':
    task_config_dict = import_task_config(sys.argv[1])
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    # device = "cpu" #torch.device("cpu")

    model_name = task_config_dict['model_name']
    model_class = get_model_class(model_name)
    model = model_class(model_path=sys.argv[3], model_name=model_name, device=device_str)

    temp_generation_tokenizer = TemplateGenerationTokenizer(
        tokenizer=model.tokenizer,
        json_filename=sys.argv[2]
    )
    identity_tokenizer = IdentityTokenizer()

    grammar = Grammar(sys.argv[4])
    if DEBUG:
        grammar.print_out()

    decoder = Decoder(
        grammar=grammar,
        tokenizer=temp_generation_tokenizer,
        prediction_proxy=EDPredictionProxy(model),
        template_names=list(task_config_dict['slots_ordering_dict'].keys())
    )

    training_set, test_dataset = import_train_test_data_from_task_config(
        task_config_dict,
        temp_generation_tokenizer  # temp_generation_tokenizer ################### modify
    )

    dataset_to_use = test_dataset

    max_slots_cardinalities = None
    if os.path.isfile(sys.argv[6]):
        with open(sys.argv[6]) as fp:
            max_slots_cardinalities = json.load(fp)

    # decode documents in test set
    result_dict = dict()
    for document, _ in dataset_to_use:  # TODO!
        if DEBUG:
            print(f'decoding document {document.get_id()}')
        output_str, output_ids, output_logits = decoder.decode_document_greedy(
            document=document,
            start_symbol='#PUBLICATION_HEAD',
            max_len=1020,
            max_slots_cardinalities=max_slots_cardinalities  # None
        )
        result_dict[document.get_id()] = output_str
        if DEBUG:
            print('======================================')

    # save output strings to file
    fp = open(sys.argv[5], 'w')
    json.dump(result_dict, fp)
    fp.close()
