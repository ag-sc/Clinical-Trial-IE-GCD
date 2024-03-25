import os
import sys
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Any, Iterable

import lark
import numpy as np
import torch
from torch_scatter import scatter

from generative_approach.models.ed_model import EDModel
from generative_approach.models.flan_t5 import FlanT5
from generative_approach.utils import TemplateGenerationTokenizer


@dataclass
class EncoderDecoderPrediction:
    token_logits: Optional[torch.Tensor] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    decoder_last_hidden_state: Optional[torch.Tensor] = None
    start_pos_logits: Optional[torch.Tensor] = None
    end_pos_logits: Optional[torch.Tensor] = None
    past_key_values: Any = None
    last_cross_attention: Any = None


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
        global_attention_mask = torch.ones_like(input_ids_tensor)
        # global_attention_mask[:, 0] = 1  # TODO: this might be bad, only first token marked??

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

        encoder_decoder_prediction = EncoderDecoderPrediction(
            token_logits=outputs.logits[0, pos, :],
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            decoder_last_hidden_state=outputs.decoder_hidden_states[-1][:, pos:pos + 1, :],
            past_key_values=outputs.past_key_values,
            last_cross_attention=outputs.cross_attentions[-1][:, :, pos:pos + 1, :] if outputs.cross_attentions is not None else None
        )
        if outputs.cross_attentions is None:
            print("Warning: No cross attentions found!")

        return encoder_decoder_prediction


class LarkDecoder:
    def __init__(self,
                 edmodel: EDModel,
                 tgtokenizer: TemplateGenerationTokenizer,
                 grammar_file: Optional[str] = None,
                 ptr_decoding=False,
                 ptr_model=False,
                 max_length=1020):
        if ptr_model and ptr_decoding:
            print("Error: Pointer model and pointer decoding are both enabled!", flush=True)
        assert not (ptr_model and ptr_decoding)
        self.ptr_decoding = ptr_decoding
        self.ptr_model = ptr_model

        if grammar_file is None:
            grammar_file = os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources",
                                        'grammar_full.lark')

        self.edmodel = edmodel
        self.max_length = max_length
        self.grammar = open(grammar_file).read()

        #vocab_rule = "\nPOINT: /[^[]+/"
        vocab_rule = "\nPOINT: /"+self.create_not_string_regex("[start:")+"|"+self.create_not_string_regex("[end:")+"/"

        # vocab = set(self.edmodel.tokenizer.get_vocab().keys())
        # vocab.add("<")
        # vocab_rule = "\nPOINT.-1: /" + "|".join(["".join([
        #     "[" + c.replace("/", "\\/") + "]" if c != "]" else c for c in k
        # ]) for k in vocab]) + "/"

        self.grammar += vocab_rule
        # print(self.grammar)
        self.parser = lark.Lark(self.grammar, start='publication', parser='lalr')

        self.tokenizer = tgtokenizer

        self._prediction_proxy = EDPredictionProxy(self.edmodel)

        self._start_of_sentence_token_id = None
        self._end_of_sentence_token_id = None
        self._pad_token_id = None
        if self.tokenizer.start_of_sentence_token is not None:
            self._start_of_sentence_token_id = self.tokenizer.convert_tokens_to_ids(
                tokens=[self.tokenizer.start_of_sentence_token]
            )[0]
        if self.tokenizer.end_of_sentence_token is not None:
            self._end_of_sentence_token_id = self.tokenizer.convert_tokens_to_ids(
                tokens=[self.tokenizer.end_of_sentence_token]
            )[0]
        if self.tokenizer.pad_token is not None:
            self._pad_token_id = self.tokenizer.convert_tokens_to_ids(
                tokens=[self.tokenizer.pad_token]
            )[0]

    @staticmethod
    def create_not_string_regex(not_str: str):
        regex_list = []
        for i in range(len(not_str)):
            regex_list.append(not_str[:i] + "[^" + not_str[i] + "][^[]*")
        return "|".join(regex_list)

    @staticmethod
    def create_token_mask(logits: torch.Tensor, token_ids: Iterable[int]):
        mask = np.ones(logits.shape)
        # mask = torch.ones_like(logits)

        # for token_id in token_ids:
        #    mask[token_id] = 0
        mask[list(token_ids)] = 0

        mask = mask * (-1e9)

        return torch.from_numpy(mask).to(logits.device)

    def decode_document_greedy(self, document, masked_vocab=True):
        device = self.edmodel.model.device

        interactive = self.parser.parse_interactive("")

        # create list of token ids for each sentence in document
        sentences_token_ids = [self.tokenizer.convert_tokens_to_ids(sentence.get_tokens())
                               for sentence in document.get_sentences()]
        sos = [self._start_of_sentence_token_id] if self._start_of_sentence_token_id is not None else []
        eos = [self._end_of_sentence_token_id] if self._end_of_sentence_token_id is not None else []
        input_ids = [sos + sentence_token_ids + eos
                     for sentence_token_ids in sentences_token_ids]

        input_ids = list(chain(*input_ids))

        # save last encode hidden state for better decoding speed
        encoder_last_hidden_state = None

        token_ids = list()
        if self._start_of_sentence_token_id is not None:
            token_ids.append(self._start_of_sentence_token_id)
        elif isinstance(self.edmodel, FlanT5):
            token_ids.append(self._pad_token_id)

        past_key_values = None

        sequence_logits = []

        # predict tokens
        while len(token_ids) < self.max_length and interactive.accepts() != {"$END"}:
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

            mask = torch.zeros(logits.shape)

            if masked_vocab:
                accepted_tokens = interactive.accepts()
                next_possible_tokens = [x.pattern.value for x in self.parser.terminals if x.name in accepted_tokens]
                # TODO: add all tokens for POINT token also in this regex grammar version
                next_possible_token_ids = self.tokenizer.convert_tokens_to_ids(next_possible_tokens)
                if "POINT" in [x.name for x in self.parser.terminals if x.name in accepted_tokens]:
                    next_possible_token_ids = list(set(next_possible_token_ids + input_ids).difference(sos + eos))

                # mask invalid tokens
                mask = self.create_token_mask(
                    logits=encoder_decoder_prediction.token_logits,
                    token_ids=next_possible_token_ids
                )

                # times softmax not working because in "all 0" case arbitrary token like pad can be generated
                logits = final_dist + mask  # * mask.softmax(dim=-1)  # + mask

            # estimate predicted token by argmax
            token_id = logits.argmax(-1).item()
            [token] = self.tokenizer.convert_ids_to_tokens([token_id])
            chosen_logits = logits

            if (self.ptr_decoding # only if regular prediction does not predict end token
                    and not any([t.type != "POINT" for t in self.parser.lex(token)])
                    and "POINT" in interactive.accepts()):  # only if free text is allowed
                attention_dist = torch.mean(encoder_decoder_prediction.last_cross_attention, dim=1)
                tinput_ids = input_ids
                input_ids_tensor = torch.tensor(tinput_ids, dtype=torch.long, device=attention_dist.device)

                ptr_logits = (scatter(attention_dist, input_ids_tensor, dim_size=len(logits),
                                      reduce="sum").flatten() + mask).softmax(dim=-1)

                token_id = ptr_logits.argmax(-1).item()  # tinput_ids[attention_dist.argmax(-1).item()]
                [token] = self.tokenizer.convert_ids_to_tokens([token_id])
                chosen_logits = ptr_logits

            if self._end_of_sentence_token_id is None:
                print("Warning: End of sentence token is None, decoding won't stop probably!")
            if token_id == self._end_of_sentence_token_id:
                break

            sequence_logits.append(chosen_logits.softmax(dim=-1))
            token_ids.append(token_id)

            if token == '[end:Publication]': # end pub should break too but with adding the token
                break

            if masked_vocab:
                for tok in self.parser.lex(token):
                    interactive.feed_token(tok)
                # try:
                interactive.exhaust_lexer()

            # except TypeError as e:
            #    pass

        if masked_vocab and len(token_ids) >= self.max_length:
            while interactive.accepts() != {"$END"}:
                token_candidates = [x.pattern.value for x in self.parser.terminals if x.name in interactive.accepts() and x.name != "POINT" and x.pattern.value.startswith("[end:")]#
                if len(token_candidates) == 0:
                    token_candidates = [x.pattern.value for x in self.parser.terminals if x.name in interactive.accepts() and x.name != "POINT"]
                    print("No token candidate, fallback to ", token_candidates)

                if len(token_candidates) > 0:
                    token_ids.extend(self.tokenizer.convert_tokens_to_ids(token_candidates[0]))
                    for tok in self.parser.lex(token_candidates[0]):
                        interactive.feed_token(tok)
                else:  # max length reached just when POINT schould have been generated, add dummy POINT
                    token_ids.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
                    for tok in self.parser.lex(self.tokenizer.pad_token):
                        interactive.feed_token(tok)
                interactive.exhaust_lexer()

        # convert predicted token ids to tokens and return
        return self.tokenizer.convert_ids_to_tokens(token_ids + eos), token_ids + eos, sequence_logits
