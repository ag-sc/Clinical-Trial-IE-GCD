import sys

from data_handling import *
from template_lib.data_handling import import_train_test_data_from_task_config


class LEDPredictionProxy:
    def __init__(self, led_model):
        self._led_model = led_model
        self._past_key_values = None

    def predict_next_max_token(
            self,
            input_ids: List[int],
            output_ids: List[int],
            allowed_ids: Set[int] = None
    ):
        assert len(input_ids) > 0 and len(output_ids) > 0

        pos = len(output_ids) - 1
        input_ids = torch.tensor([input_ids])
        output_ids = torch.tensor([output_ids])
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            outputs = self._led_model(
                input_ids=input_ids,
                decoder_input_ids=output_ids,
                global_attention_mask=global_attention_mask,
                return_dict=True,
            )

        logits = outputs.token_logits[0, pos, :]
        return logits.argmax(-1).item(), logits.max(-1)


# arg1: task config file name
# arg2: special tokens json file name
# arg3: directory of trained model
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

    # load training data #######################################################################
    training_dataset, test_dataset = import_train_test_data_from_task_config(
        task_config_dict,
        temp_generation_tokenizer
    )

    test_data_elements = extract_data_elements_from_dataset(
        dataset=test_dataset,
        tokenizer=temp_generation_tokenizer,
        template_encoder=template_encoder,
        max_input_seq_len=model.max_encoder_position_embeddings,
        max_output_seq_len=model.max_decoder_position_embeddings,
        encode_flat=False,
        per_sentence=False
    )
    train_data_elements = extract_data_elements_from_dataset(
        dataset=test_dataset,
        tokenizer=temp_generation_tokenizer,
        template_encoder=template_encoder,
        max_input_seq_len=model.max_encoder_position_embeddings,
        max_output_seq_len=model.max_decoder_position_embeddings,
        encode_flat=False,
        per_sentence=False
    )
    print('number of test data elements: ', len(test_data_elements))

    # prediction ###############################################################################
    print('prediction...')
    output_ids = temp_generation_tokenizer.convert_tokens_to_ids([temp_generation_tokenizer.start_of_sentence_token])
    prediction_proxy = LEDPredictionProxy(model.model)

    for i in range(800):
        predicted_id, _ = prediction_proxy.predict_next_max_token(
            input_ids=test_data_elements[0].input_token_ids,
            output_ids=output_ids
        )

        token = temp_generation_tokenizer.convert_ids_to_tokens([predicted_id])[0]
        print(token)
        output_ids.append(predicted_id)
