import os
from typing import Optional

import torch

from generative_approach.utils import import_task_config, TemplateGenerationTokenizer, get_model_class
import generative_approach.evaluation as gen_eval
from template_lib.data_handling import import_train_test_valid_data_from_task_config
from template_lib.max_cardinalities_estimator import gen_max_slot_cardinalities


def single_result_gen(task_config: str, model_path: str, tokenizer_path: Optional[str] = None):
    torch.cuda.empty_cache()
    task_config_dict = import_task_config(task_config)

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

    # load training data ###############################################
    training_dataset, test_dataset, validation_dataset = import_train_test_valid_data_from_task_config(task_config_dict,
                                                                                                       temp_generation_tokenizer)

    max_slots_cardinalities = gen_max_slot_cardinalities(training_dataset)

    #pmid = temp_generation_tokenizer.tokenize("27740719")
    studydata = [(doc, tc) for doc, tc in test_dataset]
    #assert len(studydata) == 1


    #doc = [doc for doc, tc in test_dataset][0] #PMID = 27740719 temp_generation_tokenizer.tokenize(["27740719"])
    #tc = [tc for doc, tc in test_dataset][0]
    doc = studydata[0][0]
    tc = studydata[0][1]
    predicted_template_collection = gen_eval.single_run(
        document=doc,
        model=model,
        temp_generation_tokenizer=temp_generation_tokenizer,
        task_config_dict=task_config_dict,
        max_slots_cardinalities=max_slots_cardinalities,
        device=device
    )

    return tc, predicted_template_collection, temp_generation_tokenizer


def test_gen():
    model = "gen-gl-t5" #"gen-gl-led"
    test_model_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "test", model)
    res = single_result_gen(task_config=os.path.join(test_model_base_path, "config.json"),
                            model_path=os.path.join(test_model_base_path, "model"),
                            tokenizer_path=os.path.join(test_model_base_path, "tokenizer.json"))
    print(res)
